#!/usr/bin/env python3
"""Aggregate three compatible MSA-VAE evaluation manifests."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROTOCOL_VERSION = "msa-vae-standard-v2"
TARGET_METRICS = (
    "fid",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "accel_mm_per_frame2",
    "skating_percent",
    "t2m_r1_percent",
    "t2m_r5_percent",
    "t2m_medr",
    "m2t_r1_percent",
    "m2t_r5_percent",
    "m2t_medr",
)
TABLE_HEADERS = (
    "Variant",
    "FID↓",
    "MPJPE↓",
    "P-MPJPE↓",
    "ACCEL↓",
    "Skating%↓",
    "T2M R@1↑",
    "T2M R@5↑",
    "T2M MedR↓",
    "M2T R@1↑",
    "M2T R@5↑",
    "M2T MedR↓",
)
TRAINING_RUN_SPECIFIC_FIELDS = frozenset(
    (
        "seed",
        "exp_name",
        "resume_cnn_pth",
        "resume_cnn_sha256",
    )
)
CHECKPOINT_METADATA_FIELDS = (
    "format_version",
    "phase",
    "sequence_mode",
    "window_size",
    "full_seq_batch_size",
    "window_replay_interval",
    "down_t",
    "stride_t",
    "unit_length",
    "latent_dim",
    "normalized_loss_version",
)


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load one evaluator output and require a JSON object at the root."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return manifest


def _nested(manifest: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    value: Any = manifest
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing {label}: {'.'.join(keys)}")
        value = value[key]
    return value


def _same_identity(
    manifests: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    label: str,
) -> Any:
    values = [_nested(manifest, keys, label) for manifest in manifests]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} mismatch across manifests")
    return values[0]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _canonical_training_configuration(metadata: Any) -> Dict[str, Any]:
    """Remove only per-run identity while retaining scientific settings."""
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint training configuration is missing")
    configuration = {
        field: _nested(metadata, (field,), "training configuration")
        for field in CHECKPOINT_METADATA_FIELDS
    }
    training_args = _nested(
        metadata,
        ("training_args",),
        "training configuration",
    )
    if not isinstance(training_args, Mapping):
        raise ValueError("checkpoint training configuration is missing")
    configuration["training_args"] = {
        key: value
        for key, value in training_args.items()
        if key not in TRAINING_RUN_SPECIFIC_FIELDS
    }

    lineage = metadata.get("lineage")
    if lineage is not None:
        if not isinstance(lineage, Mapping):
            raise ValueError("checkpoint training configuration is invalid")
        parent = lineage.get("parent_checkpoint_metadata")
        configuration["parent_checkpoint"] = (
            _canonical_training_configuration(parent)
        )
    return configuration


def _valid_internal_validation_identity(training_args: Any) -> bool:
    """Return whether deterministic complete-validation settings are usable."""
    if not isinstance(training_args, Mapping):
        return False
    validation_seed = training_args.get("validation_seed")
    validation_batch_size = training_args.get("validation_batch_size")
    return (
        isinstance(validation_seed, int)
        and not isinstance(validation_seed, bool)
        and isinstance(validation_batch_size, int)
        and not isinstance(validation_batch_size, bool)
        and validation_batch_size > 0
    )


def _validate_official_two_stage_protocol(metadata: Any) -> None:
    """Require the final HumanML result protocol approved for the paper."""
    if not isinstance(metadata, Mapping):
        raise ValueError("official two-stage protocol metadata is missing")
    training_args = metadata.get("training_args")
    lineage = metadata.get("lineage")
    if not isinstance(training_args, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("official two-stage protocol lineage is missing")
    parent_path = lineage.get("parent_checkpoint_path")
    parent = lineage.get("parent_checkpoint_metadata")
    eval_iter = training_args.get("eval_iter")
    if (
        metadata.get("phase") != 2
        or metadata.get("sequence_mode") != "mixed"
        or training_args.get("msa_data_mode") != "humanml_full"
        or training_args.get("use_ft_split") is not False
        or training_args.get("num_gpus") != 2
        or not _valid_internal_validation_identity(training_args)
        or isinstance(eval_iter, bool)
        or not isinstance(eval_iter, int)
        or eval_iter < 1
        or not isinstance(parent_path, str)
        or not parent_path
        or not isinstance(parent, Mapping)
    ):
        raise ValueError("checkpoint does not follow official two-stage protocol")
    parent_args = parent.get("training_args")
    parent_eval_iter = (
        parent_args.get("eval_iter")
        if isinstance(parent_args, Mapping)
        else None
    )
    if (
        parent.get("phase") != 1
        or parent.get("sequence_mode") != "full"
        or not isinstance(parent_args, Mapping)
        or parent_args.get("msa_data_mode") != "humanml_full"
        or parent_args.get("use_ft_split") is not False
        or parent_args.get("num_gpus") != 2
        or not _valid_internal_validation_identity(parent_args)
        or isinstance(parent_eval_iter, bool)
        or not isinstance(parent_eval_iter, int)
        or parent_eval_iter < 1
    ):
        raise ValueError("checkpoint does not follow official two-stage protocol")


def aggregate_variant(
    variant: str,
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate and aggregate exactly three independently trained seeds."""
    if len(manifests) != 3:
        raise ValueError("exactly three evaluation manifests are required")
    if not variant:
        raise ValueError("variant must be non-empty")
    if any(not isinstance(manifest, Mapping) for manifest in manifests):
        raise ValueError("each manifest must be a JSON object")

    protocols = [_nested(item, ("protocol",), "protocol") for item in manifests]
    versions = [
        _nested(item, ("protocol", "version"), "protocol")
        for item in manifests
    ]
    if any(version != PROTOCOL_VERSION for version in versions):
        raise ValueError(
            f"protocol version must be {PROTOCOL_VERSION} for every manifest"
        )
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("protocol mismatch across manifests")

    evaluator_sha = _same_identity(
        manifests,
        ("evaluator", "sha256"),
        "evaluator",
    )
    dataset_identity = {
        "sample_hash": _same_identity(
            manifests,
            ("dataset", "sample_hash"),
            "dataset",
        ),
        "sample_count": _same_identity(
            manifests,
            ("dataset", "sample_count"),
            "dataset",
        ),
    }
    skating = _same_identity(manifests, ("skating",), "skating")
    model_values = _same_identity(
        manifests,
        ("model_config", "values"),
        "model",
    )
    evaluation_seed = _same_identity(
        manifests,
        ("seed",),
        "evaluation seed",
    )
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
        raise ValueError("evaluation seed must be an integer")
    evaluation_batch_size = _same_identity(
        manifests,
        ("batch_size",),
        "evaluation batch size",
    )
    if (
        isinstance(evaluation_batch_size, bool)
        or not isinstance(evaluation_batch_size, int)
        or evaluation_batch_size < 1
    ):
        raise ValueError("evaluation batch size must be a positive integer")

    checkpoint_shas = [
        _nested(item, ("checkpoint", "sha256"), "checkpoint")
        for item in manifests
    ]
    if any(not isinstance(sha, str) or not sha for sha in checkpoint_shas):
        raise ValueError("checkpoint SHA-256 values must be non-empty strings")
    if len(set(checkpoint_shas)) != 3:
        raise ValueError("checkpoint SHA-256 values must be distinct")

    checkpoint_paths = [
        _nested(item, ("checkpoint", "path"), "checkpoint")
        for item in manifests
    ]
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).name != "net_last.pth"
        for path in checkpoint_paths
    ):
        raise ValueError(
            "formal aggregation requires each Phase 2 checkpoint "
            "to be net_last.pth"
        )

    seeds = [
        _nested(
            item,
            ("checkpoint", "metadata", "training_args", "seed"),
            "training seeds",
        )
        for item in manifests
    ]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("training seeds must be integers")
    if len(set(seeds)) != 3:
        raise ValueError("training seeds must be distinct")

    alignment_weights = {}
    for key in ("global_align_weight", "local_align_weight"):
        values = [
            _finite_number(
                _nested(
                    item,
                    ("checkpoint", "metadata", "training_args", key),
                    "alignment weights",
                ),
                f"alignment weights ({key})",
            )
            for item in manifests
        ]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"alignment weights mismatch for {key}")
        alignment_weights[key] = values[0]

    tae_path = _same_identity(
        manifests,
        (
            "checkpoint",
            "metadata",
            "training_args",
            "resume_cnn_pth",
        ),
        "TAE",
    )
    tae_sha256 = _same_identity(
        manifests,
        (
            "checkpoint",
            "metadata",
            "training_args",
            "resume_cnn_sha256",
        ),
        "TAE",
    )
    if not isinstance(tae_path, str) or not tae_path:
        raise ValueError("TAE checkpoint path must be a non-empty string")
    if (
        not isinstance(tae_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", tae_sha256) is None
    ):
        raise ValueError("TAE checkpoint SHA-256 must be 64 hexadecimal characters")

    training_configurations = [
        _canonical_training_configuration(
            _nested(
                item,
                ("checkpoint", "metadata"),
                "training configuration",
            )
        )
        for item in manifests
    ]
    if any(
        configuration != training_configurations[0]
        for configuration in training_configurations[1:]
    ):
        raise ValueError("training configuration mismatch across manifests")
    for item in manifests:
        _validate_official_two_stage_protocol(
            _nested(
                item,
                ("checkpoint", "metadata"),
                "official two-stage protocol",
            )
        )

    aggregated_metrics = {}
    for name in TARGET_METRICS:
        values = []
        for item in manifests:
            metrics = _nested(item, ("metrics",), "metric")
            if not isinstance(metrics, Mapping) or name not in metrics:
                raise ValueError(f"metric {name} is missing")
            values.append(_finite_number(metrics[name], f"metric {name}"))
        aggregated_metrics[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
        }

    sources = []
    for seed, manifest, checkpoint_path in sorted(
        zip(seeds, manifests, checkpoint_paths),
        key=lambda item: item[0],
    ):
        sources.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": _nested(
                    manifest,
                    ("checkpoint", "sha256"),
                    "checkpoint",
                ),
            }
        )

    return {
        "variant": variant,
        "seed_count": 3,
        "seeds": sorted(seeds),
        "evaluation_seed": evaluation_seed,
        "evaluation_batch_size": evaluation_batch_size,
        "protocol": protocols[0],
        "evaluator_sha256": evaluator_sha,
        "dataset": dataset_identity,
        "skating": skating,
        "model_config": model_values,
        "alignment_weights": alignment_weights,
        "training_configuration": training_configurations[0],
        "tae_checkpoint": {
            "path": tae_path,
            "sha256": tae_sha256.lower(),
        },
        "metrics": aggregated_metrics,
        "sources": sources,
    }


def write_aggregate_artifacts(
    result: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write machine-readable JSON/CSV and the requested paper table row."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "aggregate.json"
    csv_path = output_dir / "aggregate.csv"
    markdown_path = output_dir / "table.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    row = {
        "variant": result["variant"],
        "seed_count": result["seed_count"],
        "seeds": ",".join(str(seed) for seed in result["seeds"]),
    }
    for name in TARGET_METRICS:
        row[f"{name}_mean"] = result["metrics"][name]["mean"]
        row[f"{name}_std"] = result["metrics"][name]["std"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    values = [str(result["variant"])]
    for name in TARGET_METRICS:
        metric = result["metrics"][name]
        values.append(f'{metric["mean"]:.3f} ± {metric["std"]:.3f}')
    separator = ["---"] * len(TABLE_HEADERS)
    markdown = (
        "| " + " | ".join(TABLE_HEADERS) + " |\n"
        "| " + " | ".join(separator) + " |\n"
        "| " + " | ".join(values) + " |\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate exactly three compatible MSA-VAE metrics.json files."
        )
    )
    parser.add_argument("metrics_json", nargs=3, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifests = [load_manifest(path) for path in args.metrics_json]
    result = aggregate_variant(args.variant, manifests)
    paths = write_aggregate_artifacts(result, args.output_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
