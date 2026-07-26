#!/usr/bin/env python3
"""Aggregate three compatible MSA-VAE evaluation manifests."""

import argparse
import csv
import json
import math
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

    checkpoint_shas = [
        _nested(item, ("checkpoint", "sha256"), "checkpoint")
        for item in manifests
    ]
    if any(not isinstance(sha, str) or not sha for sha in checkpoint_shas):
        raise ValueError("checkpoint SHA-256 values must be non-empty strings")
    if len(set(checkpoint_shas)) != 3:
        raise ValueError("checkpoint SHA-256 values must be distinct")

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
    for seed, manifest in sorted(zip(seeds, manifests), key=lambda pair: pair[0]):
        sources.append(
            {
                "seed": seed,
                "checkpoint_path": _nested(
                    manifest,
                    ("checkpoint", "path"),
                    "checkpoint",
                ),
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
        "protocol": protocols[0],
        "evaluator_sha256": evaluator_sha,
        "dataset": dataset_identity,
        "skating": skating,
        "model_config": model_values,
        "alignment_weights": alignment_weights,
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
