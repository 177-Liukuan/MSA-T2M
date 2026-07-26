#!/usr/bin/env python3
"""Contract and result collector for the MSA-VAE alignment pilot."""

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PILOT_SEED = 123
PHASE_ITERATIONS = 25000
EVAL_INTERVAL = 5000
VALIDATION_SEED = 123
VALIDATION_BATCH_SIZE = 32
TAE_CHECKPOINT = (
    "Experiments/causal_TAE_t2m_272_h100_20260203/"
    "net_best_mpjpe.pth"
)
TAE_SHA256 = (
    "7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8"
)
DEFAULT_OUTPUT_ROOT = (
    "Experiments/msa_vae_alignment_realism_pilot_s123_20260726"
)
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


@dataclass(frozen=True)
class PilotVariant:
    label: str
    slug: str
    training_gpus: str
    evaluation_gpu: str
    screen_session: str
    phase1_global: float
    phase1_local: float
    phase2_global: float
    phase2_local: float


PILOT_VARIANTS = (
    PilotVariant(
        "No Alignment",
        "no_align",
        "0,1",
        "0",
        "msa_pilot_no_align_s123",
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    PilotVariant(
        "Global Only",
        "global_only",
        "2,3",
        "2",
        "msa_pilot_global_s123",
        0.2,
        0.0,
        0.05,
        0.0,
    ),
    PilotVariant(
        "Local Only",
        "local_only",
        "4,5",
        "4",
        "msa_pilot_local_s123",
        0.0,
        0.2,
        0.0,
        0.05,
    ),
    PilotVariant(
        "Global + Local",
        "global_local",
        "6,7",
        "6",
        "msa_pilot_both_s123",
        0.2,
        0.2,
        0.05,
        0.05,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_root(value: Any) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _weight_name(value: float) -> str:
    return format(float(value), "g").replace(".", "p")


def phase_experiment_name(variant: PilotVariant, phase: int) -> str:
    if phase == 1:
        global_weight = variant.phase1_global
        local_weight = variant.phase1_local
    elif phase == 2:
        global_weight = variant.phase2_global
        local_weight = variant.phase2_local
    else:
        raise ValueError("phase must be 1 or 2")
    return (
        f"{variant.slug}_s{PILOT_SEED}_phase{phase}_25k_"
        f"g{_weight_name(global_weight)}_l{_weight_name(local_weight)}"
    )


def phase_checkpoint_path(
    output_root: Path,
    variant: PilotVariant,
    phase: int,
) -> Path:
    return (
        Path(output_root)
        / phase_experiment_name(variant, phase)
        / "net_last.pth"
    )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def contract_payload(output_root: Path) -> Dict[str, Any]:
    output_root = _resolve_output_root(output_root)
    variants = []
    for variant in PILOT_VARIANTS:
        item = asdict(variant)
        item["phase1_experiment"] = phase_experiment_name(variant, 1)
        item["phase2_experiment"] = phase_experiment_name(variant, 2)
        variants.append(item)
    return {
        "format_version": 1,
        "qualification": "single-seed pilot",
        "git_commit": _git_commit(),
        "output_root": str(output_root),
        "seed": PILOT_SEED,
        "phase_iterations": PHASE_ITERATIONS,
        "eval_interval": EVAL_INTERVAL,
        "validation_seed": VALIDATION_SEED,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "tae_checkpoint": TAE_CHECKPOINT,
        "tae_sha256": TAE_SHA256,
        "variants": variants,
    }


def emit_contract(output_root: Path, format_name: str) -> str:
    if format_name == "json":
        return json.dumps(
            contract_payload(output_root),
            indent=2,
            sort_keys=True,
        ) + "\n"
    if format_name != "tsv":
        raise ValueError("contract format must be json or tsv")
    lines = []
    for variant in PILOT_VARIANTS:
        fields = (
            variant.slug,
            variant.label,
            variant.training_gpus,
            variant.evaluation_gpu,
            variant.screen_session,
            variant.phase1_global,
            variant.phase1_local,
            variant.phase2_global,
            variant.phase2_local,
        )
        lines.append("\t".join(str(value) for value in fields))
    return "\n".join(lines) + "\n"


def _nested(value: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing {label}: {'.'.join(keys)}")
        current = current[key]
    return current


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_equal(values: Sequence[Any], label: str) -> Any:
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} mismatch across variants")
    return values[0]


def _validate_training_args(
    training_args: Any,
    variant: PilotVariant,
    phase: int,
) -> None:
    if not isinstance(training_args, Mapping):
        raise ValueError("training configuration is missing")
    expected_global = (
        variant.phase1_global if phase == 1 else variant.phase2_global
    )
    expected_local = (
        variant.phase1_local if phase == 1 else variant.phase2_local
    )
    if (
        _finite(
            training_args.get("global_align_weight"),
            "alignment weights",
        )
        != expected_global
        or _finite(
            training_args.get("local_align_weight"),
            "alignment weights",
        )
        != expected_local
    ):
        raise ValueError(f"alignment weights mismatch for {variant.slug}")
    if training_args.get("seed") != PILOT_SEED:
        raise ValueError("training seed must be 123")
    if (
        training_args.get("total_iter") != PHASE_ITERATIONS
        or training_args.get("eval_iter") != EVAL_INTERVAL
    ):
        raise ValueError("training budget does not match the pilot")
    if (
        training_args.get("validation_seed") != VALIDATION_SEED
        or training_args.get("validation_batch_size")
        != VALIDATION_BATCH_SIZE
    ):
        raise ValueError("validation identity does not match the pilot")
    if (
        training_args.get("num_gpus") != 2
        or training_args.get("use_ft_split") is not False
        or training_args.get("msa_data_mode") != "humanml_full"
    ):
        raise ValueError("training configuration does not match the pilot")
    if (
        training_args.get("resume_cnn_pth") != TAE_CHECKPOINT
        or training_args.get("resume_cnn_sha256") != TAE_SHA256
    ):
        raise ValueError("TAE identity does not match the fixed checkpoint")


def _validate_variant_manifest(
    manifest: Mapping[str, Any],
    variant: PilotVariant,
) -> Dict[str, float]:
    if manifest.get("seed") != PILOT_SEED:
        raise ValueError("evaluation seed must be 123")
    if manifest.get("batch_size") != VALIDATION_BATCH_SIZE:
        raise ValueError("evaluation batch size must be 32")
    if _nested(manifest, ("protocol", "version"), "protocol") != (
        PROTOCOL_VERSION
    ):
        raise ValueError(f"protocol must be {PROTOCOL_VERSION}")

    checkpoint_path = _nested(
        manifest,
        ("checkpoint", "path"),
        "checkpoint",
    )
    if (
        not isinstance(checkpoint_path, str)
        or Path(checkpoint_path).name != "net_last.pth"
    ):
        raise ValueError("best checkpoint is forbidden; use net_last.pth")
    metadata = _nested(
        manifest,
        ("checkpoint", "metadata"),
        "checkpoint metadata",
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("phase") != 2
        or metadata.get("sequence_mode") != "mixed"
        or metadata.get("normalized_loss_version") != 1
    ):
        raise ValueError("Phase 2 checkpoint metadata is invalid")
    _validate_training_args(metadata.get("training_args"), variant, phase=2)

    lineage = metadata.get("lineage")
    parent = (
        lineage.get("parent_checkpoint_metadata")
        if isinstance(lineage, Mapping)
        else None
    )
    parent_path = (
        lineage.get("parent_checkpoint_path")
        if isinstance(lineage, Mapping)
        else None
    )
    if (
        not isinstance(parent, Mapping)
        or parent.get("phase") != 1
        or parent.get("sequence_mode") != "full"
        or not isinstance(parent_path, str)
        or Path(parent_path).name != "net_last.pth"
    ):
        raise ValueError("Phase 1 lineage is invalid")
    _validate_training_args(parent.get("training_args"), variant, phase=1)

    metrics = _nested(manifest, ("metrics",), "metric")
    if not isinstance(metrics, Mapping):
        raise ValueError("metric dictionary is missing")
    return {
        name: _finite(metrics.get(name), f"metric {name}")
        for name in TARGET_METRICS
    }


def _load_pilot_manifests(output_root: Path) -> List[Dict[str, Any]]:
    output_root = _resolve_output_root(output_root)
    manifests = []
    for variant in PILOT_VARIANTS:
        path = output_root / "evaluation" / variant.slug / "metrics.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"missing metrics manifest for {variant.slug}: {path}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object")
        manifests.append(value)
    return manifests


def validate_pilot_manifests(output_root: Path) -> List[Dict[str, Any]]:
    manifests = _load_pilot_manifests(output_root)
    for manifest, variant in zip(manifests, PILOT_VARIANTS):
        _validate_variant_manifest(manifest, variant)

    common_identities = (
        (("protocol",), "protocol"),
        (("evaluator", "sha256"), "evaluator"),
        (("dataset", "sample_hash"), "dataset"),
        (("dataset", "sample_count"), "dataset"),
        (("skating",), "skating"),
        (("model_config", "values"), "model"),
        (("seed",), "evaluation seed"),
        (("batch_size",), "evaluation batch size"),
    )
    for keys, label in common_identities:
        _require_equal(
            [_nested(manifest, keys, label) for manifest in manifests],
            label,
        )
    sample_count = _nested(
        manifests[0],
        ("dataset", "sample_count"),
        "dataset",
    )
    if sample_count != 2480:
        raise ValueError(
            f"dataset sample count must be 2480, got {sample_count}"
        )
    return manifests


def write_pilot_table(output_root: Path) -> Dict[str, Path]:
    output_root = _resolve_output_root(output_root)
    manifests = validate_pilot_manifests(output_root)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_rows = []
    for variant, manifest in zip(PILOT_VARIANTS, manifests):
        metrics = _validate_variant_manifest(manifest, variant)
        row = {
            "Variant": variant.label,
            "seed": str(PILOT_SEED),
        }
        row.update(
            {
                header: metrics[name]
                for header, name in zip(TABLE_HEADERS[1:], TARGET_METRICS)
            }
        )
        rows.append(row)
        json_rows.append(
            {
                "variant": variant.label,
                "slug": variant.slug,
                "seed": PILOT_SEED,
                "checkpoint": manifest["checkpoint"],
                "metrics": metrics,
            }
        )

    json_path = summary_dir / "pilot_table.json"
    csv_path = summary_dir / "pilot_table.csv"
    markdown_path = summary_dir / "pilot_table.md"
    payload = {
        "qualification": "single-seed pilot; no uncertainty estimate",
        "seed_count": 1,
        "seed": PILOT_SEED,
        "protocol": manifests[0]["protocol"],
        "dataset": manifests[0]["dataset"],
        "rows": json_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = ("Variant", "seed") + TABLE_HEADERS[1:]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_rows = []
    for row in rows:
        values = [row["Variant"]]
        values.extend(f"{float(row[header]):.3f}" for header in TABLE_HEADERS[1:])
        markdown_rows.append("| " + " | ".join(values) + " |")
    markdown = (
        "| " + " | ".join(TABLE_HEADERS) + " |\n"
        "| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |\n"
        + "\n".join(markdown_rows)
        + "\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--format", choices=("json", "tsv"), default="json")
    subparsers.add_parser("collect")
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    output_root = _resolve_output_root(args.output_root)
    if args.command == "contract":
        print(emit_contract(output_root, args.format), end="")
        return
    paths = write_pilot_table(output_root)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
