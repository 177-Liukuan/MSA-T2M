#!/usr/bin/env python3
"""Contract and result collector for the MSA-VAE alignment pilot."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
import socket
from datetime import datetime, timezone
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
    main_process_port: int
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
        29501,
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
        29502,
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
        29503,
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
        29504,
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


def _variant_by_slug(slug: str) -> PilotVariant:
    for variant in PILOT_VARIANTS:
        if variant.slug == slug:
            return variant
    raise ValueError(f"unknown pilot variant: {slug}")


def record_run_event(
    output_root: Path,
    slug: str,
    event: str,
    exit_code: int = 0,
) -> Path:
    """Atomically append one training lifecycle event to the run manifest."""
    import fcntl

    output_root = _resolve_output_root(output_root)
    variant = _variant_by_slug(slug)
    manifest_path = output_root / "run_manifest.json"
    lock_path = output_root / ".run_manifest.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    phase1_path = phase_checkpoint_path(output_root, variant, 1)
    phase2_path = phase_checkpoint_path(output_root, variant, 2)
    phase1_dir = phase1_path.parent
    phase2_dir = phase2_path.parent
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "format_version": 1,
                "git_commit": _git_commit(),
                "tae_checkpoint": str(
                    (_repo_root() / TAE_CHECKPOINT).resolve()
                ),
                "tae_sha256": TAE_SHA256,
                "seed": PILOT_SEED,
                "training_budget_per_phase": PHASE_ITERATIONS,
                "validation": {
                    "seed": VALIDATION_SEED,
                    "batch_size": VALIDATION_BATCH_SIZE,
                    "interval": EVAL_INTERVAL,
                },
                "variants": {},
            }
        run = payload["variants"].setdefault(
            slug,
            {
                "label": variant.label,
                "gpu_pair": variant.training_gpus,
                "main_process_port": variant.main_process_port,
                "weights": {
                    "phase1": {
                        "global": variant.phase1_global,
                        "local": variant.phase1_local,
                    },
                    "phase2": {
                        "global": variant.phase2_global,
                        "local": variant.phase2_local,
                    },
                },
                "phase1": {
                    "experiment_dir": str(phase1_dir),
                    "launcher": str(_repo_root() / "TRAIN_msa_vae_phase1.sh"),
                    "command": [
                        "bash",
                        "TRAIN_msa_vae_phase1.sh",
                        "2",
                        "t2m_272",
                    ],
                    "environment": {
                        "OUT_DIR": str(output_root),
                        "EXP_NAME": phase_experiment_name(variant, 1),
                        "SEED": PILOT_SEED,
                        "TOTAL_ITER": PHASE_ITERATIONS,
                        "WARM_UP_ITER": 500,
                        "EVAL_ITER": EVAL_INTERVAL,
                        "VALIDATION_SEED": VALIDATION_SEED,
                        "VALIDATION_BATCH_SIZE": VALIDATION_BATCH_SIZE,
                        "FULL_SEQ_BATCH_SIZE": 16,
                        "LENGTH_BUCKET_SIZE": 256,
                        "GLOBAL_ALIGN_WEIGHT": variant.phase1_global,
                        "LOCAL_ALIGN_WEIGHT": variant.phase1_local,
                        "MAIN_PROCESS_PORT": variant.main_process_port,
                    },
                },
                "phase2": {
                    "experiment_dir": str(phase2_dir),
                    "launcher": str(_repo_root() / "TRAIN_msa_vae_phase2.sh"),
                    "command": [
                        "bash",
                        "TRAIN_msa_vae_phase2.sh",
                        "2",
                        "t2m_272",
                    ],
                    "environment": {
                        "OUT_DIR": str(output_root),
                        "PHASE1_DIR": str(phase1_dir),
                        "EXP_NAME": phase_experiment_name(variant, 2),
                        "SEED": PILOT_SEED,
                        "TOTAL_ITER": PHASE_ITERATIONS,
                        "WARM_UP_ITER": 1000,
                        "EVAL_ITER": EVAL_INTERVAL,
                        "VALIDATION_SEED": VALIDATION_SEED,
                        "VALIDATION_BATCH_SIZE": VALIDATION_BATCH_SIZE,
                        "FULL_SEQ_BATCH_SIZE": 8,
                        "LENGTH_BUCKET_SIZE": 256,
                        "WINDOW_REPLAY_INTERVAL": 4,
                        "GLOBAL_ALIGN_WEIGHT": variant.phase2_global,
                        "LOCAL_ALIGN_WEIGHT": variant.phase2_local,
                        "MAIN_PROCESS_PORT": variant.main_process_port,
                    },
                },
                "events": [],
            },
        )
        run["events"].append(
            {
                "event": event,
                "timestamp": timestamp,
                "exit_code": exit_code,
            }
        )
        if event == "phase1_complete":
            run["phase1"].update(
                {
                    "exit_code": exit_code,
                    "completed_at": timestamp,
                    "checkpoint_sha256": _file_sha256(phase1_path),
                }
            )
        elif event == "phase2_complete":
            run["phase2"].update(
                {
                    "exit_code": exit_code,
                    "completed_at": timestamp,
                    "checkpoint_sha256": _file_sha256(phase2_path),
                }
            )
        elif event == "started":
            run["started_at"] = timestamp
        elif event == "failed":
            run["failed_at"] = timestamp
            run["exit_code"] = exit_code
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_root,
            prefix=".run_manifest.",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(manifest_path)
    return manifest_path


def check_pilot_ports() -> None:
    """Fail unless every fixed Accelerate rendezvous port is available."""
    sockets = []
    try:
        for variant in PILOT_VARIANTS:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind(("127.0.0.1", variant.main_process_port))
            except OSError as error:
                listener.close()
                raise ValueError(
                    "Accelerate port "
                    f"{variant.main_process_port} is unavailable "
                    f"for {variant.slug}"
                ) from error
            sockets.append(listener)
    finally:
        for listener in sockets:
            listener.close()


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
            variant.main_process_port,
            format(variant.phase1_global, "g"),
            format(variant.phase1_local, "g"),
            format(variant.phase2_global, "g"),
            format(variant.phase2_local, "g"),
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
    expected_full_batch = 16 if phase == 1 else 8
    expected_warm_up = 500 if phase == 1 else 1000
    expected_sequence_mode = "full" if phase == 1 else "mixed"
    if (
        training_args.get("batch_size") != 64
        or training_args.get("full_seq_batch_size")
        != expected_full_batch
        or training_args.get("warm_up_iter") != expected_warm_up
        or training_args.get("length_bucket_size") != 256
        or training_args.get("sequence_mode")
        != expected_sequence_mode
        or (
            phase == 2
            and training_args.get("window_replay_interval") != 4
        )
    ):
        raise ValueError("training schedule does not match the pilot")
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
    tae_path = training_args.get("resume_cnn_pth")
    if not isinstance(tae_path, str):
        raise ValueError("TAE identity does not match the fixed checkpoint")
    resolved_tae = Path(tae_path)
    if not resolved_tae.is_absolute():
        resolved_tae = _repo_root() / resolved_tae
    expected_tae = (_repo_root() / TAE_CHECKPOINT).resolve()
    if (
        resolved_tae.resolve() != expected_tae
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pilot_checkpoints(output_root: Path) -> List[Dict[str, Any]]:
    """Validate four real Phase-1/Phase-2 checkpoint lineages."""
    import torch

    output_root = _resolve_output_root(output_root)
    results = []
    for variant in PILOT_VARIANTS:
        parent_path = phase_checkpoint_path(output_root, variant, 1)
        phase2_path = phase_checkpoint_path(output_root, variant, 2)
        if not parent_path.is_file() or parent_path.stat().st_size < 1:
            raise ValueError(
                f"missing Phase 1 net_last.pth for {variant.slug}: "
                f"{parent_path}"
            )
        if not phase2_path.is_file() or phase2_path.stat().st_size < 1:
            raise ValueError(
                f"missing Phase 2 net_last.pth for {variant.slug}: "
                f"{phase2_path}"
            )
        parent_payload = torch.load(
            parent_path,
            map_location="cpu",
            weights_only=False,
        )
        phase2_payload = torch.load(
            phase2_path,
            map_location="cpu",
            weights_only=False,
        )
        parent_metadata = (
            parent_payload.get("metadata")
            if isinstance(parent_payload, Mapping)
            else None
        )
        metadata = (
            phase2_payload.get("metadata")
            if isinstance(phase2_payload, Mapping)
            else None
        )
        if (
            not isinstance(parent_metadata, Mapping)
            or parent_metadata.get("phase") != 1
            or parent_metadata.get("sequence_mode") != "full"
        ):
            raise ValueError(
                f"Phase 1 checkpoint metadata is invalid for {variant.slug}"
            )
        _validate_training_args(
            parent_metadata.get("training_args"),
            variant,
            phase=1,
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("phase") != 2
            or metadata.get("sequence_mode") != "mixed"
        ):
            raise ValueError(
                f"Phase 2 checkpoint metadata is invalid for {variant.slug}"
            )
        _validate_training_args(
            metadata.get("training_args"),
            variant,
            phase=2,
        )
        lineage = metadata.get("lineage")
        embedded_parent = (
            lineage.get("parent_checkpoint_metadata")
            if isinstance(lineage, Mapping)
            else None
        )
        embedded_path = (
            lineage.get("parent_checkpoint_path")
            if isinstance(lineage, Mapping)
            else None
        )
        if embedded_parent != parent_metadata:
            raise ValueError(
                f"Phase 1 lineage metadata mismatch for {variant.slug}"
            )
        if (
            not isinstance(embedded_path, str)
            or Path(embedded_path).resolve() != parent_path.resolve()
        ):
            raise ValueError(
                f"Phase 1 lineage path mismatch for {variant.slug}"
            )
        results.append(
            {
                "slug": variant.slug,
                "phase1_path": str(parent_path),
                "phase2_path": str(phase2_path),
                "phase1_sha256": _file_sha256(parent_path),
                "phase2_sha256": _file_sha256(phase2_path),
            }
        )
    return results


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
    output_root = _resolve_output_root(output_root)
    checkpoint_records = validate_pilot_checkpoints(output_root)
    manifests = _load_pilot_manifests(output_root)
    for manifest, variant, checkpoint in zip(
        manifests,
        PILOT_VARIANTS,
        checkpoint_records,
    ):
        _validate_variant_manifest(manifest, variant)
        manifest_path = _nested(
            manifest,
            ("checkpoint", "path"),
            "checkpoint path",
        )
        if Path(manifest_path).resolve() != Path(
            checkpoint["phase2_path"]
        ).resolve():
            raise ValueError(
                f"checkpoint path mismatch for {variant.slug}"
            )
        manifest_sha = _nested(
            manifest,
            ("checkpoint", "sha256"),
            "checkpoint SHA-256",
        )
        if manifest_sha != checkpoint["phase2_sha256"]:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {variant.slug}"
            )

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
    subparsers.add_parser("verify")
    subparsers.add_parser("collect")
    subparsers.add_parser("check-ports")
    record = subparsers.add_parser("record-run")
    record.add_argument("--slug", required=True)
    record.add_argument(
        "--event",
        required=True,
        choices=(
            "started",
            "phase1_complete",
            "phase2_complete",
            "failed",
        ),
    )
    record.add_argument("--exit-code", type=int, default=0)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    output_root = _resolve_output_root(args.output_root)
    if args.command == "contract":
        print(emit_contract(output_root, args.format), end="")
        return
    if args.command == "verify":
        print(
            json.dumps(
                validate_pilot_checkpoints(output_root),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "record-run":
        print(
            record_run_event(
                output_root,
                args.slug,
                args.event,
                args.exit_code,
            )
        )
        return
    if args.command == "check-ports":
        check_pilot_ports()
        return
    paths = write_pilot_table(output_root)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
