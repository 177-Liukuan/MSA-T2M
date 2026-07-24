"""Safely archive and restore selected exploration result directories."""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


@dataclass(frozen=True)
class ArchiveEntry:
    route: str
    name: str


def entry(route, name):
    return ArchiveEntry(route=route, name=name)


ARCHIVE_ENTRIES = (
    entry("clip", "MotionStreamer_t2m_272_baseline_clip"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_200000Iter"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned_addLR"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose_fix"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo_gateclose"),
    entry("cross_attention/latent_retrieval", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_6layer_top3_ddpm"),
    entry("cross_attention/latent_retrieval", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_late_after_sa_every1layer_top3_ddpm_cfg_saca_dropout01"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L4_k3"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L4_k3_crossattn"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L8_k3"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L8_k3_crossattn"),
    entry("qformer", "QFormer_t2m_272_v1"),
    entry("qformer", "QFormer_t2m_272_v2"),
    entry("qformer", "QFormer_t2m_272_v3"),
    entry("qformer", "QFormer_t2m_272_v4"),
    entry("qformer", "QFormer_t2m_272_v5"),
    entry("representation_experiments", "SAE_v1_t2m_272"),
    entry("representation_experiments", "TAE_GAN_Loss_"),
    entry("motionstreamer_baselines", "t2m_model"),
    entry("motionstreamer_baselines", "MotionStreamer_vaebyh100_t2m_h100_20260204"),
    entry("motionstreamer_baselines", "motionstreamer_model_causal_TAE_t2m_babel_272_h100_20260205_20260209"),
    entry("motionstreamer_baselines", "MotionStreamer_8gpus_distributed"),
    entry("motionstreamer_baselines", "MotionStreamer_8gpus_distributed_mp"),
    entry("motionstreamer_baselines", "MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16"),
    entry("motionstreamer_baselines", "MotionStreamer_vae_causal_TAE_t2m_272_h100_20260203_t2m_h100_20260206"),
    entry("misc", ".ipynb_checkpoints"),
)


@dataclass(frozen=True)
class TreeSnapshot:
    byte_size: int
    file_count: int
    checkpoint_names: tuple


@dataclass(frozen=True)
class MoveRecord:
    experiments_root: Path
    entry: ArchiveEntry
    source: Path
    destination: Path
    snapshot: TreeSnapshot


def snapshot_tree(path):
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return TreeSnapshot(
        byte_size=sum(candidate.stat().st_size for candidate in files),
        file_count=len(files),
        checkpoint_names=tuple(
            sorted(
                str(candidate.relative_to(path))
                for candidate in files
                if candidate.suffix in {".pth", ".pt", ".ckpt", ".safetensors"}
            )
        ),
    )


def _nearest_existing_ancestor(path):
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError("no existing destination ancestor: {}".format(path))
        candidate = parent
    return candidate


def _reject_destination_parent_symlinks(experiments_root, destination_parent):
    try:
        relative_parent = destination_parent.relative_to(experiments_root)
    except ValueError:
        raise ValueError(
            "destination parent is outside experiments root: {}".format(
                destination_parent
            )
        )

    candidate = experiments_root
    if candidate.is_symlink():
        raise OSError("destination parent symlink: {}".format(candidate))
    for component in relative_parent.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise OSError("destination parent symlink: {}".format(candidate))


def preflight(experiments_root, entries, rollback=False):
    names = [archive_entry.name for archive_entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source name in archive manifest")

    records = []
    root_device = experiments_root.stat().st_dev
    for archive_entry in entries:
        archived = (
            experiments_root / "explorations" / archive_entry.route / archive_entry.name
        )
        root_path = experiments_root / archive_entry.name
        source, destination = (
            (archived, root_path) if rollback else (root_path, archived)
        )
        if source.is_symlink():
            raise ValueError("source symlink is not an experiment directory: {}".format(source))
        if not source.is_dir():
            raise FileNotFoundError("missing source: {}".format(source))
        _reject_destination_parent_symlinks(experiments_root, destination.parent)
        if os.path.lexists(destination):
            raise FileExistsError("destination exists: {}".format(destination))
        if source.stat().st_dev != root_device:
            raise OSError("source is on a different filesystem: {}".format(source))
        destination_ancestor = _nearest_existing_ancestor(destination.parent)
        if destination_ancestor.stat().st_dev != root_device:
            raise OSError(
                "destination is on a different filesystem: {}".format(
                    destination_ancestor
                )
            )
        records.append(
            MoveRecord(
                experiments_root=experiments_root,
                entry=archive_entry,
                source=source,
                destination=destination,
                snapshot=snapshot_tree(source),
            )
        )
    return records


def apply_moves(records):
    for record in records:
        _reject_destination_parent_symlinks(
            record.experiments_root, record.destination.parent
        )
        record.destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_destination_parent_symlinks(
            record.experiments_root, record.destination.parent
        )
        if os.path.lexists(record.destination):
            raise FileExistsError(
                "destination appeared during move: {}".format(record.destination)
            )
        if record.source.is_symlink():
            raise ValueError(
                "source symlink appeared during move: {}".format(record.source)
            )
        if not record.source.is_dir():
            raise FileNotFoundError("source disappeared during move: {}".format(record.source))
        record.source.rename(record.destination)
        verify_moves([record])


def verify_moves(records):
    for record in records:
        if record.source.exists():
            raise RuntimeError("source remains after move: {}".format(record.source))
        if not record.destination.is_dir():
            raise RuntimeError(
                "destination missing after move: {}".format(record.destination)
            )
        actual = snapshot_tree(record.destination)
        if actual != record.snapshot:
            raise RuntimeError(
                "snapshot mismatch for {}: expected {}, got {}".format(
                    record.destination, record.snapshot, actual
                )
            )


def _verification_state(records, operation):
    if operation == "rollback":
        verified = all(
            record.source.is_dir()
            and not record.destination.exists()
            and snapshot_tree(record.source) == record.snapshot
            for record in records
        )
        pending = all(
            not record.source.exists()
            and record.destination.is_dir()
            and snapshot_tree(record.destination) == record.snapshot
            for record in records
        )
    else:
        verified = all(
            not record.source.exists()
            and record.destination.is_dir()
            and snapshot_tree(record.destination) == record.snapshot
            for record in records
        )
        pending = all(
            record.source.is_dir()
            and not record.destination.exists()
            and snapshot_tree(record.source) == record.snapshot
            for record in records
        )
    if verified:
        return "passed"
    if pending:
        return "pending"
    return "inconsistent"


def render_manifest(records, operation):
    state = _verification_state(records, operation)
    payload = {
        "version": 1,
        "operation": operation,
        "records": [
            {
                "route": record.entry.route,
                "name": record.entry.name,
                "source": str(record.source),
                "destination": str(record.destination),
                "snapshot": {
                    "byte_size": record.snapshot.byte_size,
                    "file_count": record.snapshot.file_count,
                    "checkpoint_names": list(record.snapshot.checkpoint_names),
                },
            }
            for record in records
        ],
    }
    lines = [
        "# Exploration results archive manifest",
        "",
        "Operation: {}".format(operation),
        "Verification: {}".format(state),
        "",
        "| Route | Source | Destination | Bytes | Files | Checkpoints |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        checkpoints = ", ".join(record.snapshot.checkpoint_names) or "-"
        lines.append(
            "| {route} | {source} | {destination} | {bytes} | {files} | {checkpoints} |".format(
                route=record.entry.route,
                source=record.source,
                destination=record.destination,
                bytes=record.snapshot.byte_size,
                files=record.snapshot.file_count,
                checkpoints=checkpoints,
            )
        )
    lines.extend(
        [
            "",
            "<!-- ARCHIVE_MANIFEST_JSON",
            json.dumps(payload, indent=2, sort_keys=True),
            "ARCHIVE_MANIFEST_JSON -->",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_manifest_path_component(value, is_route):
    if not isinstance(value, str) or not value:
        raise ValueError("unsafe manifest path component: {!r}".format(value))
    normalized_parts = value.replace("\\", "/").split("/")
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in normalized_parts)
        or (not is_route and len(normalized_parts) != 1)
    ):
        raise ValueError("unsafe manifest path component: {!r}".format(value))


def load_manifest(path, experiments_root):
    content = Path(path).read_text()
    match = re.search(
        r"<!-- ARCHIVE_MANIFEST_JSON\s*(\{.*?\})\s*ARCHIVE_MANIFEST_JSON -->",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("manifest is missing its machine-readable JSON block")
    payload = json.loads(match.group(1))
    if payload.get("version") != 1:
        raise ValueError("unsupported manifest version")
    items = payload.get("records")
    if not isinstance(items, list) or len(items) != len(ARCHIVE_ENTRIES):
        raise ValueError("manifest must contain exactly 35 archive entries")
    entries = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("manifest record must be an object")
        route = item.get("route")
        name = item.get("name")
        _validate_manifest_path_component(route, is_route=True)
        _validate_manifest_path_component(name, is_route=False)
        entries.append(ArchiveEntry(route=route, name=name))
    if set(entries) != set(ARCHIVE_ENTRIES):
        raise ValueError("manifest entries do not match immutable archive manifest")

    records = []
    for item, entry_value in zip(items, entries):
        snapshot_value = item["snapshot"]
        snapshot = TreeSnapshot(
            byte_size=snapshot_value["byte_size"],
            file_count=snapshot_value["file_count"],
            checkpoint_names=tuple(snapshot_value["checkpoint_names"]),
        )
        records.append(
            MoveRecord(
                experiments_root=experiments_root,
                entry=entry_value,
                source=experiments_root / entry_value.name,
                destination=(
                    experiments_root
                    / "explorations"
                    / entry_value.route
                    / entry_value.name
                ),
                snapshot=snapshot,
            )
        )
    return records


def _write_manifest(path, records, operation):
    Path(path).write_text(render_manifest(records, operation))


def _parser():
    parser = argparse.ArgumentParser(
        description="Safely archive selected exploration result directories."
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("Experiments"))
    parser.add_argument("--manifest", type=Path, help="Markdown manifest path")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="archive the manifest entries")
    modes.add_argument("--verify", action="store_true", help="verify an archive manifest")
    modes.add_argument("--rollback", action="store_true", help="restore an archive manifest")
    return parser


def main(argv=None):
    parser = _parser()
    arguments = parser.parse_args(argv)
    root = arguments.experiments_root

    if arguments.verify or arguments.rollback:
        if arguments.manifest is None:
            parser.error("--manifest is required with --verify or --rollback")
        records = load_manifest(arguments.manifest, root)
        if arguments.verify:
            verify_moves(records)
            print(render_manifest(records, "verify"))
            return 0

        verify_moves(records)
        rollback_records = preflight(root, [record.entry for record in records], rollback=True)
        _write_manifest(arguments.manifest, records, "rollback")
        apply_moves(rollback_records)
        verify_moves(rollback_records)
        _write_manifest(arguments.manifest, records, "rollback")
        return 0

    records = preflight(root, ARCHIVE_ENTRIES)
    if arguments.apply:
        manifest = arguments.manifest or root / "archive_exploration_results.md"
        _write_manifest(manifest, records, "archive")
        apply_moves(records)
        verify_moves(records)
        _write_manifest(manifest, records, "archive")
        return 0

    rendered = render_manifest(records, "dry-run")
    if arguments.manifest is not None:
        _write_manifest(arguments.manifest, records, "dry-run")
    print(rendered)
    return 0


if __name__ == "__main__":
    main()
