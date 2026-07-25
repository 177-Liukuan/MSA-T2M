"""Distributed reconstruction validation for the BABEL MSA-VAE stream.

This evaluator intentionally has no dependency on the HumanML3D TMR models.
It validates the reconstruction and semantic objectives that are actually
supervised by the BABEL local-action stream.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.eval_trans import calculate_mpjpe, recover_from_local_position


@dataclass(frozen=True)
class BabelEvalResult:
    mpjpe: float
    reconstruction: float
    kl: float
    latent: float
    local_cosine: float
    local_loss: float
    global_coverage: float
    local_coverage: float
    semantic_objective: float
    best_semantic: float
    best_mpjpe: float


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _resolved_file(path, label):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, resolved))
    return resolved


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_identity(path, label):
    manifest_path = _resolved_file(path, label)
    return str(manifest_path), _file_sha256(manifest_path)


def _normalization_identity(path, label, require_nonzero=False):
    resolved = _resolved_file(path, label)
    try:
        values = np.load(str(resolved), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("cannot load {}: {}".format(label, error)) from error
    if values.shape != (272,):
        raise ValueError("{} must have shape (272,)".format(label))
    if require_nonzero and np.any(values == 0):
        raise ValueError("{} must not contain zero".format(label))
    return str(resolved), _file_sha256(resolved)


def _canonical_recorded_path(path):
    return str(Path(path).expanduser().resolve())


def _validated_sha256(value, label):
    normalized = "" if value is None else str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("{} must be an exact SHA-256 hex digest".format(label))
    return normalized


def build_msa_checkpoint_metadata(
    args,
    include_train_manifest=True,
    causal_tae_identity=None,
):
    """Build reproducibility metadata without changing checkpoint model keys."""
    mean_path, mean_sha256 = _normalization_identity(
        args.msa_mean_path, "MSA mean"
    )
    std_path, std_sha256 = _normalization_identity(
        args.msa_std_path, "MSA std", require_nonzero=True
    )
    mode = str(args.msa_data_mode)
    if mode not in ("humanml_full", "babel_sparse_global"):
        raise ValueError("unsupported MSA data mode: {}".format(mode))

    train_manifest_path = None
    train_manifest_sha256 = None
    validation_manifest_path = None
    validation_manifest_sha256 = None
    if mode == "babel_sparse_global":
        if include_train_manifest:
            train_manifest_path, train_manifest_sha256 = _manifest_identity(
                args.babel_train_cache_manifest,
                "BABEL training cache manifest",
            )
        validation_manifest_path, validation_manifest_sha256 = _manifest_identity(
            args.babel_val_cache_manifest,
            "BABEL validation cache manifest",
        )

    causal_path = None
    causal_sha256 = None
    if causal_tae_identity is not None:
        causal_path = _canonical_recorded_path(causal_tae_identity["path"])
        causal_sha256 = _validated_sha256(
            causal_tae_identity["sha256"], "Causal TAE artifact SHA-256"
        )

    return {
        "msa_data_mode": mode,
        "mean_path": mean_path,
        "mean_sha256": mean_sha256,
        "std_path": std_path,
        "std_sha256": std_sha256,
        "train_cache_manifest_path": train_manifest_path,
        "train_cache_manifest_sha256": train_manifest_sha256,
        "val_cache_manifest_path": validation_manifest_path,
        "val_cache_manifest_sha256": validation_manifest_sha256,
        "causal_tae_artifact_path": causal_path,
        "causal_tae_artifact_sha256": causal_sha256,
        "global_align_weight": float(args.global_align_weight),
        "local_align_weight": float(args.local_align_weight),
        "phase": int(args.phase),
        "training_args": _json_safe(vars(args)),
    }


def _metadata_manifest_value(metadata, split, suffix):
    key = "{}_cache_manifest_{}".format(split, suffix)
    value = metadata.get(key)
    if value is None and split == "val":
        # Task 4's first checkpoint generation used a validation-only alias.
        value = metadata.get("cache_manifest_{}".format(suffix))
    return value


def validate_msa_checkpoint_metadata(
    metadata,
    args,
    scope="standalone",
    expected_causal_sha256=None,
):
    """Reject checkpoints whose data/normalization identity differs."""
    if scope not in ("standalone", "training"):
        raise ValueError("unsupported checkpoint validation scope: {}".format(scope))
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no MSA metadata")
    expected = build_msa_checkpoint_metadata(
        args, include_train_manifest=(scope == "training")
    )
    if metadata.get("msa_data_mode") != expected["msa_data_mode"]:
        raise ValueError(
            "checkpoint data mode {!r} does not match requested {!r}".format(
                metadata.get("msa_data_mode"), expected["msa_data_mode"]
            )
        )
    for key, label in (("mean_path", "mean path"), ("std_path", "std path")):
        actual_path = metadata.get(key)
        if actual_path is None or str(Path(actual_path).expanduser().resolve()) != expected[key]:
            raise ValueError("checkpoint {} does not match requested asset".format(label))
        hash_key = key.replace("_path", "_sha256")
        actual_sha256 = metadata.get(hash_key)
        if scope == "training" and actual_sha256 is None:
            raise ValueError("checkpoint {} identity is missing".format(label))
        if actual_sha256 is not None and actual_sha256 != expected[hash_key]:
            raise ValueError("checkpoint {} identity does not match requested asset".format(label))
    if expected["msa_data_mode"] == "babel_sparse_global":
        if scope == "training" and expected_causal_sha256 is None:
            expected_causal_sha256 = _validated_sha256(
                getattr(args, "resume_cnn_sha256", None),
                "approved Causal TAE SHA-256",
            )
        splits = ("train", "val") if scope == "training" else ("val",)
        for split in splits:
            actual_path = _metadata_manifest_value(metadata, split, "path")
            expected_path = expected["{}_cache_manifest_path".format(split)]
            if (
                actual_path is None
                or str(Path(actual_path).expanduser().resolve()) != expected_path
            ):
                raise ValueError(
                    "checkpoint {} cache manifest path does not match requested cache".format(
                        split
                    )
                )
            actual_sha256 = _metadata_manifest_value(metadata, split, "sha256")
            if actual_sha256 != expected[
                "{}_cache_manifest_sha256".format(split)
            ]:
                raise ValueError(
                    "checkpoint {} cache manifest identity does not match requested cache".format(
                        split
                    )
                )
        if scope == "training":
            causal_path = metadata.get("causal_tae_artifact_path")
            causal_sha256 = metadata.get("causal_tae_artifact_sha256")
            if causal_path is None or causal_sha256 is None:
                raise ValueError(
                    "checkpoint has no inherited approved Causal TAE identity"
                )
            if _canonical_recorded_path(causal_path) != causal_path:
                raise ValueError("checkpoint Causal TAE path is not canonical")
            causal_sha256 = _validated_sha256(
                causal_sha256, "checkpoint Causal TAE SHA-256"
            )
            if (
                expected_causal_sha256 is not None
                and causal_sha256 != expected_causal_sha256
            ):
                raise ValueError(
                    "checkpoint inherited Causal TAE identity does not match "
                    "the approved SHA-256"
                )


def _training_data_identity(metadata):
    return {
        key: metadata.get(key)
        for key in (
            "msa_data_mode",
            "mean_path",
            "mean_sha256",
            "std_path",
            "std_sha256",
            "train_cache_manifest_path",
            "train_cache_manifest_sha256",
            "val_cache_manifest_path",
            "val_cache_manifest_sha256",
            "causal_tae_artifact_path",
            "causal_tae_artifact_sha256",
            "resume_checkpoint_path",
            "resume_checkpoint_sha256",
        )
    }


def _gather_envelopes(local_envelope, accelerator):
    if hasattr(accelerator, "gather_object"):
        gathered = accelerator.gather_object(local_envelope)
    else:
        from accelerate.utils import gather_object

        gathered = gather_object([local_envelope])
    if not isinstance(gathered, (list, tuple)):
        gathered = [gathered]
    return list(gathered)


def validate_distributed_msa_envelope(local_envelope, accelerator):
    """Collect local status first, then fail identically on every rank."""
    gathered = _gather_envelopes(local_envelope, accelerator)
    failures = []
    for rank, envelope in enumerate(gathered):
        if not isinstance(envelope, dict):
            failures.append("rank {}: malformed preflight envelope".format(rank))
        elif not envelope.get("ok", False):
            failures.append(
                "rank {}: {}".format(
                    rank, envelope.get("error", "unknown error")
                )
            )
    if failures:
        raise RuntimeError(
            "MSA rank-local asset preflight failed: {}".format(
                " | ".join(failures)
            )
        )
    serialized = {
        json.dumps(envelope["identity"], sort_keys=True, separators=(",", ":"))
        for envelope in gathered
    }
    if len(serialized) != 1:
        raise RuntimeError(
            "MSA data identities differ across ranks: {}".format(
                " | ".join(sorted(serialized))
            )
        )
    return local_envelope["identity"]


def validate_distributed_msa_identity(metadata, accelerator):
    """Compatibility helper for callers that already built local metadata."""
    local_identity = _training_data_identity(metadata)
    envelope = {"ok": True, "identity": local_identity, "error": None}
    return validate_distributed_msa_envelope(envelope, accelerator)


def _load_checkpoint_snapshot(path, label):
    resolved = _resolved_file(path, label)
    try:
        with resolved.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            source.seek(0)
            checkpoint = torch.load(
                source, map_location="cpu", weights_only=False
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("cannot load {}: {}".format(label, error)) from error
    return checkpoint, str(resolved), digest.hexdigest()


def _require_state_mapping(checkpoint, label):
    state = (
        checkpoint["net"]
        if isinstance(checkpoint, dict) and "net" in checkpoint
        else checkpoint
    )
    if (
        not isinstance(state, dict)
        or not state
        or not all(torch.is_tensor(value) for value in state.values())
    ):
        raise ValueError("{} does not contain a valid model state dict".format(label))
    return state


def _expected_causal_tae_shapes(args):
    from models.tae import Causal_HumanTAE

    with torch.device("meta"):
        model = Causal_HumanTAE(
            hidden_size=args.hidden_size,
            down_t=args.down_t,
            stride_t=args.stride_t,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            latent_dim=args.latent_dim,
            clip_range=[-30, 20],
        )
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }


def _map_causal_tae_state(state):
    mapped = {}
    for key, value in state.items():
        if key.startswith("tae.encoder."):
            new_key = key.replace("tae.encoder.", "msa_vae.cnn_encoder.", 1)
        elif key.startswith("tae.decoder."):
            new_key = key.replace("tae.decoder.", "msa_vae.cnn_decoder.", 1)
        elif key.startswith("tae.decode_proj."):
            new_key = key.replace(
                "tae.decode_proj.", "msa_vae.decode_proj.", 1
            )
        else:
            raise ValueError("unexpected Causal TAE key {}".format(key))
        mapped[new_key] = value
    return mapped


def _expected_msa_cnn_shapes(args):
    from models.msa_vae import MSA_HumanVAE

    with torch.device("meta"):
        model = MSA_HumanVAE(
            hidden_size=args.hidden_size,
            down_t=args.down_t,
            stride_t=args.stride_t,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            latent_dim=args.latent_dim,
            clip_range=[-30, 20],
            trans_d_model=16,
            trans_nhead=4,
            trans_enc_layers=1,
            trans_dec_layers=1,
            trans_ff_size=32,
            trans_dropout=0.0,
            clip_dim=16,
        )
    prefixes = (
        "msa_vae.cnn_encoder.",
        "msa_vae.cnn_decoder.",
        "msa_vae.decode_proj.",
    )
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def _require_causal_tae_state(checkpoint, args):
    state = _require_state_mapping(checkpoint, "resume CNN checkpoint")
    expected = _expected_causal_tae_shapes(args)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = [
        "{}: expected {}, got {}".format(
            key, expected[key], tuple(state[key].shape)
        )
        for key in sorted(set(expected) & set(state))
        if tuple(state[key].shape) != expected[key]
    ]
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append("missing keys {}".format(", ".join(missing[:8])))
        if unexpected:
            details.append(
                "unexpected keys {}".format(", ".join(unexpected[:8]))
            )
        if shape_mismatches:
            details.append(
                "shape mismatch {}".format("; ".join(shape_mismatches[:8]))
            )
        raise ValueError(
            "resume CNN checkpoint does not match the official Causal TAE: "
            + "; ".join(details)
        )
    mapped = _map_causal_tae_state(state)
    expected_mapped = _expected_msa_cnn_shapes(args)
    mapped_shapes = {
        key: tuple(value.shape) for key, value in mapped.items()
    }
    if mapped_shapes != expected_mapped:
        raise ValueError(
            "official Causal TAE mapping does not exactly match the MSA CNN"
        )
    return mapped


def _local_training_asset_probe(args):
    """Build an error envelope without raising before the collective."""
    try:
        base_metadata = build_msa_checkpoint_metadata(
            args, include_train_manifest=True
        )
        mode = args.msa_data_mode
        approved_sha256 = None
        if mode == "babel_sparse_global" and args.phase in (1, 2):
            approved_sha256 = _validated_sha256(
                getattr(args, "resume_cnn_sha256", None),
                "approved Causal TAE SHA-256",
            )

        mapped_cnn_state = None
        causal_identity = None
        if args.resume_cnn_pth:
            cnn_checkpoint, cnn_path, cnn_sha256 = _load_checkpoint_snapshot(
                args.resume_cnn_pth, "resume CNN checkpoint"
            )
            if (
                mode == "babel_sparse_global"
                and approved_sha256 != cnn_sha256
            ):
                raise ValueError(
                    "Causal TAE artifact does not match the approved SHA-256"
                )
            mapped_cnn_state = _require_causal_tae_state(
                cnn_checkpoint, args
            )
            causal_identity = {"path": cnn_path, "sha256": cnn_sha256}

        full_checkpoint = None
        full_path = None
        full_sha256 = None
        if args.resume_pth:
            full_checkpoint, full_path, full_sha256 = _load_checkpoint_snapshot(
                args.resume_pth, "resume MSA-VAE checkpoint"
            )
            _require_state_mapping(
                full_checkpoint, "resume MSA-VAE checkpoint"
            )
            checkpoint_metadata = (
                full_checkpoint.get("metadata")
                if isinstance(full_checkpoint, dict)
                else None
            )
            if checkpoint_metadata is None:
                if mode == "babel_sparse_global":
                    raise ValueError(
                        "BABEL training requires a tagged full resume checkpoint"
                    )
            else:
                validate_msa_checkpoint_metadata(
                    checkpoint_metadata,
                    args,
                    scope="training",
                    expected_causal_sha256=approved_sha256,
                )
                inherited_path = checkpoint_metadata.get(
                    "causal_tae_artifact_path"
                )
                if inherited_path is not None:
                    inherited = {
                        "path": inherited_path,
                        "sha256": _validated_sha256(
                            checkpoint_metadata.get(
                                "causal_tae_artifact_sha256"
                            ),
                            "checkpoint Causal TAE SHA-256",
                        ),
                    }
                    if (
                        causal_identity is not None
                        and inherited["sha256"] != causal_identity["sha256"]
                    ):
                        raise ValueError(
                            "direct and inherited Causal TAE identities differ"
                        )
                    causal_identity = causal_identity or inherited

        if mode == "babel_sparse_global":
            if (
                args.phase == 1
                and mapped_cnn_state is None
                and full_checkpoint is None
            ):
                raise ValueError(
                    "BABEL Phase 1 requires --resume-cnn-pth or a tagged full resume"
                )
            if args.phase == 2 and full_checkpoint is None:
                raise ValueError(
                    "BABEL Phase 2 requires a tagged BABEL Phase-1/full resume"
                )

        metadata = dict(
            base_metadata,
            causal_tae_artifact_path=(
                None
                if causal_identity is None
                else _canonical_recorded_path(causal_identity["path"])
            ),
            causal_tae_artifact_sha256=(
                None if causal_identity is None else causal_identity["sha256"]
            ),
        )
        identity = _training_data_identity(metadata)
        identity["resume_checkpoint_path"] = full_path
        identity["resume_checkpoint_sha256"] = full_sha256
        return (
            {"ok": True, "identity": identity, "error": None},
            metadata,
            full_checkpoint,
            mapped_cnn_state,
        )
    except Exception as error:
        return (
            {
                "ok": False,
                "identity": None,
                "error": "{}: {}".format(type(error).__name__, error),
            },
            None,
            None,
            None,
        )


def preflight_msa_training_assets(args, accelerator):
    """Collectively validate assets before dataset/model construction."""
    envelope, metadata, full_checkpoint, mapped_cnn_state = (
        _local_training_asset_probe(args)
    )
    validate_distributed_msa_envelope(envelope, accelerator)
    return metadata, full_checkpoint, mapped_cnn_state


def prepare_babel_validation_loader(accelerator, val_loader):
    """Shard the deterministic validation loader exactly once at setup time."""
    return accelerator.prepare(val_loader)


def build_msa_training_loaders(
    args,
    humanml_train_factory=None,
    humanml_validation_factory=None,
    babel_train_factory=None,
    babel_validation_factory=None,
):
    """Select the paired train/validation data contract for one MSA mode."""
    if args.msa_data_mode == "babel_sparse_global":
        if babel_train_factory is None or babel_validation_factory is None:
            from humanml3d_272.dataset_msa_vae_babel import (
                DATALoader,
                ValidationDATALoader,
            )

            babel_train_factory = babel_train_factory or DATALoader
            babel_validation_factory = (
                babel_validation_factory or ValidationDATALoader
            )
        shared = {
            "t5_model_path": args.t5_model_path,
            "mean_path": args.msa_mean_path,
            "std_path": args.msa_std_path,
            "window_size": args.window_size,
            "unit_length": 2 ** args.down_t,
            "text_embed_dim": args.text_embed_dim,
        }
        train_loader = babel_train_factory(
            batch_size=args.batch_size,
            bridge_split_file=args.bridge_split_file,
            bridge_motion_dir=args.bridge_motion_dir,
            bridge_text_dir=args.bridge_text_dir,
            bridge_global_embed_dir=args.bridge_global_embed_dir,
            bridge_local_embed_dir=args.bridge_local_embed_dir,
            babel_motion_dir=args.babel_train_motion_dir,
            babel_text_dir=args.babel_train_text_dir,
            babel_cache_dir=args.babel_train_t5_cache_dir,
            babel_cache_manifest=args.babel_train_cache_manifest,
            babel_split="train",
            **shared
        )
        validation_loader = babel_validation_factory(
            batch_size=args.batch_size,
            babel_motion_dir=args.babel_val_motion_dir,
            babel_text_dir=args.babel_val_text_dir,
            babel_cache_dir=args.babel_val_t5_cache_dir,
            babel_cache_manifest=args.babel_val_cache_manifest,
            babel_split="val",
            **shared
        )
        return train_loader, validation_loader, "babel_reconstruction"

    if humanml_train_factory is None:
        from humanml3d_272.dataset_msa_vae import DATALoader

        humanml_train_factory = DATALoader
    if humanml_validation_factory is None:
        from humanml3d_272.dataset_eval_t2m import DATALoader

        humanml_validation_factory = DATALoader
    train_loader = humanml_train_factory(
        args.dataname,
        args.batch_size,
        window_size=args.window_size,
        unit_length=2 ** args.down_t,
        use_ft_split=args.use_ft_split,
        text_encoder_type=args.text_encoder_type,
        clip_embed_dir=args.clip_embed_dir,
        t5_embed_dir=args.t5_embed_dir,
        text_embed_dim=args.text_embed_dim,
        use_offline_global_text=args.use_offline_global_text,
        clip_global_embed_dir=args.clip_global_embed_dir,
        t5_global_embed_dir=args.t5_global_embed_dir,
    )
    validation_loader = humanml_validation_factory(
        args.dataname,
        False,
        32,
        unit_length=2 ** args.down_t,
    )
    return train_loader, validation_loader, "humanml_tmr"


def _reduce_sum(accelerator, value):
    return accelerator.reduce(value, reduction="sum")


def _safe_mean(total, count):
    return total / count.clamp_min(1).to(total.dtype)


def _checkpoint_payload(net, metadata, global_coverage, local_coverage):
    coverage = {
        "global": float(global_coverage),
        "local": float(local_coverage),
    }
    return {
        "net": net.state_dict(),
        "metadata": dict(metadata, supervision_coverage=coverage),
    }


def _save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


@torch.no_grad()
def evaluate_msa_vae_babel(
    out_dir,
    val_loader,
    net,
    dataset,
    logger,
    writer,
    iteration,
    phase,
    best_semantic,
    best_mpjpe,
    device,
    accelerator,
    metadata,
    save_checkpoints=True,
):
    """Evaluate BABEL reconstruction and masked local semantic alignment."""
    was_training = net.training
    net.eval()

    metric_names = (
        "reconstruction_sum",
        "reconstruction_count",
        "kl_sum",
        "kl_count",
        "latent_sum",
        "latent_count",
        "local_cosine_sum",
        "local_count",
        "global_count",
        "sample_count",
        "local_possible",
        "mpjpe_sum",
        "pose_count",
    )
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in metric_names
    }

    try:
        for batch in val_loader:
            (
                motion,
                _captions,
                _global_target,
                has_global,
                local_target,
                has_local,
                _total_frames,
                _local_pooled,
            ) = batch
            motion = motion.to(device=device, dtype=torch.float32)
            local_target = local_target.to(device=device, dtype=torch.float32)
            has_global = has_global.to(device=device, dtype=torch.bool)
            has_local = has_local.to(device=device, dtype=torch.bool)
            output = net(motion)
            reconstruction = output["x_recon"]

            difference = reconstruction - motion
            totals["reconstruction_sum"] += difference.square().sum().double()
            totals["reconstruction_count"] += difference.numel()

            mu = output["mu"]
            logvar = output["logvar"]
            kl_values = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
            totals["kl_sum"] += kl_values.sum().double()
            totals["kl_count"] += kl_values.numel()

            latent_target = output.get("trans_latent_target", mu)
            latent_values = (output["mu_recon"] - latent_target).square()
            totals["latent_sum"] += latent_values.sum().double()
            totals["latent_count"] += latent_values.numel()

            local_features = output["clip_local_feat"]
            if local_features.shape != local_target.shape:
                raise ValueError(
                    "local feature shape {} does not match target {}".format(
                        tuple(local_features.shape), tuple(local_target.shape)
                    )
                )
            cosine = F.cosine_similarity(local_features, local_target, dim=-1)
            token_mask = has_local.unsqueeze(1).expand_as(cosine)
            totals["local_cosine_sum"] += cosine[token_mask].sum().double()
            totals["local_count"] += token_mask.sum().double()
            totals["local_possible"] += token_mask.numel()
            totals["global_count"] += has_global.sum().double()
            totals["sample_count"] += has_global.numel()

            motion_np = dataset.inv_transform(motion.detach().cpu().numpy())
            reconstruction_np = dataset.inv_transform(
                reconstruction.detach().cpu().numpy()
            )
            for sample_index in range(motion.shape[0]):
                target_joints = torch.from_numpy(
                    recover_from_local_position(motion_np[sample_index], 22)
                ).to(device=device, dtype=torch.float32)
                predicted_joints = torch.from_numpy(
                    recover_from_local_position(reconstruction_np[sample_index], 22)
                ).to(device=device, dtype=torch.float32)
                per_frame_mpjpe = calculate_mpjpe(target_joints, predicted_joints)
                totals["mpjpe_sum"] += per_frame_mpjpe.sum().double()
                totals["pose_count"] += per_frame_mpjpe.numel()

        reduced = {
            name: _reduce_sum(accelerator, value)
            for name, value in totals.items()
        }
        reconstruction_metric = _safe_mean(
            reduced["reconstruction_sum"], reduced["reconstruction_count"]
        )
        kl_metric = _safe_mean(reduced["kl_sum"], reduced["kl_count"])
        latent_metric = _safe_mean(reduced["latent_sum"], reduced["latent_count"])
        local_cosine = _safe_mean(
            reduced["local_cosine_sum"], reduced["local_count"]
        )
        local_loss = torch.where(
            reduced["local_count"] > 0,
            1.0 - local_cosine,
            torch.zeros_like(local_cosine),
        )
        global_coverage = _safe_mean(
            reduced["global_count"], reduced["sample_count"]
        )
        local_coverage = _safe_mean(
            reduced["local_count"], reduced["local_possible"]
        )
        mpjpe = _safe_mean(reduced["mpjpe_sum"], reduced["pose_count"]) * 1000.0
        local_weight = float(metadata.get("local_align_weight", 0.0))
        semantic_objective = latent_metric + local_weight * local_loss

        values = {
            "mpjpe": float(mpjpe.item()),
            "reconstruction": float(reconstruction_metric.item()),
            "kl": float(kl_metric.item()),
            "latent": float(latent_metric.item()),
            "local_cosine": float(local_cosine.item()),
            "local_loss": float(local_loss.item()),
            "global_coverage": float(global_coverage.item()),
            "local_coverage": float(local_coverage.item()),
            "semantic_objective": float(semantic_objective.item()),
        }

        if accelerator.is_main_process:
            logger.info(
                "BABEL MSA-VAE eval iter {}: MPJPE {:.3f}mm, recon {:.6f}, "
                "KL {:.6f}, latent {:.6f}, local cosine {:.6f}, "
                "semantic {:.6f}, coverage global {:.4f}/local {:.4f}".format(
                    iteration,
                    values["mpjpe"],
                    values["reconstruction"],
                    values["kl"],
                    values["latent"],
                    values["local_cosine"],
                    values["semantic_objective"],
                    values["global_coverage"],
                    values["local_coverage"],
                )
            )
            for name, value in values.items():
                writer.add_scalar("BABEL/{}".format(name), value, iteration)

            if save_checkpoints:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                payload = _checkpoint_payload(
                    net,
                    metadata,
                    values["global_coverage"],
                    values["local_coverage"],
                )
                if phase == 1 and values["semantic_objective"] < best_semantic:
                    logger.info(
                        "BABEL semantic objective improved from {:.6f} to {:.6f}".format(
                            best_semantic, values["semantic_objective"]
                        )
                    )
                    best_semantic = values["semantic_objective"]
                    _save_checkpoint(
                        Path(out_dir) / "net_best_semantic.pth", payload
                    )
                if phase == 2 and values["mpjpe"] < best_mpjpe:
                    logger.info(
                        "BABEL MPJPE improved from {:.6f} to {:.6f}".format(
                            best_mpjpe, values["mpjpe"]
                        )
                    )
                    best_mpjpe = values["mpjpe"]
                    _save_checkpoint(Path(out_dir) / "net_best_mpjpe.pth", payload)
                _save_checkpoint(Path(out_dir) / "net_last.pth", payload)

        accelerator.wait_for_everyone()
        return BabelEvalResult(
            best_semantic=float(best_semantic),
            best_mpjpe=float(best_mpjpe),
            **values
        )
    finally:
        net.train(was_training)
