"""Evaluate native MSA-SentenceT5 alignment and deterministic realism."""

import argparse
import csv
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from eval_msa_vae_metrics import load_evaluator_checkpoint
from humanml3d_272.dataset_eval_msa_vae_metrics import (
    MSAVAEAlignmentDataset,
    make_msa_vae_alignment_loader,
)
from utils.eval_trans import recover_from_local_position
from utils.msa_vae_alignment_metrics import (
    calculate_masked_local_cosine,
    calculate_motion_macro_cosine,
    calculate_msa_t5_retrieval,
    shuffled_text_control,
)
from utils.msa_vae_eval_config import (
    build_and_load_msa_vae,
    checkpoint_manifest,
    load_checkpoint_payload,
    resolve_msa_vae_config,
)
from utils.msa_vae_metrics import (
    ReconstructionMetricAccumulator,
    SkatingConfig,
    calculate_fid,
)


PROTOCOL_VERSION = "msa-vae-internal-alignment-v1"
GLOBAL_METRIC_KEYS = (
    "global_cosine",
    "fid",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "accel_mm_per_frame2",
    "skating_percent",
    "msa_t5_t2m_r1_percent",
    "msa_t5_t2m_r2_percent",
    "msa_t5_t2m_r3_percent",
    "msa_t5_t2m_r5_percent",
    "msa_t5_t2m_medr",
    "msa_t5_m2t_r1_percent",
    "msa_t5_m2t_r2_percent",
    "msa_t5_m2t_r3_percent",
    "msa_t5_m2t_r5_percent",
    "msa_t5_m2t_medr",
)


@dataclass(frozen=True)
class AlignmentEvaluationPaths:
    checkpoint: Path
    data_root: Path
    split_file: Path
    global_text_embed_dir: Path
    local_split_file: Optional[Path]
    local_text_embed_dir: Optional[Path]
    evaluator_root: Path
    evaluator_checkpoint: Path
    output_dir: Path


def _repository_path(repo_root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate internal MSA-SentenceT5 alignment and posterior-mean "
            "MSA-VAE reconstruction realism."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="MSA-VAE .pth checkpoint path")
    parser.add_argument("--data-root", default="humanml3d_272")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--global-text-embed-dir", default=None)
    parser.add_argument("--local-split-file", default=None)
    parser.add_argument("--local-text-embed-dir", default=None)
    parser.add_argument(
        "--local-target-scope",
        choices=("held-out", "in-sample"),
        default=None,
    )
    parser.add_argument("--evaluator-root", default="Evaluator_272")
    parser.add_argument("--evaluator-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--down-t", type=int, default=None)
    parser.add_argument("--stride-t", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--dilation-growth-rate", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--trans-d-model", type=int, default=None)
    parser.add_argument("--trans-nhead", type=int, default=None)
    parser.add_argument("--trans-enc-layers", type=int, default=None)
    parser.add_argument("--trans-dec-layers", type=int, default=None)
    parser.add_argument("--trans-ff-size", type=int, default=None)
    parser.add_argument("--trans-dropout", type=float, default=None)
    parser.add_argument("--clip-dim", type=int, default=None)
    decoupling = parser.add_mutually_exclusive_group()
    decoupling.add_argument(
        "--disable-decoupling",
        dest="disable_decoupling",
        action="store_const",
        const=True,
        default=None,
    )
    decoupling.add_argument(
        "--enable-decoupling",
        dest="disable_decoupling",
        action="store_const",
        const=False,
    )
    return parser.parse_args(argv)


def resolve_cli_paths(repo_root, args):
    repo_root = Path(repo_root).resolve()
    checkpoint = _repository_path(repo_root, args.checkpoint)
    data_root = _repository_path(repo_root, args.data_root)
    split_file = (
        _repository_path(repo_root, args.split_file)
        if args.split_file
        else data_root / "split" / "test.txt"
    )
    global_text_embed_dir = (
        _repository_path(repo_root, args.global_text_embed_dir)
        if args.global_text_embed_dir
        else data_root / "text_latents_t5"
    )
    local_split_file = (
        _repository_path(repo_root, args.local_split_file)
        if args.local_split_file
        else None
    )
    local_text_embed_dir = (
        _repository_path(repo_root, args.local_text_embed_dir)
        if args.local_text_embed_dir
        else None
    )
    evaluator_root = _repository_path(repo_root, args.evaluator_root)
    evaluator_checkpoint = (
        _repository_path(repo_root, args.evaluator_checkpoint)
        if args.evaluator_checkpoint
        else evaluator_root
        / "experiments"
        / "temos"
        / "EXP1"
        / "checkpoints"
        / "epoch=99.ckpt"
    )
    if args.output_dir:
        output_dir = _repository_path(repo_root, args.output_dir)
    else:
        output_dir = (
            repo_root
            / "output"
            / "msa_vae_alignment"
            / checkpoint.parent.name
            / checkpoint.stem
        )
    return AlignmentEvaluationPaths(
        checkpoint=checkpoint,
        data_root=data_root,
        split_file=split_file.resolve(),
        global_text_embed_dir=global_text_embed_dir.resolve(),
        local_split_file=(
            local_split_file.resolve()
            if local_split_file is not None
            else None
        ),
        local_text_embed_dir=(
            local_text_embed_dir.resolve()
            if local_text_embed_dir is not None
            else None
        ),
        evaluator_root=evaluator_root,
        evaluator_checkpoint=evaluator_checkpoint,
        output_dir=output_dir.resolve(),
    )


def validate_runtime_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    local_values = (
        args.local_split_file,
        args.local_text_embed_dir,
        args.local_target_scope,
    )
    if any(value is not None for value in local_values) and not all(
        value is not None for value in local_values
    ):
        raise ValueError(
            "--local-split-file, --local-text-embed-dir, and "
            "--local-target-scope must be supplied together"
        )


def preflight_alignment_assets(paths):
    required = [
        ("MSA-VAE checkpoint", paths.checkpoint, "file"),
        ("HumanML3D-272 data root", paths.data_root, "directory"),
        ("global evaluation split", paths.split_file, "file"),
        ("HumanML3D-272 motion_data", paths.data_root / "motion_data", "directory"),
        ("HumanML3D-272 texts", paths.data_root / "texts", "directory"),
        ("HumanML3D-272 Mean.npy", paths.data_root / "mean_std" / "Mean.npy", "file"),
        ("HumanML3D-272 Std.npy", paths.data_root / "mean_std" / "Std.npy", "file"),
        (
            "global SentenceT5 target directory",
            paths.global_text_embed_dir,
            "directory",
        ),
        ("272-D evaluator root", paths.evaluator_root, "directory"),
        (
            "272-D evaluator checkpoint",
            paths.evaluator_checkpoint,
            "file",
        ),
        (
            "272-D evaluator motion encoder source",
            paths.evaluator_root
            / "mld"
            / "models"
            / "architectures"
            / "temos"
            / "motionencoder"
            / "actor.py",
            "file",
        ),
    ]
    if paths.local_split_file is not None:
        required.extend(
            [
                (
                    "local alignment split",
                    paths.local_split_file,
                    "file",
                ),
                (
                    "local SentenceT5 target directory",
                    paths.local_text_embed_dir,
                    "directory",
                ),
            ]
        )
    missing = []
    for label, path, kind in required:
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            missing.append("{}: {}".format(label, path))
    if missing:
        raise FileNotFoundError(
            "MSA-VAE alignment evaluation preflight failed: "
            + "; ".join(missing)
        )


def architecture_overrides(args):
    names = (
        "hidden_size",
        "down_t",
        "stride_t",
        "depth",
        "dilation_growth_rate",
        "latent_dim",
        "trans_d_model",
        "trans_nhead",
        "trans_enc_layers",
        "trans_dec_layers",
        "trans_ff_size",
        "trans_dropout",
        "clip_dim",
        "disable_decoupling",
    )
    return {
        name: getattr(args, name)
        for name in names
        if getattr(args, name) is not None
    }


def set_evaluation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation was requested but CUDA is not available; "
            "pass --device cpu only for smoke tests"
        )
    return device


def _distribution_location(output, label):
    location = getattr(output, "loc", None)
    if not isinstance(location, torch.Tensor) or location.ndim != 2:
        raise ValueError(
            "{} must return a distribution with 2D .loc".format(label)
        )
    if not torch.isfinite(location).all().item():
        raise ValueError("{} returned non-finite embeddings".format(label))
    return location.detach().cpu()


def _recover_valid_joints(dataset, normalized_motion, length):
    valid = normalized_motion[:length].detach().cpu().numpy()
    features = dataset.inv_transform(valid)
    joints = recover_from_local_position(features, 22)
    joints = torch.as_tensor(joints).float()
    if joints.shape != (length, 22, 3):
        raise ValueError(
            "272-D joint recovery returned {}, expected ({}, 22, 3)".format(
                tuple(joints.shape),
                length,
            )
        )
    return joints


def _required_semantic_tensor(outputs, key, ndim):
    if not isinstance(outputs, dict) or key not in outputs:
        raise ValueError(
            "MSA-VAE semantic forward must return {}".format(key)
        )
    value = outputs[key]
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(
            "MSA-VAE semantic {} must be a {}D tensor".format(
                key,
                ndim,
            )
        )
    if not torch.isfinite(value).all().item():
        raise ValueError(
            "MSA-VAE semantic {} contains non-finite values".format(key)
        )
    return value


def _validate_motion_batch(batch, device):
    motions = batch["motions"].to(device).float()
    lengths = batch["lengths"].to(device).long()
    if motions.ndim != 3 or motions.shape[2] != 272:
        raise ValueError("evaluation motions must have shape (B, T, 272)")
    if lengths.ndim != 1 or lengths.shape[0] != motions.shape[0]:
        raise ValueError("evaluation lengths must match the motion batch")
    if torch.any(lengths <= 0).item() or torch.any(
        lengths > motions.shape[1]
    ).item():
        raise ValueError("evaluation lengths are outside the padded batch")
    if not torch.isfinite(motions).all().item():
        raise ValueError("evaluation motions contain non-finite values")
    return motions, lengths


@torch.inference_mode()
def evaluate_global_alignment_and_realism(
    model,
    motion_encoder,
    loader,
    device,
    skating_config,
    shuffled_seed=0,
):
    """Run the global alignment and posterior-mean realism branches."""

    model.eval()
    motion_encoder.eval()
    reconstruction = ReconstructionMetricAccumulator(skating_config)
    global_motion_features = []
    global_text_features = []
    text_motion_indices = []
    reference_embeddings = []
    prediction_embeddings = []
    sample_count = 0
    caption_count = 0

    for batch in loader:
        if batch.get("target_mode") != "global":
            raise ValueError("global evaluation requires global targets")
        motions, lengths = _validate_motion_batch(batch, device)
        semantic = model(
            motions,
            lengths=lengths,
            semantic_only=True,
        )
        mu = _required_semantic_tensor(semantic, "mu", ndim=3)
        global_features = _required_semantic_tensor(
            semantic,
            "clip_global_feat",
            ndim=2,
        )
        batch_size = motions.shape[0]
        if mu.shape[0] != batch_size:
            raise ValueError("semantic mu batch dimension does not match")
        if global_features.shape[0] != batch_size:
            raise ValueError(
                "global semantic feature batch dimension does not match"
            )

        predictions = model.forward_decoder(mu)
        if (
            not isinstance(predictions, torch.Tensor)
            or predictions.shape != motions.shape
        ):
            shape = (
                tuple(predictions.shape)
                if isinstance(predictions, torch.Tensor)
                else type(predictions).__name__
            )
            raise ValueError(
                "posterior-mean reconstruction shape {} does not match "
                "input {}".format(shape, tuple(motions.shape))
            )
        if not torch.isfinite(predictions).all().item():
            raise ValueError(
                "posterior-mean reconstruction contains non-finite values"
            )

        cached_targets = batch.get("global_text_embeddings")
        if (
            not isinstance(cached_targets, list)
            or len(cached_targets) != batch_size
        ):
            raise ValueError(
                "global batch must provide one caption-target tensor "
                "per motion"
            )
        batch_caption_count = 0
        for motion_index, targets in enumerate(cached_targets):
            if (
                not isinstance(targets, torch.Tensor)
                or targets.ndim != 2
                or targets.shape[0] < 1
            ):
                raise ValueError(
                    "global caption targets must be non-empty 2D tensors"
                )
            if targets.shape[1] != global_features.shape[1]:
                raise ValueError(
                    "global target and motion feature dimensions must match"
                )
            if not torch.isfinite(targets).all().item():
                raise ValueError(
                    "global caption targets contain non-finite values"
                )
            global_text_features.append(
                targets.detach().cpu().float()
            )
            target_count = targets.shape[0]
            text_motion_indices.append(
                torch.full(
                    (target_count,),
                    sample_count + motion_index,
                    dtype=torch.long,
                )
            )
            batch_caption_count += target_count
        caption_count += batch_caption_count
        global_motion_features.append(
            global_features.detach().cpu().float()
        )

        frame_index = torch.arange(
            predictions.shape[1],
            device=predictions.device,
        ).unsqueeze(0)
        padding_mask = frame_index >= lengths.unsqueeze(1)
        prediction_for_encoder = predictions.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )
        reference_embeddings.append(
            _distribution_location(
                motion_encoder(motions, lengths),
                "ground-truth motion encoder",
            )
        )
        prediction_embeddings.append(
            _distribution_location(
                motion_encoder(prediction_for_encoder, lengths),
                "reconstructed motion encoder",
            )
        )

        for index, length in enumerate(lengths.tolist()):
            target_joints = _recover_valid_joints(
                loader.dataset,
                motions[index],
                length,
            )
            prediction_joints = _recover_valid_joints(
                loader.dataset,
                predictions[index],
                length,
            )
            reconstruction.update(prediction_joints, target_joints)
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("global alignment loader yielded no samples")
    global_motion_features = torch.cat(
        global_motion_features,
        dim=0,
    )
    global_text_features = torch.cat(global_text_features, dim=0)
    text_motion_indices = torch.cat(text_motion_indices, dim=0)
    reference_embeddings = torch.cat(reference_embeddings, dim=0)
    prediction_embeddings = torch.cat(prediction_embeddings, dim=0)
    if global_motion_features.shape[0] != sample_count:
        raise ValueError(
            "global semantic feature count does not match sample count"
        )
    if (
        global_text_features.shape[0] != caption_count
        or text_motion_indices.shape[0] != caption_count
    ):
        raise ValueError("global caption count does not match targets")
    if (
        reference_embeddings.shape[0] != sample_count
        or prediction_embeddings.shape[0] != sample_count
    ):
        raise ValueError(
            "motion encoder embedding count does not match sample count"
        )

    metrics = reconstruction.compute()
    metrics["fid"] = calculate_fid(
        reference_embeddings,
        prediction_embeddings,
    )
    metrics["global_cosine"] = calculate_motion_macro_cosine(
        global_motion_features,
        global_text_features,
        text_motion_indices,
    )
    metrics.update(
        calculate_msa_t5_retrieval(
            global_text_features,
            global_motion_features,
            text_motion_indices,
        )
    )
    shuffled_targets = shuffled_text_control(
        global_text_features,
        seed=shuffled_seed,
    )
    shuffled_metrics = calculate_msa_t5_retrieval(
        shuffled_targets,
        global_motion_features,
        text_motion_indices,
    )
    return {
        "sample_count": sample_count,
        "caption_count": caption_count,
        **metrics,
        "shuffled_global_retrieval": shuffled_metrics,
    }


@torch.inference_mode()
def evaluate_local_alignment(model, loader, device):
    """Evaluate native local projection against exact cached targets."""

    model.eval()
    sample_count = 0
    token_count = 0
    cosine_weighted_sum = 0.0
    for batch in loader:
        if batch.get("target_mode") != "local":
            raise ValueError("local evaluation requires local targets")
        motions, lengths = _validate_motion_batch(batch, device)
        semantic = model(
            motions,
            lengths=lengths,
            semantic_only=True,
        )
        local_features = _required_semantic_tensor(
            semantic,
            "clip_local_feat",
            ndim=3,
        )
        targets = batch.get("local_text_embeddings")
        valid_mask = batch.get("local_mask")
        if not isinstance(targets, torch.Tensor):
            raise ValueError(
                "local batch must provide local_text_embeddings"
            )
        if not isinstance(valid_mask, torch.Tensor):
            raise ValueError("local batch must provide local_mask")
        targets = targets.to(device).float()
        valid_mask = valid_mask.to(device)
        if local_features.shape != targets.shape:
            raise ValueError(
                "local semantic and target shapes must match"
            )
        batch_size = motions.shape[0]
        if local_features.shape[0] != batch_size:
            raise ValueError(
                "local semantic feature batch dimension does not match"
            )
        batch_cosine = calculate_masked_local_cosine(
            local_features,
            targets,
            valid_mask,
        )
        cosine_weighted_sum += batch_cosine * batch_size
        sample_count += batch_size
        token_count += int(valid_mask.sum().item())
    if sample_count == 0 or token_count == 0:
        raise ValueError("local alignment loader yielded no valid targets")
    return {
        "local_sample_count": sample_count,
        "local_token_count": token_count,
        "local_cosine": cosine_weighted_sum / sample_count,
    }


def load_frozen_humanml_motion_encoder(
    evaluator_root,
    device,
    evaluator_checkpoint,
):
    """Load only the frozen 272-D TMR motion encoder used for FID."""

    evaluator_root = Path(evaluator_root).resolve()
    inserted = str(evaluator_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(evaluator_root))
    try:
        from mld.models.architectures.temos.motionencoder.actor import (
            ActorAgnosticEncoder,
        )
    finally:
        if inserted:
            sys.path.remove(str(evaluator_root))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"enable_nested_tensor is True.*",
            category=UserWarning,
        )
        motion_encoder = ActorAgnosticEncoder(
            nfeats=272,
            vae=True,
            num_layers=4,
            latent_dim=256,
            max_len=-1,
        )
    evaluator_payload = load_evaluator_checkpoint(
        evaluator_checkpoint,
        evaluator_root=evaluator_root,
    )
    state = {
        key[len("motionencoder") + 1 :]: value
        for key, value in evaluator_payload["state_dict"].items()
        if key.startswith("motionencoder.")
    }
    if not state:
        raise ValueError(
            "272-D evaluator checkpoint has no motionencoder weights"
        )
    motion_encoder.load_state_dict(state, strict=True)
    motion_encoder.eval().to(device)
    for parameter in motion_encoder.parameters():
        parameter.requires_grad = False
    return motion_encoder


def _normalized_local_scope(value):
    if value is None:
        return None
    normalized = str(value).replace("-", "_")
    if normalized not in ("held_out", "in_sample"):
        raise ValueError("local scope must be held_out or in_sample")
    return normalized


def _validated_metric_values(metrics):
    missing = [key for key in GLOBAL_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(
            "alignment result is missing metrics: {}".format(
                ", ".join(missing)
            )
        )
    values = {key: float(metrics[key]) for key in GLOBAL_METRIC_KEYS}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("alignment result metrics contain non-finite values")
    return values


def _validate_diagnostic_values(value):
    if isinstance(value, dict):
        for child in value.values():
            _validate_diagnostic_values(child)
        return
    if isinstance(value, (float, int, np.number)):
        if not np.isfinite(float(value)):
            raise ValueError("alignment diagnostics contain non-finite values")
        return
    raise TypeError("alignment diagnostics must contain numeric values")


def build_alignment_result_manifest(
    metrics,
    diagnostics,
    checkpoint,
    evaluator,
    resolved_config,
    global_dataset,
    local_dataset,
    local_scope,
    seed,
    batch_size,
    skating_config,
) -> Dict[str, Any]:
    """Build the auditable internal-alignment result contract."""

    metric_values = _validated_metric_values(metrics)
    sample_count = int(metrics.get("sample_count", -1))
    caption_count = int(metrics.get("caption_count", -1))
    if sample_count != len(global_dataset.sample_ids):
        raise ValueError(
            "global result sample count does not match dataset"
        )
    if caption_count != int(global_dataset.caption_count):
        raise ValueError(
            "global result caption count does not match dataset"
        )
    _validate_diagnostic_values(diagnostics)

    scope = _normalized_local_scope(local_scope)
    local_value = None
    in_sample_local_value = None
    if local_dataset is None:
        if scope is not None:
            raise ValueError(
                "local scope requires a local alignment dataset"
            )
        local_manifest = {
            "scope": None,
            "sample_count": 0,
            "sample_ids": [],
            "sample_hash": None,
            "split_file": None,
            "target_directory": None,
            "target_hash": None,
            "token_count": 0,
        }
    else:
        if scope is None:
            raise ValueError(
                "local alignment dataset requires an explicit scope"
            )
        local_sample_count = int(
            metrics.get("local_sample_count", -1)
        )
        local_token_count = int(
            metrics.get("local_token_count", -1)
        )
        if local_sample_count != len(local_dataset.sample_ids):
            raise ValueError(
                "local result sample count does not match dataset"
            )
        if local_token_count != int(local_dataset.local_token_count):
            raise ValueError(
                "local result token count does not match dataset"
            )
        raw_local_value = float(metrics["local_cosine"])
        if not np.isfinite(raw_local_value):
            raise ValueError(
                "alignment result metrics contain non-finite values"
            )
        if scope == "held_out":
            local_value = raw_local_value
        else:
            in_sample_local_value = raw_local_value
        local_manifest = {
            "scope": scope,
            "split": Path(local_dataset.split_file).name,
            "sample_count": local_sample_count,
            "sample_ids": list(local_dataset.sample_ids),
            "sample_hash": local_dataset.sample_hash,
            "split_file": str(local_dataset.split_file),
            "target_directory": local_dataset.target_directory,
            "target_hash": local_dataset.target_hash,
            "token_count": local_token_count,
        }
    metric_values["local_cosine"] = local_value
    metric_values["in_sample_local_cosine"] = in_sample_local_value

    return {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "dataset": "HumanML3D-272-complete-motion",
            "retrieval": (
                "MSA-global-projection-to-SentenceT5-multi-positive"
            ),
            "retrieval_similarity": "L2-normalized cosine",
            "retrieval_ties": "average rank",
            "caption_policy": (
                "all complete-motion captions; multi-positive M2T"
            ),
            "target_row_policy": "exact source-line cache rows",
            "global_aggregation": "motion-macro over captions",
            "local_aggregation": "motion-macro over valid latent tokens",
            "reconstruction_decode": "posterior_mean",
            "fid_encoder": "frozen-TMR-motion-encoder-only",
            "acceleration_scaling": (
                "finite difference; no FPS^2 multiplier"
            ),
        },
        "metrics": metric_values,
        "units": {
            "global_cosine": "cosine",
            "local_cosine": "cosine",
            "in_sample_local_cosine": "cosine",
            "fid": "embedding distance",
            "mpjpe_mm": "mm",
            "p_mpjpe_mm": "mm",
            "accel_mm_per_frame2": "mm/frame^2",
            "skating_percent": "percent",
            "msa_t5_t2m_r1_percent": "percent",
            "msa_t5_t2m_r2_percent": "percent",
            "msa_t5_t2m_r3_percent": "percent",
            "msa_t5_t2m_r5_percent": "percent",
            "msa_t5_t2m_medr": "rank",
            "msa_t5_m2t_r1_percent": "percent",
            "msa_t5_m2t_r2_percent": "percent",
            "msa_t5_m2t_r3_percent": "percent",
            "msa_t5_m2t_r5_percent": "percent",
            "msa_t5_m2t_medr": "rank",
        },
        "checkpoint": dict(checkpoint),
        "evaluator": dict(evaluator),
        "model_config": {
            "values": dict(resolved_config.values),
            "sources": dict(resolved_config.sources),
        },
        "global_realism_dataset": {
            "sample_count": sample_count,
            "sample_ids": list(global_dataset.sample_ids),
            "sample_hash": global_dataset.sample_hash,
            "split_file": str(global_dataset.split_file),
            "target_directory": global_dataset.target_directory,
            "target_hash": global_dataset.target_hash,
            "caption_count": caption_count,
            "caption_hash": global_dataset.caption_hash,
        },
        "local_alignment": local_manifest,
        "diagnostics": dict(diagnostics),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "skating": asdict(skating_config),
    }


def _metric_text(value, precision=6):
    if value is None:
        return "N/A"
    return ("{:.%df}" % precision).format(value)


def _format_final_report(manifest):
    metrics = manifest["metrics"]
    return (
        "MSA-VAE internal alignment evaluation ({protocol})\n"
        "Global samples: {samples} | captions: {captions}\n"
        "Global cosine {global_cosine} | Local cosine {local_cosine} | "
        "In-sample local cosine {in_sample_local}\n"
        "FID {fid:.6f} | MPJPE {mpjpe:.3f} mm | "
        "P-MPJPE {pmpjpe:.3f} mm | ACCEL {accel:.3f} mm/frame^2 | "
        "Skating {skating:.3f}%\n"
        "MSA-T5 text-to-motion: R@1 {t1:.3f}% | R@2 {t2:.3f}% | "
        "R@3 {t3:.3f}% | R@5 {t5:.3f}% | MedR {tm:.3f}\n"
        "MSA-T5 motion-to-text: R@1 {m1:.3f}% | R@2 {m2:.3f}% | "
        "R@3 {m3:.3f}% | R@5 {m5:.3f}% | MedR {mm:.3f}\n"
    ).format(
        protocol=manifest["protocol"]["version"],
        samples=manifest["global_realism_dataset"]["sample_count"],
        captions=manifest["global_realism_dataset"]["caption_count"],
        global_cosine=_metric_text(metrics["global_cosine"]),
        local_cosine=_metric_text(metrics["local_cosine"]),
        in_sample_local=_metric_text(
            metrics["in_sample_local_cosine"]
        ),
        fid=metrics["fid"],
        mpjpe=metrics["mpjpe_mm"],
        pmpjpe=metrics["p_mpjpe_mm"],
        accel=metrics["accel_mm_per_frame2"],
        skating=metrics["skating_percent"],
        t1=metrics["msa_t5_t2m_r1_percent"],
        t2=metrics["msa_t5_t2m_r2_percent"],
        t3=metrics["msa_t5_t2m_r3_percent"],
        t5=metrics["msa_t5_t2m_r5_percent"],
        tm=metrics["msa_t5_t2m_medr"],
        m1=metrics["msa_t5_m2t_r1_percent"],
        m2=metrics["msa_t5_m2t_r2_percent"],
        m3=metrics["msa_t5_m2t_r3_percent"],
        m5=metrics["msa_t5_m2t_r5_percent"],
        mm=metrics["msa_t5_m2t_medr"],
    )


def write_alignment_result_artifacts(manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    row = {
        "protocol_version": manifest["protocol"]["version"],
        "retrieval_protocol": manifest["protocol"]["retrieval"],
        "checkpoint_path": manifest["checkpoint"]["path"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "evaluator_path": manifest["evaluator"]["path"],
        "evaluator_sha256": manifest["evaluator"]["sha256"],
        "global_sample_count": manifest[
            "global_realism_dataset"
        ]["sample_count"],
        "global_sample_hash": manifest[
            "global_realism_dataset"
        ]["sample_hash"],
        "global_target_hash": manifest[
            "global_realism_dataset"
        ]["target_hash"],
        "caption_count": manifest[
            "global_realism_dataset"
        ]["caption_count"],
        "local_scope": manifest["local_alignment"]["scope"],
        "local_sample_count": manifest[
            "local_alignment"
        ]["sample_count"],
        "local_sample_hash": manifest[
            "local_alignment"
        ]["sample_hash"],
        "local_target_hash": manifest[
            "local_alignment"
        ]["target_hash"],
        "local_token_count": manifest[
            "local_alignment"
        ]["token_count"],
        "seed": manifest["seed"],
        "batch_size": manifest["batch_size"],
    }
    row.update(manifest["metrics"])
    with (output_dir / "metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    report = _format_final_report(manifest)
    (output_dir / "evaluation.log").write_text(
        report,
        encoding="utf-8",
    )
    return report


def main(argv=None):
    repo_root = Path(__file__).resolve().parent
    args = parse_args(argv)
    validate_runtime_args(args)
    paths = resolve_cli_paths(repo_root, args)
    preflight_alignment_assets(paths)
    device = _resolve_device(args.device)
    set_evaluation_seed(args.seed)

    overrides = architecture_overrides(args)
    payload = load_checkpoint_payload(paths.checkpoint)
    pre_model_config = resolve_msa_vae_config(
        paths.checkpoint,
        payload,
        overrides,
    )
    unit_length = (
        pre_model_config.values["stride_t"]
        ** pre_model_config.values["down_t"]
    )
    text_embed_dim = pre_model_config.values["clip_dim"]
    global_dataset = MSAVAEAlignmentDataset(
        paths.data_root,
        split_file=paths.split_file,
        unit_length=unit_length,
        target_mode="global",
        text_embed_dim=text_embed_dim,
        global_text_embed_dir=paths.global_text_embed_dir,
    )
    local_dataset = None
    if paths.local_split_file is not None:
        local_dataset = MSAVAEAlignmentDataset(
            paths.data_root,
            split_file=paths.local_split_file,
            unit_length=unit_length,
            target_mode="local",
            text_embed_dim=text_embed_dim,
            local_text_embed_dir=paths.local_text_embed_dir,
        )

    model, resolved_config, model_identity = build_and_load_msa_vae(
        paths.checkpoint,
        overrides,
        device,
    )
    global_loader = make_msa_vae_alignment_loader(
        global_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    local_loader = (
        make_msa_vae_alignment_loader(
            local_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if local_dataset is not None
        else None
    )
    motion_encoder = load_frozen_humanml_motion_encoder(
        paths.evaluator_root,
        device,
        paths.evaluator_checkpoint,
    )
    skating_config = SkatingConfig()
    metrics = evaluate_global_alignment_and_realism(
        model,
        motion_encoder,
        global_loader,
        device,
        skating_config,
        shuffled_seed=args.seed,
    )
    shuffled_retrieval = metrics.pop("shuffled_global_retrieval")
    if local_loader is not None:
        metrics.update(
            evaluate_local_alignment(
                model,
                local_loader,
                device,
            )
        )
    manifest = build_alignment_result_manifest(
        metrics=metrics,
        diagnostics={
            "shuffled_global_retrieval": shuffled_retrieval,
        },
        checkpoint=model_identity,
        evaluator=checkpoint_manifest(paths.evaluator_checkpoint),
        resolved_config=resolved_config,
        global_dataset=global_dataset,
        local_dataset=local_dataset,
        local_scope=args.local_target_scope,
        seed=args.seed,
        batch_size=args.batch_size,
        skating_config=skating_config,
    )
    report = write_alignment_result_artifacts(
        manifest,
        paths.output_dir,
    )
    print(report, end="")
    print("Artifacts: {}".format(paths.output_dir))


if __name__ == "__main__":
    main()
