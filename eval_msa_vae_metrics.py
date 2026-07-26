"""Standalone standard HumanML3D-272 evaluation for MSA-VAE checkpoints."""

import argparse
import csv
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from humanml3d_272.dataset_eval_msa_vae_metrics import (
    MSAVAEMetricsDataset,
    make_msa_vae_metrics_loader,
)
from utils.eval_trans import recover_from_local_position
from utils.msa_vae_eval_config import (
    build_and_load_msa_vae,
    checkpoint_manifest,
)
from utils.msa_vae_metrics import (
    ReconstructionMetricAccumulator,
    SkatingConfig,
    calculate_bidirectional_retrieval,
    calculate_fid,
)


METRIC_KEYS = (
    "fid",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "accel_mm_per_frame2",
    "skating_percent",
    "t2m_r1_percent",
    "t2m_r2_percent",
    "t2m_r3_percent",
    "t2m_r5_percent",
    "t2m_medr",
    "m2t_r1_percent",
    "m2t_r2_percent",
    "m2t_r3_percent",
    "m2t_r5_percent",
    "m2t_medr",
)


@dataclass(frozen=True)
class EvaluationPaths:
    checkpoint: Path
    data_root: Path
    split_file: Path
    evaluator_root: Path
    evaluator_checkpoint: Path
    distilbert_root: Path
    output_dir: Path


def _repository_path(repo_root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one HumanML3D-272 MSA-VAE checkpoint with FID, "
            "MPJPE, P-MPJPE, ACCEL, foot skating, and bidirectional "
            "TMR-full-normal retrieval."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="MSA-VAE .pth checkpoint path")
    parser.add_argument("--data-root", default="humanml3d_272")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--evaluator-root", default="Evaluator_272")
    parser.add_argument(
        "--evaluator-checkpoint",
        default=None,
    )
    parser.add_argument(
        "--distilbert-root",
        default=None,
    )
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
    distilbert_root = (
        _repository_path(repo_root, args.distilbert_root)
        if args.distilbert_root
        else evaluator_root / "deps" / "distilbert-base-uncased"
    )
    if args.output_dir:
        output_dir = _repository_path(repo_root, args.output_dir)
    else:
        output_dir = (
            repo_root
            / "output"
            / "msa_vae_metrics"
            / checkpoint.parent.name
            / checkpoint.stem
        )
    return EvaluationPaths(
        checkpoint=checkpoint,
        data_root=data_root,
        split_file=split_file.resolve(),
        evaluator_root=evaluator_root,
        evaluator_checkpoint=evaluator_checkpoint,
        distilbert_root=distilbert_root,
        output_dir=output_dir.resolve(),
    )


def preflight_evaluation_assets(paths):
    required = (
        ("MSA-VAE checkpoint", paths.checkpoint, "file"),
        ("HumanML3D-272 data root", paths.data_root, "directory"),
        ("HumanML3D-272 test split", paths.split_file, "file"),
        ("HumanML3D-272 motion_data", paths.data_root / "motion_data", "directory"),
        ("HumanML3D-272 texts", paths.data_root / "texts", "directory"),
        ("HumanML3D-272 Mean.npy", paths.data_root / "mean_std" / "Mean.npy", "file"),
        ("HumanML3D-272 Std.npy", paths.data_root / "mean_std" / "Std.npy", "file"),
        ("272-D evaluator root", paths.evaluator_root, "directory"),
        ("272-D evaluator checkpoint", paths.evaluator_checkpoint, "file"),
        ("DistilBERT evaluator dependency", paths.distilbert_root, "directory"),
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
    )
    missing = []
    for label, path, kind in required:
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            missing.append("{}: {}".format(label, path))
    if missing:
        raise FileNotFoundError(
            "MSA-VAE evaluation preflight failed:\n- " + "\n- ".join(missing)
        )


def validate_runtime_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")


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


def _distribution_location(output, label):
    location = getattr(output, "loc", None)
    if not isinstance(location, torch.Tensor) or location.ndim != 2:
        raise ValueError("{} must return a distribution with 2D .loc".format(label))
    if not torch.isfinite(location).all():
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


@torch.inference_mode()
def evaluate_msa_vae_metrics(
    model,
    evaluator,
    loader,
    device,
    skating_config,
):
    model.eval()
    text_encoder, motion_encoder = evaluator
    text_encoder.eval()
    motion_encoder.eval()
    reconstruction = ReconstructionMetricAccumulator(skating_config)
    text_embeddings = []
    reference_embeddings = []
    prediction_embeddings = []
    sample_count = 0

    for batch in loader:
        motions = batch["motions"].to(device).float()
        lengths = batch["lengths"].to(device).long()
        outputs = model(motions, lengths=lengths)
        if not isinstance(outputs, dict) or "x_recon" not in outputs:
            raise ValueError("MSA-VAE forward must return an x_recon tensor")
        predictions = outputs["x_recon"]
        if predictions.shape != motions.shape:
            raise ValueError(
                "MSA-VAE reconstruction shape {} does not match input {}".format(
                    tuple(predictions.shape),
                    tuple(motions.shape),
                )
            )
        if not torch.isfinite(predictions).all():
            raise ValueError("MSA-VAE reconstruction contains non-finite values")

        frame_index = torch.arange(
            predictions.shape[1],
            device=predictions.device,
        ).unsqueeze(0)
        padding_mask = frame_index >= lengths.unsqueeze(1)
        prediction_for_encoder = predictions.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )
        text_embeddings.append(
            _distribution_location(
                text_encoder(batch["captions"]),
                "text encoder",
            )
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
        sample_count += len(batch["sample_ids"])

    if sample_count == 0:
        raise ValueError("MSA-VAE evaluation loader yielded no samples")
    text_embeddings = torch.cat(text_embeddings, dim=0)
    reference_embeddings = torch.cat(reference_embeddings, dim=0)
    prediction_embeddings = torch.cat(prediction_embeddings, dim=0)
    embedding_counts = {
        text_embeddings.shape[0],
        reference_embeddings.shape[0],
        prediction_embeddings.shape[0],
        sample_count,
    }
    if len(embedding_counts) != 1:
        raise ValueError(
            "evaluator embedding count does not match evaluated sample count"
        )

    metrics = reconstruction.compute()
    metrics["fid"] = calculate_fid(
        reference_embeddings,
        prediction_embeddings,
    )
    metrics.update(
        calculate_bidirectional_retrieval(
            text_embeddings,
            prediction_embeddings,
        )
    )
    return {"sample_count": sample_count, **metrics}


def load_evaluator_checkpoint(path, evaluator_root=None):
    """Load the trusted frozen Lightning evaluator, including NumPy scalars."""
    root = None if evaluator_root is None else str(Path(evaluator_root).resolve())
    inserted = root is not None and root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        payload = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    finally:
        if inserted:
            sys.path.remove(root)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"),
        dict,
    ):
        raise ValueError("272-D evaluator checkpoint has no state_dict")
    return payload


def load_frozen_humanml_evaluator(
    evaluator_root,
    device,
    evaluator_checkpoint,
    distilbert_root,
):
    evaluator_root = Path(evaluator_root).resolve()
    inserted = str(evaluator_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(evaluator_root))
    try:
        from mld.models.architectures.temos.motionencoder.actor import (
            ActorAgnosticEncoder,
        )
        from mld.models.architectures.temos.textencoder.distillbert_actor import (
            DistilbertActorAgnosticEncoder,
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
        text_encoder = DistilbertActorAgnosticEncoder(
            str(distilbert_root),
            num_layers=4,
            latent_dim=256,
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
    state_dict = evaluator_payload["state_dict"]
    for prefix, encoder in (
        ("textencoder", text_encoder),
        ("motionencoder", motion_encoder),
    ):
        state = {
            key[len(prefix) + 1 :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix + ".")
        }
        if not state:
            raise ValueError(
                "272-D evaluator checkpoint has no {} weights".format(prefix)
            )
        encoder.load_state_dict(state, strict=True)
        encoder.eval().to(device)
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    return [text_encoder, motion_encoder]


def build_result_manifest(
    metrics,
    checkpoint,
    evaluator,
    resolved_config,
    dataset,
    seed,
    batch_size,
    skating_config,
):
    missing_metrics = [key for key in METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ValueError(
            "result is missing metrics: {}".format(", ".join(missing_metrics))
        )
    metric_values = {key: float(metrics[key]) for key in METRIC_KEYS}
    if not all(np.isfinite(value) for value in metric_values.values()):
        raise ValueError("result metrics contain non-finite values")
    sample_count = int(metrics["sample_count"])
    if sample_count != len(dataset.sample_ids):
        raise ValueError("result sample count does not match deterministic dataset")
    return {
        "protocol": {
            "version": "msa-vae-standard-v2",
            "dataset": "HumanML3D-272-complete-motion-test",
            "retrieval": "TMR-full-normal",
            "retrieval_similarity": "L2-normalized cosine",
            "retrieval_ties": "average rank",
            "caption_policy": "first complete-motion caption",
            "acceleration_scaling": "finite difference; no FPS^2 multiplier",
        },
        "metrics": metric_values,
        "units": {
            "fid": "embedding distance",
            "mpjpe_mm": "mm",
            "p_mpjpe_mm": "mm",
            "accel_mm_per_frame2": "mm/frame^2",
            "skating_percent": "percent",
            "t2m_r1_percent": "percent",
            "t2m_r2_percent": "percent",
            "t2m_r3_percent": "percent",
            "t2m_r5_percent": "percent",
            "t2m_medr": "rank",
            "m2t_r1_percent": "percent",
            "m2t_r2_percent": "percent",
            "m2t_r3_percent": "percent",
            "m2t_r5_percent": "percent",
            "m2t_medr": "rank",
        },
        "checkpoint": dict(checkpoint),
        "evaluator": dict(evaluator),
        "model_config": {
            "values": dict(resolved_config.values),
            "sources": dict(resolved_config.sources),
        },
        "dataset": {
            "sample_count": sample_count,
            "sample_hash": dataset.sample_hash,
            "sample_ids": list(dataset.sample_ids),
        },
        "seed": int(seed),
        "batch_size": int(batch_size),
        "skating": asdict(skating_config),
    }


def _format_final_report(manifest):
    metrics = manifest["metrics"]
    return (
        "MSA-VAE standard evaluation ({protocol})\n"
        "Samples: {samples}\n"
        "FID {fid:.6f} | MPJPE {mpjpe:.3f} mm | "
        "P-MPJPE {pmpjpe:.3f} mm | ACCEL {accel:.3f} mm/frame^2 | "
        "Skating {skating:.3f}%\n"
        "Text-to-motion: R@1 {t1:.3f}% | R@2 {t2:.3f}% | "
        "R@3 {t3:.3f}% | R@5 {t5:.3f}% | MedR {tm:.3f}\n"
        "Motion-to-text: R@1 {m1:.3f}% | R@2 {m2:.3f}% | "
        "R@3 {m3:.3f}% | R@5 {m5:.3f}% | MedR {mm:.3f}\n"
    ).format(
        protocol=manifest["protocol"]["retrieval"],
        samples=manifest["dataset"]["sample_count"],
        fid=metrics["fid"],
        mpjpe=metrics["mpjpe_mm"],
        pmpjpe=metrics["p_mpjpe_mm"],
        accel=metrics["accel_mm_per_frame2"],
        skating=metrics["skating_percent"],
        t1=metrics["t2m_r1_percent"],
        t2=metrics["t2m_r2_percent"],
        t3=metrics["t2m_r3_percent"],
        t5=metrics["t2m_r5_percent"],
        tm=metrics["t2m_medr"],
        m1=metrics["m2t_r1_percent"],
        m2=metrics["m2t_r2_percent"],
        m3=metrics["m2t_r3_percent"],
        m5=metrics["m2t_r5_percent"],
        mm=metrics["m2t_medr"],
    )


def write_result_artifacts(manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    row = {
        "protocol_version": manifest["protocol"]["version"],
        "retrieval_protocol": manifest["protocol"]["retrieval"],
        "checkpoint_path": manifest["checkpoint"]["path"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "evaluator_path": manifest["evaluator"]["path"],
        "evaluator_sha256": manifest["evaluator"]["sha256"],
        "sample_count": manifest["dataset"]["sample_count"],
        "sample_hash": manifest["dataset"]["sample_hash"],
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
    (output_dir / "evaluation.log").write_text(report, encoding="utf-8")
    return report


def _resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation was requested but CUDA is not available; "
            "pass --device cpu only for smoke tests"
        )
    return device


def main(argv=None):
    repo_root = Path(__file__).resolve().parent
    args = parse_args(argv)
    paths = resolve_cli_paths(repo_root, args)
    preflight_evaluation_assets(paths)
    validate_runtime_args(args)
    device = _resolve_device(args.device)
    set_evaluation_seed(args.seed)

    model, resolved_config, model_identity = build_and_load_msa_vae(
        paths.checkpoint,
        architecture_overrides(args),
        device,
    )
    unit_length = (
        resolved_config.values["stride_t"]
        ** resolved_config.values["down_t"]
    )
    dataset = MSAVAEMetricsDataset(
        paths.data_root,
        split_file=paths.split_file,
        unit_length=unit_length,
    )
    loader = make_msa_vae_metrics_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    evaluator = load_frozen_humanml_evaluator(
        paths.evaluator_root,
        device,
        paths.evaluator_checkpoint,
        paths.distilbert_root,
    )
    skating_config = SkatingConfig()
    metrics = evaluate_msa_vae_metrics(
        model,
        evaluator,
        loader,
        device,
        skating_config,
    )
    manifest = build_result_manifest(
        metrics=metrics,
        checkpoint=model_identity,
        evaluator=checkpoint_manifest(paths.evaluator_checkpoint),
        resolved_config=resolved_config,
        dataset=dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        skating_config=skating_config,
    )
    report = write_result_artifacts(manifest, paths.output_dir)
    print(report, end="")
    print("Artifacts: {}".format(paths.output_dir))


if __name__ == "__main__":
    main()
