"""Deterministic training-time validation helpers for MSA-VAE."""

import contextlib
import math
import os
import random
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from utils.msa_vae_training import save_msa_checkpoint


VALIDATION_METRIC_KEYS = (
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
class MSAValidationState:
    best_fid: float = float("inf")
    best_mpjpe: float = float("inf")


@contextlib.contextmanager
def isolated_validation_rng(seed):
    """Temporarily seed validation without perturbing caller RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def run_deterministic_msa_validation(
    model,
    evaluator,
    loader,
    device,
    seed,
    skating_config=None,
):
    """Run the standard metric core without changing caller RNG or mode."""
    from eval_msa_vae_metrics import evaluate_msa_vae_metrics
    from utils.msa_vae_metrics import SkatingConfig

    was_training = model.training
    try:
        with isolated_validation_rng(seed):
            return evaluate_msa_vae_metrics(
                model,
                evaluator,
                loader,
                device,
                skating_config or SkatingConfig(),
            )
    finally:
        model.train(was_training)


def _validated_result(result):
    if not isinstance(result, dict):
        raise ValueError("validation result must be a dictionary")
    sample_count = result.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("validation sample_count must be a positive integer")
    validated = {"sample_count": sample_count}
    for name in VALIDATION_METRIC_KEYS:
        value = result.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"validation metric {name} must be a finite number"
            )
        validated[name] = float(value)
    return validated


def publish_msa_validation(
    result,
    iteration,
    out_dir,
    model,
    metadata,
    state,
    logger,
    writer,
    validation_seed,
    validation_batch_size,
):
    """Log deterministic metrics and publish strict-best plus last states."""
    result = _validated_result(result)
    if not isinstance(state, MSAValidationState):
        raise ValueError("validation state has the wrong type")
    os.makedirs(os.fspath(out_dir), exist_ok=True)

    logger.info(
        "Complete deterministic val: "
        f"iteration={iteration} samples={result['sample_count']} "
        f"seed={validation_seed} batch_size={validation_batch_size} "
        f"FID={result['fid']:.6f} "
        f"MPJPE={result['mpjpe_mm']:.3f}mm"
    )
    writer.add_scalar(
        "CompleteVal/sample_count",
        result["sample_count"],
        iteration,
    )
    for name in VALIDATION_METRIC_KEYS:
        writer.add_scalar(
            f"CompleteVal/{name}",
            result[name],
            iteration,
        )

    best_fid = state.best_fid
    if result["fid"] < best_fid:
        logger.info(
            f"Complete val FID improved from {best_fid:.6f} "
            f"to {result['fid']:.6f}"
        )
        best_fid = result["fid"]
        save_msa_checkpoint(
            os.path.join(out_dir, "net_best_fid.pth"),
            model,
            metadata,
        )

    best_mpjpe = state.best_mpjpe
    if result["mpjpe_mm"] < best_mpjpe:
        logger.info(
            f"Complete val MPJPE improved from {best_mpjpe:.3f} "
            f"to {result['mpjpe_mm']:.3f}"
        )
        best_mpjpe = result["mpjpe_mm"]
        save_msa_checkpoint(
            os.path.join(out_dir, "net_best_mpjpe.pth"),
            model,
            metadata,
        )

    save_msa_checkpoint(
        os.path.join(out_dir, "net_last.pth"),
        model,
        metadata,
    )
    return MSAValidationState(
        best_fid=best_fid,
        best_mpjpe=best_mpjpe,
    )
