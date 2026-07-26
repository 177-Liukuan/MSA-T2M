"""Deterministic training-time validation helpers for MSA-VAE."""

import contextlib
import random

import numpy as np
import torch


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
