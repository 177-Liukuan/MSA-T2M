"""Shared temporal preprocessing for HumanML3D/BABEL local text targets."""

import numpy as np


def _pool_to_latent(text_window, latent_length):
    if text_window.shape[0] == latent_length:
        return text_window.astype(np.float32, copy=False)
    boundaries = np.linspace(
        0,
        text_window.shape[0],
        latent_length + 1,
    ).astype(int)
    pooled = np.empty(
        (latent_length, text_window.shape[1]),
        dtype=np.float32,
    )
    for index in range(latent_length):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            raise ValueError("latent pooling produced an empty interval")
        pooled[index] = text_window[start:end].mean(axis=0)
    return pooled


def build_local_text_target(
    local_text,
    raw_motion_length,
    view_start,
    view_length,
    latent_length,
    expected_dim,
):
    """Map frame text to a motion view and average-pool it to latent rate."""
    local_text = np.asarray(local_text, dtype=np.float32)
    raw_motion_length = int(raw_motion_length)
    view_start = int(view_start)
    view_length = int(view_length)
    latent_length = int(latent_length)
    expected_dim = int(expected_dim)
    if local_text.ndim != 2:
        raise ValueError("local text target must be a 2D array")
    if local_text.shape[0] < 1:
        raise ValueError("local text target must contain at least one frame")
    if local_text.shape[1] != expected_dim:
        raise ValueError(
            "local text target dimension does not match expected dimension"
        )
    if not np.isfinite(local_text).all():
        raise ValueError("local text target contains non-finite values")
    if raw_motion_length < 1:
        raise ValueError("raw_motion_length must be positive")
    if (
        view_start < 0
        or view_length < 1
        or view_start + view_length > raw_motion_length
    ):
        raise ValueError("view must lie inside the raw motion")
    if latent_length < 1 or latent_length > view_length:
        raise ValueError(
            "latent_length must be between one and view_length"
        )

    frame_indices = np.round(
        np.linspace(0, local_text.shape[0] - 1, raw_motion_length)
    ).astype(int)
    motion_rate = local_text[frame_indices]
    view = motion_rate[view_start:view_start + view_length]
    pooled = view.mean(axis=0).astype(np.float32)
    latent = _pool_to_latent(view, latent_length)
    return latent.astype(np.float32), pooled
