"""Standard metrics for deterministic MSA-VAE reconstruction evaluation."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.ndimage import uniform_filter1d


def _batch_compute_similarity_transform(prediction, target):
    transposed = False
    if prediction.shape[0] not in (2, 3):
        prediction = prediction.permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        transposed = True

    prediction_mean = prediction.mean(axis=-1, keepdims=True)
    target_mean = target.mean(axis=-1, keepdims=True)
    prediction_centered = prediction - prediction_mean
    target_centered = target - target_mean
    prediction_variance = torch.sum(prediction_centered ** 2, dim=1).sum(dim=1)
    covariance = prediction_centered.bmm(
        target_centered.permute(0, 2, 1)
    )
    left, _, right = torch.svd(covariance)
    reflection = torch.eye(
        left.shape[1],
        dtype=left.dtype,
        device=left.device,
    ).unsqueeze(0).repeat(left.shape[0], 1, 1)
    reflection[:, -1, -1] *= torch.sign(
        torch.det(left.bmm(right.permute(0, 2, 1)))
    )
    rotation = right.bmm(reflection.bmm(left.permute(0, 2, 1)))
    scale = torch.stack([torch.trace(item) for item in rotation.bmm(covariance)])
    scale = scale / prediction_variance
    translation = target_mean - (
        scale.unsqueeze(-1).unsqueeze(-1) * rotation.bmm(prediction_mean)
    )
    aligned = (
        scale.unsqueeze(-1).unsqueeze(-1) * rotation.bmm(prediction)
        + translation
    )
    if transposed:
        aligned = aligned.permute(0, 2, 1)
    return aligned


def calc_mpjpe(prediction, target, align_inds=(0,)):
    """MLD-compatible root-aligned per-frame MPJPE in input distance units."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target joint shapes must match")
    valid_mask = target[:, :, 0] != -2.0
    if align_inds is not None:
        prediction = prediction - prediction[:, align_inds].mean(
            dim=1,
            keepdim=True,
        )
        target = target - target[:, align_inds].mean(dim=1, keepdim=True)
    errors = torch.linalg.norm(prediction - target, dim=-1)
    return (errors * valid_mask.float()).sum(-1) / valid_mask.float().sum(-1)


def calc_pampjpe(prediction, target):
    """MLD-compatible per-frame Procrustes-aligned MPJPE."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target joint shapes must match")
    aligned = _batch_compute_similarity_transform(
        prediction.float(),
        target.float(),
    )
    return torch.linalg.norm(aligned - target.float(), dim=-1).mean(-1)


def calc_accel(prediction, target):
    """MLD-compatible per-frame joint acceleration error."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target joint shapes must match")
    prediction_accel = prediction[:-2] - 2 * prediction[1:-1] + prediction[2:]
    target_accel = target[:-2] - 2 * target[1:-1] + target[2:]
    return torch.linalg.norm(prediction_accel - target_accel, dim=-1).mean(1)


@dataclass(frozen=True)
class SkatingConfig:
    foot_indices: Tuple[int, int] = (10, 11)
    fps: float = 30.0
    height_threshold_m: float = 0.05
    velocity_threshold_mps: float = 0.50
    smoothing_window_frames: int = 8

    def __post_init__(self):
        if len(self.foot_indices) != 2:
            raise ValueError("skating requires exactly two foot indices")
        if self.fps <= 0:
            raise ValueError("skating FPS must be positive")
        if self.smoothing_window_frames <= 0:
            raise ValueError("skating smoothing window must be positive")


def _calculate_skating_ratio(joints, config):
    feet = joints[:, config.foot_indices, :].detach().cpu().numpy()
    horizontal_displacement = feet[1:, :, (0, 2)] - feet[:-1, :, (0, 2)]
    instantaneous_speed = (
        np.linalg.norm(horizontal_displacement, axis=-1).T * config.fps
    )
    smoothed_speed = uniform_filter1d(
        instantaneous_speed,
        axis=-1,
        size=config.smoothing_window_frames,
        mode="constant",
        origin=0,
    )
    heights = feet[:, :, 1].T
    contact = np.logical_and(
        heights[:, :-1] < config.height_threshold_m,
        heights[:, 1:] < config.height_threshold_m,
    )
    skating = np.logical_and(
        contact,
        instantaneous_speed > config.velocity_threshold_mps,
    )
    skating = np.logical_and(
        skating,
        smoothed_speed > config.velocity_threshold_mps,
    )
    either_foot = np.logical_or(skating[0], skating[1])
    return float(either_foot.mean())


class ReconstructionMetricAccumulator:
    """Length-safe MLD reconstruction errors plus OmniControl foot skating."""

    def __init__(self, skating_config=None):
        self.skating_config = skating_config or SkatingConfig()
        self.mpjpe_sum = 0.0
        self.p_mpjpe_sum = 0.0
        self.accel_sum = 0.0
        self.frame_count = 0
        self.accel_frame_count = 0
        self.skating_ratios = []

    @staticmethod
    def _validate(prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target joint shapes must match")
        if prediction.ndim != 3 or prediction.shape[1:] != (22, 3):
            raise ValueError("joint sequences must have shape (frames, 22, 3)")
        if prediction.shape[0] < 3:
            raise ValueError("joint sequences need at least 3 valid frames")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("joint sequences contain non-finite values")

    def update(self, prediction, target):
        prediction = torch.as_tensor(prediction).detach().cpu().float()
        target = torch.as_tensor(target).detach().cpu().float()
        self._validate(prediction, target)
        frame_count = int(prediction.shape[0])
        self.mpjpe_sum += float(calc_mpjpe(prediction, target).sum())
        self.p_mpjpe_sum += float(calc_pampjpe(prediction, target).sum())
        self.accel_sum += float(calc_accel(prediction, target).sum())
        self.frame_count += frame_count
        self.accel_frame_count += frame_count - 2
        self.skating_ratios.append(
            _calculate_skating_ratio(prediction, self.skating_config)
        )

    def compute(self):
        if self.frame_count == 0 or not self.skating_ratios:
            raise ValueError("no reconstruction samples were accumulated")
        metrics = {
            "mpjpe_mm": self.mpjpe_sum / self.frame_count * 1000.0,
            "p_mpjpe_mm": self.p_mpjpe_sum / self.frame_count * 1000.0,
            "accel_mm_per_frame2": (
                self.accel_sum / self.accel_frame_count * 1000.0
            ),
            "skating_percent": float(np.mean(self.skating_ratios) * 100.0),
        }
        _validate_finite_metrics(metrics)
        return metrics


def _as_embedding_array(value, label):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("{} embeddings must be a 2D array".format(label))
    if not np.isfinite(array).all():
        raise ValueError("{} embeddings contain non-finite values".format(label))
    return array


def calculate_fid(reference, prediction, eps=1e-6):
    reference = _as_embedding_array(reference, "reference")
    prediction = _as_embedding_array(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction embeddings must have matching shapes")
    if reference.shape[0] < 2:
        raise ValueError("FID requires at least two embedding samples")
    reference_mean = np.mean(reference, axis=0)
    prediction_mean = np.mean(prediction, axis=0)
    reference_cov = np.cov(reference, rowvar=False)
    prediction_cov = np.cov(prediction, rowvar=False)
    mean_difference = reference_mean - prediction_mean
    covariance_mean, _ = linalg.sqrtm(
        reference_cov.dot(prediction_cov),
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(reference_cov.shape[0]) * eps
        covariance_mean = linalg.sqrtm(
            (reference_cov + offset).dot(prediction_cov + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(
            np.diagonal(covariance_mean).imag,
            0,
            atol=1e-3,
        ):
            raise ValueError("FID covariance square root has a large imaginary part")
        covariance_mean = covariance_mean.real
    value = float(
        mean_difference.dot(mean_difference)
        + np.trace(reference_cov)
        + np.trace(prediction_cov)
        - 2 * np.trace(covariance_mean)
    )
    value = max(value, 0.0)
    if not np.isfinite(value):
        raise ValueError("FID is non-finite")
    return value


def _diagonal_ranks_with_average_ties(similarity):
    ranks = []
    for index, row in enumerate(similarity):
        positive = row[index]
        better = np.count_nonzero(row > positive)
        tied = np.count_nonzero(row == positive)
        ranks.append(float(better) + (float(tied) - 1.0) / 2.0)
    return np.asarray(ranks, dtype=np.float64)


def _rank_metrics(ranks, prefix):
    return {
        "{}_r1_percent".format(prefix): float(np.mean(ranks < 1) * 100.0),
        "{}_r2_percent".format(prefix): float(np.mean(ranks < 2) * 100.0),
        "{}_r3_percent".format(prefix): float(np.mean(ranks < 3) * 100.0),
        "{}_r5_percent".format(prefix): float(np.mean(ranks < 5) * 100.0),
        "{}_medr".format(prefix): float(np.median(ranks) + 1.0),
    }


def retrieval_metrics_from_similarity(similarity):
    similarity = np.asarray(similarity, dtype=np.float64)
    if (
        similarity.ndim != 2
        or similarity.shape[0] != similarity.shape[1]
        or similarity.shape[0] == 0
    ):
        raise ValueError("TMR-full-normal requires a non-empty square matrix")
    if not np.isfinite(similarity).all():
        raise ValueError("retrieval similarity contains non-finite values")
    metrics = _rank_metrics(
        _diagonal_ranks_with_average_ties(similarity),
        "t2m",
    )
    metrics.update(
        _rank_metrics(
            _diagonal_ranks_with_average_ties(similarity.T),
            "m2t",
        )
    )
    _validate_finite_metrics(metrics)
    return metrics


def calculate_bidirectional_retrieval(text_embeddings, motion_embeddings):
    text = torch.as_tensor(text_embeddings).detach().cpu().float()
    motion = torch.as_tensor(motion_embeddings).detach().cpu().float()
    if text.ndim != 2 or motion.ndim != 2:
        raise ValueError("retrieval embeddings must be 2D")
    if text.shape != motion.shape:
        raise ValueError("text and motion embeddings must have matching shapes")
    if not torch.isfinite(text).all() or not torch.isfinite(motion).all():
        raise ValueError("retrieval embeddings contain non-finite values")
    similarity = F.normalize(text, dim=-1).mm(F.normalize(motion, dim=-1).T)
    return retrieval_metrics_from_similarity(similarity.numpy())


def _validate_finite_metrics(metrics: Dict[str, float]):
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise ValueError("{} is non-finite".format(name))
