import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.msa_vae_metrics import (
    ReconstructionMetricAccumulator,
    SkatingConfig,
    calc_accel,
    calc_mpjpe,
    calc_pampjpe,
    calculate_bidirectional_retrieval,
    calculate_fid,
    retrieval_metrics_from_similarity,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_mld_reference():
    path = ROOT / "Evaluator_272" / "mld" / "models" / "metrics" / "utils.py"
    spec = importlib.util.spec_from_file_location("_mld_metric_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLD_REFERENCE = _load_mld_reference()


class ReconstructionMetricsTest(unittest.TestCase):
    @staticmethod
    def non_degenerate_joints(frames=5):
        generator = torch.Generator().manual_seed(1729)
        joints = torch.randn(frames, 22, 3, generator=generator)
        joints += torch.arange(frames, dtype=torch.float32)[:, None, None] * 0.1
        return joints

    @staticmethod
    def stationary_feet(frames):
        joints = torch.zeros(frames, 22, 3)
        joints[:, [10, 11], 1] = 0.01
        return joints

    def test_local_motion_errors_match_tracked_mld_reference(self):
        target = self.non_degenerate_joints(frames=7)
        prediction = target + torch.randn(
            target.shape,
            generator=torch.Generator().manual_seed(42),
        ) * 0.03

        np.testing.assert_allclose(
            calc_mpjpe(prediction, target).numpy(),
            MLD_REFERENCE.calc_mpjpe(prediction, target).numpy(),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            calc_pampjpe(prediction, target).numpy(),
            MLD_REFERENCE.calc_pampjpe(prediction, target).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            calc_accel(prediction, target).numpy(),
            MLD_REFERENCE.calc_accel(prediction, target).numpy(),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_mpjpe_root_alignment_removes_global_translation(self):
        target = self.non_degenerate_joints(frames=4)
        prediction = target + torch.tensor([3.0, 0.0, -2.0])
        accumulator = ReconstructionMetricAccumulator()

        accumulator.update(prediction, target)

        self.assertAlmostEqual(accumulator.compute()["mpjpe_mm"], 0.0, places=3)

    def test_p_mpjpe_removes_per_frame_similarity_transform(self):
        target = self.non_degenerate_joints(frames=4)
        prediction = 2.5 * target + torch.tensor([1.0, -3.0, 2.0])
        accumulator = ReconstructionMetricAccumulator()

        accumulator.update(prediction, target)

        self.assertAlmostEqual(
            accumulator.compute()["p_mpjpe_mm"],
            0.0,
            places=3,
        )

    def test_accel_uses_mld_millimetres_per_frame_squared(self):
        target = self.non_degenerate_joints(frames=5)
        prediction = target.clone()
        prediction[:, :, 0] += torch.arange(
            5,
            dtype=torch.float32,
        ).square()[:, None]
        accumulator = ReconstructionMetricAccumulator()

        accumulator.update(prediction, target)

        self.assertAlmostEqual(
            accumulator.compute()["accel_mm_per_frame2"],
            2000.0,
            places=3,
        )

    def test_skating_is_sample_mean_and_uses_only_each_valid_sequence(self):
        sliding = self.stationary_feet(frames=12)
        sliding[:, [10, 11], 0] = torch.arange(12)[:, None] * 0.1
        still = self.stationary_feet(frames=20)
        accumulator = ReconstructionMetricAccumulator()

        accumulator.update(sliding, sliding.clone())
        accumulator.update(still, still.clone())

        self.assertAlmostEqual(
            accumulator.compute()["skating_percent"],
            50.0,
            places=4,
        )

    def test_rejects_too_short_or_non_finite_joint_sequences(self):
        accumulator = ReconstructionMetricAccumulator()
        with self.assertRaisesRegex(ValueError, "at least 3"):
            accumulator.update(torch.zeros(2, 22, 3), torch.zeros(2, 22, 3))
        invalid = torch.zeros(3, 22, 3)
        invalid[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            accumulator.update(invalid, torch.zeros_like(invalid))


class DistributionAndRetrievalMetricsTest(unittest.TestCase):
    def test_fid_of_identical_embeddings_is_zero(self):
        embeddings = np.arange(24, dtype=np.float64).reshape(6, 4)

        fid = calculate_fid(embeddings, embeddings)

        self.assertAlmostEqual(fid, 0.0, places=8)

    def test_bidirectional_retrieval_uses_full_matrix_and_transpose(self):
        similarity = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.8, 0.0],
                [0.0, 0.7, 1.0],
            ]
        )

        metrics = retrieval_metrics_from_similarity(similarity)

        self.assertAlmostEqual(metrics["t2m_r1_percent"], 200.0 / 3.0)
        self.assertEqual(metrics["t2m_r2_percent"], 100.0)
        self.assertEqual(metrics["t2m_r3_percent"], 100.0)
        self.assertEqual(metrics["t2m_medr"], 1.0)
        self.assertEqual(metrics["m2t_r1_percent"], 100.0)
        self.assertEqual(metrics["m2t_r2_percent"], 100.0)
        self.assertEqual(metrics["m2t_r3_percent"], 100.0)
        self.assertEqual(metrics["m2t_medr"], 1.0)

    def test_retrieval_averages_exact_ties_like_tmr(self):
        metrics = retrieval_metrics_from_similarity(np.zeros((2, 2)))

        self.assertEqual(metrics["t2m_r1_percent"], 100.0)
        self.assertEqual(metrics["m2t_r1_percent"], 100.0)
        self.assertEqual(metrics["t2m_medr"], 1.5)
        self.assertEqual(metrics["m2t_medr"], 1.5)

    def test_retrieval_r5_includes_rank_four_and_excludes_rank_five(self):
        similarity = np.eye(6)
        similarity[0] = np.array([0.0, 4.0, 3.0, 2.0, 1.0, -1.0])
        similarity[1] = np.array([5.0, 0.0, 4.0, 3.0, 2.0, 1.0])

        metrics = retrieval_metrics_from_similarity(similarity)

        self.assertAlmostEqual(metrics["t2m_r5_percent"], 500.0 / 6.0)
        self.assertIn("m2t_r5_percent", metrics)

    def test_bidirectional_retrieval_l2_normalizes_embeddings(self):
        text = torch.tensor([[10.0, 0.0], [0.0, 2.0]])
        motion = torch.tensor([[3.0, 0.0], [0.0, 7.0]])

        metrics = calculate_bidirectional_retrieval(text, motion)

        self.assertEqual(metrics["t2m_r1_percent"], 100.0)
        self.assertEqual(metrics["m2t_r1_percent"], 100.0)

    def test_rejects_non_square_or_non_finite_retrieval_inputs(self):
        with self.assertRaisesRegex(ValueError, "square"):
            retrieval_metrics_from_similarity(np.zeros((2, 3)))
        invalid = np.eye(2)
        invalid[0, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            retrieval_metrics_from_similarity(invalid)


class SkatingProtocolTest(unittest.TestCase):
    def test_defaults_record_humanml_30fps_omnicontrol_adaptation(self):
        config = SkatingConfig()

        self.assertEqual(config.foot_indices, (10, 11))
        self.assertEqual(config.fps, 30.0)
        self.assertEqual(config.height_threshold_m, 0.05)
        self.assertEqual(config.velocity_threshold_mps, 0.50)
        self.assertEqual(config.smoothing_window_frames, 8)


if __name__ == "__main__":
    unittest.main()
