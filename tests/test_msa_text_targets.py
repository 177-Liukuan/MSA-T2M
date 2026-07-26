import unittest

import numpy as np

from humanml3d_272.msa_text_targets import build_local_text_target


class LocalTextTargetTest(unittest.TestCase):
    def test_resamples_to_raw_motion_then_crops_and_pools(self):
        source = np.array(
            [
                [0.0, 0.0],
                [10.0, -10.0],
                [20.0, -20.0],
                [30.0, -30.0],
            ],
            dtype=np.float32,
        )

        latent, pooled = build_local_text_target(
            source,
            raw_motion_length=6,
            view_start=0,
            view_length=4,
            latent_length=2,
            expected_dim=2,
        )

        np.testing.assert_allclose(
            latent,
            np.array([[5.0, -5.0], [15.0, -15.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            pooled,
            np.array([10.0, -10.0], dtype=np.float32),
        )

    def test_identity_rate_preserves_training_window_alignment(self):
        source = np.stack(
            [
                np.arange(70, dtype=np.float32),
                -np.arange(70, dtype=np.float32),
            ],
            axis=1,
        )

        latent, pooled = build_local_text_target(
            source,
            raw_motion_length=70,
            view_start=4,
            view_length=64,
            latent_length=16,
            expected_dim=2,
        )

        np.testing.assert_allclose(latent[0], np.array([5.5, -5.5]))
        np.testing.assert_allclose(pooled, np.array([35.5, -35.5]))

    def test_rejects_invalid_shape_length_crop_and_dimension(self):
        valid = np.ones((4, 2), dtype=np.float32)
        cases = (
            (valid[:, 0], 4, 0, 4, 1, 2, "2D"),
            (valid, 0, 0, 4, 1, 2, "raw_motion_length"),
            (valid, 4, 3, 2, 1, 2, "view"),
            (valid, 4, 0, 4, 0, 2, "latent_length"),
            (valid, 4, 0, 4, 1, 3, "dimension"),
        )
        for value, raw, start, length, latent, dim, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_local_text_target(
                        value,
                        raw_motion_length=raw,
                        view_start=start,
                        view_length=length,
                        latent_length=latent,
                        expected_dim=dim,
                    )

        non_finite = valid.copy()
        non_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_local_text_target(
                non_finite,
                raw_motion_length=4,
                view_start=0,
                view_length=4,
                latent_length=1,
                expected_dim=2,
            )


if __name__ == "__main__":
    unittest.main()
