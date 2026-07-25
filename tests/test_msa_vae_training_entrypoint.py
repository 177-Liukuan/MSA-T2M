import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.msa_vae_training import (
    build_global_alignment_target,
    build_msa_checkpoint_metadata,
    save_msa_checkpoint,
    select_training_batch,
    validate_msa_checkpoint_metadata,
)


class TrainingBatchRoutingTest(unittest.TestCase):
    def test_modes_consume_the_expected_iterators(self):
        full_iter = iter(["full-1", "full-2"])
        window_iter = iter(["window-1", "window-2"])

        batch, kind = select_training_batch(
            step=1,
            mode="full",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("full-1", "full"))

        batch, kind = select_training_batch(
            step=1,
            mode="window",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("window-1", "window"))

        batch, kind = select_training_batch(
            step=4,
            mode="mixed",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("window-2", "window"))

        batch, kind = select_training_batch(
            step=5,
            mode="mixed",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("full-2", "full"))

    def test_unknown_mode_is_rejected_without_consuming_data(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            select_training_batch(
                step=1,
                mode="prefix",
                full_iter=iter(["full"]),
                window_iter=iter(["window"]),
                replay_interval=4,
            )


class GlobalTargetRoutingTest(unittest.TestCase):
    def setUp(self):
        self.global_text = torch.tensor([[3.0, 4.0]])
        self.local_text = torch.tensor([[-3.0, -4.0]])
        self.has_local = torch.tensor([True])
        self.total_frames = torch.tensor([128])

    def test_full_and_mixed_modes_use_only_complete_caption_target(self):
        expected = F.normalize(self.global_text, dim=-1)

        for mode in ("full", "mixed"):
            actual = build_global_alignment_target(
                global_text=self.global_text,
                local_pooled=self.local_text,
                has_local=self.has_local,
                total_frames=self.total_frames,
                window_size=64,
                sequence_mode=mode,
                spotlight_alpha=-1.0,
            )
            torch.testing.assert_close(actual, expected)

    def test_window_mode_preserves_spotlight_interpolation(self):
        actual = build_global_alignment_target(
            global_text=self.global_text,
            local_pooled=self.local_text,
            has_local=self.has_local,
            total_frames=self.total_frames,
            window_size=64,
            sequence_mode="window",
            spotlight_alpha=-1.0,
        )

        torch.testing.assert_close(
            actual,
            F.normalize(
                0.5 * self.global_text + 0.5 * self.local_text,
                dim=-1,
            ),
        )


class CheckpointMetadataTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "phase": 1,
            "sequence_mode": "full",
            "window_size": 64,
            "full_seq_batch_size": 16,
            "window_replay_interval": 4,
            "down_t": 2,
            "stride_t": 2,
            "latent_dim": 16,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_checkpoint_payload_preserves_state_and_metadata(self):
        model = nn.Linear(3, 2)
        metadata = build_msa_checkpoint_metadata(self._args())

        self.assertEqual(metadata["unit_length"], 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            save_msa_checkpoint(path, model, metadata)
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(set(payload["net"]), set(model.state_dict()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(payload["net"][key], value)

    def test_structural_metadata_is_validated_but_phase_handoff_is_allowed(self):
        metadata = build_msa_checkpoint_metadata(self._args(phase=1))

        validate_msa_checkpoint_metadata(
            metadata,
            self._args(phase=2, sequence_mode="mixed"),
        )

        for field, value in (
            ("down_t", 3),
            ("stride_t", 3),
            ("latent_dim", 32),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_msa_checkpoint_metadata(
                        metadata,
                        self._args(**{field: value}),
                    )

        incompatible_unit = dict(metadata)
        incompatible_unit["unit_length"] = 8
        with self.assertRaisesRegex(ValueError, "unit_length"):
            validate_msa_checkpoint_metadata(
                incompatible_unit,
                self._args(),
            )

    def test_legacy_checkpoint_without_metadata_is_accepted(self):
        validate_msa_checkpoint_metadata(None, self._args())


if __name__ == "__main__":
    unittest.main()
