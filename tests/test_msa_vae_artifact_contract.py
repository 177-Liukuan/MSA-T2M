import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.msa_vae_training import (
    checkpoint_signature,
    prepare_extraction_roots,
)


class MSAArtifactContractTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            down_t=2,
            stride_t=2,
            latent_dim=16,
        )

    def test_prepare_roots_records_checkpoint_identity_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint-a")
            output_roots = [
                root / "z",
                root / "h_cls",
                root / "mu",
            ]
            metadata = {
                "sequence_mode": "mixed",
                "down_t": 2,
                "stride_t": 2,
                "latent_dim": 16,
            }

            payload = prepare_extraction_roots(
                output_roots,
                checkpoint,
                metadata,
                self._args(),
            )
            repeated = prepare_extraction_roots(
                output_roots,
                checkpoint,
                metadata,
                self._args(),
            )

            self.assertEqual(payload, repeated)
            self.assertEqual(
                payload["checkpoint"],
                checkpoint_signature(checkpoint),
            )
            self.assertEqual(payload["sequence_mode"], "mixed")
            for output_root in output_roots:
                manifest = json.loads(
                    (output_root / "extraction_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest, payload)

    def test_different_checkpoint_is_rejected_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_checkpoint = root / "first.pth"
            second_checkpoint = root / "second.pth"
            first_checkpoint.write_bytes(b"first")
            second_checkpoint.write_bytes(b"second-checkpoint")
            output_roots = [root / "z", root / "h_cls", root / "mu"]

            first_payload = prepare_extraction_roots(
                output_roots,
                first_checkpoint,
                {},
                self._args(),
            )
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                prepare_extraction_roots(
                    output_roots,
                    second_checkpoint,
                    {},
                    self._args(),
                )

            for output_root in output_roots:
                manifest = json.loads(
                    (output_root / "extraction_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest, first_payload)

    def test_nonempty_untracked_output_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            output_root = root / "z"
            output_root.mkdir()
            (output_root / "old.npy").write_bytes(b"old")

            with self.assertRaisesRegex(ValueError, "no extraction manifest"):
                prepare_extraction_roots(
                    [output_root],
                    checkpoint,
                    {},
                    self._args(),
                )


if __name__ == "__main__":
    unittest.main()
