import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from options import option_msa_vae


ROOT = Path(__file__).resolve().parents[1]


class MSAOptionTest(unittest.TestCase):
    def test_sequence_training_defaults_preserve_window_invocation(self):
        with mock.patch.object(sys, "argv", ["train_msa_vae.py"]):
            args = option_msa_vae.get_args_parser()

        self.assertEqual(args.sequence_mode, "window")
        self.assertEqual(args.full_seq_batch_size, 32)
        self.assertEqual(args.window_replay_interval, 4)
        self.assertEqual(args.length_bucket_size, 256)


class MSAFullSequenceLauncherTest(unittest.TestCase):
    def _run_launcher(self, script_name, extra_env):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            capture = temp_root / "args.txt"
            accelerate = temp_root / "accelerate"
            accelerate.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            accelerate.chmod(0o755)

            env = os.environ.copy()
            env.update(extra_env)
            env["CAPTURE_FILE"] = str(capture)
            env["PATH"] = f"{temp_root}:{env['PATH']}"
            completed = subprocess.run(
                ["bash", str(ROOT / script_name), "1", "t2m_272"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            return capture.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def _value_after(arguments, flag):
        return arguments[arguments.index(flag) + 1]

    def test_phase1_launches_full_sequence_training_with_overrides(self):
        arguments = self._run_launcher(
            "TRAIN_msa_vae_phase1.sh",
            {
                "FULL_SEQ_BATCH_SIZE": "7",
                "LENGTH_BUCKET_SIZE": "19",
                "CNN_CKPT": "Experiments/test-cnn.pth",
            },
        )

        self.assertEqual(
            self._value_after(arguments, "--sequence_mode"),
            "full",
        )
        self.assertEqual(
            self._value_after(arguments, "--full-seq-batch-size"),
            "7",
        )
        self.assertEqual(
            self._value_after(arguments, "--length-bucket-size"),
            "19",
        )
        self.assertIn(
            "fullseq",
            self._value_after(arguments, "--exp-name"),
        )

    def test_phase2_launches_mixed_training_from_final_phase1_state(self):
        arguments = self._run_launcher(
            "TRAIN_msa_vae_phase2.sh",
            {
                "FULL_SEQ_BATCH_SIZE": "5",
                "WINDOW_REPLAY_INTERVAL": "6",
                "LENGTH_BUCKET_SIZE": "23",
                "PHASE1_DIR": "Experiments/test-phase1",
            },
        )

        self.assertEqual(
            self._value_after(arguments, "--sequence_mode"),
            "mixed",
        )
        self.assertEqual(
            self._value_after(arguments, "--full-seq-batch-size"),
            "5",
        )
        self.assertEqual(
            self._value_after(arguments, "--window-replay-interval"),
            "6",
        )
        self.assertEqual(
            self._value_after(arguments, "--length-bucket-size"),
            "23",
        )
        self.assertEqual(
            self._value_after(arguments, "--resume-pth"),
            "Experiments/test-phase1/net_last.pth",
        )
        self.assertIn(
            "fullseq_replay6",
            self._value_after(arguments, "--exp-name"),
        )


if __name__ == "__main__":
    unittest.main()
