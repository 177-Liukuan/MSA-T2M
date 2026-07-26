import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from options import option_msa_vae


ROOT = Path(__file__).resolve().parents[1]
FAKE_CNN_CONTENT = b"fake CNN checkpoint for launcher tests\n"
FAKE_CNN_SHA256 = hashlib.sha256(FAKE_CNN_CONTENT).hexdigest()


class MSAOptionTest(unittest.TestCase):
    def test_sequence_training_defaults_preserve_window_invocation(self):
        with mock.patch.object(sys, "argv", ["train_msa_vae.py"]):
            args = option_msa_vae.get_args_parser()

        self.assertEqual(args.sequence_mode, "window")
        self.assertEqual(args.full_seq_batch_size, 32)
        self.assertEqual(args.window_replay_interval, 4)
        self.assertEqual(args.length_bucket_size, 256)
        self.assertEqual(args.validation_seed, 123)
        self.assertEqual(args.validation_batch_size, 32)

    def test_phase2_local_projection_freeze_is_explicit_and_phase_scoped(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "train_msa_vae.py",
                "--phase",
                "2",
                "--freeze-phase2-local-proj",
            ],
        ):
            enabled = option_msa_vae.get_args_parser()
        self.assertTrue(enabled.freeze_phase2_local_proj)

        with mock.patch.object(sys, "argv", ["train_msa_vae.py"]):
            default = option_msa_vae.get_args_parser()
        self.assertFalse(default.freeze_phase2_local_proj)

        for phase in ("0", "1"):
            with self.subTest(phase=phase):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "train_msa_vae.py",
                        "--phase",
                        phase,
                        "--freeze-phase2-local-proj",
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        option_msa_vae.get_args_parser()


class MSAFullSequenceLauncherTest(unittest.TestCase):
    def _invoke_launcher(self, script_name, extra_env):
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
            fake_cnn = temp_root / "fixed-tae.pth"
            fake_cnn.write_bytes(FAKE_CNN_CONTENT)

            env = os.environ.copy()
            env.update(extra_env)
            env.setdefault("CNN_CKPT", str(fake_cnn))
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
            arguments = None
            if capture.exists():
                arguments = capture.read_text(encoding="utf-8").splitlines()
            return completed, arguments

    def _run_launcher(self, script_name, extra_env):
        completed, arguments = self._invoke_launcher(script_name, extra_env)
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertIsNotNone(arguments)
        return arguments

    def _assert_launcher_rejected(self, script_name, extra_env, message):
        completed, arguments = self._invoke_launcher(script_name, extra_env)
        self.assertNotEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertIsNone(arguments)
        self.assertIn(message, completed.stdout + completed.stderr)

    @staticmethod
    def _value_after(arguments, flag):
        return arguments[arguments.index(flag) + 1]

    def test_phase1_launches_full_sequence_training_with_overrides(self):
        arguments = self._run_launcher(
            "TRAIN_msa_vae_phase1.sh",
            {
                "FULL_SEQ_BATCH_SIZE": "7",
                "LENGTH_BUCKET_SIZE": "19",
                "CNN_CKPT_SHA256": FAKE_CNN_SHA256,
                "EXP_NAME": "phase1-global-local-seed123",
                "GLOBAL_ALIGN_WEIGHT": "0.25",
                "LOCAL_ALIGN_WEIGHT": "0.05",
                "SEED": "123",
                "TOTAL_ITER": "17",
                "WARM_UP_ITER": "3",
                "EVAL_ITER": "4",
                "OUT_DIR": "Experiments/test-ablation",
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
        self.assertEqual(
            self._value_after(arguments, "--validation-seed"),
            "123",
        )
        self.assertEqual(
            self._value_after(arguments, "--validation-batch-size"),
            "32",
        )
        expected = {
            "--exp-name": "phase1-global-local-seed123",
            "--global_align_weight": "0.25",
            "--local_align_weight": "0.05",
            "--seed": "123",
            "--total-iter": "17",
            "--warm-up-iter": "3",
            "--eval-iter": "4",
            "--out-dir": "Experiments/test-ablation",
            "--resume-cnn-sha256": FAKE_CNN_SHA256,
        }
        for flag, value in expected.items():
            self.assertEqual(
                self._value_after(arguments, flag),
                value,
                msg=flag,
            )

    def test_phase2_launches_mixed_training_from_final_phase1_state(self):
        arguments = self._run_launcher(
            "TRAIN_msa_vae_phase2.sh",
            {
                "FULL_SEQ_BATCH_SIZE": "5",
                "WINDOW_REPLAY_INTERVAL": "6",
                "LENGTH_BUCKET_SIZE": "23",
                "PHASE1_DIR": "Experiments/test-phase1",
                "EXP_NAME": "phase2-global-local-seed456",
                "GLOBAL_ALIGN_WEIGHT": "0.125",
                "LOCAL_ALIGN_WEIGHT": "0.025",
                "SEED": "456",
                "TOTAL_ITER": "29",
                "WARM_UP_ITER": "2",
                "EVAL_ITER": "7",
                "OUT_DIR": "Experiments/test-ablation",
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
            self._value_after(arguments, "--validation-seed"),
            "123",
        )
        self.assertEqual(
            self._value_after(arguments, "--validation-batch-size"),
            "32",
        )
        self.assertEqual(
            self._value_after(arguments, "--resume-pth"),
            "Experiments/test-phase1/net_last.pth",
        )
        expected = {
            "--exp-name": "phase2-global-local-seed456",
            "--global_align_weight": "0.125",
            "--local_align_weight": "0.025",
            "--seed": "456",
            "--total-iter": "29",
            "--warm-up-iter": "2",
            "--eval-iter": "7",
            "--out-dir": "Experiments/test-ablation",
        }
        for flag, value in expected.items():
            self.assertEqual(
                self._value_after(arguments, flag),
                value,
                msg=flag,
            )

    def test_phase2_default_phase1_path_matches_text_encoder(self):
        arguments = self._run_launcher(
            "TRAIN_msa_vae_phase2.sh",
            {
                "TEXT_ENCODER_TYPE": "clip",
            },
        )

        self.assertEqual(
            self._value_after(arguments, "--resume-pth"),
            "Experiments/MSA_VAEv7_phase1_fullseq_t2m_272_clip_fulldb/"
            "net_last.pth",
        )

    def test_phase2_local_projection_freeze_is_default_off_and_opt_in(self):
        default_arguments = self._run_launcher(
            "TRAIN_msa_vae_phase2.sh",
            {},
        )
        self.assertNotIn(
            "--freeze-phase2-local-proj",
            default_arguments,
        )

        enabled_arguments = self._run_launcher(
            "TRAIN_msa_vae_phase2.sh",
            {"FREEZE_PHASE2_LOCAL_PROJ": "1"},
        )
        self.assertIn(
            "--freeze-phase2-local-proj",
            enabled_arguments,
        )

        self._assert_launcher_rejected(
            "TRAIN_msa_vae_phase2.sh",
            {"FREEZE_PHASE2_LOCAL_PROJ": "yes"},
            "FREEZE_PHASE2_LOCAL_PROJ",
        )

        self._assert_launcher_rejected(
            "TRAIN_msa_vae_phase2.sh",
            {"FREEZE_PHASE2_LOCAL_PROJ": ""},
            "FREEZE_PHASE2_LOCAL_PROJ",
        )

    def test_launchers_reject_empty_names_and_negative_weights(self):
        for script_name in (
            "TRAIN_msa_vae_phase1.sh",
            "TRAIN_msa_vae_phase2.sh",
        ):
            cases = (
                ({"EXP_NAME": ""}, "EXP_NAME"),
                ({"GLOBAL_ALIGN_WEIGHT": "-0.1"}, "GLOBAL_ALIGN_WEIGHT"),
                ({"LOCAL_ALIGN_WEIGHT": "-0.1"}, "LOCAL_ALIGN_WEIGHT"),
                ({"MAIN_PROCESS_PORT": "not-a-port"}, "MAIN_PROCESS_PORT"),
                ({"MAIN_PROCESS_PORT": "65536"}, "MAIN_PROCESS_PORT"),
            )
            for extra_env, message in cases:
                with self.subTest(
                    script_name=script_name,
                    extra_env=extra_env,
                ):
                    self._assert_launcher_rejected(
                        script_name,
                        extra_env,
                        message,
                    )

        self._assert_launcher_rejected(
            "TRAIN_msa_vae_phase1.sh",
            {"CNN_CKPT_SHA256": "not-a-sha256"},
            "CNN_CKPT_SHA256",
        )
        self._assert_launcher_rejected(
            "TRAIN_msa_vae_phase1.sh",
            {"CNN_CKPT_SHA256": "0" * 64},
            "does not match",
        )

    def test_launchers_forward_optional_accelerate_port(self):
        for script_name in (
            "TRAIN_msa_vae_phase1.sh",
            "TRAIN_msa_vae_phase2.sh",
        ):
            with self.subTest(script_name=script_name):
                arguments = self._run_launcher(
                    script_name,
                    {"MAIN_PROCESS_PORT": "29502"},
                )
                self.assertEqual(
                    self._value_after(arguments, "--main_process_port"),
                    "29502",
                )


if __name__ == "__main__":
    unittest.main()
