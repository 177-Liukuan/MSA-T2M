import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "EVAL_msa_vae_alignment.sh"


class MSAVAEAlignmentLauncherTest(unittest.TestCase):
    def _run_with_capture(self, arguments, active_environment):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            capture = temp_root / "arguments.txt"
            executable_name = "python" if active_environment else "conda"
            executable = temp_root / executable_name
            executable.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = "{}:{}".format(
                temp_root,
                environment["PATH"],
            )
            environment["CAPTURE_FILE"] = str(capture)
            if active_environment:
                environment["CONDA_DEFAULT_ENV"] = "mgpt"
            else:
                environment.pop("CONDA_DEFAULT_ENV", None)
            completed = subprocess.run(
                ["bash", str(LAUNCHER)] + list(arguments),
                cwd=temp_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            captured = (
                capture.read_text(encoding="utf-8").splitlines()
                if capture.exists()
                else []
            )
            return completed, captured

    def test_inactive_environment_uses_mgpt_and_forwards_local_contract(self):
        requested = [
            "Experiments/ablation/net_last.pth",
            "--local-split-file",
            "humanml3d_272/split/train_ft.txt",
            "--local-text-embed-dir",
            "humanml3d_272/t5_enc_single",
            "--local-target-scope",
            "in-sample",
            "--batch-size",
            "7",
        ]

        completed, arguments = self._run_with_capture(
            requested,
            active_environment=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(arguments[:4], ["run", "-n", "mgpt", "python"])
        self.assertEqual(
            Path(arguments[4]),
            ROOT / "eval_msa_vae_alignment.py",
        )
        self.assertEqual(arguments[5:], requested)

    def test_active_mgpt_environment_uses_python_without_nested_conda(self):
        requested = ["/models/net.pth", "--num-workers", "3"]

        completed, arguments = self._run_with_capture(
            requested,
            active_environment=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            Path(arguments[0]),
            ROOT / "eval_msa_vae_alignment.py",
        )
        self.assertEqual(arguments[1:], requested)

    def test_missing_or_option_like_checkpoint_does_not_launch(self):
        for arguments in ([], ["--device", "cpu"]):
            with self.subTest(arguments=arguments):
                completed, captured = self._run_with_capture(
                    arguments,
                    active_environment=False,
                )

                self.assertEqual(completed.returncode, 2)
                self.assertIn("Usage:", completed.stderr)
                self.assertEqual(captured, [])


if __name__ == "__main__":
    unittest.main()
