import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "EVAL_t2m_rag_t5_optimized.sh"


class OptimizedLauncherTests(unittest.TestCase):
    def test_launcher_preserves_path_arguments_and_uses_optimized_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            capture_path = temp_path / "captured arguments.txt"
            python_stub = temp_path / "python stub.sh"
            python_stub.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
                encoding="utf-8",
            )
            python_stub.chmod(
                python_stub.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": str(python_stub),
                    "CAPTURE_PATH": str(capture_path),
                    "MSA_VAE_CKPT": "/tmp/vae checkpoint.pth",
                    "RAG_CKPT": "/tmp/rag checkpoint.pth",
                    "MOTION_LATENT_DIR": "/tmp/motion latents",
                    "TEXT_LATENT_DIR": "/tmp/text latents",
                    "HCLS_DIR": "/tmp/h cls",
                    "EMPTY_TEXT_PATH": "/tmp/empty embedding.npy",
                    "T5_MODEL_PATH": "/tmp/t5 model",
                }
            )

            completed = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=str(REPO_ROOT),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[0], "eval_msa_t2m_rag_t5_optimized.py")
            self.assertNotIn("eval_msa_t2m_rag_t5.py", arguments)

            expected_pairs = {
                "--resume-pth": "/tmp/vae checkpoint.pth",
                "--resume-trans": "/tmp/rag checkpoint.pth",
                "--latent_dir": "/tmp/motion latents",
                "--text_latent_dir": "/tmp/text latents",
                "--hcls_dir": "/tmp/h cls",
                "--empty_text_path": "/tmp/empty embedding.npy",
                "--t5_model_path": "/tmp/t5 model",
                "--text_source": "online_t5",
                "--cfg_scale": "4.0",
                "--stop_threshold": "0.1",
                "--retrieval_topk": "3",
            }
            for flag, expected_value in expected_pairs.items():
                flag_index = arguments.index(flag)
                self.assertEqual(arguments[flag_index + 1], expected_value)

            exp_index = arguments.index("--exp-name")
            self.assertEqual(
                arguments[exp_index + 1],
                "MotionStreamer_t2m_272_msa_rag_t5_trans662048_"
                "vaefulldb_k3_testcode_ema_optimized",
            )


if __name__ == "__main__":
    unittest.main()
