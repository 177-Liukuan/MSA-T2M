"""Source and argv contracts for authoritative BABEL sparse-global launchers."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE1 = REPOSITORY_ROOT / "TRAIN_msa_vae_babel_phase1.sh"
PHASE2 = REPOSITORY_ROOT / "TRAIN_msa_vae_babel_phase2.sh"
EVALUATION = REPOSITORY_ROOT / "EVAL_msa_vae_babel.sh"
APPROVED_JOINT_TAE_SHA256 = (
    "c819493606aacba0b4d126871ddef7195ff45bca1b4b13792d11a13894154387"
)


class BabelMSAVAELauncherTest(unittest.TestCase):
    def read(self, path):
        self.assertTrue(path.is_file(), "missing launcher: {}".format(path.name))
        return path.read_text()

    def test_all_launchers_enter_repository_root(self):
        for launcher in (PHASE1, PHASE2, EVALUATION):
            with self.subTest(launcher=launcher.name):
                content = self.read(launcher)
                self.assertIn("SCRIPT_DIR=", content)
                self.assertIn("REPO_ROOT=", content)
                self.assertIn('cd "$REPO_ROOT"', content)

    def test_training_launchers_bind_babel_sparse_global_contract(self):
        for launcher in (PHASE1, PHASE2):
            with self.subTest(launcher=launcher.name):
                content = self.read(launcher)
                self.assertIn("--msa_data_mode babel_sparse_global", content)
                self.assertIn("--dataname t2m_babel_272", content)
                self.assertIn("--text_encoder_type t5", content)
                self.assertIn("--text_embed_dim 768", content)
                self.assertIn('--msa_mean_path "$MSA_MEAN_PATH"', content)
                self.assertIn('--msa_std_path "$MSA_STD_PATH"', content)
                self.assertIn(
                    '--babel_train_cache_manifest "$BABEL_TRAIN_MANIFEST"', content
                )
                self.assertIn(
                    '--babel_val_cache_manifest "$BABEL_VAL_MANIFEST"', content
                )
                self.assertIn('--bridge_split_file "$BRIDGE_SPLIT_FILE"', content)
                self.assertIn("babel_sparse_global", content)
                self.assertNotIn("MSA_VAEv6_phase1_t2m_272", content)

    def test_phase1_defaults_to_verified_joint_causal_tae(self):
        content = self.read(PHASE1)
        self.assertIn(
            "CNN_CKPT=${CNN_CKPT:-Experiments/causal_TAE_t2m_babel_272_h100_20260205/net_best_mpjpe.pth}",
            content,
        )
        self.assertIn(APPROVED_JOINT_TAE_SHA256, content)
        self.assertIn('--resume-cnn-pth "$CNN_CKPT"', content)
        self.assertIn('--resume-cnn-sha256 "$CNN_CKPT_SHA256"', content)

    def test_phase2_requires_the_babel_phase1_semantic_checkpoint(self):
        content = self.read(PHASE2)
        self.assertIn("PHASE1_DIR=${PHASE1_DIR:?", content)
        self.assertIn('RESUME_PTH="${PHASE1_DIR}/net_best_semantic.pth"', content)
        self.assertIn('--resume-pth "$RESUME_PTH"', content)
        self.assertNotIn("net_best_fid.pth", content)

    def test_evaluation_is_babel_validation_only(self):
        content = self.read(EVALUATION)
        self.assertIn("BABEL_CKPT=${BABEL_CKPT:?", content)
        self.assertIn('--resume-pth "$BABEL_CKPT"', content)
        self.assertIn("--msa_data_mode babel_sparse_global", content)
        self.assertIn("--dataname t2m_babel_272", content)
        self.assertIn('--babel_val_motion_dir "$BABEL_VAL_MOTION_DIR"', content)
        self.assertIn('--babel_val_text_dir "$BABEL_VAL_TEXT_DIR"', content)
        self.assertIn(
            '--babel_val_cache_manifest "$BABEL_VAL_MANIFEST"', content
        )
        self.assertNotIn("Evaluator_272", content)

    def test_operational_docs_describe_babel_reconstruction_workflow(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text()
        agents = (REPOSITORY_ROOT / "AGENTS.md").read_text()
        for content in (readme, agents):
            self.assertIn("BABEL sparse-global", content)
            self.assertIn("net_best_semantic.pth", content)
            self.assertIn("net_best_mpjpe.pth", content)
        self.assertIn("不用于 BABEL 文本到运动生成", readme)

    def test_path_overrides_are_one_argv_value_without_shell_expansion(self):
        """The harmless launcher substitutes prove paths reach argparse intact."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture_path = root / "captured.json"
            executable = root / "capture argv"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['CAPTURE_PATH'], 'w') as output:\n"
                "    json.dump(sys.argv[1:], output)\n"
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            def unsafe_value(name):
                return str(root / "{} value [*]".format(name))

            common = {
                name: unsafe_value(name)
                for name in (
                    "MSA_MEAN_PATH",
                    "MSA_STD_PATH",
                    "BRIDGE_SPLIT_FILE",
                    "BRIDGE_MOTION_DIR",
                    "BRIDGE_TEXT_DIR",
                    "BRIDGE_GLOBAL_EMBED_DIR",
                    "BRIDGE_LOCAL_EMBED_DIR",
                    "BABEL_TRAIN_MOTION_DIR",
                    "BABEL_TRAIN_TEXT_DIR",
                    "BABEL_TRAIN_CACHE_DIR",
                    "BABEL_TRAIN_MANIFEST",
                    "BABEL_VAL_MOTION_DIR",
                    "BABEL_VAL_TEXT_DIR",
                    "BABEL_VAL_CACHE_DIR",
                    "BABEL_VAL_MANIFEST",
                    "T5_MODEL_PATH",
                    "EXP_NAME",
                )
            }
            option_variables = {
                "--msa_mean_path": "MSA_MEAN_PATH",
                "--msa_std_path": "MSA_STD_PATH",
                "--bridge_split_file": "BRIDGE_SPLIT_FILE",
                "--bridge_motion_dir": "BRIDGE_MOTION_DIR",
                "--bridge_text_dir": "BRIDGE_TEXT_DIR",
                "--bridge_global_embed_dir": "BRIDGE_GLOBAL_EMBED_DIR",
                "--bridge_local_embed_dir": "BRIDGE_LOCAL_EMBED_DIR",
                "--babel_train_motion_dir": "BABEL_TRAIN_MOTION_DIR",
                "--babel_train_text_dir": "BABEL_TRAIN_TEXT_DIR",
                "--babel_train_t5_cache_dir": "BABEL_TRAIN_CACHE_DIR",
                "--babel_train_cache_manifest": "BABEL_TRAIN_MANIFEST",
                "--babel_val_motion_dir": "BABEL_VAL_MOTION_DIR",
                "--babel_val_text_dir": "BABEL_VAL_TEXT_DIR",
                "--babel_val_t5_cache_dir": "BABEL_VAL_CACHE_DIR",
                "--babel_val_cache_manifest": "BABEL_VAL_MANIFEST",
                "--t5_embed_dir": "BRIDGE_LOCAL_EMBED_DIR",
                "--t5_global_embed_dir": "BRIDGE_GLOBAL_EMBED_DIR",
                "--t5_model_path": "T5_MODEL_PATH",
                "--exp-name": "EXP_NAME",
            }

            cases = (
                (PHASE1, "ACCELERATE_BIN", {"CNN_CKPT": unsafe_value("CNN_CKPT")}),
                (PHASE2, "ACCELERATE_BIN", {"PHASE1_DIR": unsafe_value("PHASE1_DIR")}),
                (EVALUATION, "PYTHON_BIN", {"BABEL_CKPT": unsafe_value("BABEL_CKPT")}),
            )
            for launcher, executable_variable, extra in cases:
                with self.subTest(launcher=launcher.name):
                    environment = os.environ.copy()
                    environment.update(common)
                    environment.update(extra)
                    environment.update(
                        {
                            executable_variable: str(executable),
                            "CAPTURE_PATH": str(capture_path),
                        }
                    )
                    result = subprocess.run(
                        ["bash", str(launcher), "1"],
                        cwd=str(REPOSITORY_ROOT),
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                    arguments = json.loads(capture_path.read_text())
                    for option, variable in option_variables.items():
                        with self.subTest(option=option):
                            index = arguments.index(option)
                            self.assertEqual(arguments[index + 1], common[variable])

                    if launcher == PHASE1:
                        option, expected = "--resume-cnn-pth", extra["CNN_CKPT"]
                    elif launcher == PHASE2:
                        option = "--resume-pth"
                        expected = str(Path(extra["PHASE1_DIR"]) / "net_best_semantic.pth")
                    else:
                        option, expected = "--resume-pth", extra["BABEL_CKPT"]
                    index = arguments.index(option)
                    self.assertEqual(arguments[index + 1], expected)


if __name__ == "__main__":
    unittest.main()
