import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "TRAIN_t2m_rag.sh"
NO_RAG_LAUNCHER = (
    REPOSITORY_ROOT
    / "explorations"
    / "ablations"
    / "no_rag"
    / "TRAIN_t2m_no_rag.sh"
)


def write_executable(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TrainT2MRAGLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log_path = self.root / "commands.log"
        self.motion_dir = self.root / "motion"
        self.text_dir = self.root / "text"
        self.hcls_dir = self.root / "hcls"
        for directory in (self.motion_dir, self.text_dir, self.hcls_dir):
            directory.mkdir()

        self.python_stub = self.bin_dir / "python"
        self.accelerate_stub = self.bin_dir / "accelerate"
        write_executable(
            self.python_stub,
            """#!/bin/bash
printf 'cache:%s\\n' "$*" >> "$LOG_PATH"
exit "${CACHE_EXIT_CODE:-0}"
""",
        )
        write_executable(
            self.accelerate_stub,
            """#!/bin/bash
printf 'accelerate:%s\\n' "$*" >> "$LOG_PATH"
""",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_launcher(
        self,
        cache_mode="packed",
        cache_exit_code=0,
        launcher=LAUNCHER,
    ):
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": "{}:{}".format(self.bin_dir, environment["PATH"]),
                "LOG_PATH": str(self.log_path),
                "CACHE_EXIT_CODE": str(cache_exit_code),
                "PYTHON_BIN": str(self.python_stub),
                "ACCELERATE_BIN": str(self.accelerate_stub),
                "MOTION_LATENT_DIR": str(self.motion_dir),
                "TEXT_LATENT_DIR": str(self.text_dir),
                "HCLS_DIR": str(self.hcls_dir),
                "EMPTY_TEXT_PATH": str(self.root / "empty.npy"),
                "RAG_CACHE_MODE": cache_mode,
                "RAG_CACHE_DIR": str(self.root / "cache"),
                "RETRIEVAL_TOPK": "2",
                "NUM_WORKERS": "3",
                "TEXT_EMBED_DIM": "2",
            }
        )
        return subprocess.run(
            ["bash", str(launcher), "1"],
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def read_commands(self):
        if not self.log_path.exists():
            return []
        return self.log_path.read_text().splitlines()

    def test_packed_mode_builds_cache_before_accelerate(self):
        result = self.run_launcher("packed")
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = self.read_commands()
        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0].startswith("cache:"), commands)
        self.assertIn("build_msa_rag_cache.py", commands[0])
        self.assertIn("--topk 2", commands[0])
        self.assertTrue(commands[1].startswith("accelerate:"), commands)
        self.assertIn("--cache_mode packed", commands[1])
        self.assertIn("--cache_dir {}".format(self.root / "cache"), commands[1])
        self.assertIn("--num_workers 3", commands[1])
        self.assertIn("--retrieval_topk 2", commands[1])

    def test_reference_mode_skips_cache_builder(self):
        result = self.run_launcher("reference")
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = self.read_commands()
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].startswith("accelerate:"), commands)
        self.assertIn("--cache_mode reference", commands[0])

    def test_cache_builder_failure_prevents_training(self):
        result = self.run_launcher("packed", cache_exit_code=7)
        self.assertEqual(result.returncode, 7, result.stdout)
        commands = self.read_commands()
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].startswith("cache:"), commands)

    def test_no_rag_wrapper_reaches_official_launcher_and_sets_flag(self):
        result = self.run_launcher(
            "reference",
            launcher=NO_RAG_LAUNCHER,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = self.read_commands()
        self.assertEqual(len(commands), 1)
        self.assertIn("--disable_rag", commands[0])


if __name__ == "__main__":
    unittest.main()
