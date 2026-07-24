from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLORATIONS = REPO_ROOT / "explorations"

MOVED_PYTHON_TARGETS = {
    "ablations/no_rag/DEMO_msa_t2m_no_rag_t5.sh":
        "explorations.ablations.no_rag.demo_msa_t2m_no_rag_t5",
    "ablations/no_rag/EVAL_t2m_no_rag_t5.sh":
        "explorations.ablations.no_rag.eval_msa_t2m_no_rag_t5",
    "clip/EVAL_t2m_clip_baseline.sh":
        "explorations.clip.eval_t2m_clip_baseline",
    "clip/EVAL_t2m_rag.sh": "explorations.clip.eval_t2m_rag",
    "clip/TRAIN_t2m_baseline_clip.sh":
        "explorations.clip.train_t2m_baseline_clip",
    "cross_attention/local_rag/EVAL_t2m_rag_local.sh":
        "explorations.cross_attention.local_rag.eval_msa_t2m_rag_local",
    "cross_attention/local_rag/TRAIN_t2m_rag_local.sh":
        "explorations.cross_attention.local_rag.train_t2m_rag_local",
    "cross_attention/local_rag/TRAIN_THEN_EVAL_t2m_rag_local.sh": (
        "explorations.cross_attention.local_rag.train_t2m_rag_local",
        "explorations.cross_attention.local_rag.eval_msa_t2m_rag_local",
    ),
    "cross_attention/mca/EVAL_t2m_rag_mca.sh":
        "explorations.cross_attention.mca.eval_msa_t2m_rag_mca",
    "cross_attention/mca/Train_t2m_rag_multi_text_token.sh":
        "explorations.cross_attention.mca.train_t2m_rag_multi_text_token",
    "cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr.sh":
        "explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr",
    "cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr_addcfg.sh":
        "explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr_addcfg",
    "cross_attention/latent_retrieval/Train_t2m_rag_latent_retr.sh":
        "explorations.cross_attention.latent_retrieval.train_t2m_rag_latent_retr",
    "rectified_flow/EVAL_t2m_rag_t5_rf.sh":
        "explorations.rectified_flow.eval_msa_t2m_rag_t5_rf",
    "qformer/TRAIN_qformer_rag.sh":
        "explorations.qformer.train_qformer_rag",
    "motionstreamer_baselines/EVAL_t2m.sh":
        "explorations.motionstreamer_baselines.eval_t2m",
    "motionstreamer_baselines/TRAIN_motionstreamer.sh":
        "explorations.motionstreamer_baselines.train_motionstreamer",
    "motionstreamer_baselines/TRAIN_t2m.sh":
        "explorations.motionstreamer_baselines.train_t2m",
    "motionstreamer_baselines/Train_t2m_multi.sh":
        "explorations.motionstreamer_baselines.train_t2m",
    "motionstreamer_baselines/TRAIN_t2m_cached.sh":
        "explorations.motionstreamer_baselines.train_t2m_cached",
    "representation_experiments/EVAL_sae_v1.sh":
        "explorations.representation_experiments.eval_sae_v1",
    "representation_experiments/TRAIN_sae_v1.sh":
        "explorations.representation_experiments.train_sae_v1",
    "representation_experiments/TRAIN_tae_gan_v1.sh":
        "explorations.representation_experiments.train_tae_gan_v1",
}


class ExplorationLauncherTest(unittest.TestCase):
    def test_shell_launchers_enter_repository_root(self):
        shell_files = list(EXPLORATIONS.rglob("*.sh"))
        shell_files.extend(EXPLORATIONS.rglob("*.sh.bak"))
        self.assertTrue(shell_files)
        for path in shell_files:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertIn("SCRIPT_DIR=", content)
                self.assertIn("REPO_ROOT=", content)
                self.assertIn('cd "$REPO_ROOT"', content)

    def test_shell_launchers_use_archived_python_modules(self):
        for relative_path, targets in MOVED_PYTHON_TARGETS.items():
            content = (EXPLORATIONS / relative_path).read_text()
            if isinstance(targets, str):
                targets = (targets,)
            for target in targets:
                with self.subTest(path=relative_path, target=target):
                    self.assertIn(f"-m {target}", content)

    def test_cross_archive_imports_use_package_paths(self):
        expected_imports = {
            "ablations/no_rag/demo_msa_t2m_no_rag_t5.py":
                "from explorations.demos_and_diagnostics.demo_msa_t2m_t5 import main",
            "cross_attention/latent_retrieval/eval_msa_t2m_rag_latent_retr_addcfg.py": (
                "from explorations.cross_attention.latent_retrieval."
                "eval_msa_t2m_rag_latent_retr import ("
            ),
            "qformer/build_rag_db.py":
                "from explorations.qformer.train_qformer_rag import",
        }
        for relative_path, expected in expected_imports.items():
            content = (EXPLORATIONS / relative_path).read_text()
            with self.subTest(path=relative_path):
                self.assertIn(expected, content)

    def test_no_rag_launcher_forwards_ablation_flag(self):
        wrapper = (
            EXPLORATIONS / "ablations/no_rag/TRAIN_t2m_no_rag.sh"
        ).read_text()
        official = (REPO_ROOT / "TRAIN_t2m_rag.sh").read_text()
        self.assertIn('bash "$REPO_ROOT/TRAIN_t2m_rag.sh"', wrapper)
        self.assertIn("DISABLE_RAG", official)
        self.assertIn("--disable_rag", official)


if __name__ == "__main__":
    unittest.main()
