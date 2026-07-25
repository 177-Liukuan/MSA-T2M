import ast
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLORATIONS = REPO_ROOT / "explorations"


def tracked_exploration_shell_files():
    """Return launchers owned by this repository, excluding local archives."""
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "explorations"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        REPO_ROOT / relative_path.decode("utf-8")
        for relative_path in completed.stdout.split(b"\0")
        if relative_path
        and (
            relative_path.endswith(b".sh")
            or relative_path.endswith(b".sh.bak")
        )
    ]


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

EXPLORATION_OUTPUT_ROOTS = {
    "clip/TRAIN_t2m_baseline_clip.sh": "Experiments/explorations/clip",
    "rectified_flow/Train_t2m_rag_rf.sh": "Experiments/explorations/rectified_flow",
    "cross_attention/mca/Train_t2m_rag_multi_text_token.sh": (
        "Experiments/explorations/cross_attention/mca"
    ),
    "cross_attention/latent_retrieval/Train_t2m_rag_latent_retr.sh": (
        "Experiments/explorations/cross_attention/latent_retrieval"
    ),
    "cross_attention/local_rag/TRAIN_t2m_rag_local.sh": (
        "Experiments/explorations/cross_attention/local_rag"
    ),
    "cross_attention/local_rag/TRAIN_THEN_EVAL_t2m_rag_local.sh": (
        "Experiments/explorations/cross_attention/local_rag"
    ),
    "qformer/TRAIN_qformer_rag.sh": "Experiments/explorations/qformer",
    "representation_experiments/TRAIN_sae_v1.sh": (
        "Experiments/explorations/representation_experiments"
    ),
    "representation_experiments/TRAIN_tae_gan_v1.sh": (
        "Experiments/explorations/representation_experiments/TAE_GAN_Loss_"
    ),
    "motionstreamer_baselines/TRAIN_motionstreamer.sh": (
        "Experiments/explorations/motionstreamer_baselines"
    ),
    "motionstreamer_baselines/TRAIN_t2m.sh": (
        "Experiments/explorations/motionstreamer_baselines"
    ),
    "motionstreamer_baselines/Train_t2m_multi.sh": (
        "Experiments/explorations/motionstreamer_baselines"
    ),
    "motionstreamer_baselines/TRAIN_t2m_cached.sh": (
        "Experiments/explorations/motionstreamer_baselines"
    ),
}

MOVED_CHECKPOINT_REFERENCES = {
    "clip/EVAL_t2m_clip_baseline.sh": (
        "Experiments/explorations/clip/MotionStreamer_t2m_272_baseline_clip"
    ),
    "rectified_flow/EVAL_t2m_rag_t5_rf.sh": (
        "Experiments/explorations/rectified_flow/"
        "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA"
    ),
    "cross_attention/local_rag/EVAL_t2m_rag_local.sh": (
        "Experiments/explorations/cross_attention/local_rag/"
        "MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca"
    ),
}


class ExplorationLauncherTest(unittest.TestCase):
    def test_shell_launchers_enter_repository_root(self):
        shell_files = tracked_exploration_shell_files()
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

    def test_exploration_training_outputs_stay_under_archive(self):
        for relative_path, output_root in EXPLORATION_OUTPUT_ROOTS.items():
            content = (EXPLORATIONS / relative_path).read_text()
            with self.subTest(path=relative_path):
                self.assertIn(output_root, content)

    def test_exploration_evaluations_read_archived_checkpoints(self):
        for relative_path, checkpoint_path in MOVED_CHECKPOINT_REFERENCES.items():
            content = (EXPLORATIONS / relative_path).read_text()
            with self.subTest(path=relative_path):
                self.assertIn(checkpoint_path, content)

    def test_active_cross_attention_defaults_use_archived_results(self):
        latent_checkpoint = (
            "Experiments/explorations/cross_attention/latent_retrieval/"
            "MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_"
            "6layer_top3_ddpm/net_Iter100000.pth"
        )
        for relative_path in (
            "cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr.sh",
        ):
            with self.subTest(path=relative_path):
                content = (EXPLORATIONS / relative_path).read_text()
                self.assertIn(latent_checkpoint, content)
                self.assertIn("CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-4}", content)

        addcfg_content = (
            EXPLORATIONS
            / "cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr_addcfg.sh"
        ).read_text()
        self.assertIn("RAG_CKPT=${RAG_CKPT:?", addcfg_content)
        self.assertIn("CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:?", addcfg_content)
        self.assertIn("CA_INSERTION_MODE=${CA_INSERTION_MODE:?", addcfg_content)

        for relative_path in (
            "cross_attention/mca/msa_gen_motion_mca.py",
            "cross_attention/mca/msa_gen_motion_mca_op.py",
        ):
            with self.subTest(path=relative_path):
                content = (EXPLORATIONS / relative_path).read_text()
                self.assertNotIn("RESUME_TRANS_A", content)
                self.assertNotIn("RESUME_TRANS_B", content)
                resume_trans = next(
                    statement.value
                    for statement in ast.parse(content).body
                    if isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "resume_trans"
                        for target in statement.targets
                    )
                )
                self.assertEqual(ast.literal_eval(resume_trans), latent_checkpoint)
                self.assertIn("use_joint_cfg = True", content)

                tree = ast.parse(content)
                wrapper_call = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "LLaMARAGLatentRetrWrapper"
                )
                keyword_values = {
                    keyword.arg: keyword.value for keyword in wrapper_call.keywords
                }
                self.assertIn("ca_every_n_layers", keyword_values)
                if relative_path.endswith("msa_gen_motion_mca.py"):
                    self.assertEqual(
                        ast.literal_eval(keyword_values["ca_every_n_layers"]), 4
                    )
                if "ca_insertion_mode" in keyword_values:
                    insertion_mode = keyword_values["ca_insertion_mode"]
                    if isinstance(insertion_mode, ast.Name):
                        insertion_mode = next(
                            statement.value
                            for statement in tree.body
                            if isinstance(statement, ast.Assign)
                            and any(
                                isinstance(target, ast.Name)
                                and target.id == insertion_mode.id
                                for target in statement.targets
                            )
                        )
                    self.assertEqual(
                        ast.literal_eval(insertion_mode),
                        "after_sa",
                    )
                else:
                    self.fail("missing explicit ca_insertion_mode")

        mca_op_content = (
            EXPLORATIONS / "cross_attention/mca/msa_gen_motion_mca_op.py"
        ).read_text()
        self.assertIn("ca_every_n_layers_override = 4", mca_op_content)

        local_content = (
            EXPLORATIONS / "cross_attention/local_rag/msa_gen_motion_local.py"
        ).read_text()
        self.assertIn(
            "Experiments/explorations/cross_attention/local_rag/"
            "MotionStreamer_t2m_272_msa_rag_local_L4_k3_crossattn/"
            "net_Iter100000.pth",
            local_content,
        )

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
