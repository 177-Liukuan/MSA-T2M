from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_ROOT_ENTRYPOINTS = {
    "train_causal_TAE.py",
    "TRAIN_causal_TAE.sh",
    "eval_causal_TAE.py",
    "EVAL_causal_TAE.sh",
    "train_msa_vae.py",
    "TRAIN_msa_vae_phase1.sh",
    "TRAIN_msa_vae_phase2.sh",
    "eval_msa_vae.py",
    "EVAL_msa_vae.sh",
    "eval_msa_vae_metrics.py",
    "EVAL_msa_vae_metrics.sh",
    "aggregate_msa_vae_metrics.py",
    "AGGREGATE_msa_vae_metrics.sh",
    "TRAIN_msa_vae_babel_phase1.sh",
    "TRAIN_msa_vae_babel_phase2.sh",
    "EVAL_msa_vae_babel.sh",
    "dataset_clip2t5.py",
    "get_text_latent_t5.py",
    "get_msa_latent.py",
    "build_msa_rag_cache.py",
    "train_t2m_rag.py",
    "TRAIN_t2m_rag.sh",
    "eval_msa_t2m_rag_t5.py",
    "EVAL_t2m_rag_t5.sh",
    "eval_msa_t2m_rag_t5_optimized.py",
    "EVAL_t2m_rag_t5_optimized.sh",
    "msa_gen_motion.py",
    "output_vis.py",
}

ARCHIVED_ENTRYPOINTS = {
    "msa_vae_alignment_realism": {
        "pilot.py",
    },
    "ablations/no_rag": {
        "demo_msa_t2m_no_rag_t5.py",
        "DEMO_msa_t2m_no_rag_t5.sh",
        "eval_msa_t2m_no_rag_t5.py",
        "EVAL_t2m_no_rag_t5.sh",
        "TRAIN_t2m_no_rag.sh",
    },
    "clip": {
        "demo_msa_t2m_clip.py",
        "eval_t2m_clip_baseline.py",
        "EVAL_t2m_clip_baseline.sh",
        "eval_t2m_rag.py",
        "EVAL_t2m_rag.sh",
        "get_text_latent_clip.py",
        "train_t2m_baseline_clip.py",
        "TRAIN_t2m_baseline_clip.sh",
    },
    "cross_attention/local_rag": {
        "eval_msa_t2m_rag_local.py",
        "EVAL_t2m_rag_local.sh",
        "msa_gen_motion_local.py",
        "train_t2m_rag_local.py",
        "TRAIN_t2m_rag_local.sh",
        "TRAIN_THEN_EVAL_t2m_rag_local.sh",
    },
    "cross_attention/mca": {
        "eval_msa_t2m_rag_mca.py",
        "EVAL_t2m_rag_mca.sh",
        "get_text_token_latent_t5.py",
        "msa_gen_motion_mca.py",
        "msa_gen_motion_mca_op.py",
        "train_t2m_rag_multi_text_token.py",
        "Train_t2m_rag_multi_text_token.sh",
    },
    "cross_attention/latent_retrieval": {
        "build_latent_retr_library.py",
        "eval_msa_t2m_rag_latent_retr.py",
        "EVAL_t2m_rag_latent_retr.sh",
        "eval_msa_t2m_rag_latent_retr_addcfg.py",
        "EVAL_t2m_rag_latent_retr_addcfg.sh",
        "precompute_latent_retr_lookup.py",
        "train_t2m_rag_latent_retr.py",
        "Train_t2m_rag_latent_retr.sh",
    },
    "rectified_flow": {
        "eval_msa_t2m_rag_t5_rf.py",
        "EVAL_t2m_rag_t5_rf.sh",
        "Train_t2m_rag_rf.sh",
    },
    "qformer": {
        "build_rag_db.py",
        "PREPARE_text_embeddings.sh",
        "train_qformer_rag.py",
        "TRAIN_qformer_rag.sh",
    },
    "motionstreamer_baselines": {
        "demo_t2m.py",
        "eval_t2m.py",
        "EVAL_t2m.sh",
        "get_latent.py",
        "motionstreamer_gen_motion.py",
        "train_motionstreamer.py",
        "TRAIN_motionstreamer.sh",
        "train_t2m.py",
        "TRAIN_t2m.sh",
        "Train_t2m_multi.sh",
        "train_t2m_cached.py",
        "TRAIN_t2m_cached.sh",
        "TRAIN_evaluator_272.sh",
    },
    "representation_experiments": {
        "demo_msa_vae_sample.py",
        "eval_sae_v1.py",
        "EVAL_sae_v1.sh",
        "train_sae_v1.py",
        "TRAIN_sae_v1.sh",
        "train_tae_gan_v1.py",
        "TRAIN_tae_gan_v1.sh",
        "EVAL_tae_gan_v1.sh",
        "TRAIN_msa_vae.sh",
        "TRAIN_msa_vae_multi.sh",
    },
    "retrieval_baselines": {
        "demo_retrieval.py",
        "RAG2Motion.py",
        "remodiffuse_gen_motion.py",
    },
    "demos_and_diagnostics": {
        "demo_msa_t2m_t5.py",
        "demo_msa_t2m_t5_02.py",
        "demo_verify_dataset.py",
        "demo_verify_t5_conversion.py",
        "generate_motion.py",
        "inspect_latent_shapes.py",
        "msa_gen_motion_batch.py",
        "render_smpl_aitviewer_pos.py",
        "render_smpl_aitviewer_rot.py",
        "representation_272_to_bvh.py",
        "smoke_test.py",
        "verify_setup.py",
        "visualize_t2m_generation.py",
    },
    "project_history": {
        "IMPLEMENTATION_SUMMARY.py",
        "WORKFLOW_GUIDE.py",
        "run.sh",
        "run_training.sh",
        "sedbash",
        "TRAIN_msa_vae_phase1.sh.bak",
        "TRAIN_msa_vae_phase2.sh.bak",
    },
}


class ExplorationLayoutTest(unittest.TestCase):
    def test_root_contains_only_official_entrypoints(self):
        actual = {
            path.name
            for pattern in ("*.py", "*.sh", "*.sh.bak")
            for path in REPO_ROOT.glob(pattern)
        }
        actual.update(
            name for name in ("sedbash",) if (REPO_ROOT / name).exists()
        )
        self.assertEqual(actual, OFFICIAL_ROOT_ENTRYPOINTS)

    def test_every_archived_entrypoint_has_one_destination(self):
        all_names = [
            name
            for names in ARCHIVED_ENTRYPOINTS.values()
            for name in names
        ]
        self.assertEqual(len(all_names), len(set(all_names)))
        for route, names in ARCHIVED_ENTRYPOINTS.items():
            route_dir = REPO_ROOT / "explorations" / route
            self.assertTrue((route_dir / "__init__.py").is_file())
            for name in names:
                self.assertTrue(
                    (route_dir / name).is_file(),
                    f"Missing archived entrypoint: {route}/{name}",
                )


if __name__ == "__main__":
    unittest.main()
