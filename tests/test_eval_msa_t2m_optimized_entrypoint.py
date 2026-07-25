import importlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class OptimizedEntrypointTests(unittest.TestCase):
    def test_import_has_no_runtime_side_effects(self):
        original_cwd = os.getcwd()

        module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")

        self.assertEqual(os.getcwd(), original_cwd)
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.build_optimized_parser))

    def test_parser_marks_pipeline_and_uses_distinct_default_name(self):
        module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")

        args = module.build_optimized_parser([])

        self.assertTrue(args.optimized_pipeline)
        self.assertEqual(args.generative_head_type, "ddpm")
        self.assertEqual(
            args.exp_name,
            "MotionStreamer_t2m_272_msa_rag_t5_optimized",
        )

    def test_parser_rejects_non_ddpm_head(self):
        module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")

        with self.assertRaisesRegex(ValueError, "DDPM"):
            module.build_optimized_parser(
                ["--generative_head_type", "rectified_flow"]
            )

    def test_preflight_rejects_missing_checkpoint_before_model_loading(self):
        module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            text_latents = root / "text_latents"
            hcls = root / "hcls"
            t5 = root / "t5"
            latent_dir = root / "latents"
            for directory in (
                data_root / "texts",
                data_root / "split",
                text_latents,
                hcls,
                t5,
                latent_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            empty_text = text_latents / "empty_text_embedding.npy"
            reference = latent_dir / "reference.npy"
            rag_checkpoint = root / "rag.pth"
            evaluator_checkpoint = root / "evaluator.ckpt"
            for path in (
                empty_text,
                reference,
                rag_checkpoint,
                evaluator_checkpoint,
            ):
                path.touch()

            args = SimpleNamespace(
                resume_pth=str(root / "missing-vae.pth"),
                resume_trans=str(rag_checkpoint),
                latent_dir=str(latent_dir),
                text_latent_dir=str(text_latents),
                hcls_dir=str(hcls),
                empty_text_path=str(empty_text),
                t5_model_path=str(t5),
                text_source="online_t5",
                disable_rag=False,
                enable_stopping=True,
                reference_end_latent_path=str(reference),
                dataname="t2m_272",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "MSA-VAE checkpoint",
            ):
                module.validate_runtime_assets(
                    args,
                    data_root=str(data_root),
                    evaluator_checkpoint=str(evaluator_checkpoint),
                )

    def test_checkpoint_key_selection_rejects_missing_rag_component(self):
        module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")

        with self.assertRaisesRegex(KeyError, "rag/rag_ema"):
            module.select_rag_checkpoint_keys(
                {"trans_ema": {}},
                use_ema=True,
                disable_rag=False,
            )


if __name__ == "__main__":
    unittest.main()
