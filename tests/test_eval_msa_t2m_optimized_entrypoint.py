import importlib
import os
import unittest


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


if __name__ == "__main__":
    unittest.main()
