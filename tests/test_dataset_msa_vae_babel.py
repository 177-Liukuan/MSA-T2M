import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from humanml3d_272.babel_stream_t5_cache import build_cache


class _FixedEncoder:
    def encode(self, texts):
        return np.stack(
            [np.full(768, index, dtype=np.float32) for index, _ in enumerate(texts)]
        )


class BabelSparseGlobalMSAVAEDatasetTest(unittest.TestCase):
    def _write_sources(self, root):
        bridge_motion = root / "bridge_motion"
        bridge_text = root / "bridge_text"
        bridge_global = root / "bridge_global"
        bridge_local = root / "bridge_local"
        babel_motion = root / "babel_motion"
        babel_text = root / "babel_text"
        babel_cache = root / "babel_cache"
        t5_model = root / "sentence-t5"
        for directory in (
            bridge_motion,
            bridge_text,
            bridge_global,
            bridge_local,
            babel_motion,
            babel_text,
            babel_cache,
            t5_model,
        ):
            directory.mkdir()

        np.save(bridge_motion / "000001.npy", np.arange(80 * 272, dtype=np.float32).reshape(80, 272))
        (bridge_text / "000001.txt").write_text("bridge caption#bridge/NOUN#0#0\n")
        np.save(bridge_global / "000001.npy", np.ones((1, 768), dtype=np.float32))
        np.save(
            bridge_local / "000001.npy",
            np.repeat(np.arange(40, dtype=np.float32)[:, None], 768, axis=1),
        )
        (root / "train_ft.txt").write_text("000001\n")

        np.save(babel_motion / "seq_1.npy", np.zeros((80, 272), dtype=np.float32))
        (babel_text / "seq_1.txt").write_text(
            "walk#walk/VERB#0#0*run#run/VERB#0#0#40\n"
        )
        manifest_path = babel_cache / "manifest.json"
        build_cache(
            "train",
            babel_motion,
            babel_text,
            babel_cache,
            _FixedEncoder(),
            str(t5_model.resolve()),
        )
        np.save(root / "Mean.npy", np.zeros(272, dtype=np.float32))
        np.save(root / "Std.npy", np.ones(272, dtype=np.float32))
        return {
            "bridge_split_file": root / "train_ft.txt",
            "bridge_motion_dir": bridge_motion,
            "bridge_text_dir": bridge_text,
            "bridge_global_embed_dir": bridge_global,
            "bridge_local_embed_dir": bridge_local,
            "babel_motion_dir": babel_motion,
            "babel_text_dir": babel_text,
            "babel_split": "train",
            "t5_model_path": t5_model,
            "babel_cache_dir": babel_cache,
            "babel_cache_manifest": manifest_path,
            "mean_path": root / "Mean.npy",
            "std_path": root / "Std.npy",
        }

    @staticmethod
    def _dataset_kwargs(paths):
        return {key: str(value) for key, value in paths.items()}

    def test_bridge_and_babel_entries_have_sparse_global_masks_and_expected_shapes(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            dataset = BabelSparseGlobalMSAVAEDataset(
                **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
            )
            bridge = dataset[dataset.index_for("hml:000001")]
            babel = dataset[dataset.index_for("babel:seq_1")]

            self.assertTrue(bridge[3])
            self.assertTrue(bridge[5])
            self.assertFalse(babel[3])
            self.assertTrue(babel[5])
            self.assertEqual(babel[0].shape, (64, 272))
            self.assertEqual(babel[4].shape, (16, 768))
            self.assertEqual(babel[1], "None")
            np.testing.assert_array_equal(babel[2], np.zeros(768, dtype=np.float32))

    def test_bridge_local_embeddings_are_upsampled_before_shared_window_slice(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            dataset = BabelSparseGlobalMSAVAEDataset(
                **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
            )
            with patch("humanml3d_272.dataset_msa_vae_babel.random.randint", side_effect=[8, 0]):
                bridge = dataset[dataset.index_for("hml:000001")]

            source = np.arange(40, dtype=np.float32)
            upsampled = source[np.round(np.linspace(0, 39, 80)).astype(int)]
            expected = np.array(
                [chunk.mean() for chunk in np.array_split(upsampled[8:72], 16)],
                dtype=np.float32,
            )
            np.testing.assert_allclose(bridge[4][:, 0], expected)
            self.assertEqual(bridge[7].shape, (768,))

    def test_babel_cache_must_match_motion_length_and_embedding_dimension(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            self._replace_cache(paths, np.zeros((79, 768), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "babel:seq_1.*cache/motion length mismatch"):
                BabelSparseGlobalMSAVAEDataset(
                    **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
                )

            self._replace_cache(paths, np.zeros((80, 767), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "babel:seq_1.*embedding dimension"):
                BabelSparseGlobalMSAVAEDataset(
                    **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
                )

    def test_missing_required_source_targets_and_empty_dataset_fail_closed(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        for relative_path, expected in (
            ("bridge_global/000001.npy", "hml:000001.*missing global"),
            ("bridge_local/000001.npy", "hml:000001.*missing local"),
            ("babel_cache/seq_1.npy", "babel:seq_1.*missing local cache"),
        ):
            with self.subTest(path=relative_path), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                paths = self._write_sources(root)
                (root / relative_path).unlink()
                with self.assertRaisesRegex(RuntimeError, expected):
                    BabelSparseGlobalMSAVAEDataset(
                        **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write_sources(root)
            (root / "train_ft.txt").write_text("")
            (paths["babel_motion_dir"] / "seq_1.npy").unlink()
            with self.assertRaisesRegex(RuntimeError, "no valid samples"):
                BabelSparseGlobalMSAVAEDataset(
                    **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
                )

    def test_manifest_source_root_mismatch_is_rejected(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            manifest = paths["babel_cache_manifest"].read_text()
            paths["babel_cache_manifest"].write_text(
                manifest.replace(str(paths["babel_motion_dir"].resolve()), "/wrong/source/root")
            )
            with self.assertRaisesRegex(RuntimeError, "cache manifest.*motion_dir"):
                BabelSparseGlobalMSAVAEDataset(
                    **self._dataset_kwargs(paths), window_size=64, unit_length=4, text_embed_dim=768
                )

    def test_validation_windows_are_deterministic_non_overlapping_with_one_tail(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEValidationDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            np.save(paths["babel_motion_dir"] / "seq_1.npy", np.zeros((150, 272), dtype=np.float32))
            build_cache(
                "train",
                paths["babel_motion_dir"],
                paths["babel_text_dir"],
                paths["babel_cache_dir"],
                _FixedEncoder(),
                str(paths["t5_model_path"].resolve()),
                overwrite=True,
            )
            dataset = BabelSparseGlobalMSAVAEValidationDataset(
                babel_motion_dir=str(paths["babel_motion_dir"]),
                babel_text_dir=str(paths["babel_text_dir"]),
                babel_cache_dir=str(paths["babel_cache_dir"]),
                babel_cache_manifest=str(paths["babel_cache_manifest"]),
                babel_split="train",
                t5_model_path=str(paths["t5_model_path"]),
                mean_path=str(paths["mean_path"]),
                std_path=str(paths["std_path"]),
                window_size=64,
                unit_length=4,
                text_embed_dim=768,
            )
            self.assertEqual(dataset.window_starts_for("babel:seq_1"), (0, 64, 86))
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0][0].shape, (64, 272))

    def test_manifest_rejects_requested_split_model_and_text_root_mismatches(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEDataset

        for changed_key, changed_value, expected in (
            ("babel_split", "val", "cache manifest split mismatch"),
            ("t5_model_path", "other-sentence-t5", "cache manifest model_signature mismatch"),
            ("babel_text_dir", "wrong_text_root", "cache manifest text_dir mismatch"),
        ):
            with self.subTest(changed_key=changed_key), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                paths = self._write_sources(root)
                kwargs = self._dataset_kwargs(paths)
                if changed_key == "t5_model_path":
                    wrong_path = root / changed_value
                    wrong_path.mkdir()
                    kwargs[changed_key] = str(wrong_path)
                elif changed_key == "babel_text_dir":
                    wrong_path = root / changed_value
                    wrong_path.mkdir()
                    kwargs[changed_key] = str(wrong_path)
                else:
                    kwargs[changed_key] = changed_value
                with self.assertRaisesRegex(RuntimeError, expected):
                    BabelSparseGlobalMSAVAEDataset(
                        **kwargs, window_size=64, unit_length=4, text_embed_dim=768
                    )

    def test_validation_index_for_returns_first_window_of_later_source(self):
        from humanml3d_272.dataset_msa_vae_babel import BabelSparseGlobalMSAVAEValidationDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._write_sources(Path(temporary_directory))
            np.save(paths["babel_motion_dir"] / "seq_1.npy", np.zeros((128, 272), dtype=np.float32))
            np.save(paths["babel_motion_dir"] / "seq_2.npy", np.zeros((64, 272), dtype=np.float32))
            (paths["babel_text_dir"] / "seq_2.txt").write_text(
                "sit#sit/VERB#0#0*stand#stand/VERB#0#0#32\n"
            )
            build_cache(
                "train",
                paths["babel_motion_dir"],
                paths["babel_text_dir"],
                paths["babel_cache_dir"],
                _FixedEncoder(),
                str(paths["t5_model_path"].resolve()),
                overwrite=True,
            )
            dataset = BabelSparseGlobalMSAVAEValidationDataset(
                babel_motion_dir=str(paths["babel_motion_dir"]),
                babel_text_dir=str(paths["babel_text_dir"]),
                babel_cache_dir=str(paths["babel_cache_dir"]),
                babel_cache_manifest=str(paths["babel_cache_manifest"]),
                babel_split="train",
                t5_model_path=str(paths["t5_model_path"]),
                mean_path=str(paths["mean_path"]),
                std_path=str(paths["std_path"]),
                window_size=64,
                unit_length=4,
                text_embed_dim=768,
            )
            self.assertEqual(dataset.window_starts_for("babel:seq_1"), (0, 64))
            self.assertEqual(dataset.index_for("babel:seq_2"), 2)
            self.assertEqual(dataset.data[dataset.index_for("babel:seq_2")][0]["name"], "babel:seq_2")

    def test_babel_mode_parser_requires_t5_and_768(self):
        from options.option_msa_vae import get_args_parser

        with patch.object(sys, "argv", ["train", "--msa_data_mode", "babel_sparse_global"]):
            args = get_args_parser()
        self.assertEqual(args.text_encoder_type, "t5")
        self.assertEqual(args.text_embed_dim, 768)
        self.assertTrue(args.msa_mean_path)
        self.assertTrue(args.msa_std_path)

        with patch.object(
            sys,
            "argv",
            ["train", "--msa_data_mode", "babel_sparse_global", "--text_encoder_type", "clip"],
        ):
            with self.assertRaises(SystemExit):
                get_args_parser()
        with patch.object(
            sys,
            "argv",
            ["train", "--msa_data_mode", "babel_sparse_global", "--text_embed_dim", "512"],
        ):
            with self.assertRaises(SystemExit):
                get_args_parser()

    @staticmethod
    def _replace_cache(paths, array):
        """Replace one test cache while preserving its manifest content hash."""
        import hashlib
        import json

        cache_path = paths["babel_cache_dir"] / "seq_1.npy"
        np.save(cache_path, array)
        manifest = json.loads(paths["babel_cache_manifest"].read_text())
        digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        manifest["records"]["seq_1"]["array_sha256"] = digest
        paths["babel_cache_manifest"].write_text(json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
