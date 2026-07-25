import csv
import importlib
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_msa_vae_metrics import (
    EvaluationPaths,
    build_result_manifest,
    evaluate_msa_vae_metrics,
    load_evaluator_checkpoint,
    load_frozen_humanml_evaluator,
    parse_args,
    preflight_evaluation_assets,
    resolve_cli_paths,
    validate_runtime_args,
    write_result_artifacts,
)
from humanml3d_272.dataset_eval_msa_vae_metrics import (
    collate_msa_vae_metrics,
)
from utils.msa_vae_eval_config import ResolvedMSAVAEConfig
from utils.msa_vae_metrics import SkatingConfig


class _Distribution:
    def __init__(self, loc):
        self.loc = loc


class _TinyEvaluationDataset(Dataset):
    def __init__(self, lengths=(4, 5, 6)):
        self.items = []
        generator = torch.Generator().manual_seed(2026)
        for sample_index, length in enumerate(lengths):
            motion = torch.zeros(length, 272)
            motion[:, :66] = torch.randn(
                length,
                66,
                generator=generator,
            )
            motion[:, 100] = float(sample_index)
            self.items.append(
                {
                    "sample_id": "sample-{}".format(sample_index),
                    "caption": "sample-{}".format(sample_index),
                    "motion": motion,
                    "length": length,
                }
            )
        self.sample_ids = [item["sample_id"] for item in self.items]
        self.sample_hash = "ordered-sample-hash"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @staticmethod
    def inv_transform(array):
        return np.asarray(array)


class _FakeMSAVAE(torch.nn.Module):
    def __init__(self, reverse_labels=False, padded_value=None):
        super().__init__()
        self.reverse_labels = reverse_labels
        self.padded_value = padded_value
        self.forward_calls = 0

    def forward(self, motions, lengths=None):
        self.forward_calls += 1
        prediction = motions.clone()
        if self.reverse_labels:
            prediction[:, :, 100] = 2.0 - prediction[:, :, 100]
        if self.padded_value is not None:
            for index, length in enumerate(lengths.tolist()):
                prediction[index, length:] = self.padded_value
        return {"x_recon": prediction}


class _FakeTextEncoder(torch.nn.Module):
    def forward(self, captions):
        labels = torch.tensor(
            [int(caption.rsplit("-", 1)[1]) for caption in captions]
        )
        return _Distribution(F.one_hot(labels, num_classes=3).float())


class _FakeMotionEncoder(torch.nn.Module):
    def __init__(self, drop_last=False):
        super().__init__()
        self.drop_last = drop_last

    def forward(self, motions, lengths):
        labels = motions[:, 0, 100].round().long()
        embeddings = F.one_hot(labels, num_classes=3).float()
        if self.drop_last:
            embeddings = embeddings[:-1]
        return _Distribution(embeddings)


def _recover_fixture(features, joint_count):
    if joint_count != 22:
        raise AssertionError("expected 22 joints")
    return np.asarray(features)[:, :66].reshape(-1, 22, 3)


class EvalMSAVAEMetricsTest(unittest.TestCase):
    @staticmethod
    def _loader(lengths=(4, 5, 6), batch_size=2):
        dataset = _TinyEvaluationDataset(lengths=lengths)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_msa_vae_metrics,
        )

    @mock.patch(
        "eval_msa_vae_metrics.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_each_batch_is_reconstructed_once_and_retrieval_uses_prediction(
        self,
        _recover,
    ):
        model = _FakeMSAVAE(reverse_labels=True)
        evaluator = [_FakeTextEncoder(), _FakeMotionEncoder()]

        result = evaluate_msa_vae_metrics(
            model,
            evaluator,
            self._loader(),
            torch.device("cpu"),
            SkatingConfig(),
        )

        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["fid"], 0.0)
        self.assertAlmostEqual(result["t2m_r1_percent"], 100.0 / 3.0)
        self.assertAlmostEqual(result["m2t_r1_percent"], 100.0 / 3.0)
        self.assertTrue(
            all(np.isfinite(value) for value in result.values())
        )

    @mock.patch(
        "eval_msa_vae_metrics.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_valid_lengths_are_sliced_and_prediction_padding_is_zeroed(
        self,
        _recover,
    ):
        model = _FakeMSAVAE(padded_value=1e6)

        result = evaluate_msa_vae_metrics(
            model,
            [_FakeTextEncoder(), _FakeMotionEncoder()],
            self._loader(lengths=(4, 6, 6), batch_size=3),
            torch.device("cpu"),
            SkatingConfig(),
        )

        self.assertAlmostEqual(result["mpjpe_mm"], 0.0, places=5)
        self.assertAlmostEqual(result["p_mpjpe_mm"], 0.0, places=3)
        self.assertEqual(result["t2m_r1_percent"], 100.0)

    @mock.patch(
        "eval_msa_vae_metrics.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_mismatched_embedding_count_fails_closed(self, _recover):
        with self.assertRaisesRegex(ValueError, "embedding count"):
            evaluate_msa_vae_metrics(
                _FakeMSAVAE(),
                [_FakeTextEncoder(), _FakeMotionEncoder(drop_last=True)],
                self._loader(batch_size=3),
                torch.device("cpu"),
                SkatingConfig(),
            )


class ResultArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_dir = self.root / "results"
        self.metrics = {
            "sample_count": 3,
            "fid": 0.1,
            "mpjpe_mm": 20.0,
            "p_mpjpe_mm": 15.0,
            "accel_mm_per_frame2": 1.5,
            "skating_percent": 2.0,
            "t2m_r1_percent": 10.0,
            "t2m_r2_percent": 20.0,
            "t2m_r3_percent": 30.0,
            "t2m_medr": 7.0,
            "m2t_r1_percent": 11.0,
            "m2t_r2_percent": 21.0,
            "m2t_r3_percent": 31.0,
            "m2t_medr": 8.0,
        }
        self.dataset = SimpleNamespace(
            sample_ids=["a", "b", "c"],
            sample_hash="abc123",
        )
        self.resolved = ResolvedMSAVAEConfig(
            values={"trans_d_model": 768},
            sources={"trans_d_model": "metadata"},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_artifacts_record_protocol_units_identity_and_flat_csv(self):
        manifest = build_result_manifest(
            metrics=self.metrics,
            checkpoint={
                "path": "/models/net.pth",
                "size": 100,
                "mtime_ns": 123,
                "sha256": "f" * 64,
            },
            evaluator={
                "path": "/evaluator/epoch.ckpt",
                "size": 200,
                "mtime_ns": 456,
                "sha256": "e" * 64,
            },
            resolved_config=self.resolved,
            dataset=self.dataset,
            seed=123,
            skating_config=SkatingConfig(),
        )

        write_result_artifacts(manifest, self.output_dir)

        loaded = json.loads(
            (self.output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        with (self.output_dir / "metrics.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        log_text = (self.output_dir / "evaluation.log").read_text(
            encoding="utf-8"
        )
        self.assertEqual(loaded["protocol"]["retrieval"], "TMR-full-normal")
        self.assertEqual(loaded["protocol"]["version"], "msa-vae-standard-v1")
        self.assertEqual(
            loaded["protocol"]["caption_policy"],
            "first complete-motion caption",
        )
        self.assertEqual(loaded["units"]["mpjpe_mm"], "mm")
        self.assertEqual(
            loaded["units"]["accel_mm_per_frame2"],
            "mm/frame^2",
        )
        self.assertEqual(loaded["dataset"]["sample_hash"], "abc123")
        self.assertEqual(loaded["dataset"]["sample_ids"], ["a", "b", "c"])
        self.assertEqual(loaded["skating"]["smoothing_window_frames"], 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["checkpoint_sha256"], "f" * 64)
        self.assertEqual(float(rows[0]["t2m_r1_percent"]), 10.0)
        self.assertIn("TMR-full-normal", log_text)
        self.assertEqual(log_text.count("P-MPJPE"), 1)
        self.assertEqual(log_text.count("ACCEL"), 1)

    def test_manifest_rejects_non_finite_requested_metric(self):
        invalid = dict(self.metrics)
        invalid["fid"] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_result_manifest(
                metrics=invalid,
                checkpoint={"path": "model", "sha256": "f" * 64},
                evaluator={"path": "evaluator", "sha256": "e" * 64},
                resolved_config=self.resolved,
                dataset=self.dataset,
                seed=123,
                skating_config=SkatingConfig(),
            )


class EvaluationCLITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_output_is_scoped_by_experiment_and_checkpoint(self):
        args = parse_args(["Experiments/ablation/net_best.pth", "--device", "cpu"])

        paths = resolve_cli_paths(self.root, args)

        self.assertEqual(
            paths.output_dir,
            self.root / "output" / "msa_vae_metrics" / "ablation" / "net_best",
        )
        self.assertEqual(
            paths.checkpoint,
            self.root / "Experiments" / "ablation" / "net_best.pth",
        )

    def test_custom_evaluator_root_supplies_default_checkpoint_and_dependency(self):
        args = parse_args(
            [
                "model.pth",
                "--evaluator-root",
                "custom_evaluator",
                "--device",
                "cpu",
            ]
        )

        paths = resolve_cli_paths(self.root, args)

        self.assertEqual(
            paths.evaluator_checkpoint,
            self.root
            / "custom_evaluator"
            / "experiments"
            / "temos"
            / "EXP1"
            / "checkpoints"
            / "epoch=99.ckpt",
        )
        self.assertEqual(
            paths.distilbert_root,
            self.root
            / "custom_evaluator"
            / "deps"
            / "distilbert-base-uncased",
        )

    def test_preflight_lists_missing_mean_before_model_or_evaluator_loading(self):
        paths = EvaluationPaths(
            checkpoint=self.root / "model.pth",
            data_root=self.root / "humanml3d_272",
            split_file=self.root / "humanml3d_272" / "split" / "test.txt",
            evaluator_root=self.root / "Evaluator_272",
            evaluator_checkpoint=self.root / "Evaluator_272" / "epoch.ckpt",
            distilbert_root=self.root / "Evaluator_272" / "deps" / "distilbert",
            output_dir=self.root / "output",
        )

        with self.assertRaisesRegex(FileNotFoundError, "Mean.npy"):
            preflight_evaluation_assets(paths)

    def test_runtime_batch_arguments_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "batch-size"):
            validate_runtime_args(SimpleNamespace(batch_size=0, num_workers=0))
        with self.assertRaisesRegex(ValueError, "num-workers"):
            validate_runtime_args(SimpleNamespace(batch_size=1, num_workers=-1))

    def test_trusted_lightning_evaluator_payload_accepts_numpy_scalars_without_warning(self):
        checkpoint = self.root / "evaluator.ckpt"
        torch.save(
            {
                "state_dict": {"weight": torch.ones(1)},
                "legacy_scalar": np.float64(1.0),
            },
            checkpoint,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = load_evaluator_checkpoint(checkpoint)

        self.assertIn("state_dict", payload)
        self.assertFalse(
            [item for item in caught if issubclass(item.category, FutureWarning)]
        )

    def test_evaluator_root_is_available_while_unpickling_legacy_mld_objects(self):
        package = self.root / "mld"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "legacy_payload.py").write_text(
            "class LegacyValue:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(self.root))
        try:
            module = importlib.import_module("mld.legacy_payload")
            checkpoint = self.root / "legacy_evaluator.ckpt"
            torch.save(
                {
                    "state_dict": {"weight": torch.ones(1)},
                    "legacy": module.LegacyValue(7),
                },
                checkpoint,
            )
        finally:
            sys.path.remove(str(self.root))
            sys.modules.pop("mld.legacy_payload", None)
            sys.modules.pop("mld", None)

        payload = load_evaluator_checkpoint(
            checkpoint,
            evaluator_root=self.root,
        )

        self.assertEqual(payload["legacy"].value, 7)
        self.assertNotIn(str(self.root), sys.path)
        sys.modules.pop("mld.legacy_payload", None)
        sys.modules.pop("mld", None)

    def test_frozen_evaluator_hides_only_known_nested_tensor_warning(self):
        module_sources = {
            "mld/__init__.py": "",
            "mld/models/__init__.py": "",
            "mld/models/architectures/__init__.py": "",
            "mld/models/architectures/temos/__init__.py": "",
            "mld/models/architectures/temos/motionencoder/__init__.py": "",
            "mld/models/architectures/temos/textencoder/__init__.py": "",
            "mld/models/architectures/temos/motionencoder/actor.py": (
                "import warnings\n"
                "import torch\n"
                "class ActorAgnosticEncoder(torch.nn.Module):\n"
                "    def __init__(self, **kwargs):\n"
                "        super().__init__()\n"
                "        warnings.warn('enable_nested_tensor is True, but fake', UserWarning)\n"
                "        self.weight = torch.nn.Parameter(torch.zeros(1))\n"
            ),
            "mld/models/architectures/temos/textencoder/distillbert_actor.py": (
                "import warnings\n"
                "import torch\n"
                "class DistilbertActorAgnosticEncoder(torch.nn.Module):\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        super().__init__()\n"
                "        warnings.warn('enable_nested_tensor is True, but fake', UserWarning)\n"
                "        self.weight = torch.nn.Parameter(torch.zeros(1))\n"
            ),
        }
        for relative, source in module_sources.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        distilbert = self.root / "deps" / "distilbert"
        distilbert.mkdir(parents=True)
        checkpoint = self.root / "evaluator.ckpt"
        torch.save(
            {
                "state_dict": {
                    "textencoder.weight": torch.ones(1),
                    "motionencoder.weight": torch.ones(1),
                }
            },
            checkpoint,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            encoders = load_frozen_humanml_evaluator(
                self.root,
                torch.device("cpu"),
                checkpoint,
                distilbert,
            )

        self.assertEqual(len(encoders), 2)
        self.assertFalse(
            [
                item
                for item in caught
                if str(item.message).startswith("enable_nested_tensor is True")
            ]
        )
        for name in list(sys.modules):
            if name == "mld" or name.startswith("mld."):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
