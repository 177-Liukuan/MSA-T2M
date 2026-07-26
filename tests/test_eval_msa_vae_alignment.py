import csv
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

from eval_msa_vae_alignment import (
    AlignmentEvaluationPaths,
    build_alignment_result_manifest,
    evaluate_global_alignment_and_realism,
    evaluate_local_alignment,
    load_frozen_humanml_motion_encoder,
    parse_args,
    preflight_alignment_assets,
    resolve_cli_paths,
    validate_runtime_args,
    write_alignment_result_artifacts,
)
from humanml3d_272.dataset_eval_msa_vae_metrics import (
    collate_msa_vae_alignment,
)
from utils.msa_vae_eval_config import ResolvedMSAVAEConfig
from utils.msa_vae_metrics import SkatingConfig


class _Distribution:
    def __init__(self, loc):
        self.loc = loc


class _TinyAlignmentDataset(Dataset):
    def __init__(self, target_mode="global"):
        self.target_mode = target_mode
        self.items = []
        lengths = (8, 12, 16)
        caption_counts = (2, 1, 2)
        for sample_index, (length, caption_count) in enumerate(
            zip(lengths, caption_counts)
        ):
            latent_length = length // 4
            latent = torch.zeros(latent_length, 2)
            latent[:, 0] = float(sample_index + 1)
            latent[:, 1] = torch.arange(
                1,
                latent_length + 1,
                dtype=torch.float32,
            )
            motion = torch.zeros(length, 272)
            motion[:, :2] = latent.repeat_interleave(4, dim=0)
            motion[:, 100] = float(sample_index)
            item = {
                "sample_id": "sample-{}".format(sample_index),
                "caption": "caption-{}-0".format(sample_index),
                "motion": motion,
                "length": length,
                "raw_length": length,
                "target_mode": target_mode,
            }
            if target_mode == "global":
                item.update(
                    {
                        "all_captions": tuple(
                            "caption-{}-{}".format(
                                sample_index,
                                caption_index,
                            )
                            for caption_index in range(caption_count)
                        ),
                        "caption_line_indices": tuple(
                            range(caption_count)
                        ),
                        "global_text_embeddings": F.one_hot(
                            torch.full(
                                (caption_count,),
                                sample_index,
                                dtype=torch.long,
                            ),
                            num_classes=3,
                        ).float(),
                    }
                )
            else:
                item["local_text_embeddings"] = F.pad(
                    latent,
                    (0, 1),
                )
            self.items.append(item)
        self.sample_ids = [item["sample_id"] for item in self.items]
        self.sample_hash = "ordered-{}-sample-hash".format(target_mode)
        self.target_directory = "/targets/{}".format(target_mode)
        self.target_hash = "{}-target-hash".format(target_mode)
        self.caption_hash = "complete-caption-hash"
        self.caption_count = (
            sum(caption_counts) if target_mode == "global" else 0
        )
        self.local_token_count = (
            sum(length // 4 for length in lengths)
            if target_mode == "local"
            else 0
        )
        self.split_file = Path("/splits/{}.txt".format(target_mode))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @staticmethod
    def inv_transform(array):
        return np.asarray(array)


class _FakeSemanticMSAVAE(torch.nn.Module):
    def __init__(self, poison_value=0.0):
        super().__init__()
        self.poison_value = poison_value
        self.semantic_calls = 0
        self.decoder_inputs = []

    def forward(self, motions, lengths=None, semantic_only=False):
        if not semantic_only:
            raise AssertionError(
                "internal evaluation must use semantic_only=True"
            )
        self.semantic_calls += 1
        labels = motions[:, 0, 100].round().long()
        global_features = F.one_hot(labels, num_classes=3).float()
        mu = motions[:, ::4, :2].clone()
        local_features = F.pad(mu, (0, 1))
        return {
            "mu": mu,
            "clip_global_feat": global_features,
            "clip_local_feat": local_features,
            "x_recon": torch.full_like(motions, self.poison_value),
        }

    def forward_decoder(self, latent):
        self.decoder_inputs.append(latent.detach().clone())
        batch, latent_length, _ = latent.shape
        prediction = torch.zeros(
            batch,
            latent_length * 4,
            272,
            device=latent.device,
        )
        prediction[:, :, :2] = latent.repeat_interleave(4, dim=1)
        return prediction


class _FakeMotionEncoder(torch.nn.Module):
    def forward(self, motions, lengths):
        frame_index = torch.arange(
            motions.shape[1],
            device=motions.device,
        )[None, :]
        mask = frame_index < lengths[:, None]
        pooled = (
            motions[:, :, :2] * mask[:, :, None]
        ).sum(dim=1) / lengths[:, None]
        return _Distribution(pooled)


class _StaticLoader:
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def _recover_fixture(features, joint_count):
    if joint_count != 22:
        raise AssertionError("expected 22 joints")
    return np.asarray(features)[:, :66].reshape(-1, 22, 3)


class InternalAlignmentEvaluationTest(unittest.TestCase):
    @staticmethod
    def _loader(target_mode, batch_size):
        dataset = _TinyAlignmentDataset(target_mode=target_mode)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_msa_vae_alignment,
        )

    @mock.patch(
        "eval_msa_vae_alignment.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_global_pass_decodes_mu_and_ignores_forward_x_recon(
        self,
        _recover,
    ):
        clean_model = _FakeSemanticMSAVAE(poison_value=0.0)
        poisoned_model = _FakeSemanticMSAVAE(poison_value=1e9)

        clean = evaluate_global_alignment_and_realism(
            clean_model,
            _FakeMotionEncoder(),
            self._loader("global", batch_size=2),
            torch.device("cpu"),
            SkatingConfig(),
        )
        poisoned = evaluate_global_alignment_and_realism(
            poisoned_model,
            _FakeMotionEncoder(),
            self._loader("global", batch_size=2),
            torch.device("cpu"),
            SkatingConfig(),
        )

        self.assertEqual(clean, poisoned)
        self.assertEqual(clean_model.semantic_calls, 2)
        self.assertEqual(len(clean_model.decoder_inputs), 2)
        for decoder_input in clean_model.decoder_inputs:
            self.assertTrue(torch.isfinite(decoder_input).all())
        self.assertEqual(clean["sample_count"], 3)
        self.assertEqual(clean["caption_count"], 5)
        self.assertEqual(clean["global_cosine"], 1.0)
        self.assertEqual(clean["msa_t5_t2m_r1_percent"], 100.0)
        self.assertEqual(clean["msa_t5_m2t_r1_percent"], 100.0)
        self.assertAlmostEqual(clean["fid"], 0.0, places=8)
        self.assertAlmostEqual(clean["mpjpe_mm"], 0.0, places=4)

    @mock.patch(
        "eval_msa_vae_alignment.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_global_results_repeat_and_are_batch_size_invariant(
        self,
        _recover,
    ):
        model = _FakeSemanticMSAVAE()
        loader = self._loader("global", batch_size=3)

        first = evaluate_global_alignment_and_realism(
            model,
            _FakeMotionEncoder(),
            loader,
            torch.device("cpu"),
            SkatingConfig(),
        )
        second = evaluate_global_alignment_and_realism(
            model,
            _FakeMotionEncoder(),
            loader,
            torch.device("cpu"),
            SkatingConfig(),
        )
        batch_one = evaluate_global_alignment_and_realism(
            _FakeSemanticMSAVAE(),
            _FakeMotionEncoder(),
            self._loader("global", batch_size=1),
            torch.device("cpu"),
            SkatingConfig(),
        )

        self.assertEqual(first, second)
        self.assertEqual(first, batch_one)

    def test_local_pass_masks_padding_and_is_motion_macro(self):
        dataset = _TinyAlignmentDataset(target_mode="local")
        batch = collate_msa_vae_alignment(
            [dataset[0], dataset[1]]
        )
        baseline_loader = _StaticLoader(dataset, [batch])
        changed_batch = dict(batch)
        changed_targets = batch["local_text_embeddings"].clone()
        changed_targets[~batch["local_mask"]] = 1e6
        changed_batch["local_text_embeddings"] = changed_targets
        changed_loader = _StaticLoader(dataset, [changed_batch])

        baseline = evaluate_local_alignment(
            _FakeSemanticMSAVAE(),
            baseline_loader,
            torch.device("cpu"),
        )
        changed = evaluate_local_alignment(
            _FakeSemanticMSAVAE(),
            changed_loader,
            torch.device("cpu"),
        )

        self.assertEqual(baseline, changed)
        self.assertEqual(baseline["local_sample_count"], 2)
        self.assertEqual(baseline["local_token_count"], 5)
        self.assertEqual(baseline["local_cosine"], 1.0)

    def test_global_and_local_passes_fail_on_missing_semantic_keys(self):
        class MissingKeys(torch.nn.Module):
            def forward(self, motions, lengths=None, semantic_only=False):
                return {}

        with self.assertRaisesRegex(ValueError, "mu"):
            evaluate_global_alignment_and_realism(
                MissingKeys(),
                _FakeMotionEncoder(),
                self._loader("global", batch_size=3),
                torch.device("cpu"),
                SkatingConfig(),
            )
        with self.assertRaisesRegex(ValueError, "clip_local_feat"):
            evaluate_local_alignment(
                MissingKeys(),
                self._loader("local", batch_size=3),
                torch.device("cpu"),
            )


def _metric_fixture():
    metrics = {
        "sample_count": 3,
        "caption_count": 5,
        "global_cosine": 0.8,
        "fid": 0.5,
        "mpjpe_mm": 20.0,
        "p_mpjpe_mm": 15.0,
        "accel_mm_per_frame2": 1.5,
        "skating_percent": 2.0,
    }
    for direction in ("t2m", "m2t"):
        metrics.update(
            {
                "msa_t5_{}_r1_percent".format(direction): 10.0,
                "msa_t5_{}_r2_percent".format(direction): 20.0,
                "msa_t5_{}_r3_percent".format(direction): 30.0,
                "msa_t5_{}_r5_percent".format(direction): 40.0,
                "msa_t5_{}_medr".format(direction): 7.0,
            }
        )
    metrics.update(
        {
            "local_sample_count": 3,
            "local_token_count": 9,
            "local_cosine": 1.0,
        }
    )
    return metrics


class AlignmentResultArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "results"
        self.global_dataset = _TinyAlignmentDataset("global")
        self.local_dataset = _TinyAlignmentDataset("local")
        self.resolved = ResolvedMSAVAEConfig(
            values={"clip_dim": 3, "stride_t": 2, "down_t": 2},
            sources={
                "clip_dim": "metadata",
                "stride_t": "metadata",
                "down_t": "metadata",
            },
        )
        self.checkpoint = {
            "path": "/models/net.pth",
            "sha256": "f" * 64,
            "metadata": {"training_args": {"seed": 123}},
        }
        self.evaluator = {
            "path": "/evaluator/epoch.ckpt",
            "sha256": "e" * 64,
        }
        self.diagnostics = {
            "shuffled_global_retrieval": {
                "msa_t5_t2m_r1_percent": 1.0,
                "msa_t5_m2t_r1_percent": 2.0,
            }
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _manifest(self, scope):
        return build_alignment_result_manifest(
            metrics=_metric_fixture(),
            diagnostics=self.diagnostics,
            checkpoint=self.checkpoint,
            evaluator=self.evaluator,
            resolved_config=self.resolved,
            global_dataset=self.global_dataset,
            local_dataset=self.local_dataset,
            local_scope=scope,
            seed=123,
            batch_size=32,
            skating_config=SkatingConfig(),
        )

    def test_manifest_labels_in_sample_local_metric_and_protocol(self):
        manifest = self._manifest("in_sample")

        self.assertEqual(
            manifest["protocol"]["version"],
            "msa-vae-internal-alignment-v1",
        )
        self.assertEqual(
            manifest["protocol"]["reconstruction_decode"],
            "posterior_mean",
        )
        self.assertEqual(
            manifest["protocol"]["retrieval"],
            "MSA-global-projection-to-SentenceT5-multi-positive",
        )
        self.assertEqual(
            manifest["local_alignment"]["scope"],
            "in_sample",
        )
        self.assertIsNone(manifest["metrics"]["local_cosine"])
        self.assertEqual(
            manifest["metrics"]["in_sample_local_cosine"],
            1.0,
        )
        self.assertEqual(
            manifest["global_realism_dataset"]["caption_count"],
            5,
        )
        self.assertEqual(
            manifest["local_alignment"]["token_count"],
            9,
        )

    def test_held_out_scope_populates_only_paper_local_metric(self):
        manifest = self._manifest("held_out")

        self.assertEqual(manifest["metrics"]["local_cosine"], 1.0)
        self.assertIsNone(
            manifest["metrics"]["in_sample_local_cosine"]
        )

    def test_artifacts_keep_blank_non_applicable_cell_and_name_msa_t5(self):
        manifest = self._manifest("in_sample")

        report = write_alignment_result_artifacts(
            manifest,
            self.output_dir,
        )

        loaded = json.loads(
            (self.output_dir / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        with (self.output_dir / "metrics.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(
            loaded["metrics"]["in_sample_local_cosine"],
            1.0,
        )
        self.assertEqual(row["local_cosine"], "")
        self.assertEqual(float(row["in_sample_local_cosine"]), 1.0)
        self.assertIn("MSA-T5", report)
        self.assertNotIn("TMR", report)

    def test_manifest_rejects_count_mismatch_and_non_finite_metric(self):
        invalid_count = _metric_fixture()
        invalid_count["caption_count"] = 4
        with self.assertRaisesRegex(ValueError, "caption count"):
            build_alignment_result_manifest(
                invalid_count,
                self.diagnostics,
                self.checkpoint,
                self.evaluator,
                self.resolved,
                self.global_dataset,
                self.local_dataset,
                "in_sample",
                123,
                32,
                SkatingConfig(),
            )
        invalid_value = _metric_fixture()
        invalid_value["global_cosine"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_alignment_result_manifest(
                invalid_value,
                self.diagnostics,
                self.checkpoint,
                self.evaluator,
                self.resolved,
                self.global_dataset,
                self.local_dataset,
                "in_sample",
                123,
                32,
                SkatingConfig(),
            )


class AlignmentEvaluationCLITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_paths_default_to_global_cache_and_need_no_distilbert(self):
        args = parse_args(
            ["Experiments/ablation/net_last.pth", "--device", "cpu"]
        )

        paths = resolve_cli_paths(self.root, args)

        self.assertEqual(
            paths.global_text_embed_dir,
            self.root / "humanml3d_272" / "text_latents_t5",
        )
        self.assertEqual(
            paths.output_dir,
            self.root
            / "output"
            / "msa_vae_alignment"
            / "ablation"
            / "net_last",
        )
        self.assertFalse(hasattr(paths, "distilbert_root"))

    def test_local_arguments_are_all_or_none(self):
        args = parse_args(
            [
                "model.pth",
                "--local-split-file",
                "train_ft.txt",
            ]
        )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            validate_runtime_args(args)

        complete = parse_args(
            [
                "model.pth",
                "--local-split-file",
                "train_ft.txt",
                "--local-text-embed-dir",
                "local_targets",
                "--local-target-scope",
                "in-sample",
            ]
        )
        validate_runtime_args(complete)

    def test_preflight_lists_missing_normalization_and_targets(self):
        paths = AlignmentEvaluationPaths(
            checkpoint=self.root / "model.pth",
            data_root=self.root / "humanml3d_272",
            split_file=self.root / "humanml3d_272" / "test.txt",
            global_text_embed_dir=self.root / "global_targets",
            local_split_file=None,
            local_text_embed_dir=None,
            evaluator_root=self.root / "Evaluator_272",
            evaluator_checkpoint=self.root / "evaluator.ckpt",
            output_dir=self.root / "output",
        )

        with self.assertRaisesRegex(
            FileNotFoundError,
            "Mean.npy.*global SentenceT5",
        ):
            preflight_alignment_assets(paths)

    def test_motion_evaluator_loader_does_not_import_text_encoder(self):
        module_sources = {
            "mld/__init__.py": "",
            "mld/models/__init__.py": "",
            "mld/models/architectures/__init__.py": "",
            "mld/models/architectures/temos/__init__.py": "",
            "mld/models/architectures/temos/motionencoder/__init__.py": "",
            "mld/models/architectures/temos/motionencoder/actor.py": (
                "import torch\n"
                "class ActorAgnosticEncoder(torch.nn.Module):\n"
                "    def __init__(self, **kwargs):\n"
                "        super().__init__()\n"
                "        self.init_kwargs = kwargs\n"
                "        self.weight = torch.nn.Parameter(torch.zeros(1))\n"
            ),
        }
        for relative, source in module_sources.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        checkpoint = self.root / "evaluator.ckpt"
        torch.save(
            {"state_dict": {"motionencoder.weight": torch.ones(1)}},
            checkpoint,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            encoder = load_frozen_humanml_motion_encoder(
                self.root,
                torch.device("cpu"),
                checkpoint,
            )

        self.assertEqual(encoder.init_kwargs["nfeats"], 272)
        self.assertFalse(encoder.training)
        self.assertNotIn(
            "mld.models.architectures.temos.textencoder.distillbert_actor",
            sys.modules,
        )
        for name in list(sys.modules):
            if name == "mld" or name.startswith("mld."):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
