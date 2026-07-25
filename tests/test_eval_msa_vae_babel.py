"""CPU tests for BABEL-only MSA-VAE validation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class _Dataset:
    def __init__(self):
        self.mean = np.zeros(272, dtype=np.float32)
        self.std = np.ones(272, dtype=np.float32)

    def inv_transform(self, values):
        return values * self.std + self.mean


class _Accelerator:
    num_processes = 1
    is_main_process = True

    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, loader):
        self.prepare_calls += 1
        return loader

    def reduce(self, value, reduction="sum"):
        if reduction != "sum":
            raise AssertionError("BABEL metrics must use sum reductions")
        return value

    def wait_for_everyone(self):
        pass


class _Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, iteration):
        self.values[(name, iteration)] = float(value)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class _FixedNetwork(nn.Module):
    def __init__(self, perturb_joint=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0]))
        self.perturb_joint = perturb_joint

    def forward(self, motion):
        batch = motion.shape[0]
        reconstruction = motion.clone()
        if self.perturb_joint:
            reconstruction[..., 11] += 0.01
        local_feature = torch.zeros(batch, 2, 4, device=motion.device)
        local_feature[..., 0] = 1.0
        mu = torch.zeros(batch, 2, 2, device=motion.device)
        logvar = torch.zeros_like(mu)
        return {
            "x_recon": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "mu_recon": mu.clone(),
            "trans_latent_target": mu.clone(),
            "clip_local_feat": local_feature,
        }


def _valid_motion(batch=2, frames=8):
    motion = torch.zeros(batch, frames, 272)
    # Identity 6D rotation in the representation consumed by recovery.
    motion[..., 2] = 1.0
    motion[..., 6] = 1.0
    return motion


def _batch():
    local = torch.zeros(2, 2, 4)
    local[0, :, 0] = 1.0
    # Deliberately opposite to the network output; this sample is masked out.
    local[1, :, 0] = -1.0
    return (
        _valid_motion(),
        ["valid", "invalid"],
        torch.zeros(2, 4),
        torch.tensor([False, False]),
        local,
        torch.tensor([True, False]),
        torch.tensor([8, 8]),
        torch.zeros(2, 4),
    )


class BabelMSAVAEEvaluationTest(unittest.TestCase):
    def test_mode_loader_selection_keeps_babel_independent_of_humanml_tmr_data(self):
        from eval_msa_vae import build_validation_loader

        def forbidden_humanml_loader(*_args, **_kwargs):
            raise AssertionError("BABEL mode must not initialize the HumanML loader")

        sentinel = object()
        args = SimpleNamespace(
            msa_data_mode="babel_sparse_global",
            batch_size=4,
            window_size=64,
            down_t=2,
            text_embed_dim=768,
            t5_model_path="t5",
            msa_mean_path="Mean.npy",
            msa_std_path="Std.npy",
            babel_val_motion_dir="motion",
            babel_val_text_dir="text",
            babel_val_t5_cache_dir="cache",
            babel_val_cache_manifest="manifest.json",
        )

        loader = build_validation_loader(
            args,
            humanml_loader_factory=forbidden_humanml_loader,
            babel_loader_factory=lambda **_kwargs: sentinel,
        )
        self.assertIs(loader, sentinel)

    def test_training_loader_selection_returns_babel_train_and_validation_streams(self):
        from utils.eval_msa_vae_babel import build_msa_training_loaders

        train_sentinel = object()
        validation_sentinel = object()
        args = SimpleNamespace(
            msa_data_mode="babel_sparse_global",
            batch_size=4,
            window_size=64,
            down_t=2,
            text_embed_dim=768,
            t5_model_path="t5",
            msa_mean_path="Mean.npy",
            msa_std_path="Std.npy",
            bridge_split_file="train_ft.txt",
            bridge_motion_dir="bridge_motion",
            bridge_text_dir="bridge_text",
            bridge_global_embed_dir="bridge_global",
            bridge_local_embed_dir="bridge_local",
            babel_train_motion_dir="train_motion",
            babel_train_text_dir="train_text",
            babel_train_t5_cache_dir="train_cache",
            babel_train_cache_manifest="train_manifest.json",
            babel_val_motion_dir="val_motion",
            babel_val_text_dir="val_text",
            babel_val_t5_cache_dir="val_cache",
            babel_val_cache_manifest="val_manifest.json",
        )

        train_loader, validation_loader, backend = build_msa_training_loaders(
            args,
            humanml_train_factory=lambda *_args, **_kwargs: self.fail(
                "BABEL mode selected HumanML training"
            ),
            humanml_validation_factory=lambda *_args, **_kwargs: self.fail(
                "BABEL mode selected HumanML validation"
            ),
            babel_train_factory=lambda **_kwargs: train_sentinel,
            babel_validation_factory=lambda **_kwargs: validation_sentinel,
        )

        self.assertIs(train_loader, train_sentinel)
        self.assertIs(validation_loader, validation_sentinel)
        self.assertEqual(backend, "babel_reconstruction")

    def test_exact_reconstruction_and_masked_local_alignment(self):
        from utils.eval_msa_vae_babel import evaluate_msa_vae_babel

        accelerator = _Accelerator()
        network = _FixedNetwork()
        network.train()
        result = evaluate_msa_vae_babel(
            out_dir="unused",
            val_loader=[_batch()],
            net=network,
            dataset=_Dataset(),
            logger=_Logger(),
            writer=_Writer(),
            iteration=7,
            phase=1,
            best_semantic=-1.0,
            best_mpjpe=-1.0,
            device=torch.device("cpu"),
            accelerator=accelerator,
            metadata={"local_align_weight": 0.2},
            save_checkpoints=False,
        )

        self.assertEqual(result.mpjpe, 0.0)
        self.assertEqual(result.reconstruction, 0.0)
        self.assertAlmostEqual(result.local_cosine, 1.0)
        self.assertAlmostEqual(result.local_loss, 0.0)
        self.assertAlmostEqual(result.local_coverage, 0.5)
        self.assertAlmostEqual(result.global_coverage, 0.0)
        self.assertTrue(network.training)
        # Preparation is owned by the caller so repeated validation does not
        # wrap an already prepared loader.
        self.assertEqual(accelerator.prepare_calls, 0)

    def test_phase_specific_selection_and_checkpoint_payload_compatibility(self):
        from utils.eval_msa_vae_babel import evaluate_msa_vae_babel

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            metadata = {
                "msa_data_mode": "babel_sparse_global",
                "local_align_weight": 0.25,
            }
            phase_one = evaluate_msa_vae_babel(
                str(output),
                [_batch()],
                _FixedNetwork(),
                _Dataset(),
                _Logger(),
                _Writer(),
                iteration=1,
                phase=1,
                best_semantic=float("inf"),
                best_mpjpe=float("inf"),
                device=torch.device("cpu"),
                accelerator=_Accelerator(),
                metadata=metadata,
            )
            self.assertEqual(phase_one.best_semantic, 0.0)
            self.assertTrue((output / "net_best_semantic.pth").is_file())
            self.assertFalse((output / "net_best_mpjpe.pth").exists())

            payload = torch.load(
                output / "net_best_semantic.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(set(payload), {"net", "metadata"})
            self.assertEqual(set(payload["net"]), {"weight"})
            self.assertEqual(payload["metadata"]["msa_data_mode"], "babel_sparse_global")
            self.assertEqual(
                payload["metadata"]["supervision_coverage"],
                {"global": 0.0, "local": 0.5},
            )

            worse_semantic_network = _FixedNetwork()
            worse_semantic_network.weight.data.fill_(9.0)
            not_improved = evaluate_msa_vae_babel(
                str(output),
                [_batch()],
                worse_semantic_network,
                _Dataset(),
                _Logger(),
                _Writer(),
                iteration=2,
                phase=1,
                best_semantic=-1.0,
                best_mpjpe=float("inf"),
                device=torch.device("cpu"),
                accelerator=_Accelerator(),
                metadata=metadata,
            )
            self.assertEqual(not_improved.best_semantic, -1.0)
            unchanged = torch.load(
                output / "net_best_semantic.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(unchanged["net"]["weight"].item(), 3.0)

            phase_two = evaluate_msa_vae_babel(
                str(output),
                [_batch()],
                _FixedNetwork(perturb_joint=True),
                _Dataset(),
                _Logger(),
                _Writer(),
                iteration=3,
                phase=2,
                best_semantic=phase_one.best_semantic,
                best_mpjpe=float("inf"),
                device=torch.device("cpu"),
                accelerator=_Accelerator(),
                metadata=metadata,
            )
            self.assertGreater(phase_two.mpjpe, 0.0)
            self.assertTrue((output / "net_best_mpjpe.pth").is_file())

    def test_standalone_mode_does_not_write_checkpoints(self):
        from utils.eval_msa_vae_babel import evaluate_msa_vae_babel

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = evaluate_msa_vae_babel(
                str(output),
                [_batch()],
                _FixedNetwork(),
                _Dataset(),
                _Logger(),
                _Writer(),
                iteration=0,
                phase=2,
                best_semantic=float("inf"),
                best_mpjpe=float("inf"),
                device=torch.device("cpu"),
                accelerator=_Accelerator(),
                metadata={"local_align_weight": 0.2},
                save_checkpoints=False,
            )
            self.assertEqual(result.best_semantic, float("inf"))
            self.assertEqual(result.best_mpjpe, float("inf"))
            self.assertEqual(list(output.glob("*.pth")), [])

    def test_metadata_identity_and_mode_validation(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            validate_msa_checkpoint_metadata,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mean_path = root / "Mean.npy"
            std_path = root / "Std.npy"
            manifest_path = root / "manifest.json"
            np.save(mean_path, np.zeros(272))
            np.save(std_path, np.ones(272))
            manifest_path.write_text('{"cache": "identity"}')
            args = SimpleNamespace(
                msa_data_mode="babel_sparse_global",
                msa_mean_path=str(mean_path),
                msa_std_path=str(std_path),
                babel_val_cache_manifest=str(manifest_path),
                global_align_weight=0.5,
                local_align_weight=0.2,
                phase=1,
            )

            metadata = build_msa_checkpoint_metadata(args)
            self.assertEqual(metadata["msa_data_mode"], "babel_sparse_global")
            self.assertEqual(metadata["mean_path"], str(mean_path.resolve()))
            self.assertEqual(metadata["std_path"], str(std_path.resolve()))
            self.assertEqual(len(metadata["cache_manifest_sha256"]), 64)
            validate_msa_checkpoint_metadata(metadata, args)

            wrong_mode = dict(metadata, msa_data_mode="humanml_full")
            with self.assertRaisesRegex(ValueError, "data mode"):
                validate_msa_checkpoint_metadata(wrong_mode, args)


if __name__ == "__main__":
    unittest.main()
