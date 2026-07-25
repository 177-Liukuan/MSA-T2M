"""CPU tests for BABEL-only MSA-VAE validation."""

import hashlib
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


class _IdentityAccelerator(_Accelerator):
    def __init__(self, remote_identity=None):
        super().__init__()
        self.remote_identity = remote_identity
        self.gather_calls = 0

    def gather_object(self, local_identity):
        self.gather_calls += 1
        remote = local_identity if self.remote_identity is None else self.remote_identity
        return [local_identity, remote]


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
    APPROVED_SHA256 = "a" * 64

    @staticmethod
    def _metadata_args(root, mode="babel_sparse_global", phase=1):
        root.mkdir(parents=True, exist_ok=True)
        mean_path = root / "Mean.npy"
        std_path = root / "Std.npy"
        train_manifest = root / "train-manifest.json"
        validation_manifest = root / "val-manifest.json"
        np.save(mean_path, np.zeros(272))
        np.save(std_path, np.ones(272))
        train_manifest.write_text('{"split": "train"}')
        validation_manifest.write_text('{"split": "val"}')
        return SimpleNamespace(
            msa_data_mode=mode,
            msa_mean_path=str(mean_path),
            msa_std_path=str(std_path),
            babel_train_cache_manifest=str(train_manifest),
            babel_val_cache_manifest=str(validation_manifest),
            global_align_weight=0.5,
            local_align_weight=0.2,
            phase=phase,
            resume_cnn_pth=None,
            resume_cnn_sha256=BabelMSAVAEEvaluationTest.APPROVED_SHA256,
            resume_pth=None,
            hidden_size=16,
            down_t=2,
            stride_t=2,
            depth=3,
            dilation_growth_rate=3,
            latent_dim=4,
        )

    @classmethod
    def _tagged_metadata(cls, args):
        from utils.eval_msa_vae_babel import build_msa_checkpoint_metadata

        return build_msa_checkpoint_metadata(
            args,
            causal_tae_identity={
                "path": str(Path(args.msa_mean_path).parent / "approved-joint-tae.pth"),
                "sha256": args.resume_cnn_sha256,
            },
        )

    @staticmethod
    def _write_official_causal_checkpoint(args, path):
        from models.tae import Causal_HumanTAE

        model = Causal_HumanTAE(
            hidden_size=args.hidden_size,
            down_t=args.down_t,
            stride_t=args.stride_t,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            latent_dim=args.latent_dim,
            clip_range=[-30, 20],
        )
        torch.save({"net": model.state_dict()}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return model.state_dict(), digest

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
            args = self._metadata_args(root)

            metadata = self._tagged_metadata(args)
            self.assertEqual(metadata["msa_data_mode"], "babel_sparse_global")
            self.assertEqual(metadata["mean_path"], str(Path(args.msa_mean_path).resolve()))
            self.assertEqual(metadata["std_path"], str(Path(args.msa_std_path).resolve()))
            self.assertEqual(
                metadata["train_cache_manifest_path"],
                str(Path(args.babel_train_cache_manifest).resolve()),
            )
            self.assertEqual(
                metadata["val_cache_manifest_path"],
                str(Path(args.babel_val_cache_manifest).resolve()),
            )
            self.assertEqual(len(metadata["train_cache_manifest_sha256"]), 64)
            self.assertEqual(len(metadata["val_cache_manifest_sha256"]), 64)
            self.assertEqual(
                metadata["causal_tae_artifact_sha256"],
                self.APPROVED_SHA256,
            )
            validate_msa_checkpoint_metadata(metadata, args, scope="training")

            wrong_mode = dict(metadata, msa_data_mode="humanml_full")
            with self.assertRaisesRegex(ValueError, "data mode"):
                validate_msa_checkpoint_metadata(wrong_mode, args, scope="training")

    def test_standalone_validation_requires_only_the_actual_validation_cache_identity(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            validate_msa_checkpoint_metadata,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_args = self._metadata_args(root)
            metadata = self._tagged_metadata(training_args)
            Path(training_args.babel_train_cache_manifest).unlink()
            standalone_args = SimpleNamespace(**vars(training_args))
            standalone_args.babel_train_cache_manifest = str(root / "not-present.json")

            validate_msa_checkpoint_metadata(
                metadata, standalone_args, scope="standalone"
            )

    def test_distributed_identity_rejects_a_rank_with_different_cache_hash(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            validate_distributed_msa_identity,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = self._metadata_args(Path(temporary_directory))
            metadata = self._tagged_metadata(args)
            remote = dict(
                metadata,
                train_cache_manifest_sha256="0" * 64,
            )
            identity_keys = (
                "msa_data_mode",
                "mean_path",
                "mean_sha256",
                "std_path",
                "std_sha256",
                "train_cache_manifest_path",
                "train_cache_manifest_sha256",
                "val_cache_manifest_path",
                "val_cache_manifest_sha256",
                "causal_tae_artifact_path",
                "causal_tae_artifact_sha256",
                "resume_checkpoint_path",
                "resume_checkpoint_sha256",
            )
            remote_envelope = {
                "ok": True,
                "identity": {key: remote.get(key) for key in identity_keys},
                "error": None,
            }
            with self.assertRaisesRegex(RuntimeError, "differ across ranks"):
                validate_distributed_msa_identity(
                    metadata, _IdentityAccelerator(remote_envelope)
                )

    def test_rank_local_asset_failure_reaches_collective_and_fails_both_sides(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        remote_error = {
            "ok": False,
            "identity": None,
            "error": "MSA mean not found on remote rank",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            normal_args = self._metadata_args(root / "normal", phase=0)
            normal_accelerator = _IdentityAccelerator(remote_error)
            with self.assertRaisesRegex(RuntimeError, "rank-local asset"):
                preflight_msa_training_assets(normal_args, normal_accelerator)
            self.assertEqual(normal_accelerator.gather_calls, 1)

            missing_args = self._metadata_args(root / "missing", phase=0)
            Path(missing_args.msa_mean_path).unlink()
            remote_success = {
                "ok": True,
                "identity": {"msa_data_mode": "babel_sparse_global"},
                "error": None,
            }
            missing_accelerator = _IdentityAccelerator(remote_success)
            with self.assertRaisesRegex(RuntimeError, "rank-local asset"):
                preflight_msa_training_assets(missing_args, missing_accelerator)
            self.assertEqual(missing_accelerator.gather_calls, 1)

    def test_babel_phases_fail_closed_without_required_checkpoints(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        for phase, expected in (
            (1, "Phase 1 requires"),
            (2, "Phase 2 requires"),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary_directory:
                args = self._metadata_args(Path(temporary_directory), phase=phase)
                with self.assertRaisesRegex(RuntimeError, expected):
                    preflight_msa_training_assets(args, _IdentityAccelerator())

    def test_missing_and_structurally_invalid_causal_checkpoints_fail_closed(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root)
            args.resume_cnn_pth = str(root / "missing-causal-tae.pth")
            with self.assertRaisesRegex(RuntimeError, "resume CNN checkpoint"):
                preflight_msa_training_assets(args, _IdentityAccelerator())

            malformed = root / "malformed-causal-tae.pth"
            torch.save(
                {
                    "net": {
                        "tae.encoder.x": torch.ones(1),
                        "tae.decoder.y": torch.ones(1),
                        "tae.decode_proj.z": torch.ones(1),
                    }
                },
                malformed,
            )
            args.resume_cnn_pth = str(malformed)
            args.resume_cnn_sha256 = hashlib.sha256(malformed.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "Causal TAE"):
                preflight_msa_training_assets(args, _IdentityAccelerator())

    def test_phase_one_accepts_a_structurally_valid_causal_tae(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root)
            causal_checkpoint = root / "causal-tae.pth"
            official_state, artifact_sha256 = self._write_official_causal_checkpoint(
                args, causal_checkpoint
            )
            args.resume_cnn_pth = str(causal_checkpoint)
            args.resume_cnn_sha256 = artifact_sha256

            metadata, full_checkpoint, mapped_cnn_state = preflight_msa_training_assets(
                args, _IdentityAccelerator()
            )
            self.assertEqual(metadata["phase"], 1)
            self.assertEqual(len(official_state), 70)
            self.assertEqual(
                metadata["causal_tae_artifact_sha256"], artifact_sha256
            )
            self.assertIsNone(full_checkpoint)
            self.assertEqual(len(mapped_cnn_state), len(official_state))
            self.assertTrue(
                any(
                    key.startswith("msa_vae.cnn_encoder.")
                    for key in mapped_cnn_state
                )
            )

    def test_causal_tae_shape_mismatch_is_rejected(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root)
            causal_checkpoint = root / "shape-mismatch.pth"
            official_state, _ = self._write_official_causal_checkpoint(
                args, causal_checkpoint
            )
            first_key = next(iter(official_state))
            malformed = dict(official_state)
            malformed[first_key] = torch.ones(1)
            torch.save({"net": malformed}, causal_checkpoint)
            args.resume_cnn_pth = str(causal_checkpoint)
            args.resume_cnn_sha256 = hashlib.sha256(
                causal_checkpoint.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                preflight_msa_training_assets(args, _IdentityAccelerator())

    def test_causal_tae_expected_sha_is_external_and_must_match(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root)
            causal_checkpoint = root / "causal-tae.pth"
            self._write_official_causal_checkpoint(args, causal_checkpoint)
            args.resume_cnn_pth = str(causal_checkpoint)
            args.resume_cnn_sha256 = "0" * 64

            with self.assertRaisesRegex(RuntimeError, "approved SHA-256"):
                preflight_msa_training_assets(args, _IdentityAccelerator())

    def test_preflight_returns_snapshot_used_after_causal_path_replacement(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root)
            causal_checkpoint = root / "causal-tae.pth"
            official_state, artifact_sha256 = self._write_official_causal_checkpoint(
                args, causal_checkpoint
            )
            args.resume_cnn_pth = str(causal_checkpoint)
            args.resume_cnn_sha256 = artifact_sha256
            _, _, mapped_cnn_state = preflight_msa_training_assets(
                args, _IdentityAccelerator()
            )

            torch.save({"net": {"replacement": torch.zeros(1)}}, causal_checkpoint)
            self.assertEqual(len(mapped_cnn_state), len(official_state))
            self.assertNotIn("replacement", mapped_cnn_state)

    def test_training_preflight_rejects_tagged_cross_domain_full_resumes(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            preflight_msa_training_assets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            babel_args = self._metadata_args(root / "babel")
            human_args = self._metadata_args(root / "human", mode="humanml_full")
            human_checkpoint = root / "human.pth"
            torch.save(
                {
                    "net": {"weight": torch.tensor([1.0])},
                    "metadata": build_msa_checkpoint_metadata(human_args),
                },
                human_checkpoint,
            )
            babel_args.resume_pth = str(human_checkpoint)
            with self.assertRaisesRegex(RuntimeError, "data mode"):
                preflight_msa_training_assets(
                    babel_args, _IdentityAccelerator()
                )

            babel_checkpoint = root / "babel.pth"
            torch.save(
                {
                    "net": {"weight": torch.tensor([2.0])},
                    "metadata": self._tagged_metadata(babel_args),
                },
                babel_checkpoint,
            )
            human_args.resume_pth = str(babel_checkpoint)
            with self.assertRaisesRegex(RuntimeError, "data mode"):
                preflight_msa_training_assets(
                    human_args, _IdentityAccelerator()
                )

    def test_phase_two_accepts_a_matching_tagged_babel_full_resume(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            preflight_msa_training_assets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            phase_one_args = self._metadata_args(root, phase=1)
            checkpoint_path = root / "phase-one.pth"
            torch.save(
                {
                    "net": {"weight": torch.tensor([4.0])},
                    "metadata": self._tagged_metadata(phase_one_args),
                },
                checkpoint_path,
            )
            phase_two_args = SimpleNamespace(**vars(phase_one_args))
            phase_two_args.phase = 2
            phase_two_args.resume_pth = str(checkpoint_path)

            metadata, checkpoint, cnn_state = preflight_msa_training_assets(
                phase_two_args, _IdentityAccelerator()
            )
            self.assertEqual(metadata["phase"], 2)
            self.assertEqual(checkpoint["net"]["weight"].item(), 4.0)
            self.assertIsNone(cnn_state)

    def test_phase_two_preflight_reaches_tagged_resume_with_approved_identity(self):
        """A launcher-supplied SHA lets Phase 2 pass its earliest identity gate."""
        from utils.eval_msa_vae_babel import _local_training_asset_probe

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            phase_one_args = self._metadata_args(root, phase=1)
            checkpoint_path = root / "phase-one.pth"
            torch.save(
                {
                    "net": {"weight": torch.tensor([7.0])},
                    "metadata": self._tagged_metadata(phase_one_args),
                },
                checkpoint_path,
            )
            phase_two_args = SimpleNamespace(**vars(phase_one_args))
            phase_two_args.phase = 2
            phase_two_args.resume_pth = str(checkpoint_path)
            phase_two_args.resume_cnn_sha256 = self.APPROVED_SHA256

            envelope, metadata, checkpoint, cnn_state = _local_training_asset_probe(
                phase_two_args
            )

            self.assertTrue(envelope["ok"], envelope["error"])
            self.assertEqual(metadata["phase"], 2)
            self.assertEqual(checkpoint["net"]["weight"].item(), 7.0)
            self.assertIsNone(cnn_state)

    def test_tagged_babel_resume_rejects_changed_training_cache_identity(self):
        from utils.eval_msa_vae_babel import (
            build_msa_checkpoint_metadata,
            preflight_msa_training_assets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root, phase=2)
            checkpoint_path = root / "phase-one.pth"
            torch.save(
                {
                    "net": {"weight": torch.tensor([6.0])},
                    "metadata": self._tagged_metadata(args),
                },
                checkpoint_path,
            )
            Path(args.babel_train_cache_manifest).write_text(
                '{"split": "train", "generation": 2}'
            )
            args.resume_pth = str(checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "train cache manifest identity"):
                preflight_msa_training_assets(args, _IdentityAccelerator())

    def test_humanml_training_preserves_legacy_untagged_full_resume(self):
        from utils.eval_msa_vae_babel import preflight_msa_training_assets

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._metadata_args(root, mode="humanml_full", phase=2)
            checkpoint_path = root / "legacy-humanml.pth"
            torch.save({"net": {"weight": torch.tensor([5.0])}}, checkpoint_path)
            args.resume_pth = str(checkpoint_path)

            metadata, checkpoint, cnn_state = preflight_msa_training_assets(
                args, _IdentityAccelerator()
            )
            self.assertEqual(metadata["msa_data_mode"], "humanml_full")
            self.assertEqual(checkpoint["net"]["weight"].item(), 5.0)
            self.assertIsNone(cnn_state)


if __name__ == "__main__":
    unittest.main()
