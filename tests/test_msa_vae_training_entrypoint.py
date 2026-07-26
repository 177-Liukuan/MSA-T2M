import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.msa_vae_training import (
    build_phase2_optimizer_param_groups,
    build_global_alignment_target,
    build_msa_checkpoint_metadata,
    configure_msa_vae_trainability,
    inherit_msa_checkpoint_lineage,
    save_msa_checkpoint,
    select_training_batch,
    validate_msa_checkpoint_metadata,
    validate_phase2_parent_metadata,
    validate_phase2_resume_requirement,
)

ROOT = Path(__file__).resolve().parents[1]


class TrainingBatchRoutingTest(unittest.TestCase):
    def test_modes_consume_the_expected_iterators(self):
        full_iter = iter(["full-1", "full-2"])
        window_iter = iter(["window-1", "window-2"])

        batch, kind = select_training_batch(
            step=1,
            mode="full",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("full-1", "full"))

        batch, kind = select_training_batch(
            step=1,
            mode="window",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("window-1", "window"))

        batch, kind = select_training_batch(
            step=4,
            mode="mixed",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("window-2", "window"))

        batch, kind = select_training_batch(
            step=5,
            mode="mixed",
            full_iter=full_iter,
            window_iter=window_iter,
            replay_interval=4,
        )
        self.assertEqual((batch, kind), ("full-2", "full"))

    def test_unknown_mode_is_rejected_without_consuming_data(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            select_training_batch(
                step=1,
                mode="prefix",
                full_iter=iter(["full"]),
                window_iter=iter(["window"]),
                replay_interval=4,
            )


class GlobalTargetRoutingTest(unittest.TestCase):
    def setUp(self):
        self.global_text = torch.tensor([[3.0, 4.0]])
        self.local_text = torch.tensor([[-3.0, -4.0]])
        self.has_local = torch.tensor([True])
        self.total_frames = torch.tensor([128])

    def test_full_and_mixed_modes_use_only_complete_caption_target(self):
        expected = F.normalize(self.global_text, dim=-1)

        for mode in ("full", "mixed"):
            actual = build_global_alignment_target(
                global_text=self.global_text,
                local_pooled=self.local_text,
                has_local=self.has_local,
                total_frames=self.total_frames,
                window_size=64,
                sequence_mode=mode,
                spotlight_alpha=-1.0,
            )
            torch.testing.assert_close(actual, expected)

    def test_window_mode_preserves_spotlight_interpolation(self):
        actual = build_global_alignment_target(
            global_text=self.global_text,
            local_pooled=self.local_text,
            has_local=self.has_local,
            total_frames=self.total_frames,
            window_size=64,
            sequence_mode="window",
            spotlight_alpha=-1.0,
        )

        torch.testing.assert_close(
            actual,
            F.normalize(
                0.5 * self.global_text + 0.5 * self.local_text,
                dim=-1,
            ),
        )


class TinyMSACore(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = nn.Linear(3, 2, bias=False)
        self.cnn_decoder = nn.Linear(2, 3, bias=False)
        self.decode_proj = nn.Linear(2, 2, bias=False)
        self.trans_encoder = nn.Linear(2, 2, bias=False)
        self.trans_decoder = nn.Linear(2, 2, bias=False)
        self.global_proj = nn.Linear(2, 4, bias=False)
        self.local_proj = nn.Linear(2, 4, bias=False)


class TinyMSAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.msa_vae = TinyMSACore()


class PhaseTrainabilityTest(unittest.TestCase):
    def test_phase2_frozen_local_projection_preserves_requested_module_states(self):
        model = TinyMSAWrapper()

        state = configure_msa_vae_trainability(
            model,
            phase=2,
            freeze_phase2_local_proj=True,
        )

        self.assertFalse(state["local_proj"])
        for name in (
            "cnn_encoder",
            "cnn_decoder",
            "decode_proj",
            "trans_encoder",
            "trans_decoder",
            "global_proj",
        ):
            self.assertTrue(state[name], msg=name)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.msa_vae.cnn_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.msa_vae.local_proj.parameters()
            )
        )

    def test_local_loss_crosses_frozen_projection_and_optimizer_excludes_it(self):
        torch.manual_seed(123)
        model = TinyMSAWrapper()
        configure_msa_vae_trainability(
            model,
            phase=2,
            freeze_phase2_local_proj=True,
        )
        groups = build_phase2_optimizer_param_groups(
            model,
            lr=1e-3,
            cnn_lr_scale=0.1,
        )
        optimizer = torch.optim.SGD(groups)
        local_parameter_ids = {
            id(parameter)
            for parameter in model.msa_vae.local_proj.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        self.assertTrue(local_parameter_ids.isdisjoint(optimizer_parameter_ids))

        encoder_before = model.msa_vae.cnn_encoder.weight.detach().clone()
        projection_before = model.msa_vae.local_proj.weight.detach().clone()
        inputs = torch.ones(4, 3)
        mu = model.msa_vae.cnn_encoder(inputs)
        prediction = model.msa_vae.local_proj(mu)
        loss = (prediction - torch.ones_like(prediction)).square().mean()
        loss.backward()

        encoder_grad = model.msa_vae.cnn_encoder.weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all())
        self.assertGreater(encoder_grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.msa_vae.local_proj.weight.grad)

        optimizer.step()
        self.assertFalse(
            torch.equal(
                encoder_before,
                model.msa_vae.cnn_encoder.weight.detach(),
            )
        )
        torch.testing.assert_close(
            projection_before,
            model.msa_vae.local_proj.weight.detach(),
        )

    def test_phase0_and_phase1_preserve_existing_trainability_contracts(self):
        model = TinyMSAWrapper()
        for parameter in model.parameters():
            parameter.requires_grad = False

        phase0 = configure_msa_vae_trainability(model, phase=0)
        self.assertTrue(all(phase0.values()))

        phase1 = configure_msa_vae_trainability(model, phase=1)
        for name in ("cnn_encoder", "cnn_decoder", "decode_proj"):
            self.assertFalse(phase1[name], msg=name)
        for name in (
            "trans_encoder",
            "trans_decoder",
            "global_proj",
            "local_proj",
        ):
            self.assertTrue(phase1[name], msg=name)

    def test_missing_required_module_and_empty_optimizer_group_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "local_proj"):
            configure_msa_vae_trainability(
                nn.Linear(2, 2),
                phase=2,
                freeze_phase2_local_proj=True,
            )

        model = TinyMSAWrapper()
        model.msa_vae.trans_encoder = nn.Identity()
        model.msa_vae.trans_decoder = nn.Identity()
        model.msa_vae.global_proj = nn.Identity()
        model.msa_vae.local_proj = nn.Identity()
        configure_msa_vae_trainability(model, phase=2)
        with self.assertRaisesRegex(ValueError, "top"):
            build_phase2_optimizer_param_groups(
                model,
                lr=1e-3,
                cnn_lr_scale=0.1,
            )


class CheckpointMetadataTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "phase": 1,
            "sequence_mode": "full",
            "window_size": 64,
            "full_seq_batch_size": 16,
            "window_replay_interval": 4,
            "length_bucket_size": 256,
            "validation_seed": 123,
            "validation_batch_size": 32,
            "down_t": 2,
            "stride_t": 2,
            "latent_dim": 16,
            "hidden_size": 1024,
            "depth": 3,
            "dilation_growth_rate": 3,
            "trans_d_model": 768,
            "trans_nhead": 8,
            "trans_enc_layers": 6,
            "trans_dec_layers": 6,
            "trans_ff_size": 2048,
            "trans_dropout": 0.1,
            "clip_dim": 768,
            "disable_decoupling": False,
            "dataname": "t2m_272",
            "batch_size": 128,
            "use_ft_split": False,
            "seed": 123,
            "total_iter": 50000,
            "warm_up_iter": 500,
            "eval_iter": 2500,
            "lr": 1e-4,
            "lr_scheduler": [50000],
            "gamma": 0.05,
            "weight_decay": 0.0,
            "cnn_lr_scale": 0.1,
            "spotlight_alpha": -1.0,
            "global_align_weight": 0.25,
            "local_align_weight": 0.05,
            "latent_recon_weight": 1.0,
            "root_loss": 7.0,
            "exp_name": "global_local_seed123",
            "msa_data_mode": "humanml_full",
            "text_encoder_type": "t5",
            "text_embed_dim": 768,
            "use_offline_global_text": True,
            "num_gpus": 2,
            "resume_cnn_pth": "Experiments/causal-tae/net.pth",
            "resume_cnn_sha256": "a" * 64,
            "resume_pth": None,
            "freeze_phase2_local_proj": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_checkpoint_payload_preserves_state_and_metadata(self):
        model = nn.Linear(3, 2)
        metadata = build_msa_checkpoint_metadata(self._args())

        self.assertEqual(metadata["unit_length"], 4)
        self.assertEqual(metadata["training_args"]["seed"], 123)
        self.assertEqual(
            metadata["training_args"]["global_align_weight"],
            0.25,
        )
        self.assertEqual(metadata["training_args"]["trans_enc_layers"], 6)
        self.assertEqual(
            metadata["training_args"]["exp_name"],
            "global_local_seed123",
        )
        self.assertEqual(
            metadata["training_args"]["resume_cnn_sha256"],
            "a" * 64,
        )
        self.assertEqual(metadata["training_args"]["eval_iter"], 2500)
        self.assertEqual(metadata["training_args"]["validation_seed"], 123)
        self.assertEqual(
            metadata["training_args"]["validation_batch_size"],
            32,
        )
        self.assertFalse(
            metadata["training_args"]["freeze_phase2_local_proj"]
        )

        enabled = build_msa_checkpoint_metadata(
            self._args(phase=2, freeze_phase2_local_proj=True)
        )
        self.assertTrue(
            enabled["training_args"]["freeze_phase2_local_proj"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            save_msa_checkpoint(path, model, metadata)
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(set(payload["net"]), set(model.state_dict()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(payload["net"][key], value)

    def test_checkpoint_metadata_defaults_missing_frozen_projection_flag(self):
        args = self._args()
        del args.freeze_phase2_local_proj

        metadata = build_msa_checkpoint_metadata(args)

        self.assertFalse(
            metadata["training_args"]["freeze_phase2_local_proj"]
        )

    def test_structural_metadata_is_validated_but_phase_handoff_is_allowed(self):
        metadata = build_msa_checkpoint_metadata(self._args(phase=1))

        validate_msa_checkpoint_metadata(
            metadata,
            self._args(phase=2, sequence_mode="mixed"),
        )

        for field, value in (
            ("down_t", 3),
            ("stride_t", 3),
            ("latent_dim", 32),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_msa_checkpoint_metadata(
                        metadata,
                        self._args(**{field: value}),
                    )

        incompatible_unit = dict(metadata)
        incompatible_unit["unit_length"] = 8
        with self.assertRaisesRegex(ValueError, "unit_length"):
            validate_msa_checkpoint_metadata(
                incompatible_unit,
                self._args(),
            )

    def test_legacy_checkpoint_without_metadata_is_accepted(self):
        validate_msa_checkpoint_metadata(None, self._args())

    def test_phase2_inherits_fixed_tae_identity_and_parent_metadata(self):
        parent = build_msa_checkpoint_metadata(self._args(phase=1))
        current = build_msa_checkpoint_metadata(
            self._args(
                phase=2,
                sequence_mode="mixed",
                exp_name="phase2_seed123",
                resume_cnn_pth=None,
                resume_cnn_sha256=None,
            )
        )

        inherited = inherit_msa_checkpoint_lineage(
            current,
            parent,
            "/models/phase1_seed123/net_last.pth",
        )

        self.assertEqual(
            inherited["training_args"]["resume_cnn_pth"],
            "Experiments/causal-tae/net.pth",
        )
        self.assertEqual(
            inherited["training_args"]["resume_cnn_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            inherited["lineage"]["parent_checkpoint_path"],
            "/models/phase1_seed123/net_last.pth",
        )
        self.assertEqual(
            inherited["lineage"]["parent_checkpoint_metadata"],
            parent,
        )

        mismatched_seed = build_msa_checkpoint_metadata(
            self._args(
                phase=2,
                sequence_mode="mixed",
                seed=456,
                resume_cnn_pth=None,
                resume_cnn_sha256=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            inherit_msa_checkpoint_lineage(
                mismatched_seed,
                parent,
                "/models/phase1_seed123/net_last.pth",
            )

    def test_phase2_requires_fresh_full_sequence_phase1_parent(self):
        phase2_args = self._args(phase=2, sequence_mode="mixed")
        parent = build_msa_checkpoint_metadata(self._args(phase=1))

        validate_phase2_parent_metadata(parent, phase2_args)

        invalid_cases = (
            (None, "metadata"),
            (
                build_msa_checkpoint_metadata(
                    self._args(phase=2, sequence_mode="mixed")
                ),
                "phase",
            ),
            (
                build_msa_checkpoint_metadata(
                    self._args(phase=1, sequence_mode="window")
                ),
                "sequence_mode",
            ),
        )
        for metadata, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_phase2_parent_metadata(metadata, phase2_args)

        missing_tae = build_msa_checkpoint_metadata(self._args(phase=1))
        missing_tae["training_args"]["resume_cnn_sha256"] = None
        with self.assertRaisesRegex(ValueError, "TAE"):
            validate_phase2_parent_metadata(missing_tae, phase2_args)

    def test_phase2_requires_resume_path_before_training_starts(self):
        validate_phase2_resume_requirement(self._args(phase=1))
        validate_phase2_resume_requirement(
            self._args(
                phase=2,
                sequence_mode="mixed",
                resume_pth="/models/phase1/net_last.pth",
            )
        )

        with self.assertRaisesRegex(ValueError, "resume-pth"):
            validate_phase2_resume_requirement(
                self._args(phase=2, sequence_mode="mixed")
            )


class DeterministicValidationSourceContractTest(unittest.TestCase):
    def test_training_uses_standard_complete_validation_without_rendering(self):
        source = (ROOT / "train_msa_vae.py").read_text(encoding="utf-8")

        for symbol in (
            "MSAVAEMetricsDataset",
            "make_msa_vae_metrics_loader",
            "run_deterministic_msa_validation",
            "publish_msa_validation",
            "MSAValidationState",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, source)
        self.assertNotIn("dataset_eval_t2m.DATALoader(", source)
        self.assertNotIn("evaluation_msa_vae_multi(", source)
        self.assertNotIn("class EvalCompat", source)
        self.assertNotIn("tensorborad_add_video_xyz(", source)
        self.assertRegex(
            source,
            r"ActorAgnosticEncoder\([^)]*max_len=-1",
        )


if __name__ == "__main__":
    unittest.main()
