import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from explorations.msa_vae_alignment_realism.pilot import (
    EVAL_INTERVAL,
    PHASE_ITERATIONS,
    PILOT_SEED,
    PILOT_VARIANTS,
    TAE_CHECKPOINT,
    TAE_SHA256,
    TARGET_METRICS,
    VALIDATION_BATCH_SIZE,
    VALIDATION_SEED,
    validate_pilot_manifests,
    write_pilot_table,
)


class PilotContractTest(unittest.TestCase):
    def test_contract_matches_approved_matrix(self):
        expected = {
            "no_align": ((0.0, 0.0), (0.0, 0.0), "0,1", "0"),
            "global_only": ((0.2, 0.0), (0.05, 0.0), "2,3", "2"),
            "local_only": ((0.0, 0.2), (0.0, 0.05), "4,5", "4"),
            "global_local": ((0.2, 0.2), (0.05, 0.05), "6,7", "6"),
        }
        actual = {
            item.slug: (
                (item.phase1_global, item.phase1_local),
                (item.phase2_global, item.phase2_local),
                item.training_gpus,
                item.evaluation_gpu,
            )
            for item in PILOT_VARIANTS
        }

        self.assertEqual(actual, expected)
        self.assertEqual(PILOT_SEED, 123)
        self.assertEqual(PHASE_ITERATIONS, 25000)
        self.assertEqual(EVAL_INTERVAL, 5000)
        self.assertEqual(VALIDATION_SEED, 123)
        self.assertEqual(VALIDATION_BATCH_SIZE, 32)
        self.assertEqual(
            TAE_CHECKPOINT,
            "Experiments/causal_TAE_t2m_272_h100_20260203/"
            "net_best_mpjpe.pth",
        )
        self.assertEqual(
            TAE_SHA256,
            "7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8",
        )


class PilotCollectorTest(unittest.TestCase):
    @staticmethod
    def _training_args(variant, phase):
        if phase == 1:
            global_weight = variant.phase1_global
            local_weight = variant.phase1_local
            sequence_mode = "full"
            full_batch_size = 16
            warm_up_iter = 500
        else:
            global_weight = variant.phase2_global
            local_weight = variant.phase2_local
            sequence_mode = "mixed"
            full_batch_size = 8
            warm_up_iter = 1000
        return {
            "dataname": "t2m_272",
            "batch_size": 64,
            "use_ft_split": False,
            "length_bucket_size": 256,
            "hidden_size": 1024,
            "down_t": 2,
            "stride_t": 2,
            "depth": 3,
            "dilation_growth_rate": 3,
            "latent_dim": 16,
            "trans_d_model": 768,
            "trans_nhead": 8,
            "trans_enc_layers": 6,
            "trans_dec_layers": 6,
            "trans_ff_size": 2048,
            "trans_dropout": 0.1,
            "clip_dim": 768,
            "disable_decoupling": False,
            "total_iter": 25000,
            "warm_up_iter": warm_up_iter,
            "eval_iter": 5000,
            "validation_seed": 123,
            "validation_batch_size": 32,
            "num_gpus": 2,
            "seed": 123,
            "global_align_weight": global_weight,
            "local_align_weight": local_weight,
            "latent_recon_weight": 1.0,
            "root_loss": 7.0,
            "exp_name": (
                f"{variant.slug}_s123_phase{phase}_25k_"
                f"g{global_weight}_l{local_weight}"
            ),
            "msa_data_mode": "humanml_full",
            "text_encoder_type": "t5",
            "text_embed_dim": 768,
            "use_offline_global_text": True,
            "resume_cnn_pth": TAE_CHECKPOINT,
            "resume_cnn_sha256": TAE_SHA256,
            "sequence_mode": sequence_mode,
            "full_seq_batch_size": full_batch_size,
        }

    @classmethod
    def _metadata(cls, variant):
        parent_args = cls._training_args(variant, phase=1)
        phase2_args = cls._training_args(variant, phase=2)
        parent = {
            "format_version": 1,
            "phase": 1,
            "sequence_mode": "full",
            "window_size": 64,
            "full_seq_batch_size": 16,
            "window_replay_interval": 4,
            "down_t": 2,
            "stride_t": 2,
            "unit_length": 4,
            "latent_dim": 16,
            "normalized_loss_version": 1,
            "training_args": parent_args,
        }
        return {
            "format_version": 1,
            "phase": 2,
            "sequence_mode": "mixed",
            "window_size": 64,
            "full_seq_batch_size": 8,
            "window_replay_interval": 4,
            "down_t": 2,
            "stride_t": 2,
            "unit_length": 4,
            "latent_dim": 16,
            "normalized_loss_version": 1,
            "training_args": phase2_args,
            "lineage": {
                "parent_checkpoint_path": (
                    f"/pilot/{variant.slug}/phase1/net_last.pth"
                ),
                "parent_checkpoint_metadata": parent,
            },
        }

    @classmethod
    def _manifest(cls, variant, metric_offset):
        return {
            "seed": 123,
            "batch_size": 32,
            "protocol": {
                "version": "msa-vae-standard-v2",
                "retrieval": "TMR-full-normal",
            },
            "metrics": {
                name: float(metric_offset + index)
                for index, name in enumerate(TARGET_METRICS)
            },
            "checkpoint": {
                "path": f"/pilot/{variant.slug}/phase2/net_last.pth",
                "sha256": (
                    format(metric_offset + 1, "x")[-1] * 64
                ),
                "metadata": cls._metadata(variant),
            },
            "evaluator": {"sha256": "e" * 64},
            "model_config": {
                "values": {
                    "latent_dim": 16,
                    "trans_d_model": 768,
                },
            },
            "dataset": {
                "sample_hash": "humanml-test-hash",
                "sample_count": 2480,
            },
            "skating": {
                "foot_indices": [10, 11],
                "fps": 30.0,
                "height_threshold_m": 0.05,
                "velocity_threshold_mps": 0.5,
                "smoothing_window_frames": 8,
            },
        }

    @classmethod
    def _write_manifests(cls, output_root):
        manifests = []
        for index, variant in enumerate(PILOT_VARIANTS, start=1):
            manifest = cls._manifest(variant, index)
            path = output_root / "evaluation" / variant.slug / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            manifests.append(manifest)
        return manifests

    def test_validates_and_writes_requested_raw_single_seed_table(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp)
            self._write_manifests(output_root)

            validated = validate_pilot_manifests(output_root)
            paths = write_pilot_table(output_root)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            with paths["csv"].open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(validated), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["seed"] for row in rows], ["123"] * 4)
        self.assertEqual(
            [row["Variant"] for row in rows],
            [variant.label for variant in PILOT_VARIANTS],
        )
        self.assertEqual(payload["seed_count"], 1)
        self.assertEqual(payload["seed"], 123)
        self.assertIn("single-seed pilot", payload["qualification"])
        self.assertNotIn("±", markdown)
        self.assertEqual(markdown.count("\n"), 6)
        for header in (
            "FID↓",
            "MPJPE↓",
            "P-MPJPE↓",
            "ACCEL↓",
            "Skating%↓",
            "T2M R@1↑",
            "T2M R@5↑",
            "T2M MedR↓",
            "M2T R@1↑",
            "M2T R@5↑",
            "M2T MedR↓",
        ):
            self.assertIn(header, markdown)

    def test_rejects_protocol_and_scientific_identity_mismatches(self):
        cases = (
            ("best checkpoint", lambda manifests: manifests[0]["checkpoint"].update(
                {"path": "/pilot/net_best_fid.pth"}
            )),
            ("training seed", lambda manifests: manifests[0]["checkpoint"][
                "metadata"
            ]["training_args"].update({"seed": 456})),
            ("training budget", lambda manifests: manifests[0]["checkpoint"][
                "metadata"
            ]["training_args"].update({"total_iter": 50000})),
            ("alignment weights", lambda manifests: manifests[0]["checkpoint"][
                "metadata"
            ]["training_args"].update({"global_align_weight": 0.1})),
            ("TAE", lambda manifests: manifests[0]["checkpoint"]["metadata"][
                "lineage"
            ]["parent_checkpoint_metadata"]["training_args"].update(
                {"resume_cnn_sha256": "f" * 64}
            )),
            ("dataset", lambda manifests: manifests[0]["dataset"].update(
                {"sample_hash": "different"}
            )),
            ("evaluator", lambda manifests: manifests[0]["evaluator"].update(
                {"sha256": "f" * 64}
            )),
            ("evaluation seed", lambda manifests: manifests[0].update(
                {"seed": 456}
            )),
            ("metric", lambda manifests: manifests[0]["metrics"].update(
                {"fid": math.nan}
            )),
        )
        for message, mutate in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temp:
                    output_root = Path(temp)
                    manifests = self._write_manifests(output_root)
                    mutate(manifests)
                    for variant, manifest in zip(PILOT_VARIANTS, manifests):
                        path = (
                            output_root
                            / "evaluation"
                            / variant.slug
                            / "metrics.json"
                        )
                        path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        validate_pilot_manifests(output_root)
