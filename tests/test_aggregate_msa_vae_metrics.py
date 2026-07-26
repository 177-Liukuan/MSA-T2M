import copy
import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from aggregate_msa_vae_metrics import (
    TARGET_METRICS,
    aggregate_variant,
    load_manifest,
    write_aggregate_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "AGGREGATE_msa_vae_metrics.sh"


class AggregateMSAVAEMetricsTest(unittest.TestCase):
    @staticmethod
    def _manifest(seed, checkpoint_suffix, metric_value):
        metrics = {
            name: float(metric_value)
            for name in TARGET_METRICS
        }
        parent_metadata = {
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
            "training_args": {
                "seed": seed,
                "exp_name": f"phase1-seed-{seed}",
                "global_align_weight": 0.25,
                "local_align_weight": 0.05,
                "root_loss": 7.0,
                "latent_recon_weight": 1.0,
                "eval_iter": 2500,
                "validation_seed": 123,
                "validation_batch_size": 32,
                "msa_data_mode": "humanml_full",
                "use_ft_split": False,
                "num_gpus": 2,
                "resume_cnn_pth": "/models/fixed-tae.pth",
                "resume_cnn_sha256": "d" * 64,
            },
        }
        return {
            "seed": 2026,
            "batch_size": 32,
            "protocol": {
                "version": "msa-vae-standard-v2",
                "retrieval": "TMR-full-normal",
            },
            "metrics": metrics,
            "checkpoint": {
                "sha256": checkpoint_suffix * 64,
                "path": f"/models/seed-{seed}/net_last.pth",
                "metadata": {
                    "format_version": 1,
                    "phase": 2,
                    "sequence_mode": "mixed",
                    "window_size": 64,
                    "full_seq_batch_size": 16,
                    "window_replay_interval": 4,
                    "down_t": 2,
                    "stride_t": 2,
                    "unit_length": 4,
                    "latent_dim": 16,
                    "normalized_loss_version": 1,
                    "lineage": {
                        "parent_checkpoint_path": (
                            f"/models/phase1-seed-{seed}.pth"
                        ),
                        "parent_checkpoint_metadata": parent_metadata,
                    },
                    "training_args": {
                        "seed": seed,
                        "exp_name": f"variant-seed-{seed}",
                        "global_align_weight": 0.25,
                        "local_align_weight": 0.05,
                        "root_loss": 7.0,
                        "latent_recon_weight": 1.0,
                        "eval_iter": 2500,
                        "validation_seed": 123,
                        "validation_batch_size": 32,
                        "msa_data_mode": "humanml_full",
                        "use_ft_split": False,
                        "num_gpus": 2,
                        "resume_cnn_pth": "/models/fixed-tae.pth",
                        "resume_cnn_sha256": "d" * 64,
                    }
                },
            },
            "evaluator": {"sha256": "e" * 64},
            "model_config": {
                "values": {
                    "latent_dim": 16,
                    "trans_d_model": 768,
                }
            },
            "dataset": {
                "sample_hash": "sample-hash",
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

    def _manifests(self):
        return [
            self._manifest(123, "a", 1.0),
            self._manifest(456, "b", 2.0),
            self._manifest(789, "c", 3.0),
        ]

    def test_aggregates_exactly_three_distinct_seeds_with_sample_std(self):
        result = aggregate_variant("Global + Local", self._manifests())

        self.assertEqual(result["variant"], "Global + Local")
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["seeds"], [123, 456, 789])
        self.assertEqual(result["metrics"]["fid"]["mean"], 2.0)
        self.assertEqual(result["metrics"]["fid"]["std"], 1.0)
        self.assertEqual(len(result["sources"]), 3)

    def test_rejects_incompatible_or_unverifiable_inputs(self):
        mutations = {}

        two_inputs = self._manifests()[:2]
        mutations["exactly three"] = two_inputs

        old_protocol = self._manifests()
        old_protocol[0]["protocol"]["version"] = "msa-vae-standard-v1"
        mutations["protocol"] = old_protocol

        duplicate_seed = self._manifests()
        duplicate_seed[1]["checkpoint"]["metadata"]["training_args"]["seed"] = 123
        mutations["training seeds"] = duplicate_seed

        duplicate_checkpoint = self._manifests()
        duplicate_checkpoint[1]["checkpoint"]["sha256"] = "a" * 64
        mutations["checkpoint"] = duplicate_checkpoint

        evaluator_mismatch = self._manifests()
        evaluator_mismatch[1]["evaluator"]["sha256"] = "f" * 64
        mutations["evaluator"] = evaluator_mismatch

        sample_mismatch = self._manifests()
        sample_mismatch[1]["dataset"]["sample_hash"] = "different"
        mutations["dataset"] = sample_mismatch

        skating_mismatch = self._manifests()
        skating_mismatch[1]["skating"]["fps"] = 20.0
        mutations["skating"] = skating_mismatch

        model_mismatch = self._manifests()
        model_mismatch[1]["model_config"]["values"]["latent_dim"] = 32
        mutations["model"] = model_mismatch

        weight_mismatch = self._manifests()
        weight_mismatch[1]["checkpoint"]["metadata"]["training_args"][
            "global_align_weight"
        ] = 0.5
        mutations["alignment weights"] = weight_mismatch

        tae_mismatch = self._manifests()
        tae_mismatch[1]["checkpoint"]["metadata"]["training_args"][
            "resume_cnn_sha256"
        ] = "c" * 64
        mutations["TAE"] = tae_mismatch

        evaluation_seed_mismatch = self._manifests()
        evaluation_seed_mismatch[1]["seed"] = 2027
        mutations["evaluation seed"] = evaluation_seed_mismatch

        evaluation_batch_mismatch = self._manifests()
        evaluation_batch_mismatch[1]["batch_size"] = 16
        mutations["evaluation batch size"] = evaluation_batch_mismatch

        training_config_mutations = (
            (("phase",), 1),
            (("sequence_mode",), "window"),
            (("window_replay_interval",), 8),
            (("training_args", "root_loss"), 8.0),
            (("training_args", "latent_recon_weight"), 0.5),
            (("training_args", "msa_data_mode"), "babel_sparse_global"),
            (("training_args", "use_ft_split"), True),
            (("training_args", "num_gpus"), 1),
        )
        for keys, value in training_config_mutations:
            incompatible = self._manifests()
            target = incompatible[1]["checkpoint"]["metadata"]
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(training_configuration=keys):
                with self.assertRaisesRegex(
                    ValueError,
                    "training configuration",
                ):
                    aggregate_variant("variant", incompatible)

        missing_metric = self._manifests()
        del missing_metric[1]["metrics"]["fid"]
        mutations["metric"] = missing_metric

        non_finite = self._manifests()
        non_finite[1]["metrics"]["fid"] = float("nan")
        mutations["finite"] = non_finite

        for message, manifests in mutations.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    aggregate_variant("variant", manifests)

    def test_rejects_runs_outside_official_two_stage_humanml_protocol(self):
        cases = (
            (("phase",), 1),
            (("sequence_mode",), "full"),
            (("training_args", "msa_data_mode"), "babel_sparse_global"),
            (("training_args", "use_ft_split"), True),
            (("lineage",), None),
            (
                ("lineage", "parent_checkpoint_metadata", "phase"),
                2,
            ),
            (
                (
                    "lineage",
                    "parent_checkpoint_metadata",
                    "sequence_mode",
                ),
                "window",
            ),
        )
        for keys, value in cases:
            manifests = self._manifests()
            for manifest in manifests:
                target = manifest["checkpoint"]["metadata"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
            with self.subTest(keys=keys, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "official two-stage protocol",
                ):
                    aggregate_variant("variant", manifests)

    def test_rejects_missing_or_invalid_eval_interval_in_either_phase(self):
        phase_paths = (
            ("training_args",),
            ("lineage", "parent_checkpoint_metadata", "training_args"),
        )
        for phase_path in phase_paths:
            for invalid_value in (None, 0, True, 1.5):
                manifests = self._manifests()
                for manifest in manifests:
                    target = manifest["checkpoint"]["metadata"]
                    for key in phase_path:
                        target = target[key]
                    if invalid_value is None:
                        del target["eval_iter"]
                    else:
                        target["eval_iter"] = invalid_value
                with self.subTest(
                    phase_path=phase_path,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "official two-stage protocol",
                    ):
                        aggregate_variant("variant", manifests)

    def test_rejects_missing_or_invalid_internal_validation_identity(self):
        phase_paths = (
            ("training_args",),
            ("lineage", "parent_checkpoint_metadata", "training_args"),
        )
        invalid_fields = (
            ("validation_seed", None),
            ("validation_seed", True),
            ("validation_seed", 1.5),
            ("validation_batch_size", None),
            ("validation_batch_size", 0),
            ("validation_batch_size", True),
            ("validation_batch_size", 1.5),
        )
        for phase_path in phase_paths:
            for field, invalid_value in invalid_fields:
                manifests = self._manifests()
                for manifest in manifests:
                    target = manifest["checkpoint"]["metadata"]
                    for key in phase_path:
                        target = target[key]
                    if invalid_value is None:
                        del target[field]
                    else:
                        target[field] = invalid_value
                with self.subTest(
                    phase_path=phase_path,
                    field=field,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "official two-stage protocol",
                    ):
                        aggregate_variant("variant", manifests)

    def test_rejects_cross_seed_internal_validation_mismatch(self):
        for field, value in (
            ("validation_seed", 456),
            ("validation_batch_size", 16),
        ):
            manifests = self._manifests()
            manifests[1]["checkpoint"]["metadata"]["training_args"][
                field
            ] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    "training configuration",
                ):
                    aggregate_variant("variant", manifests)

    def test_formal_aggregation_requires_phase2_net_last(self):
        for filename in ("net_best_fid.pth", "net_best_mpjpe.pth"):
            manifests = self._manifests()
            manifests[1]["checkpoint"]["path"] = (
                f"/models/seed-456/{filename}"
            )
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "net_last.pth"):
                    aggregate_variant("variant", manifests)

    def test_writes_json_numeric_csv_and_requested_markdown_row(self):
        result = aggregate_variant("Global + Local", self._manifests())

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_aggregate_artifacts(result, output)
            loaded = json.loads(
                (output / "aggregate.json").read_text(encoding="utf-8")
            )
            with (output / "aggregate.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                row = next(csv.DictReader(handle))
            markdown = (output / "table.md").read_text(encoding="utf-8")

        self.assertEqual(loaded["variant"], "Global + Local")
        self.assertEqual(float(row["fid_mean"]), 2.0)
        self.assertEqual(float(row["fid_std"]), 1.0)
        self.assertIn("| Global + Local |", markdown)
        self.assertEqual(markdown.count("2.000 ± 1.000"), len(TARGET_METRICS))
        self.assertNotIn("R@2", markdown)
        self.assertNotIn("R@3", markdown)

    def test_load_manifest_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.json"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_manifest(path)


class AggregateLauncherTest(unittest.TestCase):
    def test_active_mgpt_environment_forwards_all_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            capture = temp_root / "arguments.txt"
            python = temp_root / "python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{temp_root}:{env['PATH']}"
            env["CAPTURE_FILE"] = str(capture)
            env["CONDA_DEFAULT_ENV"] = "mgpt"
            arguments = [
                "--variant",
                "Global + Local",
                "--output-dir",
                "output/table",
                "a/metrics.json",
                "b/metrics.json",
                "c/metrics.json",
            ]

            completed = subprocess.run(
                ["bash", str(LAUNCHER)] + arguments,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            captured = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                Path(captured[0]),
                ROOT / "aggregate_msa_vae_metrics.py",
            )
            self.assertEqual(captured[1:], arguments)


if __name__ == "__main__":
    unittest.main()
