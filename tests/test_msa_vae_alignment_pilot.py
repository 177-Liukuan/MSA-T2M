import csv
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

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
    emit_contract,
    phase_checkpoint_path,
    validate_pilot_checkpoints,
    validate_pilot_manifests,
    write_pilot_table,
)

ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_ROOT = (
    ROOT / "explorations" / "msa_vae_alignment_realism"
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

    def test_tsv_uses_canonical_weight_strings_for_shell_directory_names(self):
        lines = emit_contract(Path("."), "tsv").splitlines()
        fields = [line.split("\t") for line in lines]

        self.assertEqual(fields[0][5:], ["0", "0", "0", "0"])
        self.assertEqual(fields[1][5:], ["0.2", "0", "0.05", "0"])
        self.assertEqual(fields[2][5:], ["0", "0.2", "0", "0.05"])
        self.assertEqual(
            fields[3][5:],
            ["0.2", "0.2", "0.05", "0.05"],
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

    def test_accepts_canonical_absolute_fixed_tae_path(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp)
            manifests = self._write_manifests(output_root)
            absolute_tae = str((ROOT / TAE_CHECKPOINT).resolve())
            for manifest in manifests:
                metadata = manifest["checkpoint"]["metadata"]
                metadata["training_args"]["resume_cnn_pth"] = absolute_tae
                metadata["lineage"]["parent_checkpoint_metadata"][
                    "training_args"
                ]["resume_cnn_pth"] = absolute_tae
            for variant, manifest in zip(PILOT_VARIANTS, manifests):
                path = (
                    output_root
                    / "evaluation"
                    / variant.slug
                    / "metrics.json"
                )
                path.write_text(json.dumps(manifest), encoding="utf-8")

            validated = validate_pilot_manifests(output_root)

        self.assertEqual(len(validated), 4)


class PilotRunnerTest(unittest.TestCase):
    @staticmethod
    def _fake_launcher(path, capture_variable, exit_code=0, checkpoint=True):
        checkpoint_command = (
            'mkdir -p "$OUT_DIR/$EXP_NAME"\n'
            'printf "checkpoint\\n" > "$OUT_DIR/$EXP_NAME/net_last.pth"\n'
            if checkpoint
            else ""
        )
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'capture="${{{capture_variable}}}"\n'
            "{\n"
            '  printf "arg1=%s\\n" "$1"\n'
            '  printf "arg2=%s\\n" "$2"\n'
            '  printf "cuda=%s\\n" "$CUDA_VISIBLE_DEVICES"\n'
            '  printf "out_dir=%s\\n" "$OUT_DIR"\n'
            '  printf "exp_name=%s\\n" "$EXP_NAME"\n'
            '  printf "seed=%s\\n" "$SEED"\n'
            '  printf "total_iter=%s\\n" "$TOTAL_ITER"\n'
            '  printf "eval_iter=%s\\n" "$EVAL_ITER"\n'
            '  printf "validation_seed=%s\\n" "$VALIDATION_SEED"\n'
            '  printf "validation_batch=%s\\n" "$VALIDATION_BATCH_SIZE"\n'
            '  printf "global=%s\\n" "$GLOBAL_ALIGN_WEIGHT"\n'
            '  printf "local=%s\\n" "$LOCAL_ALIGN_WEIGHT"\n'
            '  printf "cnn=%s\\n" "${CNN_CKPT:-}"\n'
            '  printf "cnn_sha=%s\\n" "${CNN_CKPT_SHA256:-}"\n'
            '  printf "phase1_dir=%s\\n" "${PHASE1_DIR:-}"\n'
            '} > "$capture"\n'
            f"{checkpoint_command}"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    @staticmethod
    def _read_capture(path):
        return dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )

    def _run_variant(
        self,
        temp_root,
        phase1_exit=0,
        phase1_checkpoint=True,
    ):
        phase1 = temp_root / "phase1.sh"
        phase2 = temp_root / "phase2.sh"
        phase1_capture = temp_root / "phase1.txt"
        phase2_capture = temp_root / "phase2.txt"
        self._fake_launcher(
            phase1,
            "CAPTURE_PHASE1",
            exit_code=phase1_exit,
            checkpoint=phase1_checkpoint,
        )
        self._fake_launcher(phase2, "CAPTURE_PHASE2")
        env = os.environ.copy()
        env.update(
            {
                "PILOT_ROOT": str(temp_root / "pilot"),
                "TAE_CHECKPOINT": "/fixed/tae.pth",
                "TAE_SHA256": TAE_SHA256,
                "PHASE1_LAUNCHER": str(phase1),
                "PHASE2_LAUNCHER": str(phase2),
                "CAPTURE_PHASE1": str(phase1_capture),
                "CAPTURE_PHASE2": str(phase2_capture),
            }
        )
        completed = subprocess.run(
            [
                "bash",
                str(EXPLORATION_ROOT / "run_variant.sh"),
                "global_only",
                "2,3",
                "0.2",
                "0",
                "0.05",
                "0",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed, phase1_capture, phase2_capture

    def test_runs_fresh_phase1_then_its_own_phase2_with_fixed_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            completed, phase1_path, phase2_path = self._run_variant(
                temp_root
            )
            phase1 = self._read_capture(phase1_path)
            phase2 = self._read_capture(phase2_path)
            status = (
                temp_root
                / "pilot"
                / "status"
                / "global_only.status"
            ).read_text(encoding="utf-8")

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        for capture in (phase1, phase2):
            self.assertEqual(capture["arg1"], "2")
            self.assertEqual(capture["arg2"], "t2m_272")
            self.assertEqual(capture["cuda"], "2,3")
            self.assertEqual(capture["seed"], "123")
            self.assertEqual(capture["total_iter"], "25000")
            self.assertEqual(capture["eval_iter"], "5000")
            self.assertEqual(capture["validation_seed"], "123")
            self.assertEqual(capture["validation_batch"], "32")
        self.assertEqual(phase1["global"], "0.2")
        self.assertEqual(phase1["local"], "0")
        self.assertEqual(phase1["cnn"], "/fixed/tae.pth")
        self.assertEqual(phase1["cnn_sha"], TAE_SHA256)
        self.assertEqual(phase2["global"], "0.05")
        self.assertEqual(phase2["local"], "0")
        self.assertEqual(
            phase2["phase1_dir"],
            phase1["out_dir"] + "/" + phase1["exp_name"],
        )
        self.assertIn("state=complete", status)

    def test_phase1_failure_or_missing_checkpoint_never_starts_phase2(self):
        cases = (
            (7, True, "Phase 1 launcher failed"),
            (0, False, "Phase 1 net_last.pth is missing"),
        )
        for exit_code, checkpoint, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temp:
                    temp_root = Path(temp)
                    completed, _, phase2_path = self._run_variant(
                        temp_root,
                        phase1_exit=exit_code,
                        phase1_checkpoint=checkpoint,
                    )
                    status = (
                        temp_root
                        / "pilot"
                        / "status"
                        / "global_only.status"
                    ).read_text(encoding="utf-8")
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(phase2_path.exists())
                self.assertIn("state=failed", status)
                self.assertIn(message, completed.stdout + completed.stderr)


class PilotScreenEntrypointTest(unittest.TestCase):
    @staticmethod
    def _write_executable(path, source):
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def _fake_tools(self, temp_root, busy=False):
        screen_capture = temp_root / "screen.txt"
        screen = temp_root / "screen"
        nvidia_smi = temp_root / "nvidia-smi"
        sha256sum = temp_root / "sha256sum"
        self._write_executable(
            screen,
            "#!/usr/bin/env bash\n"
            "if [[ ${1:-} == -ls ]]; then\n"
            '  printf "No Sockets found.\\n"\n'
            "  exit 1\n"
            "fi\n"
            'printf "%s" "$1" >> "$SCREEN_CAPTURE"\n'
            'shift\n'
            'printf " %s" "$@" >> "$SCREEN_CAPTURE"\n'
            'printf "\\n" >> "$SCREEN_CAPTURE"\n',
        )
        compute_output = (
            'printf "GPU-deadbeef, 999, 100 MiB\\n"\n'
            if busy
            else ":"
        )
        self._write_executable(
            nvidia_smi,
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *"--query-compute-apps="* ]]; then\n'
            f"  {compute_output}\n"
            "  exit 0\n"
            "fi\n"
            "for index in 0 1 2 3 4 5 6 7; do\n"
            '  printf "%s, 1, 24564, 0\\n" "$index"\n'
            "done\n",
        )
        self._write_executable(
            sha256sum,
            "#!/usr/bin/env bash\n"
            f'printf "{TAE_SHA256}  %s\\n" "${{!#}}"\n',
        )
        return screen, nvidia_smi, sha256sum, screen_capture

    def _launch(
        self,
        temp_root,
        dry_run=False,
        busy=False,
        contract_blank_line=False,
    ):
        screen, nvidia, sha256sum, capture = self._fake_tools(
            temp_root,
            busy=busy,
        )
        tae = temp_root / "tae.pth"
        tae.write_bytes(b"fixed TAE fixture")
        output_root = temp_root / "pilot"
        env = os.environ.copy()
        env.update(
            {
                "PILOT_ROOT": str(output_root),
                "TAE_CHECKPOINT": str(tae),
                "SCREEN_BIN": str(screen),
                "NVIDIA_SMI_BIN": str(nvidia),
                "SHA256SUM_BIN": str(sha256sum),
                "SCREEN_CAPTURE": str(capture),
                "PILOT_DRY_RUN": "1" if dry_run else "0",
            }
        )
        if contract_blank_line:
            conda = temp_root / "conda"
            self._write_executable(
                conda,
                "#!/usr/bin/env bash\n"
                "shift 4\n"
                'python "$@"\n'
                "code=$?\n"
                'printf "\\n"\n'
                "exit $code\n",
            )
            env["CONDA_DEFAULT_ENV"] = "base"
            env["PATH"] = f"{temp_root}:{env['PATH']}"
        completed = subprocess.run(
            ["bash", str(EXPLORATION_ROOT / "RUN_PILOT.sh")],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed, output_root, capture

    def test_dry_run_prints_four_sessions_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            completed, output_root, capture = self._launch(
                Path(temp),
                dry_run=True,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertFalse(output_root.exists())
        self.assertFalse(capture.exists())
        for variant in PILOT_VARIANTS:
            self.assertIn(variant.screen_session, completed.stdout)
            self.assertIn(variant.training_gpus, completed.stdout)

    def test_launches_exactly_four_logged_detached_screens(self):
        with tempfile.TemporaryDirectory() as temp:
            completed, output_root, capture = self._launch(Path(temp))
            screen_lines = capture.read_text(encoding="utf-8").splitlines()
            contract = json.loads(
                (output_root / "contract.json").read_text(encoding="utf-8")
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(len(screen_lines), 4)
        self.assertEqual(contract["seed"], 123)
        for variant, line in zip(PILOT_VARIANTS, screen_lines):
            self.assertIn("-dmS", line)
            self.assertIn(variant.screen_session, line)
            self.assertIn(variant.training_gpus, line)
            self.assertIn("run_variant.sh", line)

    def test_ignores_conda_run_trailing_blank_contract_line(self):
        with tempfile.TemporaryDirectory() as temp:
            completed, _, capture = self._launch(
                Path(temp),
                contract_blank_line=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            screen_lines = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(screen_lines), 4)

    def test_rejects_busy_gpu_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            completed, output_root, capture = self._launch(
                Path(temp),
                busy=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("GPU compute process", completed.stderr)
        self.assertFalse(output_root.exists())
        self.assertFalse(capture.exists())


class PilotEvaluationTest(unittest.TestCase):
    @classmethod
    def _write_checkpoints(cls, output_root):
        for variant in PILOT_VARIANTS:
            parent_path = phase_checkpoint_path(output_root, variant, 1)
            phase2_path = phase_checkpoint_path(output_root, variant, 2)
            parent_path.parent.mkdir(parents=True)
            phase2_path.parent.mkdir(parents=True)
            metadata = PilotCollectorTest._metadata(variant)
            metadata["lineage"]["parent_checkpoint_path"] = str(parent_path)
            parent_metadata = metadata["lineage"][
                "parent_checkpoint_metadata"
            ]
            torch.save(
                {"net": {}, "metadata": parent_metadata},
                parent_path,
            )
            torch.save({"net": {}, "metadata": metadata}, phase2_path)

    def test_checkpoint_preflight_validates_all_phase_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp)
            self._write_checkpoints(output_root)
            result = validate_pilot_checkpoints(output_root)
        self.assertEqual(
            [item["slug"] for item in result],
            [variant.slug for variant in PILOT_VARIANTS],
        )
        self.assertTrue(
            all(item["phase2_sha256"] for item in result)
        )

    def test_eval_runner_uses_complete_test_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            output_root = temp_root / "pilot"
            checkpoint = temp_root / "net_last.pth"
            checkpoint.write_bytes(b"checkpoint")
            capture = temp_root / "eval.txt"
            evaluator = temp_root / "fake-eval.sh"
            evaluator.write_text(
                "#!/usr/bin/env bash\n"
                'printf "%s\\n" "$@" > "$EVAL_CAPTURE"\n'
                "output=''\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  if [[ $1 == --output-dir ]]; then output=$2; fi\n"
                "  shift\n"
                "done\n"
                'mkdir -p "$output"\n'
                'printf "{}\\n" > "$output/metrics.json"\n',
                encoding="utf-8",
            )
            evaluator.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PILOT_ROOT": str(output_root),
                    "EVAL_COMMAND": str(evaluator),
                    "EVAL_CAPTURE": str(capture),
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(EXPLORATION_ROOT / "eval_variant.sh"),
                    "global_only",
                    "2",
                    str(checkpoint),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            status = (
                output_root
                / "status"
                / "global_only.evaluation.status"
            ).read_text(encoding="utf-8")
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(arguments[0], str(checkpoint))
        self.assertEqual(
            arguments[arguments.index("--split-file") + 1],
            "humanml3d_272/split/test.txt",
        )
        self.assertEqual(
            arguments[arguments.index("--batch-size") + 1],
            "32",
        )
        self.assertEqual(arguments[arguments.index("--seed") + 1], "123")
        self.assertIn("state=evaluation_complete", status)
