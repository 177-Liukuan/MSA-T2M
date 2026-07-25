import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from models.msa_vae import MSA_HumanVAE
from utils.msa_vae_eval_config import (
    build_and_load_msa_vae,
    checkpoint_manifest,
    load_checkpoint_payload,
    resolve_msa_vae_config,
)


class MSAEvalConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "net_best.pth"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def synthetic_state_dict(
        hidden_size=1024,
        latent_dim=16,
        trans_d_model=512,
        enc_layers=5,
        dec_layers=4,
        ff_size=1536,
        clip_dim=768,
    ):
        state = {
            "msa_vae.cnn_encoder.model.0.conv.weight": torch.zeros(
                hidden_size,
                272,
                3,
            ),
            "msa_vae.trans_encoder.input_proj.weight": torch.zeros(
                trans_d_model,
                latent_dim,
            ),
            "msa_vae.local_proj.weight": torch.zeros(clip_dim, latent_dim),
        }
        for index in range(enc_layers):
            state[
                "msa_vae.trans_encoder.transformer_encoder.layers."
                f"{index}.linear1.weight"
            ] = torch.zeros(ff_size, trans_d_model)
        for index in range(dec_layers):
            state[
                "msa_vae.trans_decoder.transformer_decoder.layers."
                f"{index}.linear1.weight"
            ] = torch.zeros(ff_size, trans_d_model)
        return state

    def _write_run_log(self, values):
        (self.root / "run.log").write_text(
            "2026-01-01 INFO launcher started\n"
            "2026-01-01 INFO {}\n"
            "2026-01-01 INFO ignored trailing text\n".format(
                json.dumps(values, indent=2)
            ),
            encoding="utf-8",
        )

    def test_resolution_precedence_is_cli_metadata_log_state_default(self):
        self._write_run_log({"trans_nhead": 2, "hidden_size": 512})
        payload = {
            "net": self.synthetic_state_dict(),
            "metadata": {
                "training_args": {"trans_nhead": 4, "depth": 2},
            },
        }

        resolved = resolve_msa_vae_config(
            self.checkpoint,
            payload,
            {"trans_nhead": 8},
        )

        self.assertEqual(resolved.values["trans_nhead"], 8)
        self.assertEqual(resolved.sources["trans_nhead"], "cli")
        self.assertEqual(resolved.values["depth"], 2)
        self.assertEqual(resolved.sources["depth"], "metadata")
        self.assertEqual(resolved.values["hidden_size"], 512)
        self.assertEqual(resolved.sources["hidden_size"], "run.log")
        self.assertEqual(resolved.values["trans_enc_layers"], 5)
        self.assertEqual(resolved.values["trans_dec_layers"], 4)
        self.assertEqual(resolved.values["trans_ff_size"], 1536)
        self.assertEqual(resolved.sources["trans_ff_size"], "state_dict")
        self.assertEqual(resolved.values["stride_t"], 2)
        self.assertEqual(resolved.sources["stride_t"], "default")

    def test_legacy_mainline_defaults_cover_non_inferable_fields(self):
        payload = {"net": self.synthetic_state_dict(trans_d_model=512)}

        resolved = resolve_msa_vae_config(self.checkpoint, payload, {})

        self.assertEqual(resolved.values["down_t"], 2)
        self.assertEqual(resolved.values["stride_t"], 2)
        self.assertEqual(resolved.values["trans_nhead"], 8)
        self.assertEqual(resolved.values["dilation_growth_rate"], 3)
        self.assertEqual(resolved.values["trans_d_model"], 512)
        self.assertEqual(resolved.values["clip_dim"], 768)

    def test_run_log_scanner_skips_malformed_prefix_and_reads_first_json(self):
        (self.root / "run.log").write_text(
            "INFO not-json {broken}\n"
            "INFO {\n"
            '  "depth": 1,\n'
            '  "text_embed_dim": 512\n'
            "}\n"
            'INFO {"depth": 99}\n',
            encoding="utf-8",
        )

        resolved = resolve_msa_vae_config(
            self.checkpoint,
            {"net": self.synthetic_state_dict(clip_dim=512)},
            {},
        )

        self.assertEqual(resolved.values["depth"], 1)
        self.assertEqual(resolved.values["clip_dim"], 512)
        self.assertEqual(resolved.values["trans_d_model"], 512)

    def test_rejects_invalid_metadata_dimensions(self):
        payload = {
            "net": self.synthetic_state_dict(),
            "metadata": {"training_args": {"trans_nhead": 7}},
        }

        with self.assertRaisesRegex(ValueError, "trans_d_model.*trans_nhead"):
            resolve_msa_vae_config(self.checkpoint, payload, {})

    def test_load_checkpoint_accepts_wrapped_and_raw_state_dicts(self):
        state = self.synthetic_state_dict()
        torch.save({"net": state}, self.checkpoint)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapped = load_checkpoint_payload(self.checkpoint)
        torch.save(state, self.checkpoint)
        raw = load_checkpoint_payload(self.checkpoint)

        self.assertIn("net", wrapped)
        self.assertEqual(set(raw["net"]), set(state))
        self.assertFalse(
            [item for item in caught if issubclass(item.category, FutureWarning)]
        )

    def test_checkpoint_manifest_records_stable_file_identity(self):
        self.checkpoint.write_bytes(b"checkpoint-bytes")

        manifest = checkpoint_manifest(self.checkpoint)

        self.assertEqual(manifest["path"], str(self.checkpoint.resolve()))
        self.assertEqual(manifest["size"], len(b"checkpoint-bytes"))
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(
            manifest,
            checkpoint_manifest(self.checkpoint),
        )

    def test_build_strictly_loads_a_real_small_msa_vae(self):
        config = {
            "hidden_size": 8,
            "down_t": 1,
            "stride_t": 2,
            "depth": 1,
            "dilation_growth_rate": 1,
            "latent_dim": 2,
            "trans_d_model": 8,
            "trans_nhead": 2,
            "trans_enc_layers": 1,
            "trans_dec_layers": 1,
            "trans_ff_size": 16,
            "trans_dropout": 0.0,
            "clip_dim": 4,
            "disable_decoupling": False,
        }
        source_model = MSA_HumanVAE(**config)
        torch.save(
            {"net": source_model.state_dict(), "metadata": {"training_args": config}},
            self.checkpoint,
        )

        loaded, resolved, manifest = build_and_load_msa_vae(
            self.checkpoint,
            overrides={},
            device=torch.device("cpu"),
        )

        self.assertEqual(resolved.values, config)
        self.assertEqual(loaded.state_dict().keys(), source_model.state_dict().keys())
        self.assertEqual(manifest["path"], str(self.checkpoint.resolve()))
        self.assertFalse(loaded.training)

    def test_build_rejects_partial_checkpoint_instead_of_partial_loading(self):
        config = {
            "hidden_size": 8,
            "down_t": 1,
            "stride_t": 2,
            "depth": 1,
            "dilation_growth_rate": 1,
            "latent_dim": 2,
            "trans_d_model": 8,
            "trans_nhead": 2,
            "trans_enc_layers": 1,
            "trans_dec_layers": 1,
            "trans_ff_size": 16,
            "trans_dropout": 0.0,
            "clip_dim": 4,
            "disable_decoupling": False,
        }
        state = MSA_HumanVAE(**config).state_dict()
        del state["msa_vae.trans_encoder.input_proj.weight"]
        torch.save(
            {"net": state, "metadata": {"training_args": config}},
            self.checkpoint,
        )

        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            build_and_load_msa_vae(
                self.checkpoint,
                overrides={},
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
