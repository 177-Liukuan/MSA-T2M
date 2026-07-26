"""Resolve and strictly load new and legacy MSA-VAE checkpoints."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from models.msa_vae import MSA_HumanVAE


CONFIG_FIELDS = (
    "hidden_size",
    "down_t",
    "stride_t",
    "depth",
    "dilation_growth_rate",
    "latent_dim",
    "trans_d_model",
    "trans_nhead",
    "trans_enc_layers",
    "trans_dec_layers",
    "trans_ff_size",
    "trans_dropout",
    "clip_dim",
    "disable_decoupling",
)

INTEGER_FIELDS = {
    "hidden_size",
    "down_t",
    "stride_t",
    "depth",
    "dilation_growth_rate",
    "latent_dim",
    "trans_d_model",
    "trans_nhead",
    "trans_enc_layers",
    "trans_dec_layers",
    "trans_ff_size",
    "clip_dim",
}

MAINLINE_DEFAULTS = {
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
}


@dataclass(frozen=True)
class ResolvedMSAVAEConfig:
    values: Dict[str, Any]
    sources: Dict[str, str]


def load_checkpoint_payload(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("MSA-VAE checkpoint does not exist: {}".format(resolved))
    loaded = torch.load(
        str(resolved),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("MSA-VAE checkpoint must contain a state dictionary")
    if "net" in loaded:
        if not isinstance(loaded["net"], Mapping):
            raise ValueError("MSA-VAE checkpoint 'net' must be a state dictionary")
        return dict(loaded)
    if loaded and all(isinstance(value, torch.Tensor) for value in loaded.values()):
        return {"net": dict(loaded)}
    raise ValueError("MSA-VAE checkpoint has neither 'net' nor a raw state dictionary")


def _first_json_object(path):
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _layer_count(state, pattern, label):
    indices = {
        int(match.group(1))
        for key in state
        for match in [pattern.match(key)]
        if match is not None
    }
    if not indices:
        return None
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError("{} state-dict layer indices are not contiguous".format(label))
    return len(indices)


def _infer_from_state_dict(state):
    inferred = {}
    cnn_weight = state.get("msa_vae.cnn_encoder.model.0.conv.weight")
    if isinstance(cnn_weight, torch.Tensor) and cnn_weight.ndim == 3:
        inferred["hidden_size"] = int(cnn_weight.shape[0])

    input_projection = state.get("msa_vae.trans_encoder.input_proj.weight")
    if isinstance(input_projection, torch.Tensor) and input_projection.ndim == 2:
        inferred["trans_d_model"] = int(input_projection.shape[0])
        inferred["latent_dim"] = int(input_projection.shape[1])

    encoder_pattern = re.compile(
        r"^msa_vae\.trans_encoder\.transformer_encoder\.layers\.(\d+)\."
    )
    decoder_pattern = re.compile(
        r"^msa_vae\.trans_decoder\.transformer_decoder\.layers\.(\d+)\."
    )
    encoder_layers = _layer_count(state, encoder_pattern, "encoder")
    decoder_layers = _layer_count(state, decoder_pattern, "decoder")
    if encoder_layers is not None:
        inferred["trans_enc_layers"] = encoder_layers
    if decoder_layers is not None:
        inferred["trans_dec_layers"] = decoder_layers

    feed_forward_sizes = set()
    for key, value in state.items():
        if (
            encoder_pattern.match(key) or decoder_pattern.match(key)
        ) and key.endswith(".linear1.weight") and value.ndim == 2:
            feed_forward_sizes.add(int(value.shape[0]))
    if len(feed_forward_sizes) > 1:
        raise ValueError("state-dict Transformer feed-forward sizes disagree")
    if feed_forward_sizes:
        inferred["trans_ff_size"] = feed_forward_sizes.pop()

    local_projection = state.get("msa_vae.local_proj.weight")
    global_projection = state.get("msa_vae.global_proj.weight")
    if isinstance(local_projection, torch.Tensor) and local_projection.ndim == 2:
        inferred["clip_dim"] = int(local_projection.shape[0])
    elif isinstance(global_projection, torch.Tensor) and global_projection.ndim == 2:
        inferred["clip_dim"] = int(global_projection.shape[0])
    elif "trans_d_model" in inferred:
        inferred["clip_dim"] = inferred["trans_d_model"]
    return inferred


def _known_values(raw):
    if not isinstance(raw, Mapping):
        return {}
    values = {field: raw[field] for field in CONFIG_FIELDS if field in raw}
    text_embed_dim = raw.get("text_embed_dim")
    if text_embed_dim is not None:
        try:
            text_embed_dim = int(text_embed_dim)
        except (TypeError, ValueError):
            raise ValueError(
                "invalid MSA-VAE configuration field text_embed_dim={!r}".format(
                    raw.get("text_embed_dim")
                )
            )
        if text_embed_dim > 0:
            values.setdefault("clip_dim", text_embed_dim)
            values.setdefault("trans_d_model", text_embed_dim)
    return values


def _metadata_values(payload):
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    values = _known_values(metadata.get("training_args", {}))
    values.update(_known_values(metadata))
    return values


def _coerce_config(values):
    coerced = {}
    for field in CONFIG_FIELDS:
        value = values[field]
        try:
            if field in INTEGER_FIELDS:
                value = int(value)
            elif field == "trans_dropout":
                value = float(value)
            elif field == "disable_decoupling":
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized not in ("true", "false"):
                        raise ValueError
                    value = normalized == "true"
                elif not isinstance(value, bool):
                    raise ValueError
        except (TypeError, ValueError):
            raise ValueError("invalid MSA-VAE configuration field {}={!r}".format(
                field,
                values[field],
            ))
        coerced[field] = value

    for field in INTEGER_FIELDS:
        if coerced[field] <= 0:
            raise ValueError("{} must be positive".format(field))
    if not 0.0 <= coerced["trans_dropout"] < 1.0:
        raise ValueError("trans_dropout must be in [0, 1)")
    if coerced["trans_d_model"] % coerced["trans_nhead"] != 0:
        raise ValueError("trans_d_model must be divisible by trans_nhead")
    return coerced


def resolve_msa_vae_config(checkpoint_path, payload, overrides):
    if not isinstance(payload, Mapping) or not isinstance(payload.get("net"), Mapping):
        raise ValueError("checkpoint payload must contain a 'net' state dictionary")
    values = dict(MAINLINE_DEFAULTS)
    sources = {field: "default" for field in CONFIG_FIELDS}

    tiers = (
        ("state_dict", _infer_from_state_dict(payload["net"])),
        (
            "run.log",
            _known_values(
                _first_json_object(
                    Path(checkpoint_path).expanduser().resolve().parent / "run.log"
                )
            ),
        ),
        ("metadata", _metadata_values(payload)),
        (
            "cli",
            {
                key: value
                for key, value in _known_values(overrides).items()
                if value is not None
            },
        ),
    )
    for source, tier_values in tiers:
        for field, value in tier_values.items():
            values[field] = value
            sources[field] = source
    return ResolvedMSAVAEConfig(
        values=_coerce_config(values),
        sources=sources,
    )


def checkpoint_manifest(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("MSA-VAE checkpoint does not exist: {}".format(resolved))
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def build_and_load_msa_vae(path, overrides, device):
    payload = load_checkpoint_payload(path)
    resolved = resolve_msa_vae_config(path, payload, overrides)
    model = MSA_HumanVAE(
        hidden_size=resolved.values["hidden_size"],
        down_t=resolved.values["down_t"],
        stride_t=resolved.values["stride_t"],
        depth=resolved.values["depth"],
        dilation_growth_rate=resolved.values["dilation_growth_rate"],
        activation="relu",
        latent_dim=resolved.values["latent_dim"],
        clip_range=[-30, 20],
        trans_d_model=resolved.values["trans_d_model"],
        trans_nhead=resolved.values["trans_nhead"],
        trans_enc_layers=resolved.values["trans_enc_layers"],
        trans_dec_layers=resolved.values["trans_dec_layers"],
        trans_ff_size=resolved.values["trans_ff_size"],
        trans_dropout=resolved.values["trans_dropout"],
        clip_dim=resolved.values["clip_dim"],
        disable_decoupling=resolved.values["disable_decoupling"],
    )
    model.load_state_dict(payload["net"], strict=True)
    model.to(device)
    model.eval()
    identity = checkpoint_manifest(path)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        identity["metadata"] = dict(metadata)
    return model, resolved, identity
