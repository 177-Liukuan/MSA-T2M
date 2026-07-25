"""BABEL sparse-global MSA-VAE datasets.

The bridge HumanML3D entries receive both global and local supervision.  The
BABEL stream entries receive only exact-frame local supervision.  Keeping the
two sources explicit prevents a missing target from silently changing a loss
mask during training.
"""

import codecs as cs
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from humanml3d_272.babel_stream_t5_cache import CACHE_VERSION, validate_cache_manifest
from humanml3d_272.dataset_msa_vae import _pool_to_latent, collate_fn


MOTION_DIM = 272
MAX_REPORTED_FAILURES = 12


def _canonical(path):
    return str(Path(path).expanduser().resolve())


def _load_array(path, label):
    try:
        return np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("cannot load {}: {}".format(label, error)) from error


def _require_2d(array, expected_dim, label):
    if array.ndim != 2:
        raise ValueError("{} must be two-dimensional".format(label))
    if array.shape[1] != expected_dim:
        raise ValueError(
            "{} embedding dimension {} does not match {}".format(
                label, array.shape[1], expected_dim
            )
        )


def _full_motion_captions(text_path):
    captions = []
    with cs.open(str(text_path), "r") as source:
        for line_index, line in enumerate(source):
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0].strip()
            if not caption:
                continue
            try:
                from_tag = 0.0 if parts[2] == "nan" else float(parts[2])
                to_tag = 0.0 if parts[3] == "nan" else float(parts[3])
            except ValueError as error:
                raise ValueError("invalid caption frame tag: {}".format(error)) from error
            if from_tag == 0.0 and to_tag == 0.0:
                captions.append((caption, line_index))
    return captions


def _read_manifest(manifest_path, babel_motion_dir, babel_cache_dir, text_embed_dim):
    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot read BABEL cache manifest: {}".format(error)) from error

    expected_motion_root = _canonical(babel_motion_dir)
    expected_cache_root = _canonical(babel_cache_dir)
    errors = []
    if manifest_path.parent != Path(expected_cache_root):
        errors.append("cache directory does not match manifest parent")
    if manifest.get("version") != CACHE_VERSION:
        errors.append("cache manifest version mismatch")
    if manifest.get("motion_dir") != expected_motion_root:
        errors.append("cache manifest motion_dir mismatch")
    if manifest.get("embedding_dim") != int(text_embed_dim):
        errors.append("cache manifest embedding dimension mismatch")
    if not isinstance(manifest.get("text_dir"), str) or not Path(manifest["text_dir"]).is_dir():
        errors.append("cache manifest text_dir is missing")
    records = manifest.get("records")
    if not isinstance(records, dict):
        errors.append("cache manifest records must be a mapping")
        records = {}
    if manifest.get("valid_samples") != len(records) or manifest.get("rejected_samples") != 0:
        errors.append("cache manifest sample counts are invalid")
    if errors:
        raise RuntimeError("BABEL cache manifest validation failed: {}".format("; ".join(errors)))
    return manifest_path, manifest


def _validate_cache_contents(manifest_path, manifest):
    """Run Task 1's full source/hash validation after useful local diagnostics."""
    expected = {
        "split": manifest["split"],
        "model_signature": manifest["model_signature"],
        "embedding_dim": manifest["embedding_dim"],
        "motion_dir": manifest["motion_dir"],
        "text_dir": manifest["text_dir"],
    }
    try:
        validate_cache_manifest(manifest_path, expected)
    except (OSError, ValueError) as error:
        raise RuntimeError("BABEL cache manifest content validation failed: {}".format(error)) from error


def _load_normalization(mean_path, std_path):
    mean = _load_array(Path(mean_path), "MSA mean")
    std = _load_array(Path(std_path), "MSA std")
    if mean.shape != (MOTION_DIM,) or std.shape != (MOTION_DIM,):
        raise ValueError("MSA mean/std must each have shape ({},)".format(MOTION_DIM))
    if np.any(std == 0):
        raise ValueError("MSA std must not contain zero")
    return mean.astype(np.float32), std.astype(np.float32)


def _format_failures(failures, bridge_count, babel_count, has_entries):
    messages = list(failures[:MAX_REPORTED_FAILURES])
    if not has_entries:
        messages.append("no valid samples")
    return (
        "BABEL sparse-global dataset validation failed "
        "(bridge={}, babel={}, failures={}): {}".format(
            bridge_count, babel_count, len(failures), "; ".join(messages)
        )
    )


class _BabelSourceMixin:
    def _discover_babel_entries(
        self, babel_motion_dir, babel_cache_dir, babel_cache_manifest, text_embed_dim
    ):
        manifest_path, manifest = _read_manifest(
            babel_cache_manifest, babel_motion_dir, babel_cache_dir, text_embed_dim
        )
        motion_root = Path(babel_motion_dir).expanduser().resolve()
        cache_root = Path(babel_cache_dir).expanduser().resolve()
        if not motion_root.is_dir():
            raise RuntimeError("BABEL motion directory not found: {}".format(motion_root))
        if not cache_root.is_dir():
            raise RuntimeError("BABEL cache directory not found: {}".format(cache_root))

        record_ids = set(manifest["records"])
        motion_ids = {path.stem for path in motion_root.glob("*.npy")}
        cache_ids = {
            path.stem
            for path in cache_root.glob("*.npy")
            if path.name != "manifest.json"
        }
        failures = []
        for name in sorted(record_ids - motion_ids):
            failures.append("babel:{} missing motion".format(name))
        for name in sorted(motion_ids - record_ids):
            failures.append("babel:{} absent from cache manifest".format(name))
        for name in sorted(cache_ids - record_ids):
            failures.append("babel:{} cache absent from cache manifest".format(name))
        for name in sorted(record_ids - cache_ids):
            failures.append("babel:{} missing local cache".format(name))

        entries = []
        for name in sorted(record_ids & motion_ids):
            motion_path = motion_root / "{}.npy".format(name)
            cache_path = cache_root / "{}.npy".format(name)
            try:
                motion = _load_array(motion_path, "babel:{} motion".format(name))
                if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
                    raise ValueError("motion must have shape (T, {})".format(MOTION_DIM))
                if motion.shape[0] < self.window_size:
                    raise ValueError("motion shorter than window_size")
                if not cache_path.is_file():
                    raise ValueError("missing local cache")
                local = _load_array(cache_path, "babel:{} local cache".format(name))
                _require_2d(local, text_embed_dim, "babel:{} local cache".format(name))
                if local.shape[0] != motion.shape[0]:
                    raise ValueError("cache/motion length mismatch")
                entries.append(
                    {
                        "name": "babel:{}".format(name),
                        "source": "babel",
                        "motion": motion.astype(np.float32),
                        "local": local.astype(np.float32),
                    }
                )
            except (OSError, ValueError) as error:
                failures.append("babel:{} {}".format(name, error))

        if not failures:
            try:
                _validate_cache_contents(manifest_path, manifest)
            except RuntimeError as error:
                failures.append("babel cache manifest content validation {}".format(error))
        return entries, failures


class BabelSparseGlobalMSAVAEDataset(_BabelSourceMixin, data.Dataset):
    """Uniformly concatenated HumanML bridge and BABEL local-only samples."""

    def __init__(
        self,
        bridge_split_file,
        bridge_motion_dir,
        bridge_text_dir,
        bridge_global_embed_dir,
        bridge_local_embed_dir,
        babel_motion_dir,
        babel_cache_dir,
        babel_cache_manifest,
        mean_path,
        std_path,
        window_size,
        unit_length,
        text_embed_dim,
    ):
        if window_size <= 0 or unit_length <= 0 or window_size % unit_length:
            raise ValueError("window_size must be positive and divisible by unit_length")
        self.window_size = int(window_size)
        self.unit_length = int(unit_length)
        self.text_embed_dim = int(text_embed_dim)
        self.mean, self.std = _load_normalization(mean_path, std_path)

        bridge_entries, bridge_failures = self._discover_bridge_entries(
            bridge_split_file,
            bridge_motion_dir,
            bridge_text_dir,
            bridge_global_embed_dir,
            bridge_local_embed_dir,
        )
        babel_entries, babel_failures = self._discover_babel_entries(
            babel_motion_dir, babel_cache_dir, babel_cache_manifest, self.text_embed_dim
        )
        failures = bridge_failures + babel_failures
        self.data = bridge_entries + babel_entries
        self.bridge_count = len(bridge_entries)
        self.babel_count = len(babel_entries)
        if failures or not self.data:
            raise RuntimeError(
                _format_failures(failures, self.bridge_count, self.babel_count, bool(self.data))
            )
        self._index_by_name = {entry["name"]: index for index, entry in enumerate(self.data)}
        print(
            "BABEL sparse-global MSA-VAE: {} bridge + {} BABEL local-only samples".format(
                self.bridge_count, self.babel_count
            )
        )

    def _discover_bridge_entries(
        self,
        bridge_split_file,
        bridge_motion_dir,
        bridge_text_dir,
        bridge_global_embed_dir,
        bridge_local_embed_dir,
    ):
        split_path = Path(bridge_split_file).expanduser().resolve()
        try:
            bridge_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        except OSError as error:
            raise RuntimeError("cannot read bridge split file: {}".format(error)) from error

        motion_root = Path(bridge_motion_dir).expanduser().resolve()
        text_root = Path(bridge_text_dir).expanduser().resolve()
        global_root = Path(bridge_global_embed_dir).expanduser().resolve()
        local_root = Path(bridge_local_embed_dir).expanduser().resolve()
        entries = []
        failures = []
        for name in bridge_ids:
            try:
                motion = _load_array(motion_root / "{}.npy".format(name), "hml:{} motion".format(name))
                if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
                    raise ValueError("motion must have shape (T, {})".format(MOTION_DIM))
                if motion.shape[0] < self.window_size:
                    raise ValueError("motion shorter than window_size")

                text_path = text_root / "{}.txt".format(name)
                if not text_path.is_file():
                    raise ValueError("missing text")
                captions = _full_motion_captions(text_path)
                if not captions:
                    raise ValueError("missing full-motion caption")

                global_path = global_root / "{}.npy".format(name)
                if not global_path.is_file():
                    raise ValueError("missing global target")
                global_values = _load_array(global_path, "hml:{} global target".format(name))
                _require_2d(global_values, self.text_embed_dim, "hml:{} global target".format(name))
                if max(index for _, index in captions) >= global_values.shape[0]:
                    raise ValueError("global target has no matching caption row")

                local_path = local_root / "{}.npy".format(name)
                if not local_path.is_file():
                    raise ValueError("missing local target")
                local_values = _load_array(local_path, "hml:{} local target".format(name))
                _require_2d(local_values, self.text_embed_dim, "hml:{} local target".format(name))
                if local_values.shape[0] <= 0:
                    raise ValueError("local target has no frames")

                entries.append(
                    {
                        "name": "hml:{}".format(name),
                        "source": "hml",
                        "motion": motion.astype(np.float32),
                        "captions": captions,
                        "global": global_values.astype(np.float32),
                        "local": local_values.astype(np.float32),
                    }
                )
            except (OSError, ValueError) as error:
                failures.append("hml:{} {}".format(name, error))
        return entries, failures

    def __len__(self):
        return len(self.data)

    def index_for(self, source_name):
        return self._index_by_name[source_name]

    def inv_transform(self, values):
        return values * self.std + self.mean

    def __getitem__(self, item):
        entry = self.data[item]
        motion = entry["motion"]
        start = random.randint(0, motion.shape[0] - self.window_size)
        motion_window = motion[start : start + self.window_size]
        local_window = self._local_window(entry, start)
        local_pooled = local_window.mean(axis=0).astype(np.float32)
        local_latent = _pool_to_latent(
            local_window, self.window_size // self.unit_length
        ).astype(np.float32)

        if entry["source"] == "hml":
            caption_index = random.randint(0, len(entry["captions"]) - 1)
            caption, global_index = entry["captions"][caption_index]
            global_embed = entry["global"][global_index].astype(np.float32)
            has_global = True
        else:
            caption = "None"
            global_embed = np.zeros((self.text_embed_dim,), dtype=np.float32)
            has_global = False

        return (
            ((motion_window - self.mean) / self.std).astype(np.float32),
            caption,
            global_embed,
            has_global,
            local_latent,
            True,
            motion.shape[0],
            local_pooled,
        )

    def _local_window(self, entry, start):
        if entry["source"] == "babel":
            return entry["local"][start : start + self.window_size]
        local_20fps = entry["local"]
        indices = np.round(
            np.linspace(0, local_20fps.shape[0] - 1, entry["motion"].shape[0])
        ).astype(int)
        return local_20fps[indices][start : start + self.window_size]


class BabelSparseGlobalMSAVAEValidationDataset(_BabelSourceMixin, data.Dataset):
    """Deterministic BABEL-only reconstruction/local-alignment windows."""

    def __init__(
        self,
        babel_motion_dir,
        babel_cache_dir,
        babel_cache_manifest,
        mean_path,
        std_path,
        window_size,
        unit_length,
        text_embed_dim,
    ):
        if window_size <= 0 or unit_length <= 0 or window_size % unit_length:
            raise ValueError("window_size must be positive and divisible by unit_length")
        self.window_size = int(window_size)
        self.unit_length = int(unit_length)
        self.text_embed_dim = int(text_embed_dim)
        self.mean, self.std = _load_normalization(mean_path, std_path)
        source_entries, failures = self._discover_babel_entries(
            babel_motion_dir, babel_cache_dir, babel_cache_manifest, self.text_embed_dim
        )
        if failures or not source_entries:
            raise RuntimeError(_format_failures(failures, 0, len(source_entries), bool(source_entries)))

        self.data = []
        self._starts_by_name = {}
        for source_entry in source_entries:
            motion_frames = source_entry["motion"].shape[0]
            starts = list(range(0, motion_frames - self.window_size + 1, self.window_size))
            tail_start = motion_frames - self.window_size
            if starts[-1] != tail_start:
                starts.append(tail_start)
            self._starts_by_name[source_entry["name"]] = tuple(starts)
            for start in starts:
                self.data.append((source_entry, start))
        self._index_by_name = {
            name: index for index, name in enumerate(self._starts_by_name)
        }

    def __len__(self):
        return len(self.data)

    def index_for(self, source_name):
        return self._index_by_name[source_name]

    def window_starts_for(self, source_name):
        return self._starts_by_name[source_name]

    def inv_transform(self, values):
        return values * self.std + self.mean

    def __getitem__(self, item):
        entry, start = self.data[item]
        motion_window = entry["motion"][start : start + self.window_size]
        local_window = entry["local"][start : start + self.window_size]
        local_pooled = local_window.mean(axis=0).astype(np.float32)
        local_latent = _pool_to_latent(
            local_window, self.window_size // self.unit_length
        ).astype(np.float32)
        return (
            ((motion_window - self.mean) / self.std).astype(np.float32),
            "None",
            np.zeros((self.text_embed_dim,), dtype=np.float32),
            False,
            local_latent,
            True,
            entry["motion"].shape[0],
            local_pooled,
        )


def DATALoader(batch_size, num_workers=8, **dataset_kwargs):
    dataset = BabelSparseGlobalMSAVAEDataset(**dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


def ValidationDATALoader(batch_size, num_workers=8, **dataset_kwargs):
    dataset = BabelSparseGlobalMSAVAEValidationDataset(**dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
