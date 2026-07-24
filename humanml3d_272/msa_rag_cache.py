"""Packed, validated cache for the official MSA-T2M global-RAG dataset."""

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch


CACHE_SCHEMA_VERSION = 1
ARRAY_FILES = (
    "motion_values.npy",
    "motion_offsets.npy",
    "motion_lengths.npy",
    "text_values.npy",
    "text_offsets.npy",
    "text_counts.npy",
    "hcls_values.npy",
    "retrieval_indices.npy",
    "retrieval_scores.npy",
)


class CacheValidationError(ValueError):
    """Raised when a packed RAG cache does not match its source features."""


def _split_path(dataset_name):
    if dataset_name != "t2m_272":
        raise ValueError("Unsupported dataset: {}".format(dataset_name))
    return Path("humanml3d_272") / "split" / "train.txt"


def _canonical(path):
    return str(Path(path).expanduser().resolve())


def _file_metadata(path):
    stat = path.stat()
    return {
        "name": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _discover_sources(dataset_name, motion_latent_dir, text_latent_dir, hcls_dir):
    split_path = _split_path(dataset_name)
    if not split_path.is_file():
        raise FileNotFoundError("Training split not found: {}".format(split_path))

    directories = {
        "motion": Path(motion_latent_dir).expanduser().resolve(),
        "text": Path(text_latent_dir).expanduser().resolve(),
        "hcls": Path(hcls_dir).expanduser().resolve(),
    }
    for label, directory in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError("{} directory not found: {}".format(label, directory))

    sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    sources = []
    for sample_id in sample_ids:
        paths = {
            label: directory / "{}.npy".format(sample_id)
            for label, directory in directories.items()
        }
        if all(path.is_file() for path in paths.values()):
            sources.append({"id": sample_id, "paths": paths})

    if not sources:
        raise RuntimeError("No valid samples found in the motion/text/h_cls intersection.")
    return directories, sources


def _source_records(sources):
    records = []
    for source in sources:
        records.append(
            {
                "id": source["id"],
                "motion": _file_metadata(source["paths"]["motion"]),
                "text": _file_metadata(source["paths"]["text"]),
                "hcls": _file_metadata(source["paths"]["hcls"]),
            }
        )
    return records


def _load_sources(sources, text_embed_dim):
    motions = []
    texts = []
    hcls_values = []
    for source in sources:
        sample_id = source["id"]
        motion = np.ascontiguousarray(
            np.load(source["paths"]["motion"]).astype(np.float32)
        )
        text = np.ascontiguousarray(
            np.load(source["paths"]["text"]).astype(np.float32)
        )
        hcls = np.load(source["paths"]["hcls"]).astype(np.float32)

        if motion.ndim != 2:
            raise ValueError(
                "motion latent for {} must be 2-D, got {}".format(sample_id, motion.shape)
            )
        if text.ndim == 1:
            text = text.reshape(1, -1)
        if text.ndim != 2 or text.shape[1] != text_embed_dim:
            raise ValueError(
                "text embedding for {} must have final dim {}, got {}".format(
                    sample_id, text_embed_dim, text.shape
                )
            )
        if hcls.ndim == 2:
            hcls = hcls.mean(axis=0)
        else:
            hcls = hcls.reshape(-1)
        hcls = np.ascontiguousarray(hcls.astype(np.float32))
        if hcls.shape[0] != text_embed_dim:
            raise ValueError(
                "h_cls for {} must have dim {}, got {}".format(
                    sample_id, text_embed_dim, hcls.shape
                )
            )

        motions.append(motion)
        texts.append(text)
        hcls_values.append(hcls)
    return motions, texts, np.ascontiguousarray(np.stack(hcls_values))


def _pack_sequences(sequences):
    lengths = np.asarray([sequence.shape[0] for sequence in sequences], dtype=np.int64)
    offsets = np.zeros(len(sequences), dtype=np.int64)
    if len(sequences) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    values = np.ascontiguousarray(np.concatenate(sequences, axis=0).astype(np.float32))
    return values, offsets, lengths


def _compute_retrieval(text_values, text_counts, hcls_values, topk, exclude_self, batch_size):
    if batch_size <= 0:
        raise ValueError("retrieval_batch_size must be positive")
    if topk <= 0 or topk > hcls_values.shape[0]:
        raise ValueError(
            "topk must be in [1, {}], got {}".format(hcls_values.shape[0], topk)
        )

    library = torch.from_numpy(hcls_values).float()
    library = library / (library.norm(dim=-1, keepdim=True) + 1e-6)
    caption_source_indices = np.repeat(
        np.arange(len(text_counts), dtype=np.int64), text_counts
    )
    all_indices = np.empty((len(text_values), topk), dtype=np.int64)
    all_scores = np.empty((len(text_values), topk), dtype=np.float32)

    for start in range(0, len(text_values), batch_size):
        end = min(start + batch_size, len(text_values))
        query = torch.from_numpy(text_values[start:end]).float()
        query = query / (query.norm(dim=-1, keepdim=True) + 1e-6)
        similarity = torch.matmul(query, library.t())
        if exclude_self:
            rows = torch.arange(end - start)
            columns = torch.from_numpy(caption_source_indices[start:end])
            similarity[rows, columns] = -1e6
        scores, indices = torch.topk(similarity, k=topk, dim=1)
        all_indices[start:end] = indices.numpy()
        all_scores[start:end] = scores.numpy()
    return all_indices, all_scores


def _array_manifest(cache_dir):
    arrays = {}
    for filename in ARRAY_FILES:
        array = np.load(cache_dir / filename, mmap_mode="r")
        arrays[filename] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    return arrays


def _write_cache(
    target_dir,
    dataset_name,
    directories,
    sources,
    topk,
    text_embed_dim,
    exclude_self,
    retrieval_batch_size,
):
    motions, texts, hcls_values = _load_sources(sources, text_embed_dim)
    motion_values, motion_offsets, motion_lengths = _pack_sequences(motions)
    text_values, text_offsets, text_counts = _pack_sequences(texts)
    retrieval_indices, retrieval_scores = _compute_retrieval(
        text_values,
        text_counts,
        hcls_values,
        topk,
        exclude_self,
        retrieval_batch_size,
    )

    arrays = {
        "motion_values.npy": motion_values,
        "motion_offsets.npy": motion_offsets,
        "motion_lengths.npy": motion_lengths,
        "text_values.npy": text_values,
        "text_offsets.npy": text_offsets,
        "text_counts.npy": text_counts,
        "hcls_values.npy": hcls_values,
        "retrieval_indices.npy": retrieval_indices,
        "retrieval_scores.npy": retrieval_scores,
    }
    for filename, array in arrays.items():
        np.save(target_dir / filename, array, allow_pickle=False)

    sample_ids = [source["id"] for source in sources]
    (target_dir / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "source_directories": {
            label: str(path) for label, path in directories.items()
        },
        "sources": _source_records(sources),
        "sample_count": len(sample_ids),
        "caption_count": int(text_values.shape[0]),
        "topk": int(topk),
        "text_embed_dim": int(text_embed_dim),
        "exclude_self": bool(exclude_self),
        "arrays": _array_manifest(target_dir),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _load_manifest(cache_dir):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CacheValidationError("Missing manifest.json in cache: {}".format(cache_dir))
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise CacheValidationError(
            "Cannot read manifest.json in {}: {}".format(cache_dir, error)
        )


def _validate_arrays(cache_dir, manifest):
    expected_arrays = manifest.get("arrays", {})
    for filename in ARRAY_FILES:
        path = cache_dir / filename
        if not path.is_file():
            raise CacheValidationError("Missing cache array: {}".format(filename))
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise CacheValidationError("Cannot load {}: {}".format(filename, error))
        expected = expected_arrays.get(filename)
        if expected is None:
            raise CacheValidationError("manifest.json has no metadata for {}".format(filename))
        if list(array.shape) != expected.get("shape"):
            raise CacheValidationError(
                "{} shape mismatch: {} != {}".format(
                    filename, list(array.shape), expected.get("shape")
                )
            )
        if str(array.dtype) != expected.get("dtype"):
            raise CacheValidationError(
                "{} dtype mismatch: {} != {}".format(
                    filename, array.dtype, expected.get("dtype")
                )
            )


def _rebuild_hint(cache_dir):
    return "Rebuild with build_msa_rag_cache.py --cache-dir {} --force.".format(cache_dir)


def validate_cache(
    cache_dir,
    dataset_name,
    motion_latent_dir,
    text_latent_dir,
    hcls_dir,
    topk,
    text_embed_dim,
    exclude_self=True,
):
    cache_path = Path(cache_dir).expanduser().resolve()
    manifest = _load_manifest(cache_path)
    try:
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CacheValidationError("schema_version mismatch")
        if manifest.get("dataset_name") != dataset_name:
            raise CacheValidationError("dataset_name mismatch")
        if manifest.get("topk", 0) < topk:
            raise CacheValidationError(
                "topk mismatch: cache has {}, requested {}".format(
                    manifest.get("topk"), topk
                )
            )
        if manifest.get("text_embed_dim") != text_embed_dim:
            raise CacheValidationError("text_embed_dim mismatch")
        if manifest.get("exclude_self") != bool(exclude_self):
            raise CacheValidationError("exclude_self mismatch")

        directories, sources = _discover_sources(
            dataset_name, motion_latent_dir, text_latent_dir, hcls_dir
        )
        expected_directories = {
            label: str(path) for label, path in directories.items()
        }
        if manifest.get("source_directories") != expected_directories:
            raise CacheValidationError("source_directories mismatch")
        current_records = _source_records(sources)
        cached_records = manifest.get("sources")
        if cached_records != current_records:
            cached_by_id = {
                record.get("id"): record for record in (cached_records or [])
            }
            current_by_id = {record["id"]: record for record in current_records}
            differing_ids = [
                sample_id
                for sample_id in sorted(set(cached_by_id) | set(current_by_id))
                if cached_by_id.get(sample_id) != current_by_id.get(sample_id)
            ]
            detail = "{}.npy".format(differing_ids[0]) if differing_ids else "source list"
            raise CacheValidationError("source metadata mismatch for {}".format(detail))
        _validate_arrays(cache_path, manifest)
    except CacheValidationError as error:
        raise CacheValidationError("{} {}".format(error, _rebuild_hint(cache_path)))
    return manifest


def build_cache(
    dataset_name,
    motion_latent_dir,
    text_latent_dir,
    hcls_dir,
    cache_dir,
    topk,
    text_embed_dim,
    exclude_self=True,
    force=False,
    retrieval_batch_size=256,
):
    cache_path = Path(cache_dir).expanduser().resolve()
    if cache_path.exists() and not force:
        return validate_cache(
            str(cache_path),
            dataset_name,
            motion_latent_dir,
            text_latent_dir,
            hcls_dir,
            topk,
            text_embed_dim,
            exclude_self,
        )

    directories, sources = _discover_sources(
        dataset_name, motion_latent_dir, text_latent_dir, hcls_dir
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    temporary_path = cache_path.parent / ".{}.tmp-{}".format(cache_path.name, token)
    stale_path = cache_path.parent / ".{}.stale-{}".format(cache_path.name, token)
    temporary_path.mkdir()
    try:
        _write_cache(
            temporary_path,
            dataset_name,
            directories,
            sources,
            topk,
            text_embed_dim,
            exclude_self,
            retrieval_batch_size,
        )
        validate_cache(
            str(temporary_path),
            dataset_name,
            motion_latent_dir,
            text_latent_dir,
            hcls_dir,
            topk,
            text_embed_dim,
            exclude_self,
        )
        if cache_path.exists():
            os.replace(str(cache_path), str(stale_path))
        try:
            os.replace(str(temporary_path), str(cache_path))
        except Exception:
            if stale_path.exists() and not cache_path.exists():
                os.replace(str(stale_path), str(cache_path))
            raise
        if stale_path.exists():
            shutil.rmtree(str(stale_path))
    finally:
        if temporary_path.exists():
            shutil.rmtree(str(temporary_path))

    return validate_cache(
        str(cache_path),
        dataset_name,
        motion_latent_dir,
        text_latent_dir,
        hcls_dir,
        topk,
        text_embed_dim,
        exclude_self,
    )


class PackedMSARAGCache:
    """Memory-mapped access to a completed packed MSA-RAG cache."""

    def __init__(self, cache_dir, requested_topk):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest = _load_manifest(self.cache_dir)
        if self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CacheValidationError("schema_version mismatch in {}".format(self.cache_dir))
        if requested_topk <= 0 or requested_topk > self.manifest.get("topk", 0):
            raise CacheValidationError(
                "requested topk {} exceeds cached topk {}".format(
                    requested_topk, self.manifest.get("topk")
                )
            )
        _validate_arrays(self.cache_dir, self.manifest)
        sample_ids_path = self.cache_dir / "sample_ids.txt"
        if not sample_ids_path.is_file():
            raise CacheValidationError("Missing sample_ids.txt in {}".format(self.cache_dir))
        self.sample_ids = [
            line.strip() for line in sample_ids_path.read_text().splitlines() if line.strip()
        ]
        if len(self.sample_ids) != self.manifest.get("sample_count"):
            raise CacheValidationError("sample_ids.txt count does not match manifest.json")

        self.requested_topk = int(requested_topk)
        for filename in ARRAY_FILES:
            attribute = filename[:-4]
            setattr(
                self,
                attribute,
                np.load(self.cache_dir / filename, mmap_mode="r", allow_pickle=False),
            )

    def __len__(self):
        return len(self.sample_ids)

    def get(self, sample_idx, caption_idx):
        if sample_idx < 0 or sample_idx >= len(self):
            raise IndexError("sample_idx out of range: {}".format(sample_idx))
        caption_count = int(self.text_counts[sample_idx])
        if caption_idx < 0 or caption_idx >= caption_count:
            raise IndexError(
                "caption_idx out of range for {}: {}".format(
                    self.sample_ids[sample_idx], caption_idx
                )
            )

        text_row = int(self.text_offsets[sample_idx]) + int(caption_idx)
        motion_start = int(self.motion_offsets[sample_idx])
        motion_end = motion_start + int(self.motion_lengths[sample_idx])
        indices = np.asarray(
            self.retrieval_indices[text_row, : self.requested_topk], dtype=np.int64
        )
        return (
            np.asarray(self.text_values[text_row], dtype=np.float32).copy(),
            np.asarray(self.hcls_values[indices], dtype=np.float32).copy(),
            np.asarray(
                self.retrieval_scores[text_row, : self.requested_topk],
                dtype=np.float32,
            ).copy(),
            np.asarray(
                self.motion_values[motion_start:motion_end], dtype=np.float32
            ).copy(),
        )
