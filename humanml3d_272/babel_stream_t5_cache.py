"""Build and validate frame-level SentenceT5 targets for BABEL stream data."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


CACHE_VERSION = 1
MOTION_DIM = 272


class CacheBuildError(ValueError):
    """Raised when source records are invalid before a cache is published."""

    def __init__(self, rejections: Iterable[str]):
        self.rejections = tuple(rejections)
        super().__init__("Cache build rejected samples: {}".format("; ".join(self.rejections)))


@dataclass(frozen=True)
class BabelStreamRecord:
    first_text: str
    second_text: str
    boundary: int


def parse_babel_stream_text(line: str) -> BabelStreamRecord:
    """Parse the strict two-action BABEL-stream text record format."""
    segments = line.strip().split("*")
    if len(segments) != 2:
        raise ValueError("expected exactly two segments")

    first_fields = segments[0].split("#")
    second_fields = segments[1].split("#")
    first_text = first_fields[0].strip() if first_fields else ""
    second_text = second_fields[0].strip() if second_fields else ""
    if not first_text or not second_text:
        raise ValueError("action text must not be blank")
    if len(second_fields) < 5 or not second_fields[-1].strip():
        raise ValueError("missing boundary")
    try:
        boundary = int(second_fields[-1].strip())
    except ValueError as error:
        raise ValueError("invalid boundary") from error
    if boundary <= 0:
        raise ValueError("boundary must be positive")
    return BabelStreamRecord(first_text, second_text, boundary)


def expand_segment_embeddings(
    record: BabelStreamRecord,
    motion_frames: int,
    embedding_by_text: Dict[str, np.ndarray],
) -> np.ndarray:
    """Expand two sentence embeddings to one exact-length frame-level target."""
    if motion_frames <= 0:
        raise ValueError("motion_frames must be positive")
    if record.boundary <= 0 or record.boundary >= motion_frames:
        raise ValueError("boundary must lie inside motion frames")
    try:
        first_embedding = np.asarray(embedding_by_text[record.first_text])
        second_embedding = np.asarray(embedding_by_text[record.second_text])
    except KeyError as error:
        raise ValueError("missing embedding for {}".format(error.args[0])) from error
    if first_embedding.ndim != 1 or second_embedding.ndim != 1:
        raise ValueError("segment embeddings must be one-dimensional")
    if first_embedding.shape != second_embedding.shape or first_embedding.size == 0:
        raise ValueError("segment embedding dimensions must match and be non-empty")

    output = np.empty((motion_frames, first_embedding.shape[0]), dtype=np.float32)
    output[: record.boundary] = first_embedding.astype(np.float32, copy=False)
    output[record.boundary :] = second_embedding.astype(np.float32, copy=False)
    return output


def _canonical(path: Path) -> str:
    return str(Path(path).expanduser().resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_signature(stem: str, motion_path: Path, text_path: Path, frames: int) -> dict:
    return {
        "frames": int(frames),
        "text_sha256": _sha256(text_path),
        "motion_size": int(motion_path.stat().st_size),
        "motion_sha256": _sha256(motion_path),
    }


def _discover_source_paths(motion_dir: Path, text_dir: Path) -> Tuple[Path, Path, Tuple[str, ...]]:
    motion_root = Path(motion_dir).expanduser().resolve()
    text_root = Path(text_dir).expanduser().resolve()
    if not motion_root.is_dir():
        raise FileNotFoundError("motion directory not found: {}".format(motion_root))
    if not text_root.is_dir():
        raise FileNotFoundError("text directory not found: {}".format(text_root))

    motion_stems = {path.stem for path in motion_root.iterdir() if path.is_file() and path.suffix == ".npy"}
    text_stems = {path.stem for path in text_root.iterdir() if path.is_file() and path.suffix == ".txt"}
    missing_text = sorted(motion_stems - text_stems)
    missing_motion = sorted(text_stems - motion_stems)
    rejections = ["{}: missing text file".format(stem) for stem in missing_text]
    rejections.extend("{}: missing motion file".format(stem) for stem in missing_motion)
    if rejections:
        raise CacheBuildError(rejections)
    if not motion_stems:
        raise CacheBuildError(("no matching .npy/.txt source records",))
    return motion_root, text_root, tuple(sorted(motion_stems))


def _read_records(motion_dir: Path, text_dir: Path, stems: Iterable[str]):
    records = []
    rejections = []
    for stem in stems:
        motion_path = motion_dir / "{}.npy".format(stem)
        text_path = text_dir / "{}.txt".format(stem)
        try:
            record = parse_babel_stream_text(text_path.read_text())
            motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
            if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
                raise ValueError("motion must have shape (T, {})".format(MOTION_DIM))
            if motion.shape[0] <= record.boundary:
                raise ValueError("boundary must lie inside motion frames")
            records.append(
                {
                    "stem": stem,
                    "record": record,
                    "motion_path": motion_path,
                    "text_path": text_path,
                    "frames": int(motion.shape[0]),
                }
            )
        except (OSError, ValueError) as error:
            rejections.append("{}: {}".format(stem, error))
    if rejections:
        raise CacheBuildError(tuple(sorted(rejections)))
    return records


def _encode_unique_texts(records, encoder, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    texts = sorted(
        {
            text
            for item in records
            for text in (item["record"].first_text, item["record"].second_text)
        }
    )
    embeddings = {}
    embedding_dim = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        values = np.asarray(encoder.encode(batch), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(batch) or values.shape[1] <= 0:
            raise ValueError("encoder must return one non-empty 1-D embedding per text")
        if embedding_dim is None:
            embedding_dim = int(values.shape[1])
        elif values.shape[1] != embedding_dim:
            raise ValueError("encoder returned inconsistent embedding dimensions")
        embeddings.update(zip(batch, values))
    return embeddings, int(embedding_dim)


def _atomic_save_array(output_dir: Path, filename: str, array: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output_dir), prefix=".{}-".format(filename), suffix=".npy"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.save(temporary_path, array, allow_pickle=False)
        probe = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if probe.shape != array.shape or probe.dtype != np.dtype("float32"):
            raise ValueError("temporary cache array validation failed for {}".format(filename))
        os.replace(str(temporary_path), str(output_dir / filename))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_manifest(output_dir: Path, manifest: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output_dir), prefix=".manifest-", suffix=".json"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(str(temporary_path), str(output_dir / "manifest.json"))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _manifest_expected_from_sources(split, model_signature, embedding_dim, motion_dir, text_dir):
    return {
        "split": split,
        "model_signature": model_signature,
        "embedding_dim": int(embedding_dim),
        "motion_dir": _canonical(motion_dir),
        "text_dir": _canonical(text_dir),
    }


def validate_cache_manifest(manifest_path, expected) -> dict:
    """Validate a published BABEL cache against its exact source signatures."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("missing manifest.json: {}".format(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("cannot read manifest.json: {}".format(error)) from error

    checks = {
        "split": expected.get("split"),
        "model_signature": expected.get("model_signature"),
        "embedding_dim": expected.get("embedding_dim"),
        "motion_dir": expected.get("motion_dir"),
        "text_dir": expected.get("text_dir"),
    }
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError("cache version mismatch")
    for key, value in checks.items():
        if value is None:
            raise ValueError("expected cache field is required: {}".format(key))
        if manifest.get(key) != value:
            raise ValueError("cache {} mismatch".format(key))

    motion_dir, text_dir, stems = _discover_source_paths(
        Path(expected["motion_dir"]), Path(expected["text_dir"])
    )
    source_records = _read_records(motion_dir, text_dir, stems)
    current_records = {
        item["stem"]: _source_signature(
            item["stem"], item["motion_path"], item["text_path"], item["frames"]
        )
        for item in source_records
    }
    if manifest.get("records") != current_records:
        raise ValueError("cache source records mismatch")
    if manifest.get("valid_samples") != len(current_records):
        raise ValueError("cache valid sample count mismatch")
    if manifest.get("rejected_samples") != 0:
        raise ValueError("cache contains rejected samples")

    output_dir = manifest_path.parent
    embedding_dim = int(expected["embedding_dim"])
    for stem, record in current_records.items():
        array_path = output_dir / "{}.npy".format(stem)
        if not array_path.is_file():
            raise ValueError("missing cache array: {}".format(array_path.name))
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError("cannot load cache array {}: {}".format(array_path.name, error)) from error
        if array.shape != (record["frames"], embedding_dim) or array.dtype != np.dtype("float32"):
            raise ValueError("cache array shape or dtype mismatch: {}".format(array_path.name))
    return manifest


def build_cache(
    split,
    motion_dir,
    text_dir,
    output_dir,
    encoder,
    model_signature,
    overwrite=False,
    batch_size=256,
) -> dict:
    """Build a local BABEL SentenceT5 target cache and publish its manifest last."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("cannot read existing manifest.json: {}".format(error)) from error
        return validate_cache_manifest(
            manifest_path,
            _manifest_expected_from_sources(
                split,
                model_signature,
                existing.get("embedding_dim"),
                motion_dir,
                text_dir,
            ),
        )

    motion_dir, text_dir, stems = _discover_source_paths(Path(motion_dir), Path(text_dir))
    source_records = _read_records(motion_dir, text_dir, stems)
    embeddings, embedding_dim = _encode_unique_texts(source_records, encoder, batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = {}
    for item in source_records:
        stem = item["stem"]
        expanded = expand_segment_embeddings(item["record"], item["frames"], embeddings)
        _atomic_save_array(output_dir, "{}.npy".format(stem), expanded)
        manifest_records[stem] = _source_signature(
            stem, item["motion_path"], item["text_path"], item["frames"]
        )

    manifest = {
        "version": CACHE_VERSION,
        "split": str(split),
        "model_signature": str(model_signature),
        "embedding_dim": embedding_dim,
        "motion_dir": _canonical(motion_dir),
        "text_dir": _canonical(text_dir),
        "valid_samples": len(source_records),
        "rejected_samples": 0,
        "records": manifest_records,
    }
    _atomic_write_manifest(output_dir, manifest)
    return manifest
