"""Dependency-light data model and statistics for inference benchmarks."""

from __future__ import annotations

import dataclasses
import math
import platform
import statistics
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = 1
MOTION_DIM = 272
VIDEO_FPS = 30.0


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    method: str
    frames: Sequence[int] = (60, 120, 196)
    num_runs: int = 20
    warmups: int = 2
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "native"

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["frames"] = list(self.frames)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def build_manifest(
    captions: Sequence[str],
    frames: Sequence[int],
    num_runs: int = 20,
    warmups: int = 2,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create one deterministic prompt/length manifest for all methods."""
    import random

    required = int(num_runs) + int(warmups)
    unique = []
    seen = set()
    for caption in captions:
        text = " ".join(str(caption).strip().split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            unique.append(text)
    if len(unique) < required:
        raise ValueError("need at least num_runs + warmups unique captions")
    selected = random.Random(seed).sample(unique, required)
    result: List[Dict[str, Any]] = []
    for frame_count in frames:
        frame_count = int(frame_count)
        if frame_count <= 0 or frame_count % 4:
            raise ValueError("benchmark frame counts must be positive multiples of 4")
        for index, caption in enumerate(selected):
            result.append(
                {
                    "prompt_id": "test_%04d" % index,
                    "caption": caption,
                    "frames": frame_count,
                    "warmup": index < int(warmups),
                    "seed": int(seed),
                }
            )
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p95": _percentile(values, 95.0),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate non-warmup samples by method and requested frame count."""
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        validate_sample(sample)
        if not sample.get("warmup", False):
            key = "%s@%d" % (sample["method"], int(sample["frames"]))
            groups[key].append(sample)

    output: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "groups": {}}
    timing_keys = ("text_ms", "retrieval_ms", "generation_ms", "decode_ms", "e2e_ms")
    for key, entries in sorted(groups.items()):
        frame_count = int(entries[0]["frames"])
        row: Dict[str, Any] = {"count": len(entries), "frames": frame_count}
        for timing_key in timing_keys:
            values = [entry["timings_ms"].get(timing_key) for entry in entries]
            values = [value for value in values if value is not None]
            if values:
                row[timing_key] = _describe(values)
        e2e = [entry["timings_ms"].get("e2e_ms") for entry in entries]
        e2e = [float(value) for value in e2e if value is not None and value > 0]
        row["effective_fps"] = _describe([frame_count * 1000.0 / value for value in e2e])
        row["rtf"] = _describe([value / 1000.0 / (frame_count / VIDEO_FPS) for value in e2e])
        for key_name in ("first_token_ms", "first_output_ms", "other_token_ms"):
            values = [entry["timings_ms"].get(key_name) for entry in entries]
            values = [value for value in values if value is not None]
            if values:
                row[key_name] = _describe(values)
        output["groups"][key] = row
    return output


def validate_sample(sample: Mapping[str, Any]) -> None:
    if int(sample.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema_version")
    frames = int(sample.get("frames", 0))
    shape = sample.get("output_shape")
    if shape is not None and list(shape) != [frames, MOTION_DIM]:
        raise ValueError("output_shape must be [frames, 272]")
    timings = sample.get("timings_ms")
    if not isinstance(timings, Mapping) or "e2e_ms" not in timings:
        raise ValueError("timings_ms.e2e_ms is required")
    for name, value in timings.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            values = value
        else:
            values = (value,)
        if any(float(item) < 0 for item in values):
            raise ValueError("timings must be non-negative")


def environment_metadata() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def load_humanml_test_captions(data_root: str) -> List[str]:
    """Load unique HumanML3D captions from the official test split."""
    root = Path(data_root)
    split = root / "split" / "test.txt"
    text_root = root / "texts"
    if not split.is_file():
        raise FileNotFoundError("HumanML3D split file not found: %s" % split)
    captions: List[str] = []
    for line in split.read_text(encoding="utf-8").splitlines():
        sample_id = line.strip()
        if not sample_id:
            continue
        text_file = text_root / (sample_id + ".txt")
        if not text_file.is_file():
            continue
        for text_line in text_file.read_text(encoding="utf-8").splitlines():
            caption = text_line.split("#", 1)[0].strip()
            if caption:
                captions.append(caption)
    return captions


def timed_call(fn, device: str = "cpu"):
    """Run ``fn`` with CUDA synchronization and return ``(result, ms)``."""
    def sync():
        if str(device).startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize(torch.device(device))
            except ImportError:
                pass
    sync()
    start = time.perf_counter()
    value = fn()
    sync()
    return value, (time.perf_counter() - start) * 1000.0


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
