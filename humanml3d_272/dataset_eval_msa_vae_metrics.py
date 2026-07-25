"""Deterministic complete-motion dataset for standalone MSA-VAE metrics."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    caption: str
    motion_path: Path
    length: int


def _as_complete_tag(value):
    value = float(value)
    return 0.0 if math.isnan(value) else value


def _first_complete_caption(text_path):
    with text_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("#")
            if len(fields) < 4:
                raise ValueError(
                    "{}:{} must contain caption#tokens#from#to".format(
                        text_path,
                        line_number,
                    )
                )
            from_tag = _as_complete_tag(fields[2])
            to_tag = _as_complete_tag(fields[3])
            if from_tag == 0.0 and to_tag == 0.0:
                return fields[0]
    return None


class MSAVAEMetricsDataset(Dataset):
    """One deterministic full-motion/caption pair per HumanML3D test ID."""

    def __init__(
        self,
        data_root,
        split_file=None,
        unit_length=4,
        min_motion_length=60,
        max_motion_length=300,
    ):
        self.data_root = Path(data_root).resolve()
        self.motion_dir = self.data_root / "motion_data"
        self.text_dir = self.data_root / "texts"
        self.unit_length = int(unit_length)
        self.min_motion_length = int(min_motion_length)
        self.max_motion_length = int(max_motion_length)
        if self.unit_length <= 0:
            raise ValueError("unit_length must be positive")

        mean_path = self.data_root / "mean_std" / "Mean.npy"
        std_path = self.data_root / "mean_std" / "Std.npy"
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)
        if self.mean.shape != (272,) or self.std.shape != (272,):
            raise ValueError("HumanML3D-272 mean and std must each have shape (272,)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("HumanML3D-272 normalization contains non-finite values")
        if np.any(self.std == 0):
            raise ValueError("HumanML3D-272 standard deviation contains zero")

        resolved_split = (
            Path(split_file)
            if split_file is not None
            else self.data_root / "split" / "test.txt"
        )
        self.records = self._build_records(resolved_split)
        self.sample_ids = [record.sample_id for record in self.records]
        if not self.records:
            raise ValueError("deterministic MSA-VAE evaluation set is empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("deterministic MSA-VAE evaluation IDs are not unique")
        self.sample_hash = hashlib.sha256(
            "\n".join(self.sample_ids).encode("utf-8")
        ).hexdigest()

    def _build_records(self, split_file):
        with split_file.open("r", encoding="utf-8") as handle:
            sample_ids = [line.strip() for line in handle if line.strip()]

        records = []
        for sample_id in sample_ids:
            motion_path = self.motion_dir / "{}.npy".format(sample_id)
            text_path = self.text_dir / "{}.txt".format(sample_id)
            motion = np.load(motion_path, mmap_mode="r")
            if motion.ndim != 2 or motion.shape[1] != 272:
                raise ValueError(
                    "{} must have shape (frames, 272), got {}".format(
                        motion_path,
                        motion.shape,
                    )
                )
            raw_length = int(motion.shape[0])
            if (
                raw_length < self.min_motion_length
                or raw_length >= self.max_motion_length
            ):
                continue
            caption = _first_complete_caption(text_path)
            if caption is None:
                continue
            length = raw_length // self.unit_length * self.unit_length
            records.append(
                EvaluationRecord(
                    sample_id=sample_id,
                    caption=caption,
                    motion_path=motion_path,
                    length=length,
                )
            )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        motion = np.load(record.motion_path)[: record.length].astype(np.float32)
        normalized = (motion - self.mean) / self.std
        return {
            "sample_id": record.sample_id,
            "caption": record.caption,
            "motion": torch.from_numpy(normalized.astype(np.float32)),
            "length": record.length,
        }

    def inv_transform(self, array):
        return np.asarray(array) * self.std + self.mean


def collate_msa_vae_metrics(batch):
    if not batch:
        raise ValueError("cannot collate an empty MSA-VAE evaluation batch")
    max_length = max(int(item["length"]) for item in batch)
    motions = torch.zeros(len(batch), max_length, 272, dtype=torch.float32)
    lengths = torch.empty(len(batch), dtype=torch.long)
    sample_ids: List[str] = []
    captions: List[str] = []
    for index, item in enumerate(batch):
        length = int(item["length"])
        motions[index, :length] = item["motion"]
        lengths[index] = length
        sample_ids.append(item["sample_id"])
        captions.append(item["caption"])
    return {
        "sample_ids": sample_ids,
        "captions": captions,
        "motions": motions,
        "lengths": lengths,
    }


def make_msa_vae_metrics_loader(
    dataset,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_msa_vae_metrics,
        drop_last=False,
        pin_memory=bool(pin_memory),
    )
