"""Deterministic complete-motion dataset for standalone MSA-VAE metrics."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from humanml3d_272.msa_text_targets import build_local_text_target


@dataclass(frozen=True)
class CompleteCaption:
    text: str
    line_index: int


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    captions: Tuple[CompleteCaption, ...]
    motion_path: Path
    length: int
    raw_length: int

    @property
    def caption(self):
        return self.captions[0].text


def _as_complete_tag(value):
    value = float(value)
    return 0.0 if math.isnan(value) else value


def _complete_captions(text_path):
    captions = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("#")
            if len(fields) < 4:
                raise ValueError(
                    "{}:{} must contain caption#tokens#from#to".format(
                        text_path,
                        line_index + 1,
                    )
                )
            from_tag = _as_complete_tag(fields[2])
            to_tag = _as_complete_tag(fields[3])
            if from_tag == 0.0 and to_tag == 0.0:
                caption = fields[0].strip()
                if caption:
                    captions.append(
                        CompleteCaption(
                            text=caption,
                            line_index=line_index,
                        )
                    )
    return tuple(captions)


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
        self.split_file = resolved_split.resolve()
        self.records = self._build_records(self.split_file)
        self.sample_ids = [record.sample_id for record in self.records]
        if not self.records:
            raise ValueError("deterministic MSA-VAE evaluation set is empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("deterministic MSA-VAE evaluation IDs are not unique")
        self.sample_hash = hashlib.sha256(
            "\n".join(self.sample_ids).encode("utf-8")
        ).hexdigest()
        caption_digest = hashlib.sha256()
        for record in self.records:
            for caption in record.captions:
                caption_digest.update(record.sample_id.encode("utf-8"))
                caption_digest.update(b"\0")
                caption_digest.update(str(caption.line_index).encode("ascii"))
                caption_digest.update(b"\0")
                caption_digest.update(caption.text.encode("utf-8"))
                caption_digest.update(b"\n")
        self.caption_hash = caption_digest.hexdigest()
        self.caption_count = sum(
            len(record.captions) for record in self.records
        )

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
            captions = _complete_captions(text_path)
            if not captions:
                continue
            length = raw_length // self.unit_length * self.unit_length
            records.append(
                EvaluationRecord(
                    sample_id=sample_id,
                    captions=captions,
                    motion_path=motion_path,
                    length=length,
                    raw_length=raw_length,
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


def _hash_array(digest, sample_id, array):
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0float32\0")
    digest.update(contiguous.tobytes(order="C"))
    digest.update(b"\n")


class MSAVAEAlignmentDataset(MSAVAEMetricsDataset):
    """Strict complete-motion SentenceT5 targets for internal alignment."""

    def __init__(
        self,
        data_root,
        split_file=None,
        unit_length=4,
        target_mode="global",
        text_embed_dim=768,
        global_text_embed_dir=None,
        local_text_embed_dir=None,
        min_motion_length=60,
        max_motion_length=300,
    ):
        if target_mode not in ("global", "local"):
            raise ValueError("target_mode must be global or local")
        self.target_mode = target_mode
        self.text_embed_dim = int(text_embed_dim)
        if self.text_embed_dim <= 0:
            raise ValueError("text_embed_dim must be positive")
        root = Path(data_root).resolve()
        if target_mode == "global":
            target_directory = (
                Path(global_text_embed_dir)
                if global_text_embed_dir is not None
                else root / "text_latents_t5"
            )
        else:
            target_directory = (
                Path(local_text_embed_dir)
                if local_text_embed_dir is not None
                else root / "t5_enc_single"
            )
        self._target_directory_path = target_directory.resolve()
        self.target_directory = str(self._target_directory_path)
        if not self._target_directory_path.is_dir():
            raise FileNotFoundError(
                "{} alignment target directory is missing: {}".format(
                    target_mode,
                    self._target_directory_path,
                )
            )

        super().__init__(
            data_root=data_root,
            split_file=split_file,
            unit_length=unit_length,
            min_motion_length=min_motion_length,
            max_motion_length=max_motion_length,
        )
        self._target_paths = {}
        target_digest = hashlib.sha256()
        local_token_count = 0
        for record in self.records:
            target_path = (
                self._target_directory_path
                / "{}.npy".format(record.sample_id)
            )
            if not target_path.is_file():
                raise FileNotFoundError(
                    "missing {} alignment target for {}: {}".format(
                        target_mode,
                        record.sample_id,
                        target_path,
                    )
                )
            target = np.load(target_path)
            if target.ndim != 2:
                raise ValueError(
                    "{} alignment target for {} must be 2D".format(
                        target_mode,
                        record.sample_id,
                    )
                )
            if target.shape[1] != self.text_embed_dim:
                raise ValueError(
                    "{} alignment target dimension for {} must be {}".format(
                        target_mode,
                        record.sample_id,
                        self.text_embed_dim,
                    )
                )
            if not np.isfinite(target).all():
                raise ValueError(
                    "{} alignment target for {} contains non-finite values".format(
                        target_mode,
                        record.sample_id,
                    )
                )
            if target_mode == "global":
                indices = tuple(
                    caption.line_index for caption in record.captions
                )
                if target.shape[0] <= max(indices):
                    raise ValueError(
                        "global alignment target for {} is missing a "
                        "caption row".format(record.sample_id)
                    )
                selected = target[list(indices)]
                _hash_array(target_digest, record.sample_id, selected)
            else:
                if target.shape[0] < 1:
                    raise ValueError(
                        "local alignment target for {} has no frames".format(
                            record.sample_id,
                        )
                    )
                _hash_array(target_digest, record.sample_id, target)
                local_token_count += record.length // self.unit_length
            self._target_paths[record.sample_id] = target_path
        self.target_hash = target_digest.hexdigest()
        self.local_token_count = local_token_count
        if target_mode == "local":
            self.caption_count = 0

    def __getitem__(self, index):
        item = super().__getitem__(index)
        record = self.records[index]
        item["raw_length"] = record.raw_length
        item["target_mode"] = self.target_mode
        target = np.load(self._target_paths[record.sample_id]).astype(
            np.float32
        )
        if self.target_mode == "global":
            indices = tuple(
                caption.line_index for caption in record.captions
            )
            item["all_captions"] = tuple(
                caption.text for caption in record.captions
            )
            item["caption_line_indices"] = indices
            item["global_text_embeddings"] = torch.from_numpy(
                target[list(indices)].copy()
            )
        else:
            latent_length = record.length // self.unit_length
            local_target, _ = build_local_text_target(
                target,
                raw_motion_length=record.raw_length,
                view_start=0,
                view_length=record.length,
                latent_length=latent_length,
                expected_dim=self.text_embed_dim,
            )
            item["local_text_embeddings"] = torch.from_numpy(local_target)
        return item


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


def collate_msa_vae_alignment(batch):
    if not batch:
        raise ValueError("cannot collate an empty MSA-VAE alignment batch")
    modes = {item.get("target_mode") for item in batch}
    if len(modes) != 1 or None in modes:
        raise ValueError("alignment batch must contain one target_mode")
    target_mode = modes.pop()
    result = collate_msa_vae_metrics(batch)
    result["target_mode"] = target_mode
    if target_mode == "global":
        result["all_captions"] = [
            item["all_captions"] for item in batch
        ]
        result["caption_line_indices"] = [
            item["caption_line_indices"] for item in batch
        ]
        result["global_text_embeddings"] = [
            item["global_text_embeddings"] for item in batch
        ]
        return result
    if target_mode != "local":
        raise ValueError("unknown alignment target_mode")

    max_latent_length = max(
        item["local_text_embeddings"].shape[0] for item in batch
    )
    text_dim = batch[0]["local_text_embeddings"].shape[1]
    local_targets = torch.zeros(
        len(batch),
        max_latent_length,
        text_dim,
        dtype=torch.float32,
    )
    local_mask = torch.zeros(
        len(batch),
        max_latent_length,
        dtype=torch.bool,
    )
    for index, item in enumerate(batch):
        target = item["local_text_embeddings"]
        if target.ndim != 2 or target.shape[1] != text_dim:
            raise ValueError("local alignment target shapes must match")
        length = target.shape[0]
        local_targets[index, :length] = target
        local_mask[index, :length] = True
    result["local_text_embeddings"] = local_targets
    result["local_mask"] = local_mask
    return result


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


def make_msa_vae_alignment_loader(
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
        collate_fn=collate_msa_vae_alignment,
        drop_last=False,
        pin_memory=bool(pin_memory),
    )
