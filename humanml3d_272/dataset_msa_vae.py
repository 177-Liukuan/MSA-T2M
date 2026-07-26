"""
DataLoader for MSA-VAE training.

Loads motion (272d, 30fps) + global text (HumanML3D captions) + local text
embeddings (BABEL frame-level, 20fps->30fps upsampled) for the HumanML3D∩BABEL
intersection (train_ft.txt).

Each __getitem__ returns:
  motion:              (T, 272) - normalized window or complete motion
  global_text:         str - randomly chosen HumanML3D caption for this sample
  global_text_embed:   (D,) - precomputed global text embedding (offline)
  has_global_embed:    bool - whether offline global embedding is available
  local_text_embed:    (T_latent, D) - local text embeddings pooled to latent rate
  has_local_embed:     bool - whether local text embeddings are available
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
import random
import math
import codecs as cs
from tqdm import tqdm


class MSAVAEDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4,
                 use_ft_split=True, text_encoder_type='clip',
                 clip_embed_dir=None, t5_embed_dir=None, text_embed_dim=None,
                 use_offline_global_text=True,
                 clip_global_embed_dir=None, t5_global_embed_dir=None,
                 sequence_mode='window'):
        if sequence_mode not in ('window', 'full'):
            raise ValueError(
                f'sequence_mode must be window/full, got {sequence_mode}'
            )
        self.window_size = window_size
        self.unit_length = unit_length
        self.sequence_mode = sequence_mode
        self.dataset_name = dataset_name
        self.text_encoder_type = text_encoder_type.lower()
        self.use_offline_global_text = use_offline_global_text

        if self.text_encoder_type not in ('clip', 't5'):
            raise ValueError(f'text_encoder_type must be clip/t5, got {text_encoder_type}')

        default_dim = 512 if self.text_encoder_type == 'clip' else 768
        self.text_embed_dim = default_dim if text_embed_dim is None or text_embed_dim <= 0 else int(text_embed_dim)

        if dataset_name in ('t2m_272', 't2m_babel_272'):
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            self.joints_num = 22
            self.max_motion_length = 300

            default_clip_dir = pjoin(self.data_root, 'clip_enc_single')
            default_t5_dir = pjoin(self.data_root, 't5_enc_single')
            self.clip_embed_dir = clip_embed_dir if clip_embed_dir else default_clip_dir
            self.t5_embed_dir = t5_embed_dir if t5_embed_dir else default_t5_dir
            self.local_text_dir = self.clip_embed_dir if self.text_encoder_type == 'clip' else self.t5_embed_dir

            default_clip_global_dir = pjoin(self.data_root, 'text_latents_clip')
            default_t5_global_dir = pjoin(self.data_root, 'text_latents_t5')
            self.clip_global_embed_dir = clip_global_embed_dir if clip_global_embed_dir else default_clip_global_dir
            self.t5_global_embed_dir = t5_global_embed_dir if t5_global_embed_dir else default_t5_global_dir
            self.global_embed_dir = self.clip_global_embed_dir if self.text_encoder_type == 'clip' else self.t5_global_embed_dir

            if use_ft_split:
                split_file = pjoin(self.data_root, 'split', 'train_ft.txt')
            else:
                split_file = pjoin(self.data_root, 'split', 'train.txt')
        else:
            raise ValueError(f'Dataset {dataset_name} not supported')

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.data = []
        self.lengths = []
        self.skipped_subunit_count = 0
        n_with_local = 0
        n_with_global = 0

        for name in tqdm(id_list, desc='Loading MSA-VAE data'):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.unit_length:
                    self.skipped_subunit_count += 1
                    continue
                if (self.sequence_mode == 'window'
                        and motion.shape[0] < self.window_size):
                    continue

                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                with cs.open(text_path, 'r') as f:
                    text_lines = f.readlines()

                captions_all = []
                global_captions = []
                global_indices = []
                for li, line in enumerate(text_lines):
                    parts = line.strip().split('#')
                    caption = parts[0].strip() if len(parts) > 0 else ''
                    captions_all.append(caption)
                    if len(parts) >= 4:
                        f_tag = float(parts[2]) if parts[2] != 'nan' else 0.0
                        to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
                        if f_tag == 0.0 and to_tag == 0.0 and caption:
                            global_captions.append(caption)
                            global_indices.append(li)

                if not global_captions:
                    # fallback: use first caption and index 0
                    fallback = captions_all[0] if captions_all else ''
                    global_captions = [fallback]
                    global_indices = [0]

                text_path_local = pjoin(self.local_text_dir, name + '.npy')
                has_local = os.path.exists(text_path_local)
                if has_local:
                    try:
                        probe = np.load(text_path_local, mmap_mode='r')
                        if probe.ndim != 2 or probe.shape[1] != self.text_embed_dim:
                            has_local = False
                    except Exception:
                        has_local = False

                global_path = pjoin(self.global_embed_dir, name + '.npy')
                has_global = self.use_offline_global_text and os.path.exists(global_path)
                if has_global:
                    try:
                        probe_g = np.load(global_path, mmap_mode='r')
                        max_idx = max(global_indices) if len(global_indices) > 0 else 0
                        if (probe_g.ndim != 2
                                or probe_g.shape[1] != self.text_embed_dim
                                or probe_g.shape[0] <= max_idx):
                            has_global = False
                    except Exception:
                        has_global = False

                if has_local:
                    n_with_local += 1
                if has_global:
                    n_with_global += 1

                entry = {
                    'name': name,
                    'motion': motion,
                    'captions': global_captions,
                    'caption_indices': global_indices,
                    'local_text_path': text_path_local if has_local else None,
                    'has_local': has_local,
                    'global_text_path': global_path if has_global else None,
                    'has_global': has_global,
                }
                self.data.append(entry)
                if self.sequence_mode == 'window':
                    self.lengths.append(
                        motion.shape[0] - self.window_size + 1
                    )
                else:
                    self.lengths.append(1)
            except MotionSequenceTooShortError:
                raise
            except Exception:
                pass

        self.mean = mean
        self.std = std
        print(f'MSA-VAE training: {len(self.data)} samples '
              f'({n_with_local} with local {self.text_encoder_type.upper()} embeddings, '
              f'{n_with_global} with offline global embeddings, '
              f'{self.skipped_subunit_count} shorter than one latent unit skipped, '
              f'dim={self.text_embed_dim})')

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.get_item(item, self.sequence_mode)

    def get_item(self, item, sequence_mode):
        if sequence_mode not in ('window', 'full'):
            raise ValueError(
                f'sequence_mode must be window/full, got {sequence_mode}'
            )

        entry = self.data[item]
        motion = entry['motion']

        if sequence_mode == 'window':
            motion_length = self.window_size
            idx = random.randint(0, len(motion) - motion_length)
        else:
            motion_length = (len(motion) // self.unit_length) * self.unit_length
            if motion_length < self.unit_length:
                raise ValueError(
                    f'Motion {entry["name"]} is too short for one latent token'
                )
            idx = 0

        motion_view = motion[idx:idx + motion_length]
        motion_view = (motion_view - self.mean) / self.std

        # sample one global caption and its line index
        cidx = random.randint(0, len(entry['captions']) - 1)
        caption = entry['captions'][cidx]
        caption_line_idx = entry['caption_indices'][cidx]

        # local frame-level text embeddings
        has_local = entry['has_local']
        latent_len = motion_length // self.unit_length
        if has_local:
            local_text_20 = np.load(entry['local_text_path'])  # (T_20fps, D)
            T_20 = local_text_20.shape[0]
            T_30 = len(motion)
            indices_30 = np.round(np.linspace(0, T_20 - 1, T_30)).astype(int)
            local_text_30 = local_text_20[indices_30]
            local_text_view = local_text_30[idx:idx + motion_length]
            local_text_pooled = local_text_view.mean(axis=0)  # (D,)
            local_text_latent = _pool_to_latent(local_text_view, latent_len)
        else:
            local_text_latent = np.zeros((latent_len, self.text_embed_dim), dtype=np.float32)
            local_text_pooled = np.zeros((self.text_embed_dim,), dtype=np.float32)

        # global text embedding (offline)
        has_global = entry['has_global']
        if has_global:
            try:
                global_all = np.load(entry['global_text_path'])
                if global_all.ndim == 2 and global_all.shape[1] == self.text_embed_dim and global_all.shape[0] > caption_line_idx:
                    global_text_embed = global_all[caption_line_idx]
                else:
                    has_global = False
                    global_text_embed = np.zeros((self.text_embed_dim,), dtype=np.float32)
            except Exception:
                has_global = False
                global_text_embed = np.zeros((self.text_embed_dim,), dtype=np.float32)
        else:
            global_text_embed = np.zeros((self.text_embed_dim,), dtype=np.float32)

        return (
            motion_view.astype(np.float32),
            caption,
            global_text_embed.astype(np.float32),
            has_global,
            local_text_latent.astype(np.float32),
            has_local,
            len(motion),
            local_text_pooled.astype(np.float32),
            motion_length,
        )


class MotionSequenceTooShortError(ValueError):
    """A motion cannot produce even one complete temporal latent token."""


def source_dataset_sequence_mode(sequence_mode):
    """Keep every full-sequence record when mixed replay is requested."""
    if sequence_mode == 'mixed':
        return 'full'
    if sequence_mode in ('window', 'full'):
        return sequence_mode
    raise ValueError(f'unknown sequence_mode: {sequence_mode}')


class MSAVAESequenceView(data.Dataset):
    """Select a sequence view without duplicating the loaded source records."""

    def __init__(self, dataset, sequence_mode):
        if sequence_mode not in ('window', 'full'):
            raise ValueError(
                f'sequence_mode must be window/full, got {sequence_mode}'
            )
        self.dataset = dataset
        self.sequence_mode = sequence_mode
        if sequence_mode == 'window':
            self.indices = [
                index
                for index, entry in enumerate(dataset.data)
                if len(entry['motion']) >= dataset.window_size
            ]
        else:
            self.indices = list(range(len(dataset.data)))

    @property
    def source_lengths(self):
        if self.sequence_mode == 'window':
            return [self.dataset.window_size] * len(self.indices)
        return [
            (len(self.dataset.data[index]['motion']) // self.dataset.unit_length)
            * self.dataset.unit_length
            for index in self.indices
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset.get_item(self.indices[item], self.sequence_mode)


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Shuffle similarly sized sequences together to limit padding."""

    def __init__(self, lengths, batch_size, bucket_size, drop_last=True,
                 seed=123):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if bucket_size < 1:
            raise ValueError('bucket_size must be positive')
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.bucket_size = max(int(bucket_size), self.batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _bucket_sizes(self):
        return [
            min(self.bucket_size, len(self.lengths) - start)
            for start in range(0, len(self.lengths), self.bucket_size)
        ]

    def __len__(self):
        if self.drop_last:
            return sum(size // self.batch_size for size in self._bucket_sizes())
        return sum(
            math.ceil(size / self.batch_size) for size in self._bucket_sizes()
        )

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        sorted_indices = sorted(
            range(len(self.lengths)), key=self.lengths.__getitem__
        )
        batches = []
        for start in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[start:start + self.bucket_size]
            generator.shuffle(bucket)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        generator.shuffle(batches)
        return iter(batches)


def _pool_to_latent(text_window, latent_len):
    """Average-pool frame-level text embedding (T, D) to (latent_len, D)."""
    T = text_window.shape[0]
    if T == latent_len:
        return text_window
    indices = np.linspace(0, T, latent_len + 1).astype(int)
    pooled = np.zeros((latent_len, text_window.shape[1]), dtype=np.float32)
    for i in range(latent_len):
        pooled[i] = text_window[indices[i]:indices[i + 1]].mean(axis=0)
    return pooled


def collate_fn(batch):
    (motions, captions, global_texts, has_globals, local_texts, has_locals,
     total_frames, local_pooled, motion_lengths) = zip(*batch)

    max_motion_len = max(motion.shape[0] for motion in motions)
    motion_dim = motions[0].shape[1]
    motion_batch = torch.zeros(
        len(motions), max_motion_len, motion_dim, dtype=torch.float32
    )
    for index, motion in enumerate(motions):
        motion_batch[index, :motion.shape[0]] = torch.from_numpy(motion)

    max_local_len = max(local.shape[0] for local in local_texts)
    text_dim = local_texts[0].shape[1]
    local_batch = torch.zeros(
        len(local_texts), max_local_len, text_dim, dtype=torch.float32
    )
    for index, local in enumerate(local_texts):
        local_batch[index, :local.shape[0]] = torch.from_numpy(local)

    global_texts = torch.from_numpy(np.stack(global_texts))
    has_globals = torch.tensor(has_globals, dtype=torch.bool)
    has_locals = torch.tensor(has_locals, dtype=torch.bool)
    total_frames = torch.tensor(total_frames, dtype=torch.long)
    local_pooled = torch.from_numpy(np.stack(local_pooled))
    motion_lengths = torch.tensor(motion_lengths, dtype=torch.long)
    return (
        motion_batch, list(captions), global_texts, has_globals,
        local_batch, has_locals, total_frames, local_pooled, motion_lengths,
    )


def make_loader(dataset, sequence_mode, batch_size, num_workers=8,
                bucket_size=0, drop_last=True, seed=123):
    view = MSAVAESequenceView(dataset, sequence_mode)
    if sequence_mode == 'full' and bucket_size > 0:
        batch_sampler = LengthBucketBatchSampler(
            view.source_lengths,
            batch_size=batch_size,
            bucket_size=bucket_size,
            drop_last=drop_last,
            seed=seed,
        )
        return torch.utils.data.DataLoader(
            view,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return torch.utils.data.DataLoader(
        view,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def DATALoader(dataset_name, batch_size, num_workers=8,
               window_size=64, unit_length=4, use_ft_split=True,
               text_encoder_type='clip', clip_embed_dir=None,
               t5_embed_dir=None, text_embed_dim=None,
               use_offline_global_text=True,
               clip_global_embed_dir=None, t5_global_embed_dir=None,
               sequence_mode='window', bucket_size=0, drop_last=True,
               seed=123):
    trainSet = MSAVAEDataset(
        dataset_name, window_size=window_size,
        unit_length=unit_length, use_ft_split=use_ft_split,
        text_encoder_type=text_encoder_type,
        clip_embed_dir=clip_embed_dir,
        t5_embed_dir=t5_embed_dir,
        text_embed_dim=text_embed_dim,
        use_offline_global_text=use_offline_global_text,
        clip_global_embed_dir=clip_global_embed_dir,
        t5_global_embed_dir=t5_global_embed_dir,
        sequence_mode=sequence_mode,
    )
    return make_loader(
        trainSet,
        sequence_mode=sequence_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        bucket_size=bucket_size,
        drop_last=drop_last,
        seed=seed,
    )


def cycle(iterable):
    epoch = 0
    while True:
        if hasattr(iterable, 'set_epoch'):
            iterable.set_epoch(epoch)
        elif hasattr(iterable, 'batch_sampler') and hasattr(
                iterable.batch_sampler, 'set_epoch'):
            iterable.batch_sampler.set_epoch(epoch)
        for x in iterable:
            yield x
        epoch += 1
