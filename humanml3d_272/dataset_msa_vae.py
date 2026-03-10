"""
DataLoader for MSA-VAE training.

Loads motion (272d, 30fps) + global text (HumanML3D captions) + local CLIP
embeddings (BABEL frame-level, 20fps→30fps upsampled) for the HumanML3D∩BABEL
intersection (train_ft.txt).

Each __getitem__ returns:
  motion:           (window_size, 272) - normalized motion window
  global_text:      str - randomly chosen HumanML3D caption for this sample
  local_clip_enc:   (T_latent, 512) - CLIP embeddings downsampled to latent rate
  has_local:        bool - whether this sample has local CLIP embeddings
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
import random
import codecs as cs
from tqdm import tqdm


class MSAVAEDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4,
                 use_ft_split=True):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name in ('t2m_272', 't2m_babel_272'):
            self.data_root = './humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.clip_enc_dir = pjoin(self.data_root, 'clip_enc_single')
            self.meta_dir = pjoin(self.data_root, 'mean_std')
            self.joints_num = 22
            self.max_motion_length = 300
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

        # Pre-load motion and text (small), but LAZY-load clip_enc in __getitem__
        self.data = []
        self.lengths = []
        n_with_local = 0

        for name in tqdm(id_list, desc='Loading MSA-VAE data'):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue

                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                with cs.open(text_path, 'r') as f:
                    text_lines = f.readlines()

                global_captions = []
                for line in text_lines:
                    parts = line.strip().split('#')
                    if len(parts) >= 4:
                        caption = parts[0].strip()
                        f_tag = float(parts[2]) if parts[2] != 'nan' else 0.0
                        to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
                        if f_tag == 0.0 and to_tag == 0.0 and caption:
                            global_captions.append(caption)
                if not global_captions:
                    parts = text_lines[0].strip().split('#')
                    global_captions = [parts[0].strip()]

                clip_path = pjoin(self.clip_enc_dir, name + '.npy')
                has_local = os.path.exists(clip_path)
                if has_local:
                    n_with_local += 1

                entry = {
                    'name': name,
                    'motion': motion,
                    'captions': global_captions,
                    'clip_path': clip_path if has_local else None,
                    'has_local': has_local,
                }
                self.data.append(entry)
                self.lengths.append(motion.shape[0] - self.window_size)
            except Exception:
                pass

        self.mean = mean
        self.std = std
        print(f'MSA-VAE training: {len(self.data)} samples '
              f'({n_with_local} with local CLIP embeddings)')

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        motion = entry['motion']

        # Random crop a window
        idx = random.randint(0, len(motion) - self.window_size)
        motion_window = motion[idx:idx + self.window_size]

        # Normalize
        motion_window = (motion_window - self.mean) / self.std

        # Random global caption
        caption = random.choice(entry['captions'])

        # Local CLIP embeddings: lazy-load + upsample + crop + pool
        has_local = entry['has_local']
        latent_len = self.window_size // self.unit_length

        if has_local:
            clip_enc_20 = np.load(entry['clip_path'])  # (T_20fps, 512)
            T_20 = clip_enc_20.shape[0]
            T_30 = len(motion)
            # Upsample 20fps -> 30fps via nearest-neighbor
            indices_30 = np.round(np.linspace(0, T_20 - 1, T_30)).astype(int)
            clip_enc_30 = clip_enc_20[indices_30]
            # Crop to window
            local_clip_window = clip_enc_30[idx:idx + self.window_size]  # (64, 512)
            # Mean-pool all 64 frames -> single semantic vector for Spotlight
            local_clip_pooled = local_clip_window.mean(axis=0)  # (512,)
            # Average-pool to latent rate for local alignment loss
            local_clip_latent = _pool_to_latent(local_clip_window, latent_len)
        else:
            local_clip_latent = np.zeros((latent_len, 512), dtype=np.float32)
            local_clip_pooled = np.zeros((512,), dtype=np.float32)

        return (
            motion_window.astype(np.float32),
            caption,
            local_clip_latent.astype(np.float32),
            has_local,
            len(motion),  # total_frames for spotlight alpha
            local_clip_pooled.astype(np.float32),  # (512,) mean of 64 raw frames
        )


def _pool_to_latent(clip_window, latent_len):
    """Average-pool frame-level CLIP (T, 512) to (latent_len, 512)."""
    T = clip_window.shape[0]
    if T == latent_len:
        return clip_window
    indices = np.linspace(0, T, latent_len + 1).astype(int)
    pooled = np.zeros((latent_len, clip_window.shape[1]), dtype=np.float32)
    for i in range(latent_len):
        pooled[i] = clip_window[indices[i]:indices[i + 1]].mean(axis=0)
    return pooled


def collate_fn(batch):
    """Custom collate: tensors + strings + bools + total_frames + pooled local."""
    motions, captions, local_clips, has_locals, total_frames, local_pooled = zip(*batch)
    motions = torch.from_numpy(np.stack(motions))
    local_clips = torch.from_numpy(np.stack(local_clips))
    has_locals = torch.tensor(has_locals, dtype=torch.bool)
    total_frames = torch.tensor(total_frames, dtype=torch.long)
    local_pooled = torch.from_numpy(np.stack(local_pooled))
    return motions, list(captions), local_clips, has_locals, total_frames, local_pooled


def DATALoader(dataset_name, batch_size, num_workers=8,
               window_size=64, unit_length=4, use_ft_split=True):
    trainSet = MSAVAEDataset(
        dataset_name, window_size=window_size,
        unit_length=unit_length, use_ft_split=use_ft_split,
    )
    train_loader = torch.utils.data.DataLoader(
        trainSet, batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
