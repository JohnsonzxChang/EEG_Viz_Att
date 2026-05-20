"""
EEG Data Augmentation
=====================
Wraps an existing EEG dataset to apply online augmentations during training.

Three augmentations designed for ERP-locked EEG:
1. Gaussian noise injection  — regularises without changing class membership
2. Channel dropout           — prevents reliance on specific electrodes
3. Temporal smoothing        — simulates mild preprocessing variation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AugmentedEEGDataset(Dataset):
    """Wrap an existing dataset and apply EEG augmentations in __getitem__.

    Only augments during training (is_train=True).  Validation wrapping
    with is_train=False passes data through unchanged.

    Args:
        dataset:      base dataset returning (data, label, subject_id)
        is_train:     whether to apply augmentations
        noise_std:    Gaussian noise as fraction of per-channel std (default 0.05)
        chan_drop:     max number of channels to zero out (default 3)
        smooth_k:     temporal smoothing kernel size (default 3, 0 to disable)
        noise_prob:   probability of applying noise (default 0.5)
        drop_prob:    probability of applying channel dropout (default 0.5)
        smooth_prob:  probability of applying smoothing (default 0.3)
    """

    def __init__(self, dataset, is_train=True, noise_std=0.05, chan_drop=3,
                 smooth_k=3, noise_prob=0.5, drop_prob=0.5, smooth_prob=0.3):
        self.dataset = dataset
        self.is_train = is_train
        self.noise_std = noise_std
        self.chan_drop = chan_drop
        self.smooth_k = smooth_k
        self.noise_prob = noise_prob
        self.drop_prob = drop_prob
        self.smooth_prob = smooth_prob

        # pre-compute Gaussian smoothing kernel (fixed, not learned)
        if smooth_k > 0:
            k = torch.ones(1, 1, smooth_k) / smooth_k
            self.register_buffer_kernel = k  # stored as attribute, not nn.Parameter
        else:
            self.register_buffer_kernel = None

        # proxy attributes that the training pipeline may read
        if hasattr(dataset, 'pos_weight'):
            self.pos_weight = dataset.pos_weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label, subject_id = self.dataset[idx]

        if self.is_train:
            data = self._augment(data)

        return data, label, subject_id

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a single EEG sample (C, T)."""
        # 1. Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x = x + torch.randn_like(x) * (self.noise_std * std)

        # 2. Channel dropout
        if torch.rand(1).item() < self.drop_prob and self.chan_drop > 0:
            n_drop = torch.randint(1, self.chan_drop + 1, (1,)).item()
            channels = torch.randperm(x.size(0))[:n_drop]
            x[channels] = 0.0

        # 3. Temporal smoothing (moving average)
        if (torch.rand(1).item() < self.smooth_prob
                and self.register_buffer_kernel is not None):
            C = x.size(0)
            kernel = self.register_buffer_kernel.to(x.device)
            # apply per-channel: (C, T) → (C, 1, T) → conv1d → (C, T)
            pad = self.smooth_k // 2
            x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')
            x = F.conv1d(x_padded, kernel).squeeze(1)

        return x

    # ── Proxy methods for pipeline compatibility ─────────────────────

    def get_flag(self, flag):
        """Delegate to underlying dataset."""
        self.dataset.get_flag(flag)
        if hasattr(self.dataset, 'pos_weight'):
            self.pos_weight = self.dataset.pos_weight
