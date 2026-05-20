"""
Multi-Label Stratified Batch Sampler
====================================
Ensures each batch contains forced anchor samples from specific classes,
guaranteeing Circle Loss always has positive pairs for rare classes.

Algorithm (per epoch):
  1. Shuffle the C class indices.
  2. For each class c in shuffled order:
     a. Sample min(K_anchor, |pool_c|) indices from samples containing class c.
     b. Fill rest of batch with inverse-frequency weighted random draw.
     c. Yield the batch.
  3. Stop after num_batches batches.

This solves the "missing pair" problem: rare classes like toaster (~10 samples)
are guaranteed to have at least K_anchor=2 co-occurring samples per batch,
providing Circle Loss with meaningful positive pairs.
"""

import math
import numpy as np
from torch.utils.data import Sampler


class MultiLabelStratifiedBatchSampler(Sampler):
    """Batch sampler that forces rare-class coverage for metric learning.

    Args:
        labels:       (N, C) numpy float32 multi-hot label matrix.
        batch_size:   int, target batch size.
        K_anchor:     int, forced anchor samples per class per batch. Default 2.
        num_batches:  int or None, batches per epoch. None = ceil(N / batch_size).
    """

    def __init__(self, labels: np.ndarray, batch_size: int,
                 K_anchor: int = 2, num_batches: int = None):
        self.labels = labels
        self.N, self.C = labels.shape
        self.batch_size = batch_size
        self.K_anchor = K_anchor
        self.num_batches = num_batches or math.ceil(self.N / batch_size)

        # Pre-compute per-class index lists
        self.class_indices = []
        for c in range(self.C):
            idx = np.where(labels[:, c] > 0)[0]
            self.class_indices.append(idx)

        # Inverse-frequency sample weights for random fill
        class_freq = labels.sum(axis=0).clip(min=1)          # (C,)
        # Each sample's weight = max inverse-freq among its active labels
        inv_freq = 1.0 / class_freq                           # (C,)
        rare_weight = (labels * inv_freq[None, :]).max(axis=1) # (N,)
        rare_weight = np.where(rare_weight > 0, rare_weight, 1.0 / self.N)
        self.sample_weights = rare_weight / rare_weight.sum()  # (N,) normalized

    def __iter__(self):
        class_order = np.random.permutation(self.C)
        all_indices = np.arange(self.N)
        batches_yielded = 0

        for c in class_order:
            if batches_yielded >= self.num_batches:
                break

            pool = self.class_indices[c]
            if len(pool) == 0:
                continue

            # Force K_anchor samples from this class
            k = min(self.K_anchor, len(pool))
            anchors = np.random.choice(pool, size=k, replace=False)

            # Fill rest with weighted random draw (exclude anchors)
            n_fill = self.batch_size - k
            mask = np.ones(self.N, dtype=bool)
            mask[anchors] = False
            fill_weights = self.sample_weights.copy()
            fill_weights[~mask] = 0.0
            w_sum = fill_weights.sum()
            if w_sum > 0:
                fill_weights /= w_sum
            else:
                fill_weights = np.ones(self.N) / self.N

            n_fill = min(n_fill, mask.sum())
            fill = np.random.choice(all_indices, size=n_fill,
                                    replace=False, p=fill_weights)

            batch = np.concatenate([anchors, fill])
            np.random.shuffle(batch)
            yield batch.tolist()
            batches_yielded += 1

    def __len__(self):
        return self.num_batches
