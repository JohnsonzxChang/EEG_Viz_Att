"""PyTorch Dataset that joins EpochBundle + cached image embeddings (+ patch tokens)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from process1_data_process.data_io import EpochBundle


class EEGImgAttentionDataset(Dataset):
    """Each sample = (eeg, img_emb, patch_tokens, hint_label, idx).

    `patch_tokens` may be a zero tensor of shape (1,1) if not in cache.
    """
    def __init__(self, bundle: EpochBundle, img_cache: dict,
                 split: str | None = None, normalize: bool = True,
                 want_patches: bool = True) -> None:
        if split is not None:
            mask = bundle.eeg_split == split
        else:
            mask = np.ones(len(bundle), dtype=bool)
        idx = np.nonzero(mask)[0]

        # EEG: drop gaze channels
        ch_keep = [i for i, n in enumerate(bundle.ch_names)
                   if n not in ("gaze_x", "gaze_y")]
        X = bundle.data[idx][:, ch_keep, :].astype(np.float32)
        if normalize:
            mu = X.mean(axis=(0, 2), keepdims=True)
            sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
            X = (X - mu) / sd
        self.eeg = X

        # Hint labels
        classes = sorted(set(bundle.hint.tolist()))
        # Drop empty string class if present (failed-join trials)
        classes = [c for c in classes if c != ""]
        self.classes = classes
        self.cls2id = {c: i for i, c in enumerate(classes)}

        # Filter trials to those with a known hint
        keep_local = np.array([bundle.hint[ei] in self.cls2id for ei in idx])
        idx = idx[keep_local]
        self.eeg = self.eeg[keep_local]

        self.y = np.array([self.cls2id[bundle.hint[ei]] for ei in idx],
                          dtype=np.int64)

        # Global image features
        ids_in_cache = {str(s): i for i, s
                        in enumerate(img_cache["image_ids"].tolist())}
        feats = img_cache["features"]
        img_dim = feats.shape[1]
        img_emb = np.zeros((len(idx), img_dim), dtype=np.float32)

        patches = img_cache["patch_tokens"] if (want_patches and "patch_tokens" in img_cache.files) else None
        if patches is not None:
            P, Dp = patches.shape[1], patches.shape[2]
            img_pat = np.zeros((len(idx), P, Dp), dtype=np.float32)
        else:
            img_pat = np.zeros((len(idx), 1, 1), dtype=np.float32)

        missing = 0
        for k, ei in enumerate(idx):
            iid = str(bundle.image_id[ei])
            j = ids_in_cache.get(iid)
            if j is None:
                missing += 1
                continue
            img_emb[k] = feats[j]
            if patches is not None:
                img_pat[k] = patches[j]
        self.img = img_emb
        self.img_patches = img_pat
        self.n_missing_img = missing
        self.grid = tuple(img_cache["grid"].tolist()) if "grid" in img_cache.files else None
        self.image_id = bundle.image_id[idx]
        self.hint = bundle.hint[idx]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.eeg[i]),
                torch.from_numpy(self.img[i]),
                torch.from_numpy(self.img_patches[i]),
                int(self.y[i]),
                int(i))
