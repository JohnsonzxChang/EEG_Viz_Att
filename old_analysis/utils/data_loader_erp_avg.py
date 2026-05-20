"""
ERP-Averaged & K-trial Sub-averaged Dataset for EEG-COCO Classification.

References:
  - NICE (Song et al., 2023): K-trial sub-averaging for SNR improvement
  - ATM  (Li et al., NeurIPS 2024): ERP averaging per image for visual decoding
  - EEG-CLIP (Li et al., 2024): Contrastive EEG→CLIP alignment

Two modes:
  1) erp_mode='image_avg'  — average ALL trials per image_id → 1 sample per image
     Maximizes SNR but reduces dataset size.
  2) erp_mode='ktrial_avg' — randomly sample K trials of the same image, average them
     Generates multiple augmented samples with moderate SNR improvement.
  3) erp_mode='both'       — use image-level averaging for validation,
     K-trial sub-averaging for training (best of both worlds).

Can optionally load pre-computed CLIP embeddings per image for fusion.
"""

import os
import sys
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from sklearn.preprocessing import MultiLabelBinarizer

from conf import BaseConfig
from .data_loader import ALL_CLA

ROOT_PATH = os.environ.get('DATA_ROOT', os.getcwd())
UDF_VIZ_PATH = f'{ROOT_PATH}/data/udf_viz'


class Dataset_UDF_VIZ_ERP_Avg(Dataset):
    """EEG dataset with ERP averaging strategies + optional CLIP embeddings.

    Args:
        args: BaseConfig with extra fields:
            erp_mode:   'image_avg' | 'ktrial_avg' | 'both'
            erp_k:      int, number of trials to sub-average (for ktrial mode)
            erp_n_aug:  int, number of augmented samples per image (for ktrial mode)
            clip_cache_path: str, path to precomputed CLIP embeddings (.npz)
            use_clip_feat:   bool, whether to load and return CLIP features
        seeds: random seed for reproducibility
    """

    def __init__(self, args: BaseConfig, seeds: int = None):
        super().__init__()
        if seeds is not None:
            np.random.seed(seeds)

        self.args = args
        self.erp_mode = getattr(args, 'erp_mode', 'image_avg')
        self.erp_k = getattr(args, 'erp_k', 4)
        self.erp_n_aug = getattr(args, 'erp_n_aug', 5)
        self.use_clip_feat = getattr(args, 'use_clip_feat', True)

        assert os.path.exists(UDF_VIZ_PATH), f"Dataset path {UDF_VIZ_PATH} does not exist"

        # --- Subject file list ---
        all_files = [os.path.join(UDF_VIZ_PATH, 'zx-1122', 'erp.fif')]
        file_balance = [20]

        # --- COCO setup ---
        coco = COCO(os.path.join(ROOT_PATH, 'data', 'coco', 'annotations', 'instances_train2017.json'))
        mlb = MultiLabelBinarizer(classes=np.arange(len(ALL_CLA)))
        mlb.fit([[]])

        # ── Phase 1: Collect all trials grouped by image_id ──
        # {img_id: {'data': [trial1, trial2, ...], 'label': multi-hot}}
        self.img_trials = {}

        for i, file in enumerate(all_files):
            epochs = mne.read_epochs(file, preload=True, verbose=False)
            print(f"Loaded {file} with {len(epochs)} epochs, tmin={epochs.tmin}, tmax={epochs.tmax}")

            for tgt in ALL_CLA:
                ep = epochs[tgt]
                code_to_name = {code: name for name, code in ep.event_id.items()}
                event_codes = ep.events[:, 2]
                raw_data = ep.get_data() * 1e6  # (n_trials, n_chns, n_times)

                # Balance sampling (same as original)
                n_trials = raw_data.shape[0]
                if n_trials >= file_balance[i]:
                    sel = np.random.choice(n_trials, size=file_balance[i], replace=False)
                else:
                    sel = np.arange(n_trials)

                for idx in sel:
                    code = event_codes[idx]
                    img_name = code_to_name[code]
                    img_id = int(img_name.split('/')[-1].split('.')[0])
                    trial_data = raw_data[idx]  # (n_chns, n_times)

                    if img_id not in self.img_trials:
                        # Compute multi-label for this image
                        ann_ids = coco.getAnnIds(imgIds=[img_id])
                        anns = coco.loadAnns(ann_ids)
                        cat_ids = sorted({ann["category_id"] for ann in anns})
                        cats = coco.loadCats(cat_ids)
                        cat_indices = [ALL_CLA.index(c["name"]) for c in cats if c["name"] in ALL_CLA]
                        label = mlb.transform([cat_indices])[0]  # (80,)
                        self.img_trials[img_id] = {
                            'data': [],
                            'label': label,
                            'subject': i,
                        }
                    self.img_trials[img_id]['data'].append(trial_data)

        # Convert trial lists to arrays
        for img_id in self.img_trials:
            self.img_trials[img_id]['data'] = np.stack(
                self.img_trials[img_id]['data'], axis=0
            )  # (n_trials_for_img, n_chns, n_times)

        img_ids_sorted = sorted(self.img_trials.keys())
        print(f"Total unique images: {len(img_ids_sorted)}")
        trial_counts = [self.img_trials[iid]['data'].shape[0] for iid in img_ids_sorted]
        print(f"Trials per image: min={min(trial_counts)}, max={max(trial_counts)}, "
              f"mean={np.mean(trial_counts):.1f}, median={np.median(trial_counts):.0f}")

        # ── Phase 2: Build samples based on erp_mode ──
        # Store raw grouped data for train/val split, then build samples in get_flag()
        self._img_ids_sorted = img_ids_sorted

        # Train/val split by IMAGE (not by trial)
        N = len(img_ids_sorted)
        all_idx = list(range(N))
        val_idx = list(np.random.choice(all_idx, size=int(0.2 * N), replace=False))
        trn_idx = list(set(all_idx) - set(val_idx))
        self.trn_img_ids = [img_ids_sorted[i] for i in sorted(trn_idx)]
        self.val_img_ids = [img_ids_sorted[i] for i in sorted(val_idx)]
        print(f"Train images: {len(self.trn_img_ids)}, Val images: {len(self.val_img_ids)}")

        # ── Phase 3: Load CLIP embeddings if requested ──
        self._clip_cache = {}
        if self.use_clip_feat:
            clip_path = getattr(args, 'clip_cache_path', None)
            if clip_path is None:
                clip_model_name = getattr(args, "clip_model_name", "openai/clip-vit-large-patch14")
                model_tag = clip_model_name.replace("/", "-").replace(" ", "")
                cand = os.path.join("./data/coco/processed_train2017", f"clip_embeds_{model_tag}.npz")
                if os.path.isfile(cand):
                    clip_path = cand
                # Also check subject-specific path
                cand2 = os.path.join("./data/coco/processed_train2017", "zx",
                                     f"clip_embeds_{model_tag}.npz")
                if os.path.isfile(cand2):
                    clip_path = cand2

            if clip_path and os.path.isfile(clip_path):
                print(f"Loading CLIP embeddings from: {clip_path}")
                npz = np.load(clip_path, allow_pickle=True)
                cache_ids = npz["img_ids"].astype(int)
                cache_img = npz["img_emb"].astype(np.float32)
                cache_cap = npz["cap_emb"].astype(np.float32)
                for j, iid in enumerate(cache_ids.tolist()):
                    self._clip_cache[int(iid)] = (cache_img[j], cache_cap[j])
                print(f"CLIP cache loaded: {len(self._clip_cache)} images")
            else:
                print(f"WARNING: CLIP cache not found at {clip_path}, disabling CLIP features")
                self.use_clip_feat = False

        # Placeholder — filled by get_flag()
        self.all_data = None
        self.all_regs = None
        self.all_img = None
        self.all_subjects = None
        self.all_clip_img = None
        self.all_clip_cap = None
        self.pos_weight = None
        self.flag = None
        self._is_train = False

    def _extract_window(self, data: np.ndarray) -> np.ndarray:
        """Extract temporal window and select channels.
        data: (..., n_chns, n_times) → (..., n_sel_chns, t_len)
        """
        # Channel selection
        data = data[..., self.args.chn_sel, :]

        t_len = self.args.t_len
        mid = self.args.t0 + t_len // 2
        start = mid - t_len // 2
        end = mid + t_len // 2
        return data[..., start:end].astype(np.float32)

    def _build_image_avg_samples(self, img_ids):
        """Average ALL trials per image → 1 high-SNR sample per image."""
        data_list, label_list, img_list, subj_list = [], [], [], []
        clip_img_list, clip_cap_list = [], []

        for img_id in img_ids:
            info = self.img_trials[img_id]
            trials = info['data']  # (n_trials, chns, times)
            erp = trials.mean(axis=0)  # (chns, times) — ERP average

            erp_win = self._extract_window(erp)  # (sel_chns, t_len)
            data_list.append(erp_win)
            label_list.append(info['label'].astype(np.float32))
            img_list.append(img_id)
            subj_list.append(info['subject'])

            if self.use_clip_feat and img_id in self._clip_cache:
                ci, cc = self._clip_cache[img_id]
                clip_img_list.append(ci)
                clip_cap_list.append(cc)
            elif self.use_clip_feat:
                # Zero padding for missing
                dim = 768
                clip_img_list.append(np.zeros(dim, dtype=np.float32))
                clip_cap_list.append(np.zeros(dim, dtype=np.float32))

        result = {
            'data': np.stack(data_list),
            'regs': np.stack(label_list),
            'img': np.array(img_list, dtype=int),
            'subjects': np.array(subj_list, dtype=int),
        }
        if self.use_clip_feat:
            result['clip_img'] = np.stack(clip_img_list)
            result['clip_cap'] = np.stack(clip_cap_list)
        return result

    def _build_ktrial_avg_samples(self, img_ids):
        """K-trial sub-averaging: for each image, create n_aug averaged samples.
        Each sample averages K randomly chosen trials (with replacement if needed).
        This is a data augmentation strategy that improves SNR while preserving dataset size.
        """
        K = self.erp_k
        n_aug = self.erp_n_aug

        data_list, label_list, img_list, subj_list = [], [], [], []
        clip_img_list, clip_cap_list = [], []

        for img_id in img_ids:
            info = self.img_trials[img_id]
            trials = info['data']  # (n_trials, chns, times)
            n_trials = trials.shape[0]

            # Determine how many augmented samples
            n_samples = n_aug if n_trials >= K else 1

            for _ in range(n_samples):
                if n_trials >= K:
                    sel = np.random.choice(n_trials, size=K, replace=False)
                else:
                    # If fewer trials than K, use all trials
                    sel = np.arange(n_trials)
                avg = trials[sel].mean(axis=0)  # (chns, times)
                avg_win = self._extract_window(avg)

                data_list.append(avg_win)
                label_list.append(info['label'].astype(np.float32))
                img_list.append(img_id)
                subj_list.append(info['subject'])

                if self.use_clip_feat and img_id in self._clip_cache:
                    ci, cc = self._clip_cache[img_id]
                    clip_img_list.append(ci)
                    clip_cap_list.append(cc)
                elif self.use_clip_feat:
                    dim = 768
                    clip_img_list.append(np.zeros(dim, dtype=np.float32))
                    clip_cap_list.append(np.zeros(dim, dtype=np.float32))

        result = {
            'data': np.stack(data_list),
            'regs': np.stack(label_list),
            'img': np.array(img_list, dtype=int),
            'subjects': np.array(subj_list, dtype=int),
        }
        if self.use_clip_feat:
            result['clip_img'] = np.stack(clip_img_list)
            result['clip_cap'] = np.stack(clip_cap_list)
        return result

    def get_flag(self, flag: str):
        """Build the actual sample arrays based on flag and erp_mode."""
        assert flag in ['trn', 'val'], f"Flag must be 'trn' or 'val', got {flag}"
        self._is_train = (flag == 'trn')

        if flag == 'trn':
            img_ids = self.trn_img_ids
        else:
            img_ids = self.val_img_ids

        # Choose averaging strategy
        if self.erp_mode == 'image_avg':
            samples = self._build_image_avg_samples(img_ids)
        elif self.erp_mode == 'ktrial_avg':
            samples = self._build_ktrial_avg_samples(img_ids)
        elif self.erp_mode == 'both':
            if flag == 'trn':
                # Training: K-trial sub-averaging (data augmentation)
                samples = self._build_ktrial_avg_samples(img_ids)
            else:
                # Validation: full image averaging (best SNR)
                samples = self._build_image_avg_samples(img_ids)
        else:
            raise ValueError(f"Unknown erp_mode: {self.erp_mode}")

        self.all_data = samples['data']
        self.all_regs = samples['regs']
        self.all_img = samples['img']
        self.all_subjects = samples['subjects']
        if self.use_clip_feat:
            self.all_clip_img = samples.get('clip_img')
            self.all_clip_cap = samples.get('clip_cap')

        # Compute pos_weight for training set
        if flag == 'trn':
            pos_count = self.all_regs.sum(axis=0)
            neg_count = self.all_regs.shape[0] - pos_count
            self.pos_weight = np.where(pos_count > 0, neg_count / pos_count, 100.0).astype(np.float32)

        print(f"[{flag.upper()}] erp_mode={self.erp_mode}, samples={len(self.all_data)}, "
              f"unique images={len(set(self.all_img.tolist()))}")

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(self.all_data[idx].copy())
        regs = torch.from_numpy(self.all_regs[idx])
        subjects = torch.tensor(int(self.all_subjects[idx]), dtype=torch.long)

        # Training augmentation
        if self._is_train:
            # Gaussian noise (σ = 5% of signal std)
            data = data + torch.randn_like(data) * data.std() * 0.05
            # Time masking: zero out random 5% temporal segment
            T = data.shape[-1]
            mask_len = max(1, int(T * 0.05))
            t_start = torch.randint(0, T - mask_len, (1,)).item()
            data[:, t_start:t_start + mask_len] = 0.0

        result = {
            'data': data,
            'regs': regs,
            'subjects': subjects,
        }

        if self.use_clip_feat and self.all_clip_img is not None:
            result['clip_img'] = torch.from_numpy(self.all_clip_img[idx])
            result['clip_cap'] = torch.from_numpy(self.all_clip_cap[idx])

        return result
