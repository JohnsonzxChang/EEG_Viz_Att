"""
RSVP-COCO Dataset for EEG classification and CLIP-based retrieval.

Loads RSVP EEG epochs, performs ERP averaging per image,
and provides CLIP embeddings for contrastive learning.

Marker format in .fif: "image_id/category"
Data: 32 EEG channels, tmin=-0.1s, tmax=+0.6s, 1000Hz
360 unique images = 12 categories x 30 images x 10-20 repeats
"""

import os
import sys
import gc
import random
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

RSVP_CATEGORIES = [
    "dog", "cat", "car", "chair", "banana", "pizza",
    "cup", "couch", "bed", "laptop", "teddy bear", "umbrella",
]
N_RSVP_CLASSES = len(RSVP_CATEGORIES)


class Dataset_RSVP_COCO(Dataset):
    """RSVP-COCO EEG dataset with ERP averaging and optional CLIP embeddings.

    Compatible with ``get_data_loader_cutt`` via the ``get_flag`` API.
    """

    def __init__(self, args, seeds=None):
        super().__init__()
        self.args = args
        if seeds is not None:
            np.random.seed(seeds)

        fif_path = getattr(args, 'rsvp_fif_path', None)
        assert fif_path and os.path.exists(fif_path), f"RSVP fif not found: {fif_path}"

        erp_k = getattr(args, 'erp_k', None)  # None = avg ALL, 0 = raw trials
        use_post_stim = getattr(args, 'use_post_stim', True)
        raw_trials = getattr(args, 'raw_trials', False) or (erp_k == 0)

        # ── Load epochs ──────────────────────────────────────────────────
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        raw_data = epochs.get_data().astype(np.float32) * 1e6  # → µV
        events = epochs.events[:, 2]
        code_to_name = {code: name for name, code in epochs.event_id.items()}
        sfreq = epochs.info['sfreq']
        times = epochs.times

        # ── Parse markers ────────────────────────────────────────────────
        cat2idx = {c: i for i, c in enumerate(RSVP_CATEGORIES)}
        img_ids, labels = [], []
        for i in range(len(raw_data)):
            name = code_to_name[events[i]]
            parts = name.split('/')
            img_ids.append(int(parts[0]))
            labels.append(cat2idx[parts[1]])
        img_ids = np.array(img_ids)
        labels = np.array(labels)

        if raw_trials:
            # ── RAW TRIAL MODE: use all epochs directly ──────────────────
            out_data = raw_data
            out_labels = labels
            out_img_ids = img_ids
            print(f"[RSVP] Raw trial mode: {len(out_data)} trials (no ERP averaging)")
        else:
            # ── ERP AVERAGE MODE ─────────────────────────────────────────
            groups = defaultdict(list)
            for i in range(len(raw_data)):
                groups[(img_ids[i], labels[i])].append(i)

            avg_data, avg_labels, avg_img_ids = [], [], []
            for (img_id, lbl), indices in groups.items():
                trials = raw_data[indices]
                if erp_k is not None and erp_k < len(indices):
                    sel = np.random.choice(len(indices), size=erp_k, replace=False)
                    trials = trials[sel]
                avg_data.append(trials.mean(axis=0))
                avg_labels.append(lbl)
                avg_img_ids.append(img_id)

            out_data = np.stack(avg_data)
            out_labels = np.array(avg_labels)
            out_img_ids = np.array(avg_img_ids)
            print(f"[RSVP] ERP avg mode: {len(out_data)} images "
                  f"(erp_k={'all' if erp_k is None else erp_k})")

        # ── Time-window selection ────────────────────────────────────────
        t_len = getattr(args, 't_len', 500)
        if use_post_stim:
            onset_idx = int(round(-times[0] * sfreq))
            t_end = min(onset_idx + t_len, out_data.shape[2])
            self.all_data = out_data[:, :, onset_idx:t_end].astype(np.float32)
        else:
            self.all_data = out_data[:, :, :t_len].astype(np.float32)

        actual_t = self.all_data.shape[2]
        if actual_t < t_len:
            print(f"[RSVP] Warning: actual T={actual_t} < requested t_len={t_len}")

        self.all_labels = out_labels
        self.all_img_ids = out_img_ids
        self.all_subjects = np.zeros(len(out_data), dtype=int)

        # ── CLIP embeddings ──────────────────────────────────────────────
        self.all_img_emb = None
        self.all_cap_emb = None
        self._load_clip_embeddings()

        # ── Train / val split ────────────────────────────────────────────
        if raw_trials:
            # Split by IMAGE_ID to prevent data leakage
            unique_imgs = sorted(set(out_img_ids.tolist()))
            n_val_imgs = max(1, int(0.2 * len(unique_imgs)))
            val_imgs = set(np.random.choice(
                unique_imgs, size=n_val_imgs, replace=False).tolist())
            self.val_sel = [i for i in range(len(self.all_data))
                           if int(self.all_img_ids[i]) in val_imgs]
            self.trn_sel = [i for i in range(len(self.all_data))
                           if int(self.all_img_ids[i]) not in val_imgs]
        else:
            self.trn_sel = list(range(len(self.all_data)))
            self.val_sel = list(np.random.choice(
                self.trn_sel, size=int(0.2 * len(self.trn_sel)), replace=False))
            self.trn_sel = sorted(set(self.trn_sel) - set(self.val_sel))
        self.flag = None

        print(f"[RSVP] {len(self.all_data)} samples | "
              f"train={len(self.trn_sel)} val={len(self.val_sel)} | "
              f"shape={self.all_data.shape} | {N_RSVP_CLASSES} classes")

        del epochs, raw_data
        gc.collect()

    # ── CLIP embedding helpers ───────────────────────────────────────────

    def _load_clip_embeddings(self):
        """Load pre-computed CLIP cache; compute online for missing images."""
        cache_path = getattr(self.args, 'clip_cache_path', None)

        # 1) Try loading an existing cache
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if cache_path and os.path.isfile(cache_path):
            npz = np.load(cache_path, allow_pickle=True)
            ids_arr = npz['img_ids'].astype(int)
            img_arr = npz['img_emb'].astype(np.float32)
            cap_arr = npz.get('cap_emb', img_arr).astype(np.float32)
            for i, iid in enumerate(ids_arr):
                cache[int(iid)] = (img_arr[i], cap_arr[i])
            print(f"[RSVP] Loaded {len(cache)} CLIP embeds from {cache_path}")

        # 2) Identify missing ids
        unique_ids = sorted(set(int(x) for x in self.all_img_ids))
        missing = [iid for iid in unique_ids if iid not in cache]

        if missing:
            print(f"[RSVP] Computing CLIP for {len(missing)}/{len(unique_ids)} images ...")
            self._compute_clip_online(missing, cache)
            # Save updated cache
            save_path = cache_path or os.path.join(
                os.path.dirname(getattr(self.args, 'rsvp_fif_path', '.')),
                'rsvp_clip_cache.npz')
            all_ids = sorted(cache.keys())
            np.savez(save_path,
                     img_ids=np.array(all_ids),
                     img_emb=np.stack([cache[i][0] for i in all_ids]),
                     cap_emb=np.stack([cache[i][1] for i in all_ids]))
            print(f"[RSVP] Saved CLIP cache → {save_path}")
            # Update cache_path for future use
            if not cache_path:
                self.args.clip_cache_path = save_path

        # 3) Materialize per-sample
        self.all_img_emb = np.stack(
            [cache[int(i)][0] for i in self.all_img_ids]).astype(np.float32)
        self.all_cap_emb = np.stack(
            [cache[int(i)][1] for i in self.all_img_ids]).astype(np.float32)

    def _compute_clip_online(self, missing_ids, cache):
        """Compute CLIP embeddings for images not in cache."""
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            print("[RSVP] WARNING: transformers unavailable → random CLIP embeds")
            dim = getattr(self.args, 'proj_dim', 768)
            for iid in missing_ids:
                v = np.random.randn(dim).astype(np.float32)
                cache[iid] = (v / np.linalg.norm(v), v / np.linalg.norm(v))
            return

        from PIL import Image

        clip_name = getattr(self.args, 'clip_model_name',
                            'openai/clip-vit-large-patch14')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(clip_name)

        img_root = getattr(self.args, 'coco_img_path', None)
        if img_root is None:
            if 'win' in sys.platform:
                img_root = None  # <SET YOUR DATA PATH>
            else:
                img_root = None  # set via config.coco_img_path

        # Captions (optional)
        coco_cap = None
        cap_file = os.path.join(os.environ.get('DATA_ROOT', os.getcwd()),
                                'data', 'coco', 'annotations',
                                'captions_train2017.json')
        if os.path.isfile(cap_file):
            from pycocotools.coco import COCO
            coco_cap = COCO(cap_file)

        for idx, iid in enumerate(missing_ids):
            fname = str(int(iid)).zfill(12) + '.jpg'
            path = os.path.join(img_root, fname)
            if not os.path.isfile(path):
                dim = getattr(self.args, 'proj_dim', 768)
                cache[iid] = (np.zeros(dim, np.float32),
                              np.zeros(dim, np.float32))
                continue

            img = Image.open(path).convert('RGB')

            caption = ""
            if coco_cap:
                ann_ids = coco_cap.getAnnIds(imgIds=[int(iid)])
                anns = coco_cap.loadAnns(ann_ids)
                if anns:
                    caption = anns[0].get('caption', '')

            with torch.no_grad():
                inp_img = processor(images=img, return_tensors='pt').to(device)
                ie = model.get_image_features(**inp_img)
                ie = ie / ie.norm(dim=-1, keepdim=True)

                if caption:
                    inp_txt = processor(text=caption, return_tensors='pt',
                                        padding=True, truncation=True).to(device)
                    ce = model.get_text_features(**inp_txt)
                    ce = ce / ce.norm(dim=-1, keepdim=True)
                else:
                    ce = ie

            cache[iid] = (ie[0].cpu().numpy(), ce[0].cpu().numpy())
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(missing_ids)}] CLIP done")

        del model, processor
        torch.cuda.empty_cache()
        gc.collect()

    # ── Framework API ────────────────────────────────────────────────────

    def get_flag(self, flag: str):
        assert flag in ['trn', 'val'], f"Bad flag: {flag}"
        sel = self.trn_sel if flag == 'trn' else self.val_sel
        self.flag = sel
        self.all_data = self.all_data[sel]
        self.all_labels = self.all_labels[sel]
        self.all_img_ids = self.all_img_ids[sel]
        self.all_subjects = self.all_subjects[sel]
        self.all_img_emb = self.all_img_emb[sel]
        self.all_cap_emb = self.all_cap_emb[sel]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.all_data[idx].copy())
        label = torch.tensor(int(self.all_labels[idx]), dtype=torch.long)
        subjects = torch.tensor(int(self.all_subjects[idx]), dtype=torch.long)
        img_emb = torch.from_numpy(self.all_img_emb[idx])
        cap_emb = torch.from_numpy(self.all_cap_emb[idx])

        # Multi-label one-hot for compatibility with existing tasks
        regs = torch.zeros(N_RSVP_CLASSES, dtype=torch.float32)
        regs[int(self.all_labels[idx])] = 1.0

        # Augmentation (training only)
        if self.flag is self.trn_sel:
            data = data + torch.randn_like(data) * data.std() * 0.1
            T = data.shape[-1]
            mask_len = max(1, int(T * 0.1))
            t_start = random.randint(0, T - mask_len)
            data[:, t_start:t_start + mask_len] = 0.0

        return {
            "data": data,
            "regs": regs,
            "subjects": subjects,
            "label": label,
            "img_emb": img_emb,
            "cap_emb": cap_emb,
        }
