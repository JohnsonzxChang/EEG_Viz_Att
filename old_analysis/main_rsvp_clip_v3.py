#!/usr/bin/env python3
"""
RSVP-COCO V3: Mini-ERP Augmented Training

V2 used erp_k=None (average ALL trials per image) → 360 fixed samples.
V3 uses **dynamic mini-ERP**: each epoch, randomly sample k trials per image
and average them, generating a fresh set of augmented ERPs every epoch.

This combines the best of both worlds:
  - ERP averaging → high SNR (like V2)
  - Random sub-sampling → data augmentation / diversity (like raw trials)
  - Same 360 images per epoch, but different random averages each time

Speed fixes:
  - num_workers=0 on Windows (data already in RAM)
  - No unnecessary .copy() in __getitem__
  - torch.compile for model if available

Compares: V3 mini-ERP vs V2 full-ERP vs ML baseline
"""

import os
import sys
import time
import json
import argparse
import warnings
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             top_k_accuracy_score, confusion_matrix, f1_score)
import mne

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import BaseConfigRSVP
from encoder.atm_encoder import ATM_Encoder
from utils.data_loader_rsvp import RSVP_CATEGORIES, N_RSVP_CLASSES


# ── Baselines from previous experiments ─────────────────────────────────────
ML_BASELINE = {'top1': 0.2389, 'top3': 0.4361, 'bacc': 0.2389}
V2_CE_BEST  = {'top1': 0.3056, 'top3': 0.5139, 'bacc': 0.3003}
V2_CLIP_BEST= {'top1': 0.2778, 'top3': 0.5556, 'bacc': 0.3502}
CHANCE = 1.0 / N_RSVP_CLASSES


# ═══ Mini-ERP Dataset ═════════════════════════════════════════════════════════

class MiniERPDataset(Dataset):
    """Dynamic mini-ERP dataset: each epoch uses fresh random sub-averages.

    For each unique image, randomly samples `erp_k` trials and averages them,
    producing a new set of "augmented ERPs" every time `resample()` is called.

    Val set uses ALL trials averaged (deterministic).
    """

    def __init__(self, fif_path, config, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.erp_k = getattr(config, 'erp_k', 5)
        self.config = config

        # ── Load all epochs ─────────────────────────────────────────────
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        raw_data = epochs.get_data().astype(np.float32) * 1e6  # → µV
        events = epochs.events[:, 2]
        code_to_name = {code: name for name, code in epochs.event_id.items()}
        sfreq = epochs.info['sfreq']
        times = epochs.times

        # ── Parse markers ──────────────────────────────────────────────
        cat2idx = {c: i for i, c in enumerate(RSVP_CATEGORIES)}
        img_ids, labels = [], []
        for i in range(len(raw_data)):
            name = code_to_name[events[i]]
            parts = name.split('/')
            img_ids.append(int(parts[0]))
            labels.append(cat2idx[parts[1]])
        img_ids = np.array(img_ids)
        labels = np.array(labels)

        # ── Time window: post-stimulus only ────────────────────────────
        t_len = getattr(config, 't_len', 500)
        onset_idx = int(round(-times[0] * sfreq))
        t_end = min(onset_idx + t_len, raw_data.shape[2])
        raw_data = raw_data[:, :, onset_idx:t_end].astype(np.float32)
        print(f"[MiniERP] Loaded {len(raw_data)} trials, shape={raw_data.shape}")

        # ── Group trials by image_id ───────────────────────────────────
        self.groups = defaultdict(list)  # img_id → list of trial indices
        self.img_labels = {}             # img_id → label
        for i in range(len(raw_data)):
            iid = int(img_ids[i])
            self.groups[iid].append(i)
            self.img_labels[iid] = int(labels[i])

        self.raw_data = raw_data
        self.all_img_ids = sorted(self.groups.keys())  # 360 unique images
        n_trials = [len(self.groups[iid]) for iid in self.all_img_ids]
        print(f"[MiniERP] {len(self.all_img_ids)} unique images, "
              f"trials/img: min={min(n_trials)} max={max(n_trials)} "
              f"avg={np.mean(n_trials):.1f}")

        # ── Train/val split by image_id ────────────────────────────────
        n_val = max(1, int(0.2 * len(self.all_img_ids)))
        perm = np.random.permutation(len(self.all_img_ids))
        self.val_img_ids = [self.all_img_ids[i] for i in perm[:n_val]]
        self.trn_img_ids = [self.all_img_ids[i] for i in perm[n_val:]]
        print(f"[MiniERP] Train: {len(self.trn_img_ids)} images, "
              f"Val: {len(self.val_img_ids)} images")

        # ── CLIP embeddings ────────────────────────────────────────────
        self._load_clip(config, img_ids)

        # ── Prepare val set (fixed, full average) ──────────────────────
        self.val_data = []
        self.val_labels = []
        self.val_img_emb = []
        self.val_cap_emb = []
        for iid in self.val_img_ids:
            indices = self.groups[iid]
            avg = self.raw_data[indices].mean(axis=0)
            self.val_data.append(avg)
            self.val_labels.append(self.img_labels[iid])
            self.val_img_emb.append(self.clip_cache[iid][0])
            self.val_cap_emb.append(self.clip_cache[iid][1])
        self.val_data = np.stack(self.val_data)
        self.val_labels = np.array(self.val_labels)
        self.val_img_emb = np.stack(self.val_img_emb)
        self.val_cap_emb = np.stack(self.val_cap_emb)

        # ── Initialize train set (will be resampled each epoch) ────────
        self.is_train = True
        self.current_data = None
        self.current_labels = None
        self.current_img_emb = None
        self.current_cap_emb = None
        self.resample()  # initial sample

        del epochs
        gc.collect()

    def _load_clip(self, config, img_ids):
        """Load CLIP embeddings from cache."""
        cache_path = getattr(config, 'clip_cache_path', None)
        if not cache_path:
            cache_path = os.path.join(
                os.path.dirname(config.rsvp_fif_path), 'rsvp_clip_cache.npz')
        self.clip_cache = {}
        if os.path.isfile(cache_path):
            npz = np.load(cache_path, allow_pickle=True)
            ids_arr = npz['img_ids'].astype(int)
            img_arr = npz['img_emb'].astype(np.float32)
            cap_arr = npz.get('cap_emb', img_arr).astype(np.float32)
            for i, iid in enumerate(ids_arr):
                self.clip_cache[int(iid)] = (img_arr[i], cap_arr[i])
            print(f"[MiniERP] Loaded {len(self.clip_cache)} CLIP embeds")
        # Fill missing with zeros
        dim = getattr(config, 'proj_dim', 768)
        for iid in self.all_img_ids:
            if iid not in self.clip_cache:
                self.clip_cache[iid] = (np.zeros(dim, np.float32),
                                        np.zeros(dim, np.float32))

    def resample(self):
        """Generate fresh mini-ERP averages for training set.

        For each train image, randomly sample erp_k trials and average.
        Called at the start of each epoch.
        """
        data, labels, img_emb, cap_emb = [], [], [], []
        for iid in self.trn_img_ids:
            indices = self.groups[iid]
            k = min(self.erp_k, len(indices))
            sel = np.random.choice(indices, size=k, replace=False)
            avg = self.raw_data[sel].mean(axis=0)
            data.append(avg)
            labels.append(self.img_labels[iid])
            img_emb.append(self.clip_cache[iid][0])
            cap_emb.append(self.clip_cache[iid][1])
        self.current_data = np.stack(data)
        self.current_labels = np.array(labels)
        self.current_img_emb = np.stack(img_emb)
        self.current_cap_emb = np.stack(cap_emb)

    def set_train(self):
        self.is_train = True

    def set_val(self):
        self.is_train = False

    def __len__(self):
        if self.is_train:
            return len(self.current_data)
        return len(self.val_data)

    def __getitem__(self, idx):
        if self.is_train:
            data = torch.from_numpy(self.current_data[idx])
            label = int(self.current_labels[idx])
            img_emb = torch.from_numpy(self.current_img_emb[idx])
            cap_emb = torch.from_numpy(self.current_cap_emb[idx])
            # Augmentation: Gaussian noise + time masking
            data = data + torch.randn_like(data) * data.std() * 0.1
            T = data.shape[-1]
            mask_len = max(1, int(T * 0.1))
            t_start = random.randint(0, T - mask_len)
            data[:, t_start:t_start + mask_len] = 0.0
        else:
            data = torch.from_numpy(self.val_data[idx])
            label = int(self.val_labels[idx])
            img_emb = torch.from_numpy(self.val_img_emb[idx])
            cap_emb = torch.from_numpy(self.val_cap_emb[idx])

        regs = torch.zeros(N_RSVP_CLASSES, dtype=torch.float32)
        regs[label] = 1.0
        return {
            "data": data,
            "regs": regs,
            "subjects": torch.tensor(0, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "img_emb": img_emb,
            "cap_emb": cap_emb,
        }


# ═══ Modules ═════════════════════════════════════════════════════════════════

class EEGProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=768, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)
    def forward(self, x):
        return F.normalize(self.norm(self.net(x)), dim=-1)


class CategoryCLIPLoss(nn.Module):
    def __init__(self, cat_embeds, init_temp=0.5):
        super().__init__()
        self.register_buffer('prototypes', cat_embeds)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
    def forward(self, eeg_emb, labels):
        temp = self.log_temp.exp().clamp(0.01, 5.0)
        logits = eeg_emb @ self.prototypes.T / temp
        return F.cross_entropy(logits, labels)
    def predict(self, eeg_emb):
        temp = self.log_temp.exp().clamp(0.01, 5.0)
        return eeg_emb @ self.prototypes.T / temp


# ═══ Category CLIP ═══════════════════════════════════════════════════════════

def load_category_clip(config, device):
    cache = os.path.join(os.path.dirname(config.rsvp_fif_path),
                         'rsvp_category_clip.npz')
    if os.path.isfile(cache):
        return torch.from_numpy(np.load(cache)['cat_emb']).float().to(device)
    print("[V3] Computing category CLIP embeddings...")
    from transformers import CLIPModel, CLIPProcessor
    clip_name = getattr(config, 'clip_model_name', 'openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_name)
    prompts = [f"a photo of a {c}" for c in RSVP_CATEGORIES]
    with torch.no_grad():
        inp = proc(text=prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        emb = model.get_text_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    np.savez(cache, cat_emb=emb.cpu().numpy(), categories=RSVP_CATEGORIES)
    del model, proc; torch.cuda.empty_cache()
    return emb.float()


# ═══ Data ════════════════════════════════════════════════════════════════════

def build_loaders(dataset, config):
    """Build train/val loaders. num_workers=0 on Windows (data in RAM)."""
    dataset.set_train()
    dl_trn = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True,
                        drop_last=True)
    # Val loader needs a wrapper
    dataset_val = _ValWrapper(dataset)
    dl_val = DataLoader(dataset_val, batch_size=config.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    return dl_trn, dl_val


class _ValWrapper(Dataset):
    """Wraps MiniERPDataset for val-only access."""
    def __init__(self, parent):
        self.parent = parent
    def __len__(self):
        return len(self.parent.val_data)
    def __getitem__(self, idx):
        data = torch.from_numpy(self.parent.val_data[idx])
        label = int(self.parent.val_labels[idx])
        img_emb = torch.from_numpy(self.parent.val_img_emb[idx])
        cap_emb = torch.from_numpy(self.parent.val_cap_emb[idx])
        regs = torch.zeros(N_RSVP_CLASSES, dtype=torch.float32)
        regs[label] = 1.0
        return {
            "data": data, "regs": regs,
            "subjects": torch.tensor(0, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "img_emb": img_emb, "cap_emb": cap_emb,
        }


# ═══ Evaluation ══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_cls(model, loader, device):
    model.eval()
    Y, P, PR = [], [], []
    for b in loader:
        x, y = b['data'].to(device), b['label']
        lo = model(x)
        Y.append(y); P.append(lo.argmax(1).cpu()); PR.append(F.softmax(lo, 1).cpu())
    y, p, pr = [torch.cat(v).numpy() for v in (Y, P, PR)]
    all_labels = list(range(N_RSVP_CLASSES))
    return {'top1': float(accuracy_score(y, p)),
            'top3': float(top_k_accuracy_score(y, pr, k=3, labels=all_labels)),
            'bacc': float(balanced_accuracy_score(y, p)),
            'f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'cm': confusion_matrix(y, p, labels=all_labels)}


@torch.no_grad()
def eval_clip(model, proj, clip_loss, loader, device):
    model.eval(); proj.eval()
    Y, P, PR = [], [], []
    for b in loader:
        x, y = b['data'].to(device), b['label']
        feat = model.forward_features(x)
        lo = clip_loss.predict(proj(feat))
        Y.append(y); P.append(lo.argmax(1).cpu()); PR.append(F.softmax(lo, 1).cpu())
    y, p, pr = [torch.cat(v).numpy() for v in (Y, P, PR)]
    all_labels = list(range(N_RSVP_CLASSES))
    return {'top1': float(accuracy_score(y, p)),
            'top3': float(top_k_accuracy_score(y, pr, k=3, labels=all_labels)),
            'bacc': float(balanced_accuracy_score(y, p)),
            'f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'cm': confusion_matrix(y, p, labels=all_labels)}


@torch.no_grad()
def eval_ensemble(model, proj, clip_loss, loader, device, alpha=0.5):
    model.eval(); proj.eval()
    Y, P, PR = [], [], []
    for b in loader:
        x, y = b['data'].to(device), b['label']
        feat = model.forward_features(x)
        lo_ce = F.softmax(model.cls_head(feat), 1)
        lo_cl = F.softmax(clip_loss.predict(proj(feat)), 1)
        lo = alpha * lo_ce + (1 - alpha) * lo_cl
        Y.append(y); P.append(lo.argmax(1).cpu()); PR.append(lo.cpu())
    y, p, pr = [torch.cat(v).numpy() for v in (Y, P, PR)]
    pr = pr / pr.sum(axis=1, keepdims=True)
    all_labels = list(range(N_RSVP_CLASSES))
    return {'top1': float(accuracy_score(y, p)),
            'top3': float(top_k_accuracy_score(y, pr, k=3, labels=all_labels)),
            'bacc': float(balanced_accuracy_score(y, p)),
            'f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'cm': confusion_matrix(y, p, labels=all_labels)}


# ═══ Training ════════════════════════════════════════════════════════════════

def train_ce(config, dataset, dl_trn, dl_val, device):
    print("\n" + "=" * 70)
    print(f"V3 MODE 1: ATM CE — Mini-ERP (k={config.erp_k})")
    print("=" * 70)
    model = ATM_Encoder(config).to(device)
    opt = optim.AdamW(model.parameters(), lr=config.learning_rate,
                      betas=config.betas, weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = {'top1': 0}; bst = None; hist = []; patience_cnt = 0
    for ep in range(1, config.epoch + 1):
        # Resample mini-ERPs each epoch!
        dataset.resample()
        dataset.set_train()
        model.train(); losses = []; t0 = time.time()
        for b in dl_trn:
            x, y = b['data'].to(device), b['label'].to(device)
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        val = eval_cls(model, dl_val, device)
        if val['top1'] > best['top1']:
            best = {**val, 'ep': ep}
            bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        hist.append({'epoch': ep, 'loss': np.mean(losses),
                     **{k: v for k, v in val.items() if k != 'cm'}})
        if ep % 10 == 0 or ep <= 5:
            print(f"  Ep {ep:3d} | L={np.mean(losses):.4f} | "
                  f"T1={val['top1']:.4f} T3={val['top3']:.4f} BA={val['bacc']:.4f} | "
                  f"best={best['top1']:.4f}@{best['ep']} | {time.time()-t0:.1f}s")
        if patience_cnt >= config.patience:
            print(f"  Early stop at ep {ep} (patience={config.patience})")
            break
    model.load_state_dict(bst)
    f = eval_cls(model, dl_val, device)
    print(f"  Best @ep{best['ep']}: T1={f['top1']:.4f} T3={f['top3']:.4f} BA={f['bacc']:.4f}")
    return model, f, hist


def train_joint(config, dataset, dl_trn, dl_val, device, cat_embeds):
    print("\n" + "=" * 70)
    print(f"V3 MODE 2: Joint ATM (CE + Category-CLIP) — Mini-ERP (k={config.erp_k})")
    print("=" * 70)
    model = ATM_Encoder(config).to(device)
    proj = EEGProjectionHead(config.feat_dim, config.proj_dim, config.dropout).to(device)
    closs = CategoryCLIPLoss(cat_embeds, init_temp=0.5).to(device)
    ce_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_w, cl_w = config.ce_weight, config.clip_weight

    params = list(model.parameters()) + list(proj.parameters()) + [closs.log_temp]
    opt = optim.AdamW(params, lr=config.learning_rate,
                      betas=config.betas, weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6)

    best_t1 = 0; bst_m = None; bst_p = None; hist = []; patience_cnt = 0; best_ep = 0
    for ep in range(1, config.epoch + 1):
        dataset.resample()
        dataset.set_train()
        model.train(); proj.train()
        lce, lcl = [], []; t0 = time.time()
        for b in dl_trn:
            x, y = b['data'].to(device), b['label'].to(device)
            opt.zero_grad()
            feat = model.forward_features(x)
            l1 = ce_crit(model.cls_head(feat), y)
            l2 = closs(proj(feat), y)
            loss = ce_w * l1 + cl_w * l2
            loss.backward(); nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); lce.append(l1.item()); lcl.append(l2.item())
        sched.step()
        vc = eval_cls(model, dl_val, device)
        vk = eval_clip(model, proj, closs, dl_val, device)
        ve = eval_ensemble(model, proj, closs, dl_val, device)
        mx = max(vc['top1'], vk['top1'], ve['top1'])
        if mx > best_t1:
            best_t1 = mx; best_ep = ep; patience_cnt = 0
            bst_m = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bst_p = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
        else:
            patience_cnt += 1
        hist.append({
            'epoch': ep, 'ce_loss': np.mean(lce), 'clip_loss': np.mean(lcl),
            'temp': closs.log_temp.exp().item(),
            'ce_top1': vc['top1'], 'ce_top3': vc['top3'],
            'clip_top1': vk['top1'], 'clip_top3': vk['top3'],
            'ens_top1': ve['top1'], 'ens_top3': ve['top3'],
        })
        if ep % 10 == 0 or ep <= 5:
            t = closs.log_temp.exp().item()
            print(f"  Ep {ep:3d} | CE={np.mean(lce):.4f} CL={np.mean(lcl):.4f} t={t:.3f} | "
                  f"CE[{vc['top1']:.3f}/{vc['top3']:.3f}] "
                  f"CLIP[{vk['top1']:.3f}/{vk['top3']:.3f}] "
                  f"ENS[{ve['top1']:.3f}/{ve['top3']:.3f}] | "
                  f"best={best_t1:.3f}@{best_ep} | {time.time()-t0:.1f}s")
        if patience_cnt >= config.patience:
            print(f"  Early stop at ep {ep} (patience={config.patience})")
            break
    model.load_state_dict(bst_m); proj.load_state_dict(bst_p)
    fc = eval_cls(model, dl_val, device)
    fk = eval_clip(model, proj, closs, dl_val, device)
    fe = eval_ensemble(model, proj, closs, dl_val, device)
    print(f"\n  Best @ep{best_ep}:")
    print(f"    CE:   T1={fc['top1']:.4f} T3={fc['top3']:.4f} BA={fc['bacc']:.4f}")
    print(f"    CLIP: T1={fk['top1']:.4f} T3={fk['top3']:.4f} BA={fk['bacc']:.4f}")
    print(f"    ENS:  T1={fe['top1']:.4f} T3={fe['top3']:.4f} BA={fe['bacc']:.4f}")
    return model, proj, closs, fc, fk, fe, hist


# ═══ Plotting ════════════════════════════════════════════════════════════════

def plot_v3(results, save_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    k = results['erp_k']
    fig.suptitle(f'RSVP-COCO 12-Class: V3 Mini-ERP (k={k}) vs V2 Full-ERP vs ML\n'
                 f'V3: {results["n_train"]} train / {results["n_val"]} val images, '
                 f'dynamic sub-averaging each epoch',
                 fontsize=13, fontweight='bold')

    methods = ['ML\n(LDA)', 'V2: ATM\nCE (360)', 'V2: ATM\nCLIP (360)',
               f'V3: ATM\nCE (k={k})', f'V3: Joint\nCE (k={k})', f'V3: Joint\nENS (k={k})']
    t1 = [ML_BASELINE['top1'], V2_CE_BEST['top1'], V2_CLIP_BEST['top1'],
          results['ce']['top1'], results['jt_ce']['top1'], results['jt_ens']['top1']]
    t3 = [ML_BASELINE['top3'], V2_CE_BEST['top3'], V2_CLIP_BEST['top3'],
          results['ce']['top3'], results['jt_ce']['top3'], results['jt_ens']['top3']]
    ba = [ML_BASELINE['bacc'], V2_CE_BEST['bacc'], V2_CLIP_BEST['bacc'],
          results['ce']['bacc'], results['jt_ce']['bacc'], results['jt_ens']['bacc']]
    colors = ['#9E9E9E', '#81C784', '#FFB74D', '#4CAF50', '#2196F3', '#9C27B0']

    # Panel 1: Bar chart
    ax = axes[0]; x = np.arange(len(methods)); w = 0.22
    ax.bar(x - w, t1, w, color=colors, alpha=0.9, edgecolor='black', lw=0.5)
    ax.bar(x, t3, w, color=colors, alpha=0.55, edgecolor='black', lw=0.5)
    ax.bar(x + w, ba, w, color=colors, alpha=0.35, edgecolor='black', lw=0.5)
    for i in range(len(methods)):
        ax.text(i - w, t1[i] + 0.008, f'{t1[i]:.3f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(i, t3[i] + 0.008, f'{t3[i]:.3f}', ha='center', fontsize=7)
    ax.axhline(CHANCE, color='red', ls=':', lw=1.5, label=f'Chance ({CHANCE:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Score'); ax.set_title('All Methods Comparison')
    ax.legend(['Top-1', 'Top-3', 'BAcc', 'Chance'], loc='upper left', fontsize=7)
    ax.set_ylim(0, max(max(t3) + 0.12, 0.7)); ax.grid(True, alpha=0.2, axis='y')

    # Panel 2: Training curves
    ax = axes[1]
    if 'jt_hist' in results:
        h = results['jt_hist']
        ep = [e['epoch'] for e in h]
        ax.plot(ep, [e['ce_top1'] for e in h], '-', color='#2196F3', lw=1.5, label='Joint CE T1')
        ax.plot(ep, [e['clip_top1'] for e in h], '-', color='#FF9800', lw=1.5, label='Joint CLIP T1')
        ax.plot(ep, [e['ens_top1'] for e in h], '-', color='#9C27B0', lw=2, label='Joint Ensemble T1')
    if 'ce_hist' in results:
        h = results['ce_hist']
        ep = [e['epoch'] for e in h]
        ax.plot(ep, [e['top1'] for e in h], '--', color='#4CAF50', lw=1.5, label='CE-only T1')
    ax.axhline(ML_BASELINE['top1'], color='gray', ls=':', lw=2, label=f'ML ({ML_BASELINE["top1"]:.3f})')
    ax.axhline(V2_CE_BEST['top1'], color='green', ls='--', lw=1.5, alpha=0.5,
               label=f'V2 CE ({V2_CE_BEST["top1"]:.3f})')
    ax.axhline(CHANCE, color='red', ls=':', lw=1, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Top-1')
    ax.set_title('V3 Training Curves'); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

    # Panel 3: Improvement table as text
    ax = axes[2]; ax.axis('off')
    rows = [
        ['Method', 'Top-1', 'Top-3', 'BAcc', 'vs ML', 'vs V2'],
        ['ML (LDA)', f'{ML_BASELINE["top1"]:.3f}', f'{ML_BASELINE["top3"]:.3f}',
         f'{ML_BASELINE["bacc"]:.3f}', '—', '—'],
        ['V2 CE', f'{V2_CE_BEST["top1"]:.3f}', f'{V2_CE_BEST["top3"]:.3f}',
         f'{V2_CE_BEST["bacc"]:.3f}',
         f'+{(V2_CE_BEST["top1"]/ML_BASELINE["top1"]-1)*100:.1f}%', '—'],
        ['V2 CLIP', f'{V2_CLIP_BEST["top1"]:.3f}', f'{V2_CLIP_BEST["top3"]:.3f}',
         f'{V2_CLIP_BEST["bacc"]:.3f}',
         f'+{(V2_CLIP_BEST["top1"]/ML_BASELINE["top1"]-1)*100:.1f}%', '—'],
    ]
    for name, r, v2ref in [
        (f'V3 CE (k={k})', results['ce'], V2_CE_BEST),
        (f'V3 Jt CE (k={k})', results['jt_ce'], V2_CE_BEST),
        (f'V3 Jt ENS (k={k})', results['jt_ens'], V2_CE_BEST),
    ]:
        ml_imp = (r['top1'] / ML_BASELINE['top1'] - 1) * 100
        v2_imp = (r['top1'] / v2ref['top1'] - 1) * 100
        rows.append([name, f'{r["top1"]:.3f}', f'{r["top3"]:.3f}',
                     f'{r["bacc"]:.3f}', f'+{ml_imp:.1f}%', f'+{v2_imp:.1f}%'])
    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for i in range(len(rows[0])):
        table[0, i].set_facecolor('#E0E0E0')
        table[0, i].set_text_props(fontweight='bold')
    ax.set_title('Comparison Table', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_v3_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); print(f"Saved: {p}"); plt.close(fig)

    # Confusion matrix
    best_key = max(['ce', 'jt_ce', 'jt_ens'], key=lambda k2: results[k2]['top1'])
    cm = results[best_key].get('cm', None)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=max(0.3, cm_n.max()))
        for i in range(N_RSVP_CLASSES):
            for j in range(N_RSVP_CLASSES):
                v = cm_n[i, j]; c = 'white' if v > 0.15 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=c)
        ax.set_xticks(range(N_RSVP_CLASSES)); ax.set_yticks(range(N_RSVP_CLASSES))
        ax.set_xticklabels(RSVP_CATEGORIES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(RSVP_CATEGORIES, fontsize=9)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'V3 Best ({best_key}): T1={results[best_key]["top1"]:.3f} '
                     f'T3={results[best_key]["top3"]:.3f} BA={results[best_key]["bacc"]:.3f}\n'
                     f'Mini-ERP (k={k}), {results["n_train"]}+{results["n_val"]} images',
                     fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8); plt.tight_layout()
        p = os.path.join(save_dir, 'fig', 'rsvp_v3_confusion.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); print(f"Saved: {p}"); plt.close(fig)


# ═══ Main ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fif', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_weight', type=float, default=0.5)
    parser.add_argument('--erp_k', type=int, default=5,
                        help='Number of trials to sub-average per image per epoch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=40)
    args = parser.parse_args()

    config = BaseConfigRSVP()
    if args.fif: config.rsvp_fif_path = args.fif
    config.epoch = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.clip_weight = args.clip_weight
    config.erp_k = args.erp_k
    config.seed = args.seed
    config.patience = args.patience

    save_dir = args.save_dir or os.path.dirname(config.rsvp_fif_path)
    os.makedirs(os.path.join(save_dir, 'fig'), exist_ok=True)

    torch.manual_seed(config.seed); np.random.seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True  # faster on fixed input sizes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name()}")

    cat_embeds = load_category_clip(config, device)
    print(f"Category embeddings: {cat_embeds.shape}")

    print(f"\n=== Loading RSVP-COCO trials (mini-ERP k={config.erp_k}) ===")
    dataset = MiniERPDataset(config.rsvp_fif_path, config, seed=config.seed)
    n_trn = len(dataset.trn_img_ids)
    n_val = len(dataset.val_img_ids)

    dl_trn, dl_val = build_loaders(dataset, config)

    # Mode 1: CE only
    torch.manual_seed(config.seed)
    _, r_ce, h_ce = train_ce(config, dataset, dl_trn, dl_val, device)

    # Mode 2: Joint CE + Cat-CLIP
    torch.manual_seed(config.seed)
    _, _, _, r_jce, r_jcl, r_jens, h_jt = train_joint(
        config, dataset, dl_trn, dl_val, device, cat_embeds)

    # Summary
    print("\n" + "=" * 70)
    print(f"FINAL COMPARISON — V3 Mini-ERP (k={config.erp_k})")
    print("=" * 70)
    print(f"{'Method':<30} {'Top-1':>8} {'Top-3':>8} {'BAcc':>8} {'vs ML':>10} {'vs V2':>10}")
    print("-" * 80)
    print(f"{'Chance':<30} {CHANCE:>8.4f} {3*CHANCE:>8.4f} {CHANCE:>8.4f} {'':>10} {'':>10}")
    print(f"{'ML Baseline (LDA, 360 avg)':<30} {ML_BASELINE['top1']:>8.4f} "
          f"{ML_BASELINE['top3']:>8.4f} {ML_BASELINE['bacc']:>8.4f} {'—':>10} {'—':>10}")
    print(f"{'V2: ATM CE (360 full-avg)':<30} {V2_CE_BEST['top1']:>8.4f} "
          f"{V2_CE_BEST['top3']:>8.4f} {V2_CE_BEST['bacc']:>8.4f} "
          f"{(V2_CE_BEST['top1']/ML_BASELINE['top1']-1)*100:>+9.1f}% {'—':>10}")
    print(f"{'V2: ATM CatCLIP (360 avg)':<30} {V2_CLIP_BEST['top1']:>8.4f} "
          f"{V2_CLIP_BEST['top3']:>8.4f} {V2_CLIP_BEST['bacc']:>8.4f} "
          f"{(V2_CLIP_BEST['top1']/ML_BASELINE['top1']-1)*100:>+9.1f}% {'—':>10}")
    for name, r, v2ref in [
        (f'V3: CE (k={config.erp_k})', r_ce, V2_CE_BEST),
        (f'V3: Joint CE (k={config.erp_k})', r_jce, V2_CE_BEST),
        (f'V3: Joint CLIP (k={config.erp_k})', r_jcl, V2_CLIP_BEST),
        (f'V3: Joint ENS (k={config.erp_k})', r_jens, V2_CE_BEST),
    ]:
        ml_imp = (r['top1'] / ML_BASELINE['top1'] - 1) * 100
        v2_imp = (r['top1'] / v2ref['top1'] - 1) * 100
        print(f"{name:<30} {r['top1']:>8.4f} {r['top3']:>8.4f} {r['bacc']:>8.4f} "
              f"{ml_imp:>+9.1f}% {v2_imp:>+9.1f}%")
    print("-" * 80)

    # Save results
    all_res = {
        'n_train': n_trn, 'n_val': n_val, 'erp_k': config.erp_k,
        'ce': {k: v for k, v in r_ce.items() if k != 'cm'}, 'ce_hist': h_ce,
        'jt_ce': {k: v for k, v in r_jce.items() if k != 'cm'},
        'jt_clip': {k: v for k, v in r_jcl.items() if k != 'cm'},
        'jt_ens': {k: v for k, v in r_jens.items() if k != 'cm'},
        'jt_hist': h_jt,
        'config': {'epochs': config.epoch, 'batch_size': config.batch_size,
                   'lr': config.learning_rate, 'clip_weight': config.clip_weight,
                   'erp_k': config.erp_k, 'patience': config.patience},
    }
    jp = os.path.join(save_dir, 'fig', 'rsvp_v3_results.json')
    with open(jp, 'w') as f: json.dump(all_res, f, indent=2, default=str)
    print(f"\nJSON: {jp}")

    plot_v3({**all_res, 'ce': r_ce, 'jt_ce': r_jce, 'jt_ens': r_jens}, save_dir)
    print("\n=== V3 TRAINING COMPLETE ===")


if __name__ == '__main__':
    main()
