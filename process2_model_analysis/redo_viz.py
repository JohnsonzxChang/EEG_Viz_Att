"""Re-extract attention with image-deduplicated selection, then redo figs.

Run after the main pipeline. Reuses saved model state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent.parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from process1_data_process.data_io import load_epochs                  # noqa
from process2_model_analysis.dataset import EEGImgAttentionDataset     # noqa
from process2_model_analysis.models import ATMEncoder, EEGImgPatchFusion  # noqa
from process2_model_analysis import train_baselines as TB              # noqa
from process2_model_analysis import viz_results as VR                  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subj_dirs_for_viz = []
    for subj in args.subjects:
        print(f"=== {subj} ===")
        data_dir = root / "data" / subj
        fif = next(iter(data_dir.glob("epochs_big-epo.fif")))
        sess = next(iter(data_dir.glob("session_*.json")))
        cache = data_dir / "img_embeddings_vit_b_32.npz"
        out_dir = data_dir / "process2_out"
        b = load_epochs(str(fif), str(sess), pick_eeg=False, load_data=True,
                         decim=4, dtype="float32",
                         crop_tmin=-0.1, crop_tmax=0.45,
                         baseline=(-0.1, 0.0))
        img_cache = np.load(cache, allow_pickle=True)
        te = EEGImgAttentionDataset(b, img_cache, split="test",
                                     normalize=True, want_patches=True)
        te_dl = DataLoader(te, batch_size=128, shuffle=False, num_workers=0)
        n_channels = te.eeg.shape[1]; n_samples = te.eeg.shape[2]
        n_classes = len(te.classes); img_dim = te.img.shape[1]
        patch_dim = te.img_patches.shape[2]

        atm = ATMEncoder(n_channels, n_samples, n_classes,
                          d_model=128, n_heads=4, d_ff=256,
                          e_layers=1, feat_dim=256, dropout=0.3)
        m = EEGImgPatchFusion(eeg_encoder=atm, eeg_dim=atm.feat_dim,
                               patch_dim=patch_dim, img_dim=img_dim,
                               n_classes=n_classes, d_model=256, n_heads=4,
                               dropout=0.3)
        state = torch.load(out_dir/"state_eeg_img_patch.pt", map_location=device)
        m.load_state_dict(state); m.eval().to(device)
        attn = TB.extract_attention(m, te_dl, device, te, top_k=args.top_k)
        np.savez_compressed(out_dir/"attn_examples.npz",
                             attn=attn["attn"], logits=attn["logits"],
                             y_true=attn["y_true"], y_pred=attn["y_pred"],
                             conf=attn["conf"], image_id=attn["image_id"],
                             hint=attn["hint"], selected=attn["selected"],
                             grid=np.array(attn["grid"]),
                             classes=np.array(attn["classes"]))
        subj_dirs_for_viz.append(f"{subj}:{out_dir}")
        print(f"  saved {len(attn['selected'])} dedup'd attention examples")

    sys.argv = ["viz_results",
                "--subject_dirs", *subj_dirs_for_viz,
                "--selection_json", args.selection,
                "--coco_root", args.coco_root,
                "--fig_dir", args.fig_dir]
    VR.main()


if __name__ == "__main__":
    main()
