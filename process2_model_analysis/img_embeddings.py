"""Pre-compute and cache CLIP image embeddings + patch tokens.

The image set is small (~445-757 unique COCO images). We compute once and
cache to disk:
    - `features`     : global image embedding (CLS projected), shape (N, D_g)
    - `patch_tokens` : per-patch tokens (post-ln_post), shape (N, P, D_p)
    - `image_ids`    : np.ndarray of str
    - `grid`         : (gh, gw) — patch grid (e.g. 7x7 for ViT-B/32 @ 224)

Usage:
    python -m process2_model_analysis.img_embeddings \
        --selection experiment/stimuli_select/stimuli_*.json \
        --coco_root C:/Users/thlab/Desktop/ES_coco/data/coco \
        --out_cache data/zfn-0507/img_embeddings_clip_vitb32.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def extract_clip_with_patches(image_paths, device="cuda",
                              model_name="ViT-B-32",
                              pretrained="openai",
                              batch_size: int = 32):
    """Returns (global_feats (N,D_g), patch_tokens (N,P,D_p), grid (gh,gw)).

    Uses open_clip ≥3.0 API. We register a forward hook on `visual.ln_post`
    to grab tokens just before the CLS projection.
    """
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model = model.to(device).eval()

    cap = {}
    def _hook(_m, _inp, out):
        # out: (B, 1+P, D)
        cap["tokens"] = out.detach()
    h = model.visual.ln_post.register_forward_hook(_hook)

    g_feats, p_tokens = [], []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            paths = image_paths[i:i+batch_size]
            ims = [preprocess(Image.open(p).convert("RGB")) for p in paths]
            x = torch.stack(ims).to(device)
            g = model.encode_image(x)         # (B, D_g) — CLS-projected
            g = g / g.norm(dim=-1, keepdim=True)
            tok = cap["tokens"]               # (B, 1+P, D_p)
            # drop CLS token
            patches = tok[:, 1:, :]
            g_feats.append(g.float().cpu().numpy().astype(np.float32))
            p_tokens.append(patches.float().cpu().numpy().astype(np.float32))
            log.info("  CLIP %d/%d  global=%s  patches=%s",
                     i + len(paths), len(image_paths),
                     tuple(g_feats[-1].shape), tuple(p_tokens[-1].shape))
    h.remove()
    G = np.concatenate(g_feats, axis=0)
    P = np.concatenate(p_tokens, axis=0)
    # Infer square grid:
    n_patches = P.shape[1]
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise RuntimeError(f"Non-square patch grid: {n_patches}")
    return G, P, (side, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--out_cache", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sel = json.load(open(args.selection, encoding="utf-8"))
    root = Path(args.coco_root)
    image_ids, paths = [], []
    seen = set()
    for it in sel["items"]:
        iid = str(it["image_id"])
        if iid in seen:
            continue
        seen.add(iid)
        image_ids.append(iid)
        paths.append(root / it["relative_path"])
    log.info("Embedding %d unique images with %s/%s",
             len(paths), args.model, args.pretrained)
    G, P, grid = extract_clip_with_patches(paths, device=args.device,
                                            model_name=args.model,
                                            pretrained=args.pretrained)
    Path(args.out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_cache,
                         image_ids=np.array(image_ids),
                         features=G,
                         patch_tokens=P,
                         grid=np.array(grid),
                         backbone=f"{args.model}/{args.pretrained}")
    log.info("Saved %s  features=%s  patches=%s  grid=%s",
             args.out_cache, G.shape, P.shape, grid)


if __name__ == "__main__":
    main()
