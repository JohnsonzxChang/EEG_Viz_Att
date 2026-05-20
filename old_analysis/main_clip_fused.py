"""
CLIP-Fused EEG Classification with ERP Averaging.

This script trains a multi-label classifier (80 COCO classes) using:
  1. ERP-averaged EEG data (improved SNR via trial averaging)
  2. CLIP image/caption embeddings fused via gated mechanism
  3. ASL loss + optional EEG↔CLIP alignment loss

Three ERP averaging modes are available:
  - 'image_avg':  average all trials per image (best SNR, small dataset)
  - 'ktrial_avg': random K-trial sub-averaging (data augmentation)
  - 'both':       K-trial for train, image-avg for val (recommended)

Usage:
  python main_clip_fused.py                         # default: both mode + CLIP fusion
  python main_clip_fused.py --erp_mode image_avg    # image-level ERP averaging only
  python main_clip_fused.py --erp_mode ktrial_avg --erp_k 2  # K=2 sub-averaging
  python main_clip_fused.py --no_clip               # ERP averaging without CLIP fusion
  python main_clip_fused.py --model CNN             # use CNN encoder instead of ATM

References:
  - ATM (Li et al., NeurIPS 2024): Visual decoding via EEG embeddings
  - NICE (Song et al., 2023): K-trial sub-averaging
  - EEG-CLIP (Li et al., 2024): Cross-modal EEG-CLIP alignment
  - BraVL: Brain-Visual-Language trimodal learning
"""

import argparse
import torch
import numpy as np
from datetime import datetime
from conf.base_cla_clip_fused import BaseConfigClipFused
from tasks.task_classification_clip_fused import Exp_ClassificationClipFused


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-Fused EEG Classification')
    parser.add_argument('--erp_mode', type=str, default='both',
                        choices=['image_avg', 'ktrial_avg', 'both'],
                        help='ERP averaging mode')
    parser.add_argument('--erp_k', type=int, default=4,
                        help='Number of trials to sub-average in ktrial mode')
    parser.add_argument('--erp_n_aug', type=int, default=5,
                        help='Number of augmented samples per image in ktrial mode')
    parser.add_argument('--no_clip', action='store_true',
                        help='Disable CLIP feature fusion')
    parser.add_argument('--lambda_align', type=float, default=0.1,
                        help='Weight for EEG-CLIP alignment loss')
    parser.add_argument('--model', type=str, default='ATM',
                        help='Encoder model (CNN, ATM, Transformer, TViT, etc.)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: auto based on erp_mode)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=10086,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()

    config = BaseConfigClipFused()

    # Override config with command-line args
    config.erp_mode = args.erp_mode
    config.erp_k = args.erp_k
    config.erp_n_aug = args.erp_n_aug
    config.use_clip_feat = not args.no_clip
    config.lambda_align = args.lambda_align
    config.model = args.model
    config.seed = args.seed

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.epochs is not None:
        config.epoch = args.epochs
    if args.dropout is not None:
        config.dropout = args.dropout

    # Update comment
    clip_tag = "clip" if config.use_clip_feat else "noclip"
    config.comment = (
        f'{datetime.now()}-clipfused-{config.model}-'
        f'{config.erp_mode}-k{config.erp_k}-{clip_tag}'
    )

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print("CLIP-Fused EEG Classification with ERP Averaging")
    print("=" * 70)
    print(f"  Model:         {config.model}")
    print(f"  ERP mode:      {config.erp_mode}")
    print(f"  ERP K:         {config.erp_k}")
    print(f"  ERP n_aug:     {config.erp_n_aug}")
    print(f"  CLIP fusion:   {config.use_clip_feat}")
    print(f"  λ_align:       {config.lambda_align}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs:        {config.epoch}")
    print(f"  Dropout:       {config.dropout}")
    print(f"  Seed:          {config.seed}")
    print("=" * 70)

    exp = Exp_ClassificationClipFused(config)
    exp.train('clip_fused_run')


if __name__ == '__main__':
    main()
