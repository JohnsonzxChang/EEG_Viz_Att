"""
Entry point for enhanced multi-label contrastive learning.

Combines: ATM encoder + MLP projector + Circle Loss (queue-augmented) +
Proxy-Anchor Loss + Multi-Label Sampler + EEG Augmentation + Prototype eval.

Usage:
    python main_enhanced.py
"""

import torch
import numpy as np
from datetime import datetime

import argparse

from conf import BaseConfigEnhanced
from tasks import Exp_ClassificationEnhanced


def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-label Contrastive Learning')
    parser.add_argument('--epoch', type=int, default=400, help='number of training epochs')
    args = parser.parse_args()

    config = BaseConfigEnhanced()
    config.t0 = 1000
    config.epoch = args.epoch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[enhanced] model={config.model}  dropout={config.dropout}")
    print(f"[enhanced] Circle: gamma={config.circle_gamma}  lambda={config.circle_lambda}")
    print(f"[enhanced] Proxy:  scale={config.proxy_scale}  delta={config.proxy_delta}  lambda={config.proxy_lambda}")
    print(f"[enhanced] Queue:  size={config.queue_size}")
    print(f"[enhanced] Sampler: K_anchor={config.sampler_k_anchor}")
    print(f"[enhanced] lr={config.learning_rate}  wd={config.weight_decay}")

    exp = Exp_ClassificationEnhanced(config)
    exp.train('enhanced_run')


if __name__ == '__main__':
    main()
