"""
Entry point for ATM + MLP projector + Circle Loss multi-label classification.

Usage:
    python main_atm_circle.py

Changes from main_circle.py:
    - ATM_Encoder with cross-channel transformer (replaces CNN)
    - Decoupled MLP projection head for Circle Loss (SimCLR-style)
    - EEG data augmentation (noise, channel dropout, smoothing)
    - Fixed optimizer: standard AdamW betas, proper weight_decay
    - Reduced dropout (0.3 vs 0.8)
"""

import torch
import numpy as np
from datetime import datetime

from conf import BaseConfigATM
from tasks import Exp_ClassificationATM


def main():
    config = BaseConfigATM()
    config.t0 = 1000

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[ATM-circle] model={config.model}  dropout={config.dropout}")
    print(f"[ATM-circle] γ={config.circle_gamma}  m={config.circle_margin}  "
          f"λ={config.circle_lambda}  jaccard={config.circle_jaccard}")
    print(f"[ATM-circle] lr={config.learning_rate}  wd={config.weight_decay}  "
          f"betas={config.betas}")
    print(f"[ATM-circle] feat_dim={config.feat_dim}  proj_dim={config.proj_dim}")

    exp = Exp_ClassificationATM(config)
    exp.train('atm_circle_run')


if __name__ == '__main__':
    main()
