"""
Entry point for ASL + Circle Loss multi-label classification.

Usage:
    python main_circle.py

Key config knobs (edit BaseConfigCircle or override below):
    circle_gamma   : scale factor γ  (default 64)
    circle_margin  : margin m        (default 0.25)
    circle_lambda  : λ in L = L_ASL + λ·L_circle  (default 0.1)
    circle_jaccard : weight positives by Jaccard similarity  (default True)
    feat_dim       : encoder bottleneck dim  (default 256)
"""

import torch
import numpy as np
from datetime import datetime

from conf import BaseConfigCircle
from tasks import Exp_ClassificationCircle


def main():
    config = BaseConfigCircle()
    config.t0 = 1000
    config.comment = f'{datetime.now()}-circle-{config.model}'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[circle] γ={config.circle_gamma}  m={config.circle_margin}  "
          f"λ={config.circle_lambda}  jaccard={config.circle_jaccard}")

    exp = Exp_ClassificationCircle(config)
    exp.train('circle_run')


if __name__ == '__main__':
    main()
