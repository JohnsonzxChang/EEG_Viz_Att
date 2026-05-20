"""
Entry point for single-label supervised learning.

Utilizes standard ATM encoder + CrossEntropyLoss for multi-class (single-label) prediction.

Usage:
    python main_single.py
"""

import torch
import numpy as np
from datetime import datetime
import argparse

from conf.base_cla_single import BaseConfigSingle
from tasks.task_classification_single import Exp_ClassificationSingle


def main():
    parser = argparse.ArgumentParser(description='Supervised Single-Label Classification')
    parser.add_argument('--epoch', type=int, default=200, help='number of training epochs')
    parser.add_argument('--t0', type=int, default=500, help='time point for feature extraction')
    args = parser.parse_args()

    config = BaseConfigSingle()
    config.t0 = args.t0
    config.epoch = args.epoch
    config.comment = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-openimage-{config.model}-T{config.t0}'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[single_label] model={config.model}  dropout={config.dropout}")
    print(f"[single_label] lr={config.learning_rate}  wd={config.weight_decay}")

    # Use the simple logger interface directly without session
    from utils.loggers import LoggerFile
    import os
    if not os.path.exists(config.loggerdir):
        os.makedirs(config.loggerdir)
    logger = LoggerFile(config.loggerdir, config.comment, conf_class=config)
    
    exp = Exp_ClassificationSingle(config, logger=logger)
    exp.train(f'{config.comment}')


if __name__ == '__main__':
    main()
