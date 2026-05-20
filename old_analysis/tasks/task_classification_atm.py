"""
Exp_ClassificationATM
=====================
Multi-label classification with ATM encoder + decoupled MLP projector
for Circle Loss + EEG data augmentation.

Inherits from Exp_ClassificationCircle and overrides only:
  - _build_model()  → ATM_Encoder wrapped with EncoderWithProjector
  - _get_data()     → wraps training data with EEG augmentation
"""

import torch
from torch.utils.data import DataLoader

from conf import BaseConfig
from utils import LoggerFile
from encoder.atm_encoder import ATM_Encoder
from encoder.contrastive_wrapper import EncoderWithProjector
from utils.eeg_augment import AugmentedEEGDataset
from .task_classification_circle import Exp_ClassificationCircle


class Exp_ClassificationATM(Exp_ClassificationCircle):
    """ATM + MLP projector + EEG augmentation trainer."""

    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super().__init__(args, logger, session)

    def _build_model(self):
        assert self.args.task == 'classification'
        backbone = ATM_Encoder(self.args).to(self.device)

        feat_dim = getattr(self.args, 'feat_dim', 256)
        proj_dim = getattr(self.args, 'proj_dim', 128)

        model = EncoderWithProjector(backbone, feat_dim, proj_dim).to(self.device)
        return model

    def _get_data(self):
        train_data, train_loader, valid_data, valid_loader = super()._get_data()

        # Wrap training dataset with EEG augmentation
        aug_train = AugmentedEEGDataset(
            train_data,
            is_train=True,
            noise_std=getattr(self.args, 'aug_noise_std', 0.05),
            chan_drop=getattr(self.args, 'aug_chan_drop', 3),
            smooth_k=getattr(self.args, 'aug_smooth_k', 3),
        )

        # Re-create training DataLoader with augmented dataset
        train_loader = DataLoader(
            aug_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

        return aug_train, train_loader, valid_data, valid_loader
