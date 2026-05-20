from .base import BaseConfig
from datetime import datetime
import os
import sys

class BaseConfigNewAttTracking(BaseConfig):
    """EEG Spatial Attention Tracking config for Architecture 2.
    
    Integrates SAM + EEG Encoder pipeline for Preference Bbox Selection.
    """

    def __init__(self):
        super(BaseConfigNewAttTracking, self).__init__()

        self.task = 'new_attention_tracking'
        self.data = 'new_att_tracking'  # Dataset key

        self.early_stop = 10000
        self.patience = 10000
        
        # ── EEG Encoder Settings ──────────────────────────────────────────
        self.model = 'CNN'          # Placeholder/base model string
        self.n_channels = 32        # Aligned to actual loaded .fif MNE channels
        self.n_samples = 1500
        self.patch_size = 50        # ms
        self.d_eeg = 256            # Dimension for EEG tokens
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.1
        self.t_len = 1500           # Length of temporal EEG window
        
        # ── Visual & Cross-Modal Settings ─────────────────────────────────
        self.d_vis = 768            # Dimension for CLIP object features
        self.d_hidden = 256         # Shared projection dimension
        self.bbox_dim = 4

        # ── Loss Function Weights ─────────────────────────────────────────
        self.lambda1 = 0.5          # Contrastive loss weight
        self.lambda2 = 0.3          # Temporal alignment KL loss weight
        self.lambda3 = 0.1          # Bounding Box loss weight
        self.temperature = 0.07     # InfoNCE temperature (tau)
        
        # ── Optimizer & Training ──────────────────────────────────────────
        self.optimizer = 'adamw'
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.batch_size = 32
        self.epoch = 30
        self.patience = 10
        self.seed = 42

        # ── Device & Paths ────────────────────────────────────────────────
        self.device = 'cuda'
        self.loggerdir = './logs_NewAttTracking'
        self.resume = ""
        self.num_workers = min(8, os.cpu_count() or 4)
        self.pin_memory = True

        self.comment = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-NewAttTracking-EEG{self.d_eeg}-Vis{self.d_vis}'
