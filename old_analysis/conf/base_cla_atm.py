from .base_cla_circle import BaseConfigCircle
from datetime import datetime


class BaseConfigATM(BaseConfigCircle):
    """ATM encoder + decoupled MLP projector for Circle Loss.

    Key changes from BaseConfigCircle:
    - ATM_Encoder with cross-channel transformer + ShallowNet conv
    - Separate MLP projection head (proj_dim) for Circle Loss
    - Fixed optimizer hyperparameters (standard AdamW betas, proper weight_decay)
    - Reduced dropout (0.3 vs 0.8)
    - Stronger Circle Loss weighting (decoupled projector allows it)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        self.model = 'ATM'
        self.d_model = 128          # ATM ChannelTransformer hidden dim
        self.d_ff = 256             # ATM feedforward dim
        self.e_layers = 2           # fewer layers to avoid overfitting on small dataset
        self.n_heads = 4
        self.dropout = 0.4          # down from 0.8

        # ── Projection head ──────────────────────────────────────────
        self.feat_dim = 256         # backbone embedding dimension
        self.proj_dim = 128         # MLP projector output for Circle Loss

        # ── Circle Loss ──────────────────────────────────────────────
        self.circle_gamma = 80      # sharper boundary (up from 64)
        self.circle_margin = 0.25
        self.circle_lambda = 0.9    # stronger contrastive signal (up from 0.1)
        self.circle_jaccard = True

        # ── Optimizer ────────────────────────────────────────────────
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.999)   # standard AdamW betas
        self.weight_decay = 0.01    # meaningful regularization

        # ── Training ─────────────────────────────────────────────────
        self.batch_size = 256       # more batches/epoch for Circle Loss diversity
        self.epoch = 200

        # ── Data augmentation ────────────────────────────────────────
        self.batch_size = 160
        self.aug_noise_std = 0.05   # Gaussian noise: 5% of channel std
        self.aug_chan_drop = 3      # zero out up to 3 random channels
        self.aug_smooth_k = 3      # temporal smoothing kernel size

        self.comment = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-atm-mlp-circle'
