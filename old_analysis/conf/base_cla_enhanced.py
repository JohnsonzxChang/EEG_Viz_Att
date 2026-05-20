from .base_cla_circle import BaseConfigCircle
from datetime import datetime


class BaseConfigEnhanced(BaseConfigCircle):
    """Enhanced multi-label contrastive learning config.

    Adds: MLP projector, Proxy-Anchor Loss, Memory Queue, Multi-Label Sampler,
    Prototype evaluation, and EEG data augmentation — all on top of the
    existing ASL + Circle Loss pipeline.

    Encoder-agnostic: set self.model to 'ATM', 'CNN', 'EEGNet', etc.
    """

    def __init__(self):
        super().__init__()

        # ── Encoder (default ATM, change to 'CNN'/'EEGNet'/etc as needed) ──
        self.model = 'ATM'
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 2
        self.n_heads = 4
        self.dropout = 0.4

        # ── Projection head (decoupled from cls_head for contrastive) ──────
        self.feat_dim = 256         # backbone embedding dimension
        self.proj_dim = 128         # MLP projector output for Circle / Proxy

        # ── Circle Loss ────────────────────────────────────────────────────
        self.circle_gamma = 80
        self.circle_margin = 0.25
        self.circle_lambda = 0.5    # reduced from 0.9 since Proxy-Anchor also provides metric signal
        self.circle_jaccard = True

        # ── Proxy-Anchor Loss ──────────────────────────────────────────────
        self.proxy_scale = 32.0
        self.proxy_delta = 0.1
        self.proxy_lambda = 0.3     # weight in combined loss

        # ── Memory Queue ───────────────────────────────────────────────────
        self.queue_size = 512       # expands effective batch for Circle Loss

        # ── Multi-Label Sampler ────────────────────────────────────────────
        self.sampler_k_anchor = 2   # forced anchors per class per batch
        self.sampler_num_batches = None  # None = auto (ceil(N/batch_size))

        # ── Prototype Evaluation ───────────────────────────────────────────
        self.proto_tau = 10.0       # temperature for cosine → sigmoid
        self.proto_eval_freq = 5    # run prototype eval every N epochs

        # ── Optimizer ──────────────────────────────────────────────────────
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01

        # ── Training ──────────────────────────────────────────────────────
        self.batch_size = 160
        self.epoch = 200

        # ── Data augmentation ─────────────────────────────────────────────
        self.aug_noise_std = 0.05
        self.aug_chan_drop = 3
        self.aug_smooth_k = 3

        self.comment = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-enhanced-{self.model}'
