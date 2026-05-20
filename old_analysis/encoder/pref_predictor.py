import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGTemporalEncoder(nn.Module):
    def __init__(self, n_channels=128, n_samples=1500, patch_size=50, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_samples // patch_size
        
        self.patch_embed = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Dynamically slice pos_embed in case input time window < n_samples max
        tokens = self.pos_drop(tokens + self.pos_embed[:, :tokens.shape[1], :])
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        
        cls_out = tokens[:, 0, :]
        temporal_tokens = tokens[:, 1:, :]
        return temporal_tokens, cls_out

class CrossAttentionGrounding(nn.Module):
    def __init__(self, d_vis=768, d_eeg=256, d_hidden=256, n_heads=8):
        super().__init__()
        self.d_hidden = d_hidden
        self.proj_vis = nn.Sequential(
            nn.Linear(d_vis, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=n_heads, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, object_features, eeg_tokens):
        B, K, _ = object_features.shape
        Q = self.proj_vis(object_features)
        h, attn_weights = self.cross_attn(
            query=Q, key=eeg_tokens, value=eeg_tokens, need_weights=True, average_attn_weights=True
        )
        h = self.attn_norm(h + Q)
        interaction = h * Q
        combined = torch.cat([h, Q, interaction], dim=-1)
        logits = self.score_head(combined).squeeze(-1)
        return logits, attn_weights

class TextConditionedFusion(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d, d)

    def forward(self, v, t):
        t_expand = t.expand_as(v)
        gate = self.gate(torch.cat([v, t_expand], dim=-1))
        f = v + gate * self.proj(t_expand)
        return f

class PreferenceBboxPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eeg_encoder = EEGTemporalEncoder(
            n_channels=args.n_channels, n_samples=args.n_samples, patch_size=args.patch_size,
            d_model=args.d_eeg, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout
        )
        self.text_fusion = TextConditionedFusion(d=args.d_vis)
        self.grounding = CrossAttentionGrounding(d_vis=args.d_vis, d_eeg=args.d_eeg, d_hidden=args.d_hidden)

    def forward(self, eeg, object_features, object_bboxes, object_mask, text_feature=None):
        eeg_tokens, eeg_cls = self.eeg_encoder(eeg)
        if text_feature is not None:
            obj_feat = self.text_fusion(object_features, text_feature.unsqueeze(1))
        else:
            obj_feat = object_features
            
        logits, attn_maps = self.grounding(obj_feat, eeg_tokens)
        
        # Apply mask to prevent softmax from selecting zero-padded elements
        # object_mask: (B, K) where True is valid, False is padding
        logits = logits.masked_fill(~object_mask, -1e4)
        
        preference_probs = F.softmax(logits, dim=-1)
        
        if self.training:
            soft_bbox = torch.einsum('bk, bkd -> bd', preference_probs, object_bboxes)
            # return soft bbox for GIoU diff regression, also returning probabilities and cross-attention weight maps
            return soft_bbox, logits, attn_maps, eeg_cls
        else:
            best_idx = preference_probs.argmax(dim=-1)
            hard_bbox = object_bboxes[torch.arange(object_bboxes.shape[0]), best_idx]
            return hard_bbox, logits, attn_maps, eeg_cls
