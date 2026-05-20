"""
EncoderWithProjector
====================
Wraps any backbone encoder with a dedicated MLP projection head for
contrastive learning (Circle Loss), decoupled from the classification head.

Motivation (SimCLR, Chen et al. 2020):
  The contrastive objective should operate on a *separate* projection z,
  not on the backbone feature h.  This prevents the metric-learning and
  classification losses from competing on the same representation.

Data flow:
  x ──► backbone.forward_features(x) ──► h  (feat_dim)
                                          ├─► backbone.cls_head(h) ──► logits  → ASL
                                          └─► MLP projector(h)     ──► z       → Circle Loss

  forward_all(x) returns (z, logits)  — matching Exp_ClassificationCircle's interface.
  forward(x)     returns logits only  — for inference (projector is discarded).
"""

import torch
import torch.nn as nn
from transformers import SamModel


class EncoderWithTemporalCrossAttn(nn.Module):
    """
    Wrap a backbone encoder (e.g. T-ViT) with a SAM Temporal Cross-Attention head.
    The visual branch acts as queries, while the EEG temporal tokens act as keys/values.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int = 256, num_classes: int = 1, use_eeg: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_eeg = use_eeg
        
        # 1. SAM Pre-trained Mask Decoder Integration
        # We load the entire SAM model to extract its internal decoder and positional embeddings
        sam = SamModel.from_pretrained('facebook/sam-vit-base')
        self.sam_mask_decoder = sam.mask_decoder
        self.sam_pos_embed = sam.get_image_wide_positional_embeddings() # (1, 256, 64, 64)
        
        # Dense null mask embedding from SAM Prompt Encoder
        self.sam_no_mask_embed = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1) # (1, 256, 1, 1)
        
        # To avoid massive memory blows and 10x slower epoch times, we freeze the heavy decoder
        self.sam_mask_decoder.requires_grad_(False)
        
        # Free up the rest of SAM
        del sam
        
        # Visual projector maps pooled SAM Image Embeddings (256) to EEG feature space for Cross-Attention
        self.visual_projector = nn.Sequential(
            nn.Linear(256, feat_dim * 2),
            nn.BatchNorm1d(feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        
        # Project EEG Temporal Output to SAM Prompt Dimension (256)
        self.prompt_projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # 2. Cross-Attention: Visual (Query) -> EEG (Key, Value)
        # Using 1 head initially to cleanly output a single temporal preference map
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, 
            num_heads=1, 
            batch_first=True
        )
        
        # 3. Object Preference Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_classes)
        )
        
        # Spatial Bounding Box Regressor is REMOVED! We use direct SAM Decoder masks now.

        # Bounding Box Regressor in GPU and only once
        self.H, self.W = 256, 256
        y_grid = torch.arange(self.H, device='cuda:0', dtype=torch.float32) / self.H
        x_grid = torch.arange(self.W, device='cuda:0', dtype=torch.float32) / self.W
        self.y_grid = y_grid.reshape(1, self.H, 1).to('cuda:0')
        self.x_grid = x_grid.reshape(1, 1, self.W).to('cuda:0')   

    def forward(self, x, padding_mask=None, enc_self_mask=None,
                dec_self_mask=None):
        """Standard forward — returns logits only (for inference)."""
        return self.backbone(x, padding_mask, enc_self_mask, dec_self_mask)

    def forward_features(self, x, padding_mask=None, enc_self_mask=None,
                         dec_self_mask=None):
        """Return backbone embedding h (before cls_head)."""
        return self.backbone.forward_features(x, padding_mask, enc_self_mask,
                                              dec_self_mask)

    def forward_all(self, x, batch_id=None, img_emb=None, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        """
        Return preference_logits and temporal_attention_weights.
        """
        if self.use_eeg:
            # 1. EEG Temporal Tokens -> shape check
            eeg_tokens = self.backbone.forward_features(x, padding_mask, enc_self_mask, dec_self_mask)
            if len(eeg_tokens.shape) == 2:
                # If backbone is an old CNN lacking Temporal Tokens (B, D), fake a sequence of length 1: (B, 1, D)
                eeg_tokens = eeg_tokens.unsqueeze(1)
            
            # 2. SAM Visual Query Vector
            # if img_emb is not None:
                # Pool the (256, 64, 64) spatial map to (256,) for the Cross-Attention Query
            pooled_img_emb = img_emb.mean(dim=[-1, -2]) 
            v_query = self.visual_projector(pooled_img_emb).unsqueeze(1) # (B, 1, D)
            # else:
            #     # Fallback for debugging, mock missing img_emb with zeros
            #     v_query = torch.zeros(eeg_tokens.size(0), 1, eeg_tokens.size(2)).to(x.device)
                
            # 3. Cross Attention Querying
            attn_out, attn_weights = self.cross_attn(query=v_query, key=eeg_tokens, value=eeg_tokens)
            attn_out = attn_out.squeeze(1)
            attn_weights = attn_weights.squeeze(1)
        else:
            # PURE SAM BASELINE OVERRIDE (Skip EEG)
            if img_emb is not None:
                pooled_img_emb = img_emb.mean(dim=[-1, -2])
                attn_out = self.visual_projector(pooled_img_emb)
            else:
                attn_out = torch.zeros(x.size(0), self.classifier[0].in_features).to(x.device)
            # Dummy attention weights, since EEG is unbound
            attn_weights = torch.zeros(x.size(0), 1).to(x.device)
        
        # 4. Final Object Preference Score
        pref_logits = self.classifier(attn_out)
        
        # 5. Spatial SAM Mask Decoding
        # if img_emb is not None:
        # Project the learned cross-attention focus back to SAM's 256D prompt space
        # SAM expects (batch_size, point_batch_size, num_prompts, embed_dim) -> (B, 1, 1, 256)
        sparse_embeddings = self.prompt_projector(attn_out).unsqueeze(1).unsqueeze(2) # (B, 1, 1, 256)
        
        # Construct dummy dense embeddings (padding) - No gradients needed
        batch_size = img_emb.shape[0]
        dense_embeddings = self.sam_no_mask_embed.expand(batch_size, -1, 64, 64).to(x.device).detach()
        image_positional_embeddings = self.sam_pos_embed.repeat(batch_size, 1, 1, 1).to(x.device).detach()
        
        # Forward pass through HuggingFace SAM MaskDecoder
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=img_emb,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Differentiable Bounding Box Extraction via Spatial Expectation of the Mask
        mask_logits = low_res_masks.squeeze(1).squeeze(1) # (B, 1, 1, 256, 256) -> (B, 256, 256)
        probs = torch.sigmoid(mask_logits) # (B, 256, 256)
        
        x_grid = self.x_grid.expand(batch_size, self.H, self.W)
        y_grid = self.y_grid.expand(batch_size, self.H, self.W)
        
        prob_sum = probs.sum(dim=[1, 2], keepdim=True) + 1e-6
        p_norm = probs / prob_sum
        
        cx = (p_norm * x_grid).sum(dim=[1, 2])
        cy = (p_norm * y_grid).sum(dim=[1, 2])
        var_x = (p_norm * (x_grid - cx.view(batch_size, 1, 1))**2).sum(dim=[1, 2])
        var_y = (p_norm * (y_grid - cy.view(batch_size, 1, 1))**2).sum(dim=[1, 2])
        
        w = 2 * torch.sqrt(torch.clamp(var_x, min=1e-6))
        h = 2 * torch.sqrt(torch.clamp(var_y, min=1e-6))
        
        xmin = torch.clamp(cx - w/2, 0.0, 1.0)
        ymin = torch.clamp(cy - h/2, 0.0, 1.0)
        xmax = torch.clamp(cx + w/2, 0.0, 1.0)
        ymax = torch.clamp(cy + h/2, 0.0, 1.0)
        
        bbox_pred = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        # else:
        #     bbox_pred = torch.zeros(x.size(0), 4).to(x.device)
        
        return pref_logits, bbox_pred, attn_weights
