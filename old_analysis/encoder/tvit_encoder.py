import torch
import torch.nn as nn
from conf import BaseConfig

class TemporalPatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        # We use a 1D Convolution to extract features within a temporal patch
        # stride=patch_size ensures non-overlapping temporal windows
        self.proj = nn.Conv1d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.proj(x) # -> (Batch, EmbedDim, NumPatches)
        x = x.transpose(1, 2) # -> (Batch, NumPatches, EmbedDim)
        return x

class TViT_Encoder(nn.Module):
    """
    Temporal Vision Transformer for EEG.
    Splits the continuous `T` dimension into short non-overlapping patches, 
    projects them to `embed_dim`, and passes them through a Transformer Encoder.
    Outputs the full `(B, N, D)` sequence instead of pooling.
    """
    def __init__(self, args: BaseConfig):
        super().__init__()
        self.args = args
        
        # Determine number of channels (from config chn_sel)
        in_channels = len(args.chn_sel)
        
        # Hyperparameters for T-ViT (can be added to BaseConfig later, using defaults here)
        self.patch_size = getattr(args, 'patch_size', 25) # e.g. 50ms at 500Hz
        self.embed_dim = getattr(args, 'feat_dim', 128)
        self.num_heads = getattr(args, 'num_heads', 4)
        self.num_layers = getattr(args, 'num_layers', 3)
        self.dropout = getattr(args, 'dropout', 0.1)

        # 1. Patch Embedding
        self.patch_embed = TemporalPatchEmbedding(
            in_channels=in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        
        # Calculate Number of Patches 
        # (Assuming padding has been done if t_len is not perfectly divisible)
        num_patches = args.t_len // self.patch_size
        
        # 2. Positional Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 4. Final Layer Normalization
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        return self.forward_features(x, padding_mask, enc_self_mask, dec_self_mask)

    def forward_features(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        """
        Extracts temporal tokens.
        x: (B, C, T)
        Returns: 
           tokens: (B, N, D)  -> N is number of time patches
        """
        # (B, C, T) -> (B, N, D)
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer (B, N, D) -> (B, N, D)
        x = self.transformer(x)
        
        # Final Norm
        x = self.norm(x)
        
        return x
