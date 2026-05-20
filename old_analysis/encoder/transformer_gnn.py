import torch
import torch.nn as nn
import math
from conf.base import BaseConfig

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TPatch(nn.Module):
    """
    TPatch module to partition time series into patches.
    """
    def __init__(self, patch_len, stride, d_model, in_channels):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(in_channels * patch_len, d_model)

    def forward(self, x):
        # x: [B, M, T1]
        n_patches = (x.shape[2] - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [B, M, n_patches, patch_len]
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], n_patches, -1)
        # patches: [B, n_patches, M * patch_len]
        patched_x = self.projection(patches) # [B, n_patches, d_model]
        return patched_x

class SSMBlock(nn.Module):
    """
    A simplified SSM-like block using depthwise separable convolutions.
    """
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        res = x
        x = x.permute(0, 2, 1) # [B, D, L]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1) # [B, L, D]
        x = self.activation(x)
        x = self.norm(x + res)
        return x

class TransForcast(nn.Module):
    """
    Transformer-based model for time series forecasting.
    
    Args:
        args: An object containing model hyperparameters. Expected attributes:
            - enc_in (int): Number of input features/channels (M).
            - seq_len (int): Input sequence length (T1).
            - pred_len (int): Prediction sequence length (T2).
            - d_model (int): Latent dimension.
            - n_heads (int): Number of heads in multi-head attention.
            - e_layers (int): Number of encoder layers.
            - d_layers (int): Number of decoder layers.
            - d_ff (int): Dimension of the feed-forward network.
            - dropout (float): Dropout rate.
            - activation (str): Activation function ('relu' or 'gelu').
            - use_tpatch (bool): Whether to use the TPatch module.
            - use_ssm (bool): Whether to use the SSM block.
            - patch_len (int): Length of each patch (if use_tpatch is True).
            - stride (int): Stride between patches (if use_tpatch is True).
            - ssm_kernel_size (int): Kernel size for the SSM block's convolution.
    """
    def __init__(self, args:BaseConfig):
        super(TransForcast, self).__init__()
        self.args = args

        # 1. Input Embedding / Patching
        if getattr(args, 'use_tpatch', False):
            self.patching = TPatch(args.patch_len, args.stride, args.d_model, args.enc_in)
        else:
            self.input_proj = nn.Linear(args.enc_in, args.d_model)

        self.pos_encoder = PositionalEncoding(args.d_model, args.dropout)

        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model, nhead=args.n_heads, dim_feedforward=args.d_ff,
            dropout=args.dropout, activation=args.activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.e_layers)

        # 3. Optional SSM Block
        if getattr(args, 'use_ssm', False):
            self.ssm = SSMBlock(args.d_model, kernel_size=getattr(args, 'ssm_kernel_size', 3))

        # 4. Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model, nhead=args.n_heads, dim_feedforward=args.d_ff,
            dropout=args.dropout, activation=args.activation, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.d_layers)
        self.decoder_input = nn.Parameter(torch.randn(1, args.pred_len, args.d_model))

        # 5. Output Projection
        self.output_layer = nn.Linear(args.d_model, 1)

    def forward(self, x_enc, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        # x_enc: [Batch, Channels, Time] -> [B, M, T1]
        
        # 1. Patching and Embedding
        if getattr(self.args, 'use_tpatch', False):
            enc_out = self.patching(x_enc)  # [B, num_patches, d_model]
        else:
            enc_out = x_enc.permute(0, 2, 1) # [B, T1, M]
            enc_out = self.input_proj(enc_out) # [B, T1, d_model]

        # Add positional encoding
        enc_out_pos = enc_out.permute(1, 0, 2) # [L, B, D]
        enc_out_pos = self.pos_encoder(enc_out_pos)
        enc_out = enc_out_pos.permute(1, 0, 2) # [B, L, D]

        # 2. Encoder
        memory = self.transformer_encoder(enc_out, src_key_padding_mask=padding_mask)

        # 3. Optional SSM
        if getattr(self.args, 'use_ssm', False):
            memory = self.ssm(memory)

        # 4. Decoder
        tgt = self.decoder_input.repeat(x_enc.size(0), 1, 1)
        tgt_pos = tgt.permute(1, 0, 2) # [pred_len, B, D]
        tgt_pos = self.pos_encoder(tgt_pos)
        tgt = tgt_pos.permute(1, 0, 2) # [B, pred_len, D]

        dec_out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=dec_self_mask, memory_key_padding_mask=padding_mask)

        # 5. Output Projection
        output = self.output_layer(dec_out) # [B, pred_len, 1]
        
        return output.squeeze(-1) # [B, pred_len]
