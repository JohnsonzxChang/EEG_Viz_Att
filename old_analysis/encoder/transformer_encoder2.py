# pytorch >= 1.9 (2.8.0 OK)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf.base import BaseConfig

# ------------------------
# Sine-cos Positional Encoding（batch_first）
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)              # (T, D)
        pos = torch.arange(0, max_len).unsqueeze(1)     # (T,1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0).to(x.dtype)
        return self.dropout(x)

# ------------------------
# (B,M,T) -> (B,D) 回归：Encoder self-attn + Decoder self-attn + Cross-attn
# 仅需传入 src，不需要任何 mask / 额外特征
# ------------------------
class RegressionTransformer(nn.Module):
    def __init__(self, conf: BaseConfig):
        super().__init__()
        # 必需配置
        d_in   = conf.enc_in        # = M
        d_out  = conf.num_classes         # 输出维度 D
        d_model = conf.d_model
        nhead   = conf.n_heads
        e_layers = conf.e_layers
        d_layers = conf.d_layers
        d_ff     = conf.d_ff
        dropout  = conf.dropout

        # 可选：查询 token 个数（默认 1）
        self.num_queries = getattr(conf, "num_queries", 1)

        self.d_model = d_model
        self.d_out   = d_out

        # 编码端：M -> d_model
        self.src_proj = nn.Linear(d_in, d_model)
        self.src_pe = PositionalEncoding(d_model, dropout)

        # 解码端：查询标量 -> d_model（内部生成，不对外暴露）
        self.tgt_proj = nn.Linear(1, d_model)
        self.tgt_pe = PositionalEncoding(d_model, dropout)
        self.register_parameter("query_token", nn.Parameter(torch.zeros(1, 1, 1)))

        # Transformer（含 encoder self-attn、decoder self-attn、cross-attn）
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=e_layers,
            num_decoder_layers=d_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # 回归头
        self.ln_final = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, src: torch.Tensor, idid=None) -> torch.Tensor:
        """
        src: (B, M, T) —— 多变量时域信号
        return: (B, D) —— 回归输出
        """
        B, M, T = src.shape

        # 1) (B,M,T) -> (B,T,M)，投影 + 位置编码
        src_seq = src.transpose(1, 2).contiguous()          # (B,T,M)
        src_emb = self.src_pe(self.src_proj(src_seq))       # (B,T,Dm)

        # 2) 内部构造 Q 个查询 token（无额外特征）
        tgt_in  = self.query_token.expand(B, self.num_queries, 1)  # (B,Q,1)
        tgt_emb = self.tgt_pe(self.tgt_proj(tgt_in))               # (B,Q,Dm)

        # 3) Transformer 解码（带 cross-attn 到 encoder）
        dec = self.tf(src=src_emb, tgt=tgt_emb)                    # (B,Q,Dm)
        dec = self.ln_final(dec)

        # 4) 聚合为 (B,D)：默认取第一个查询，也可改为 dec.mean(dim=1)
        rep = dec.mean(dim=1) # dec[:, 0, :]                                         # (B,Dm)
        y = self.out_proj(rep)                                     # (B,D)
        return y
