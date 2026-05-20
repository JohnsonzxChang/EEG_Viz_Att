# pytorch >= 1.12（支持 Transformer batch_first）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from conf.base import BaseConfig

# --- NEW: 一个空壳的自注意力模块，只为让 TransformerEncoder 读取 batch_first ---
class _DummySelfAttn(nn.Module):
    def __init__(self, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
    def forward(self, *args, **kwargs):
        raise RuntimeError("Dummy self_attn should not be called")

# ---------- 1) HH 风格的门控 + Mamba-like 扫描 ----------
class HHGates(nn.Module):
    """
    HH 风格的慢变量门控：m, h, n, c
    - x_{t+1} = x_t + dt * (x_inf(v_t) - x_t) / tau(v_t)
    - 全部 1x1 映射，保持 O(T*D)
    """
    def __init__(self, d_model: int, tau_min: float = 1e-2):
        super().__init__()
        self.v_proj = nn.Linear(d_model, d_model)  # "membrane potential" proxy v_t

        def branch():
            return nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

        self.m_inf = branch()
        self.h_inf = branch()
        self.n_inf = branch()
        self.c_inf = branch()

        self.m_tau = nn.Linear(d_model, d_model)
        self.h_tau = nn.Linear(d_model, d_model)
        self.n_tau = nn.Linear(d_model, d_model)
        self.c_tau = nn.Linear(d_model, d_model)

        self.tau_min = tau_min

    def _update_one(self, x, v, f_inf, f_tau, dt):
        x_inf = torch.sigmoid(f_inf(v))           # (0,1)
        tau   = F.softplus(f_tau(v)) + self.tau_min
        # 指数积分会更稳：x_inf + (x - x_inf) * exp(-dt/tau)
        return x_inf + (x - x_inf) * torch.exp(-dt / tau)

    def step(self, x_t: torch.Tensor, gates: Dict[str, torch.Tensor], dt: float):
        v_t = self.v_proj(x_t)
        m = self._update_one(gates['m'], v_t, self.m_inf, self.m_tau, dt)
        h = self._update_one(gates['h'], v_t, self.h_inf, self.h_tau, dt)
        n = self._update_one(gates['n'], v_t, self.n_inf, self.n_tau, dt)
        c = self._update_one(gates['c'], v_t, self.c_inf, self.c_tau, dt)
        return {'m': m, 'h': h, 'n': n, 'c': c}, v_t


class HHMambaBlock(nn.Module):
    """
    线性时间扫描核（极简版）：
      x_{t+1} = A_t x_t + B_t u_t
      y_t     = C_t x_{t+1}
    其中 A_t = diag(exp(-dt * lambda_t))，lambda_t 由 HH 门控（n↑增强遗忘，e=m^3 h ↓遗忘）调制；
         B_t, C_t 由门控缩放（e,k,c）。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.hh = HHGates(d_model)

        # 基础对角参数（稳定参数化）
        self.lambda0 = nn.Parameter(torch.zeros(d_model))
        self.B_base  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.C_base  = nn.Parameter(torch.randn(d_model) * 0.02)

        # 门控 → 参数缩放/调制（逐通道 1x1）
        self.mix_B = nn.Linear(d_model * 3, d_model)
        self.mix_C = nn.Linear(d_model * 3, d_model)
        self.mix_l = nn.Linear(d_model * 2, d_model)

        # 可选：输入投影到 d_model（若上游不是同维度）
        self.in_proj  = nn.Identity()
        self.out_proj = nn.Identity()

    @torch.jit.export
    def _apply_pad_mask_(self, U: torch.Tensor, pad_mask: Optional[torch.Tensor], *,
                     is_hf_mask: bool = False):
        """
        统一把各种类型的 mask 转成 bool (B,T)：
        - float/bfloat/int -> 非零为 True
        - (B,T,1) -> squeeze 到 (B,T)
        - HF attention_mask (1=keep,0=pad): 传 is_hf_mask=True 会自动取反
        True 表示 padding（需要被置零）
        """
        if pad_mask is None:
            return U

        m = pad_mask
        if m.dim() == 3 and m.size(-1) == 1:
            m = m.squeeze(-1)
        if m.dtype is not torch.bool:
            # 默认：非零视为 True
            m = (m != 0)

        if is_hf_mask:
            # HF: 1=keep,0=pad；当前需要 True=pad
            m = ~m

        if m.shape != U.shape[:2]:
            raise RuntimeError(f"src_key_padding_mask shape {m.shape} must be (B,T) to match inputs {(U.shape[0], U.shape[1])}")

        m = m.to(device=U.device)
        U.masked_fill_(m.unsqueeze(-1), 0.0)
        return U


    def forward(self, U: torch.Tensor, dt: float = 1.0, pad_mask: Optional[torch.Tensor] = None):
        """
        U: (B, T, D) 输入序列
        返回: (B, T, D) 输出序列
        """
        assert U.dim() == 3 and U.size(-1) == self.d, "Expect (B,T,D=d_model)"
        B, T, D = U.shape
        U = self._apply_pad_mask_(U, pad_mask)  # 若传的是 HF 的 attention_mask，加 is_hf_mask=True

        x = torch.zeros(B, D, device=U.device, dtype=U.dtype)
        gates = {k: torch.zeros_like(x) for k in ('m', 'h', 'n', 'c')}
        Y = []

        for t in range(T):
            gates, v_t = self.hh.step(x, gates, dt)
            e = gates['m']**3 * gates['h']    # Na-like
            k = gates['n']**4                 # K-like
            c = gates['c']                    # Ca-like

            b_scale = torch.sigmoid(self.mix_B(torch.cat([e, k, c], dim=-1)))
            c_scale = torch.sigmoid(self.mix_C(torch.cat([e, k, c], dim=-1)))
            lam_add = self.mix_l(torch.cat([k, e], dim=-1))
            lam = F.softplus(self.lambda0 + lam_add)       # >= 0
            A_diag = torch.exp(-dt * lam)

            B_t = self.B_base * b_scale
            C_t = self.C_base * c_scale

            x = A_diag * x + B_t * U[:, t, :]              # Hadamard 逐通道注入
            y = C_t * x
            Y.append(y)

        return torch.stack(Y, dim=1)  # (B, T, D)


# ---------- 2) 与 TransformerEncoderLayer 同签名的自定义层 ----------
class HHMambaEncoderLayer(nn.Module):
    """
    与 nn.TransformerEncoderLayer 对齐：
    forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False)
    - 结构：Pre-LN -> HHMambaBlock -> 残差 -> Pre-LN -> FFN -> 残差
    - 支持 batch_first=True（默认）
    - src_mask/is_causal 当前不参与（Mamba 为扫描核），pad mask 用零化输入处理
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        hh_dt: float = 1.0,
        hh_tau_min: float = 1e-2,
        # --- NEW: 接收 nhead 以与官方签名对齐（不用它）
        nhead: int = 1,
    ):
        super().__init__()
        assert batch_first, "此实现假定 batch_first=True（输入形状 B,T,D）"
        self.d_model = d_model
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.hh_dt = hh_dt

        # --- NEW: 提供 self_attn 属性以兼容 torch==2.8 的内省 ---
        self.self_attn = _DummySelfAttn(batch_first=batch_first)

        self.hh = HHMambaBlock(d_model)
        self.hh.hh.tau_min = hh_tau_min
  # 直接设置门控最小时常

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # FFN 与官方保持一致
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # src: (B, T, D)
        x = src
        pad_mask = src_key_padding_mask  # (B,T) True=pad
        if self.norm_first:
            y = self.hh(self.norm1(x), dt=self.hh_dt, pad_mask=pad_mask)
            x = x + self.dropout1(y)
            y = self.linear2(self.dropout2(self.act(self.linear1(self.norm2(x)))))
            x = x + self.dropout2(y)
        else:
            y = self.hh(x, dt=self.hh_dt, pad_mask=pad_mask)
            x = self.norm1(x + self.dropout1(y))
            y = self.linear2(self.dropout2(self.act(self.linear1(x))))
            x = self.norm2(x + self.dropout2(y))
        return x

class HHMambaEncoder(nn.Module):
    def __init__(self, args:BaseConfig):
        super(HHMambaEncoder, self).__init__()

        self.encoder_layer = HHMambaEncoderLayer(
            d_model=args.enc_in,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            hh_dt=1.0,
            hh_tau_min=1e-2,
            nhead=args.n_heads,   # <- NEW
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.e_layers)
        self.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(args.enc_in * args.t_len, args.num_classes)
        )

    def forward(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        # x: [Batch, Channel, Input length]
        # TransformerEncoder expects [Input length, Batch, Channel]
        x = x.permute(2, 0, 1)
        output = self.transformer_encoder(x, src_key_padding_mask=None)
        output = output.permute(1, 0, 2)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output

# ---------- 3) 可切换的构建接口 ----------
# def build_encoder(
#     d_model: int = 256,
#     nhead: int = 8,
#     num_layers: int = 6,
#     dim_feedforward: int = 1024,
#     dropout: float = 0.1,
#     activation: str = "gelu",
#     batch_first: bool = True,
#     norm_first: bool = True,
#     use_hh_mamba: bool = False,
#     hh_dt: float = 1.0,
#     hh_tau_min: float = 1e-2,
# ) -> nn.Module:
#     """
#     返回：nn.TransformerEncoder
#     - 当 use_hh_mamba=False：内部层为官方 TransformerEncoderLayer
#     - 当 use_hh_mamba=True ：内部层为 HHMambaEncoderLayer（签名一致，可无缝复用）
#     """
#     if not use_hh_mamba:
#         layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             batch_first=batch_first,
#             norm_first=norm_first,
#         )
#     else:
#         # 注意：HHMamba 不需要 nhead；保留其他超参一致
#         layer = HHMambaEncoderLayer(
#             d_model=d_model,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             batch_first=batch_first,
#             norm_first=norm_first,
#             hh_dt=hh_dt,
#             hh_tau_min=hh_tau_min,
#             nhead=nhead,   # <- NEW
#         )
#     return nn.TransformerEncoder(layer, num_layers=num_layers)


# # ---------- 4) 简单用例 ----------
# if __name__ == "__main__":
#     from torchinfo import summary
#     B, T, D = 2, 500, 128
#     x = torch.randn(B, T, D)
#     pad_mask = torch.zeros(B, T, dtype=torch.bool)  # 无 padding

#     # A) 标准 Transformer
#     # enc_std = build_encoder(d_model=D, nhead=8, num_layers=4, use_hh_mamba=False)
#     # y_std = enc_std(x, mask=None, src_key_padding_mask=pad_mask)  # (B,T,D)

#     # B) HH-Mamba 版本（可替换，接口不变）
#     enc_hh = build_encoder(d_model=D, num_layers=4, use_hh_mamba=True, hh_dt=1.0)
#     # y_hh = enc_hh(x, mask=None, src_key_padding_mask=pad_mask)
#     summary(enc_hh, input_data=(x, pad_mask))

