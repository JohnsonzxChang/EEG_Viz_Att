from torch import nn
import torch.nn.functional as FF
import torch as th
from einops.layers import torch as eth
import math

from .mills import *
from .mills import _get_norm, _get_non_linear

class TempoEEG_Person_ID_Mux(nn.Module):
    def __init__(self, params):
        super().__init__()
        N = params['N']  # [C1, C2, C3, C4]
        M = params['M']  # EEG channels
        t = params['t']  # [t1, t2]
        s = params['s']  # [s1, s2]
        drop = params['drop']  # [drop1, drop2, drop3]
        Fs = params['Fs']  # sampling frequency
        t_filter = params['t_filter']
        T0 = params['T0']
        H = params['H']
        L = params['L']
        FeaIN = params['Feature']
        use_phase = params['use_phase']
        use_freq = params['use_freq']
        bias = [True, True]
        norm = params['norm']
        non_linear = params['non_linear']
        person = 70 if params['use_beta'] else 35

        assert len(drop) == 3
        assert len(t) == len(s) == len(N) - 2
        FreqList = th.linspace(8, 15.8, 40, requires_grad=False, device=params['device'])
        self.filter_combination = my_conv2(FeaIN, N[0], 1, 1, padding=False, groups=1, bias=bias[0],
                                           norm=norm, non_linear=non_linear, dropout=0)

        self.filter_attention = nn.Sequential(
            nn.Dropout(drop[1]),
            TempoChnFea_2dSqueeze(M=M, t0=t_filter, C=N[0], drop=0, bias=bias[1]),
        )

        self.id_embedding = nn.Embedding(person, params['emb'], max_norm=1)

        self.id_operation2 = nn.Sequential(
            nn.Linear(params['emb'], N[1], bias=True),
            # nn.Dropout(drop[1]),
        )
        self.id_operation = nn.Sequential(
            nn.Linear(params['emb'], M * M, bias=False),
            # nn.Dropout(drop[0]),
            eth.Rearrange('b (m n) -> b m n', m=M),
        )
        # self.id_operation[0].weight.data = th.eye(M).view(M * M, 1)
        # self.id_operation[0].bias.data.fill_(0)

        self.channel_combination = my_conv2(N[0], N[1], (M, 1), (1, 1), padding=False, groups=1, bias=bias[0],
                                            norm=norm, non_linear=non_linear, dropout=drop[0])

        self.channel_attention = nn.Sequential(
            nn.Dropout(drop[1]),
            TempoInfo_1dSqueeze(C=N[1], TN=T0, Fs=Fs, H=H, drop=0, conv_norm=norm, non_linear=non_linear,
                                bias=bias[1], use_phase=use_phase, use_freq=use_freq, FreqList=FreqList)
        )

        self.temporal_extraction = nn.ModuleList()
        TN = T0
        for i in range(len(t)):
            TN = 1 + (TN - t[i]) // s[i]
            self.temporal_extraction.append(
                my_conv2(N[1 + i], N[2 + i], (1, t[i]), (1, s[i]), padding=True,
                         groups=params['group'], bias=bias, norm=norm, non_linear=non_linear, dropout=drop[0]),
            )

        self.temporal_attention = nn.ModuleList()
        TN = T0
        for i in range(len(t)):
            TN = 1 + (TN - t[i]) // s[i]
            Fs = TN * Fs / T0
            self.temporal_attention.append(
                nn.Sequential(
                    nn.Dropout(drop[1]),
                    TempoChnInfo_1dSqueeze(C=N[2 + i], TN=TN, Fs=Fs, H=H, M=1, drop=0, conv_norm=norm,
                                           non_linear=non_linear,
                                           bias=bias[1], use_phase=use_phase, use_freq=use_freq, FreqList=FreqList)
                )
            )
        self.add_norm = nn.ModuleList()
        for i in range(len(N)):
            self.add_norm.append(add_norm(out_channels=N[i], norm=norm, non_linear=non_linear))

        print(f'final temporal length is{TN * N[-1]}')

        self.linear0 = nn.Sequential(
            nn.Dropout(drop[2]),
            eth.Rearrange('b c 1 t -> b (c t)'),
            nn.Linear(TN * N[-1], L),
        )

        self.TN = TN
        self.L = L
        self.loss_cel = nn.CrossEntropyLoss()

    def forward(self, x, id=None):
        assert id is not None
        id = self.id_embedding(id.long())
        x = self.filter_combination(x)
        x = self.add_norm[0](x, self.filter_attention(x))

        # id_base = rearrange(self.id_operation2(id), 'b c -> b c 1 1').expand_as(x)  # [B, N[1]]
        x = th.einsum('bmn,bcmt->bcnt', self.id_operation(id), x)  # [B, N[1]]

        x = self.channel_combination(x) # + id_base)
        x = self.add_norm[1](x, self.channel_attention(x))

        for i in range(len(self.temporal_extraction)):
            x = self.temporal_extraction[i](x)
            x = self.add_norm[2 + i](x, self.temporal_attention[i](x))

        x = self.linear0(x)
        return x
    

class my_conv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding=True, groups=1, bias=True, norm="batch", non_linear="gelu", dropout=None):
        super().__init__()
        if padding == True:
            padding = [(kernel_size[0]-1) // 2, (kernel_size[0]-1) // 2]
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                              groups=groups, bias=bias, padding_mode='replicate')
        self.norm = _get_norm(norm, out_channels)
        self.non_linear = _get_non_linear(non_linear, non_linear=0.1)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        
    def forward(self, x):
        return self.non_linear(self.norm(self.conv(self.dropout(x))))
    
class add_norm(nn.Module):
    def __init__(self, out_channels, norm='batch', non_linear='relu'):
        super().__init__()
        self.norm = _get_norm(norm, out_channels)
        self.non_linear = _get_non_linear(non_linear, non_linear=0.1)
        
    def forward(self, x, x_add):
        return self.non_linear(self.norm(x + x_add))

class freq_compress(nn.Module):
    def __init__(self, TN, Fs, H, use_phase=False, use_freq=False, eps=1e-5, FreqList=th.arange(8, 15.8, 40)):
        super().__init__()
        self.use_phase = use_phase
        self.use_freq = use_freq
        self.eps = eps
        self.H = H
        self.TN = TN
        self.Fs = Fs
        self.C = FreqList.shape[0] * H
        self.upsample = int(FreqList.max() * H * 25 / Fs)
        print(f'sample freq is {Fs} with time length {TN}, upsample is {self.upsample}')
        self.position = th.arange(0, self.TN * self.upsample - 2 * self.upsample, requires_grad=False).unsqueeze(1) / Fs

        if use_freq:
            self.ori_freq = nn.Parameter(FreqList)
        else:
            self.ori_freq = FreqList.requires_grad_(False)
        if use_phase:
            self.ori_phase = nn.Parameter(th.randn(FreqList.shape[0]) * math.pi)
        else:
            self.ori_phase = th.zeros(FreqList.shape[0], requires_grad=False)

    def get_ref(self, freq, phase):
        pe = []
        for i in range(self.H):
            pe.append(th.sin(math.pi * 2 * self.position.to(freq.device) * freq * (i + 1) + phase).unsqueeze(dim=-1))  # T, F, 1
        pe = th.cat(pe, dim=-1)
        return pe  # T, F, H

    def forward(self, x):
        # x: {B, C, T}
        assert x.shape[1:] == (self.C, self.TN), f'input shape is {x.shape}'
        x = upsample(x, self.upsample)
        # print(f'upsample shape is {x.shape}')
        ref = rearrange(self.get_ref(self.ori_freq.to(x.device), self.ori_phase.to(x.device)), 't f h -> (f h) t')
        assert x.shape[1:] == ref.shape, f'input shape is {x.shape}, ref shape is {ref.shape}'
        x = make_corr_3(x, ref, eps=self.eps)
        return x  # B, C

class senet(nn.Module):
    def __init__(self, C, drop, compress=4, no_linear='none', bias=False):
        super().__init__()
        no_linear = _get_non_linear(no_linear)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C // compress, bias=bias),
            no_linear,
            nn.Linear(C // compress, C, bias=bias),
            nn.Dropout(drop),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, C, 1, 1}
        y = rearrange(self.avg_pool(x), 'b c 1 1 -> b c')
        y = rearrange(self.fc(y), 'b c -> b c 1 1')
        return x * y.expand_as(x)


class TempoInfo_1dSqueeze(nn.Module):
    def __init__(self, C, TN, Fs, H, drop, conv_norm='batch', non_linear='gelu',
                 bias=False, use_phase=True, use_freq=True, eps=1e-5, FreqList=th.linspace(8, 15.8, 40)):
        super().__init__()
        non_linear = _get_non_linear(non_linear)
        self.embed = nn.Sequential(
            nn.Conv2d(C, FreqList.shape[0] * H, kernel_size=1, bias=bias),
            _get_norm(conv_norm, FreqList.shape[0] * H),
            non_linear,
            nn.Dropout(drop)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(FreqList.shape[0] * H, C, kernel_size=1, bias=bias),
            _get_norm(conv_norm, C),
            nn.Dropout(drop)
        )
        self.compress = freq_compress(TN, Fs, H, use_phase, use_freq, eps, FreqList)

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, C, M}
        B = x.shape[0]
        y = rearrange(self.embed(x), 'b c m t -> (b m) c t')
        y = rearrange(self.compress(y), '(b m) c -> b c m 1', b=B)
        y = FF.sigmoid(self.decode(y))
        return x * y.expand_as(x)

class TempoChnInfo_1dSqueeze(nn.Module):
    def __init__(self, C, TN, Fs, H, M, drop, conv_norm='batch', non_linear='gelu',
                 bias=False, use_phase=True, use_freq=True, eps=1e-5, FreqList=th.linspace(8, 15.8, 40)):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(C, FreqList.shape[0] * H, kernel_size=1, bias=bias),
            _get_norm(conv_norm, FreqList.shape[0] * H),
            _get_non_linear(non_linear),
            nn.Dropout(drop)
        )
        self.decode = nn.Sequential(
            nn.Linear(FreqList.shape[0] * H, C, bias=bias),
            # conv_norm(C),
            nn.Dropout(drop)
        )
        self.chncombine = nn.Sequential(
            nn.Linear(M, 1, bias=bias),
            _get_non_linear(non_linear),
        )
        self.compress = freq_compress(TN, Fs, H, use_phase, use_freq, eps, FreqList)

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, C}
        B = x.shape[0]
        y = rearrange(self.embed(x), 'b c m t -> (b m) c t')
        y = self.chncombine(rearrange(self.compress(y), '(b m) c -> b c m', b=B))
        y = rearrange(FF.sigmoid(self.decode(y.squeeze(-1))), 'b c -> b c 1 1')
        return x * y.expand_as(x)


class TempoFilterInfo_1dSqueeze(nn.Module):
    def __init__(self, C, TN, Fs, H, drop, conv_norm='batch', non_linear='gelu',
                 bias=False, use_phase=True, use_freq=True, eps=1e-5, FreqList=th.linspace(8, 15.8, 40)):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(C, FreqList.shape[0] * H, kernel_size=1, bias=bias),
            _get_norm(conv_norm, FreqList.shape[0] * H),
            _get_non_linear(non_linear),
            nn.Dropout(drop)
        )
        self.decode = nn.Sequential(
            nn.Linear(FreqList.shape[0] * H, 1, bias=bias),
            # conv_norm(C),
            nn.Dropout(drop)
        )
        self.compress = freq_compress(TN, Fs, H, use_phase, use_freq, eps, FreqList)

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, M}
        B = x.shape[0]
        y = rearrange(self.embed(x), 'b c m t -> (b m) c t')
        y = self.compress(y)
        y = rearrange(FF.sigmoid(self.decode(y)), '(b m) 1 -> b 1 m 1', b=B)
        return x * y.expand_as(x)


class TempoChnFea_2dSqueeze(nn.Module):
    def __init__(self, C, M, t0, drop, bias=False):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(C, 1, kernel_size=(M, t0), padding=(0, math.ceil((t0 - 1) / 2)), bias=bias,
                      padding_mode='replicate'),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, 1, 1, T}
        y = self.embed(x)
        y = FF.sigmoid(y)
        return x * y.expand_as(x)

class TempoFreqFea_2dSqueeze(nn.Module):
    def __init__(self, M, t0, f0, drop, bias=False):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(M, 1, kernel_size=(f0, t0), padding=(math.ceil((f0 - 1) / 2), math.ceil((t0 - 1) / 2)), bias=bias,
                      padding_mode='replicate'),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, C, 1, T}
        y = self.embed(rearrange(x, 'b c m t -> b m c t'))
        y = rearrange(FF.sigmoid(y), 'b 1 c t -> b c 1 t')
        return x * y.expand_as(x)

class TempoAllFea_2dSqueeze(nn.Module):
    def __init__(self, M, C, t0, drop, bias=False):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(M * C, 1, kernel_size=(t0), padding=math.ceil((t0 - 1) / 2), bias=bias, padding_mode='replicate'),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: {B, C, M, T}, y: {B, 1, 1, T}
        y = self.embed(rearrange(x, 'b c m t -> b (c m) t'))
        y = rearrange(FF.sigmoid(y), 'b 1 t -> b 1 1 t')
        return x * y.expand_as(x)