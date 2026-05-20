import torch
from torch import nn
import torch.nn.functional as FF
import torch as th
from einops import rearrange
import numpy as np
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def upsample(x, N=2):
    # x: {*,T}
    # return: {*,N*T}
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    sz = list(x.shape)
    T = sz[-1]
    x = x.reshape(-1, sz[-1])
    sz[-1] = N * T - 2 * N
    mats = np.zeros((T, N * T))
    for i in range(T // 2):
        for j in range(N * 2):
            mats[i * 2, i * 2 * N + j] = (j / N - 1) * (j / N - 2) / ((-1) * (-2))
            mats[min(i * 2 + 1, T - 1), i * 2 * N + j] = (j / N) * (j / N - 2) / ((1) * (-1))
            mats[min(i * 2 + 2, T - 1), i * 2 * N + j] = (j / N) * (j / N - 1) / ((1) * (2))
    return th.matmul(x, th.tensor(mats, dtype=th.float, device=x.device))[:, :-2 * N].reshape(sz).squeeze(0)

def make_corr_3(x, y, eps=1e-5):
    # x: {B, C0, T}, y: {C0, T}
    # return: {B, C0}
    x = (x - th.mean(x, dim=-1, keepdim=True)) / (eps + th.std(x, dim=-1, keepdim=True))
    xx = th.einsum('bct,bct->bc', x, x)
    yy = rearrange(th.einsum('ct,ct->c', y, y), 'c -> 1 c')
    res = th.einsum('bct,ct->bc', x, y) / th.sqrt(xx * yy.expand_as(xx) + eps)
    assert th.isnan(res).sum() == 0, f'err in make layer corr'
    return res

def freq_encoding(freq_list=list(np.linspace(8,15.8,40)), length=51, Fs=250, h=1):
    """
    :param d_model: dimension of the model (channels or dimision)
    :param length: length of positions (time point)
    :return: length*d_model position matrix
    """
    pe = th.zeros(length, len(freq_list), 2*h)
    position = th.arange(0, length).unsqueeze(1) / Fs
    freq_term = th.tensor(freq_list, dtype=th.float)
    for i in range(h):
        pe[:, :, i*2] = th.sin(math.pi * 2 * position.float() * freq_term * (i+1))
        pe[:, :, i*2+1] = th.cos(math.pi * 2 * position.float() * freq_term * (i+1))
    return pe # {T, F, 2*H}

def freq_phase_encoding(freq_list=list(np.linspace(8,15.8,40)), phase_list=[0]*40, length=51, Fs=250):
    """
    :param d_model: dimension of the model (channels or dimision)
    :param length: length of positions (time point)
    :return: length*d_model position matrix
    """
    d_model = len(freq_list)
    position = th.arange(0, length).unsqueeze(1) / Fs
    freq_term = th.tensor(freq_list, dtype=th.float)
    phase_term = th.tensor(phase_list, dtype=th.float)
    pe = th.sin(math.pi * 2 * position.float() * freq_term + phase_term)
    return pe # {T, F}

def freq_phase_harmonic_encoding(freq_list=list(np.linspace(8,15.8,40)), phase_list=[0]*40, h=1, length=51, Fs=250):
    """
    :param d_model: dimension of the model (channels or dimision)
    :param length: length of positions (time point)
    :return: length*d_model position matrix
    """
    assert h > 0, 'h must be positive'
    d_model = len(freq_list)
    pe = th.zeros(length, d_model*h)
    position = th.arange(0, length).unsqueeze(1) / Fs
    freq_term = th.tensor(freq_list, dtype=th.float)
    phase_term = th.tensor(phase_list, dtype=th.float)#.unsqueeze(0).expand_as(position.float() * freq_term)
    for i in range(h):
        pe[:,i*d_model:(i+1)*d_model] = th.sin(math.pi * 2 * position.float() * freq_term * (i+1) + phase_term)
    return pe # {T, F*H}


def make_corr_4(x, y, eps=1e-5):
    # x: {B, C0, M, T}, y: {C0, T}
    # return: {B, C0, M}
    x = (x - th.mean(x, dim=-1, keepdim=True)) / (eps + th.std(x, dim=-1, keepdim=True))
    xx = th.einsum('bcmt,bcmt->bcm', x, x)
    yy = rearrange(th.einsum('ct,ct->c', y, y), 'c -> 1 c 1')
    res = th.einsum('bcmt,ct->bcm', x, y) / th.sqrt(xx * yy.expand_as(xx) + eps)

    assert th.isnan(res).sum() == 0, f'err in make layer corr'
    return res

def make_corr_same_4(x, y, eps=1e-5):
    # x: {B, C0, M, T}, y: {B, C0, M, T}
    # return: {B, C0, M}
    x = (x - th.mean(x, dim=-1, keepdim=True)) / (eps + th.std(x, dim=-1, keepdim=True))
    y = (y - th.mean(y, dim=-1, keepdim=True)) / (eps + th.std(y, dim=-1, keepdim=True))
    res = th.einsum('bcmt,bcmt->bcm', x, y) / th.sqrt(
        th.einsum('bcmt,bcmt->bcm', x, x) * th.einsum('bcmt,bcmt->bcm', y, y) + eps
    )
    assert th.isnan(res).sum() == 0, f'err in make layer corr'
    return res

def get_max_corr(x, ref, eps=1e-5):
    # x: {B, C0, T}, ref: {F, T}
    # return: {B, C0, F}
    x = (x - th.mean(x, dim=-1, keepdim=True)) / (eps + th.std(x, dim=-1, keepdim=True))
    res = []
    for i in range(ref.shape[0]):
        xx = th.einsum('bct,bct->bc1', x, x)
        yy = th.sum(ref[i] * ref[i])
        res.append(th.einsum('bct,t->bc1', x, ref[i]) / th.sqrt(xx * yy + eps))
    res = th.cat(res, dim=-1)
    return res

def _get_norm(norm, out_channels):
    assert norm in ['batch', 'channels_first', 'channels_last', 'none']
    if norm == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif norm == 'channels_first':
        return LayerNorm(out_channels, data_format='channels_first')
    elif norm == 'channels_last':
        return LayerNorm(out_channels, data_format='channels_last')
    elif norm == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError

def _get_non_linear(m, **para):
    assert m in ['relu', 'elu', 'gelu', 'sigmoid', 'leaky_relu', 'none']
    if m == 'relu':
        return nn.ReLU(inplace=True)
    elif m == 'elu':
        return nn.ELU(alpha=para['non_linear'], inplace=True)
    elif m == 'gelu':
        return nn.GELU()
    elif m == 'sigmoid':
        return nn.Sigmoid()
    elif m == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=para['non_linear'], inplace=True)
    elif m == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return FF.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class SWATS(torch.optim.Optimizer):
    r"""Implements Switching from Adam to SGD technique. Proposed in
    `Improving Generalization Performance by Switching from Adam to SGD`
    by Nitish Shirish Keskar, Richard Socher (2017).
    The method applies Adam in the first phase of the training, then
    switches to SGD when a criteria is met.
    Implementation of Adam and SGD update are from `torch.optim.Adam` and
    `torch.optim.SGD`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, verbose=False,
                 nesterov=True):
        if not 0.0 <= lr:
            raise ValueError(
                "Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(
                "Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, phase='ADAM',
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        verbose=verbose, nesterov=nesterov)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('nesterov', False)
            group.setdefault('verbose', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional):
                A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for w in group['params']:
                if w.grad is None:
                    continue
                grad = w.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, '
                        'please consider SparseAdam instead')

                amsgrad = group['amsgrad']

                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(w.data)
                    # exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(w.data)
                    # moving average for the non-orthogonal projection scaling
                    state['exp_avg2'] = w.new(1).fill_(0)
                    if amsgrad:
                        # maintains max of all exp. moving avg.
                        # of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(w.data)

                exp_avg, exp_avg2, exp_avg_sq = \
                    state['exp_avg'], state['exp_avg2'], state['exp_avg_sq'],

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], w.data)

                # if its SGD phase, take an SGD update and continue
                if group['phase'] == 'SGD':
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(
                            grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(beta1).add_(grad)
                        grad = buf

                    grad.mul_(1 - beta1)
                    if group['nesterov']:
                        grad.add_(beta1, buf)

                    w.data.add_(-group['lr'], grad)
                    continue

                # decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # maintains the maximum of all 2nd
                    # moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    (bias_correction2 ** 0.5) / bias_correction1

                p = -step_size * (exp_avg / denom)
                w.data.add_(p)

                p_view = p.view(-1)
                pg = p_view.dot(grad.view(-1))

                if pg != 0:
                    # the non-orthognal scaling estimate
                    scaling = p_view.dot(p_view) / -pg
                    exp_avg2.mul_(beta2).add_(1 - beta2, scaling)

                    # bias corrected exponential average
                    corrected_exp_avg = exp_avg2 / bias_correction2

                    # checking criteria of switching to SGD training
                    if state['step'] > 1 and \
                            corrected_exp_avg.allclose(scaling, rtol=1e-6) and \
                            corrected_exp_avg > 0:
                        group['phase'] = 'SGD'
                        group['lr'] = corrected_exp_avg.item()
                        if group['verbose']:
                            print('Switching to SGD after '
                                  '{} steps with lr {:.5f} '
                                  'and momentum {:.5f}.'.format(
                                      state['step'], group['lr'], beta1))

        return loss