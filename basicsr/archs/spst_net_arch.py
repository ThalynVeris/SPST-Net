import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
from timm.models.layers import DropPath


# =============================================================================
# Utility Functions
# =============================================================================

def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Layer Normalization
# =============================================================================

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class CustomLayerNorm(nn.Module):
    def __init__(self, dim):
        super(CustomLayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm4D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# =============================================================================
# SPIB: Spectral Processing and Integration Block
# =============================================================================

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim > 0, "dim must be greater than 0"
        self.conv_real = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_imag = nn.Conv2d(dim, dim, kernel_size=1)

        reduction_ratio = max(1, dim // 16)
        self.channel_attention_real = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        for layer in self.channel_attention_real:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

        self.channel_attention_imag = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        for layer in self.channel_attention_imag:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        original_dtype = x.dtype

        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        with torch.cuda.amp.autocast(enabled=False):
            x = x.view(B, a, b, C)
            x_fft = torch.fft.rfft2(x.float(), dim=(1, 2), norm='ortho')
            real_part = x_fft.real.permute(0, 3, 1, 2)
            imag_part = x_fft.imag.permute(0, 3, 1, 2)

            real_conv = self.conv_real(real_part)
            imag_conv = self.conv_imag(imag_part)

            real_attention = self.channel_attention_real(real_conv)
            imag_attention = self.channel_attention_imag(imag_conv)

            real_conv = real_conv * (1 + real_attention)
            imag_conv = imag_conv * (1 + imag_attention)

            x_fft_enhanced = torch.complex(real_conv.permute(0, 2, 3, 1), imag_conv.permute(0, 2, 3, 1))
            x = torch.fft.irfft2(x_fft_enhanced, s=(a, b), dim=(1, 2), norm='ortho')
            x = x.reshape(B, N, C)

        return x.to(original_dtype)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SPIB(nn.Module):
    """Spectral Processing and Integration Block."""

    def __init__(self, dim, mlp_ratio=2., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=CustomLayerNorm):
        super(SPIB, self).__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, spatial_size):
        residual = x
        y = self.norm1(x)
        y = to_3d(y)
        y = self.filter(y, spatial_size=spatial_size)
        y = to_4d(y, spatial_size[0], spatial_size[1])
        x = residual + self.drop_path(y)

        residual = x
        y = self.norm2(x)
        y = to_3d(y)
        y = self.mlp(y)
        y = to_4d(y, spatial_size[0], spatial_size[1])
        x = residual + self.drop_path(y)
        return x


# =============================================================================
# TAB: Token Aggregation Block
# =============================================================================

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class IRCA(nn.Module):
    """Implicit Reference Centroid Attention."""

    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        if self.training:
            x_global = center_iter(F.normalize(normed_x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means
        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h d) -> h n d', h=self.heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.heads)
        return k, v, x_global.detach()


class IASA(nn.Module):
    """Intra-group Aggregation Self-Attention."""

    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)

        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)

        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)

        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)

        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)

        return self.proj(out)


class TAB(nn.Module):
    """Token Aggregation Block."""

    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128, ema_decay=0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

        self.ffn_norm = nn.LayerNorm(dim)
        self.convffn = ConvFFN(dim, mlp_dim)

        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)

        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c -> cnt (b n) c',
                                           cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            x_scores = torch.einsum('b i c, j c -> b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        y = self.iasa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w -> b (h w) c')

        x = x + self.convffn(self.ffn_norm(x), x_size=(h, w))

        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (h w) c -> b c h w', h=h)


# =============================================================================
# DS_MSA: Dual-Scale Multi-head Self-Attention
# =============================================================================

class DS_MSA(nn.Module):
    """Dual-Scale Multi-head Self-Attention with overlapping windows and RPB."""

    def __init__(self, dim, num_heads=4, window_size=16, overlap_ratio=0.5,
                 mlp_ratio=2.0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.scale = (dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                num_heads
            )
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer('relative_position_index', self._calculate_rpi_oca())

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def _calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.overlap_win_size

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size_ori - 1
        relative_coords[:, :, 1] += window_size_ori - 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_position_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, Hp, Wp = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, Hp * Wp, C)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Hp, Wp, C)

        qkv = self.qkv(x).reshape(B, Hp, Wp, 3, C).permute(3, 0, 4, 1, 2)
        q = qkv[0].permute(0, 2, 3, 1)
        kv = torch.cat((qkv[1], qkv[2]), dim=1)

        q_windows = window_partition(q, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)

        kv_windows = self.unfold(kv)
        kv_windows = rearrange(
            kv_windows,
            'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
            nc=2, ch=C, owh=self.overlap_win_size, oww=self.overlap_win_size
        ).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads

        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        x = x.view(B, Hp * Wp, self.dim)

        x = self.proj(x) + shortcut
        x = x + self.mlp(self.norm2(x))

        x = x.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# =============================================================================
# DWT Module
# =============================================================================

class DWT2d(nn.Module):
    """2D Discrete Wavelet Transform (Haar wavelet)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_ll = 0.5 * (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
                       x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2])
        x_lh = 0.5 * (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] -
                       x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2])
        x_hl = 0.5 * (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] +
                       x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2])
        x_hh = 0.5 * (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
                       x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2])
        return x_ll, x_lh, x_hl, x_hh


# =============================================================================
# DTRB: Detail Texture Refinement Block
# =============================================================================

class DTRB(nn.Module):
    """Detail Texture Refinement Block.

    Low-frequency path: Conv 3x3 for smooth structure enhancement.
    High-frequency path: DWT decomposition -> directional weighted fusion -> upsample.
    """

    def __init__(self, dim):
        super().__init__()
        self.mid_dim = dim // 2
        self.dim = dim
        self.act = nn.GELU()

        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

        self.dwt = DWT2d()
        self.direction_weights = nn.Parameter(torch.ones(3) / 3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.last_fc = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        short = x

        lfe = self.act(self.conv(x[:, :self.mid_dim, :, :]))

        high_in = x[:, self.mid_dim:, :, :]
        _, _, H, W = high_in.shape

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            high_in = F.pad(high_in, (0, pad_w, 0, pad_h), mode='reflect')

        _, x_lh, x_hl, x_hh = self.dwt(high_in)

        x_lh = self.max_pool(torch.abs(x_lh))
        x_hl = self.max_pool(torch.abs(x_hl))
        x_hh = self.max_pool(torch.abs(x_hh))

        w = F.softmax(self.direction_weights, dim=0)
        high_freq = w[0] * x_lh + w[1] * x_hl + w[2] * x_hh

        high_freq = F.interpolate(high_freq, size=(H, W), mode='bilinear', align_corners=False)

        hfe = self.act(self.fc(high_freq))

        out = torch.cat([lfe, hfe], dim=1)
        out = short + self.last_fc(out)
        return out


# =============================================================================
# Cross Attention and Gated FFN for FDCAB
# =============================================================================

class CrossAttention(nn.Module):
    """Channel-wise cross attention: Q from detail branch, KV from structural branch."""

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, low, high):
        b, c, h, w = low.shape
        q = self.q_dwconv(self.q(high))
        kv = self.kv_dwconv(self.kv(low))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network."""

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# =============================================================================
# FDCAB: Frequency-Decoupled Cross-Attention Block
# =============================================================================

class FDCAB(nn.Module):
    """Frequency-Decoupled Cross-Attention Block.

    Wraps DS_MSA (structural branch), DTRB (detail branch), and cross attention
    fusion into a unified module.

    Information flow: input -> [DS_MSA || DTRB] -> CrossAttention fusion -> GatedFFN -> output
    """

    def __init__(self, dim, num_heads=4, window_size=16, overlap_ratio=0.5, mlp_ratio=2.0):
        super().__init__()
        self.ds_msa = DS_MSA(
            dim=dim, num_heads=num_heads, window_size=window_size,
            overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio
        )
        self.dtrb = DTRB(dim)

        self.norm1 = LayerNorm4D(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm2 = LayerNorm4D(dim)
        self.ffn = GatedFFN(dim)

    def forward(self, x):
        low = self.ds_msa(x)
        high = self.dtrb(x)

        out = low + self.cross_attn(self.norm1(low), high)
        out = out + self.ffn(self.norm2(out))
        return out


# =============================================================================
# SPST-Net: Main Network
# =============================================================================

@ARCH_REGISTRY.register()
class SPSTNet(nn.Module):
    """SPST-Net for Single Image Super-Resolution.

    Architecture: SPIB -> TAB -> FDCAB -> mid_conv + residual (x block_num)
    """

    setting = dict(
        dim=48, block_num=8, qk_dim=36, mlp_dim=96, heads=4,
    )

    def __init__(
            self,
            in_chans=3,
            n_iters=[5, 5, 5, 5, 5, 5, 5, 5],
            num_tokens=[16, 32, 64, 128, 16, 32, 64, 128],
            group_size=[256, 128, 64, 32, 256, 128, 64, 32],
            upscale: int = 4,
            window_size: int = 16,
            overlap_ratio: float = 0.5,
    ):
        super().__init__()

        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.fdcab_blocks = nn.ModuleList()

        for i in range(self.block_num):
            mlp_ratio = self.mlp_dim / self.dim
            self.blocks.append(nn.ModuleList([
                SPIB(dim=self.dim, mlp_ratio=mlp_ratio),
                TAB(
                    dim=self.dim, qk_dim=self.qk_dim, mlp_dim=self.mlp_dim,
                    heads=self.heads, n_iter=self.n_iters[i],
                    num_tokens=self.num_tokens[i], group_size=self.group_size[i]
                ),
            ]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))
            self.fdcab_blocks.append(FDCAB(
                dim=self.dim, num_heads=self.heads, window_size=window_size,
                overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio
            ))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale in [2, 3]:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(0.1, True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = x
            _, _, h, w = x.shape

            spib, tab = self.blocks[i]

            x = spib(x, spatial_size=(h, w))
            x = tab(x)
            x = self.fdcab_blocks[i](x)

            x = residual + self.mid_convs[i](x)

        return x

    def forward(self, x):
        if self.upscale != 1:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else:
            base = x

        x = self.first_conv(x)
        x = self.forward_features(x) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))

        out = self.last_conv(out) + base
        return out

    def __repr__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return (f'SPSTNet | #Params: {num_params / 1e3:.2f}K | '
                f'SPIB + TAB + FDCAB | {self.block_num} blocks')
