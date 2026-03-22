"""Window-based multi-head self-attention block.

Implements W-MSA and SW-MSA (Shifted Window Multi-head Self-Attention)
following the Swin Transformer design [Liu2021], adapted for use as a
drop-in replacement for the conv-based ``AttentionBlock`` in learned
image compression models.

[Liu2021]: "Swin Transformer: Hierarchical Vision Transformer using
Shifted Windows", Liu et al., ICCV 2021.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["WindowAttentionBlock"]


def _largest_divisor(n: int, cap: int) -> int:
    """Return the largest divisor of *n* that is <= *cap*."""
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition (B, H, W, C) into (B*nW, ws*ws, C) windows."""
    B, H, W, C = x.shape
    ws = window_size
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
    return windows


def _window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """Reverse window partition: (B*nW, ws*ws, C) -> (B, H, W, C)."""
    ws = window_size
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window.

    Supports relative position bias.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.view(2, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (num_windows*B, N, C) where N = ws*ws
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """Feedforward network with GELU activation."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttentionBlock(nn.Module):
    """Window-based multi-head self-attention with optional shifted windows.

    Drop-in replacement for ``AttentionBlock``.  Input and output tensors
    are ``(B, C, H, W)`` (channels-first, matching PyTorch conv convention).

    Applies W-MSA within non-overlapping windows of size
    ``window_size x window_size``.  When ``shift=True``, features are rolled
    by ``window_size // 2`` and an attention mask prevents information leakage
    across shifted boundaries (SW-MSA, following Swin Transformer).

    Args:
        dim: Number of input/output channels.
        num_heads: Number of attention heads (adjusted down if *dim* is
            not evenly divisible).
        window_size: Side length of the attention window.
        shift: If ``True``, use shifted-window attention (SW-MSA).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = _largest_divisor(dim, num_heads)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, expansion=4)

    def _build_mask(self, H: int, W: int, device: torch.device) -> Tensor | None:
        if self.shift_size == 0:
            return None
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, ws)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)
        shortcut = x

        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        attn_mask = self._build_mask(Hp, Wp, x.device)

        x_windows = _window_partition(x, ws)
        x_windows = self.attn(x_windows, mask=attn_mask)
        x = _window_reverse(x_windows, ws, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

        if pad_b > 0 or pad_r > 0:
            x = x[:, :, :H, :W]

        return x
