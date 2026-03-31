from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeMLP(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """Non-local self-attention block for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x).view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        time_embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_mlp = TimeMLP(time_embedding_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        down_channels = [base_channels * mult for mult in channel_multipliers]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = base_channels
        for index, out_channels in enumerate(down_channels):
            self.down_blocks.append(ResBlock(current_channels, out_channels, time_embedding_dim, dropout))
            current_channels = out_channels
            if index != len(down_channels) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid_block1 = ResBlock(current_channels, current_channels, time_embedding_dim, dropout)
        self.mid_attn = SelfAttention2d(current_channels)
        self.mid_block2 = ResBlock(current_channels, current_channels, time_embedding_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(down_channels))
        for index, skip_channels in enumerate(reversed_channels):
            in_block_channels = current_channels + skip_channels
            out_channels = skip_channels
            self.up_blocks.append(ResBlock(in_block_channels, out_channels, time_embedding_dim, dropout))
            current_channels = out_channels
            if index != len(reversed_channels) - 1:
                self.upsamples.append(Upsample(current_channels))

        self.output = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        h = self.input_conv(x)

        skips = []
        downsample_index = 0
        for index, block in enumerate(self.down_blocks):
            h = block(h, time_embedding)
            skips.append(h)
            if index < len(self.down_blocks) - 1:
                h = self.downsamples[downsample_index](h)
                downsample_index += 1

        h = self.mid_block1(h, time_embedding)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_embedding)

        upsample_index = 0
        for index, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_embedding)
            if index < len(self.up_blocks) - 1:
                h = self.upsamples[upsample_index](h)
                upsample_index += 1

        return self.output(h)
