from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ResidualBlock(base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ResidualBlock(base_channels * 4),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        self.to_mean = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(x)
        return self.to_mean(hidden), self.to_logvar(hidden)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            ResidualBlock(base_channels * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoencoderKL(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(in_channels, latent_channels, base_channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + noise * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent)
        return reconstruction, mean, logvar

    @staticmethod
    def loss_function(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss = F.mse_loss(reconstruction, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kl_weight * kl_loss
        return loss, {
            "loss": float(loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
        }
