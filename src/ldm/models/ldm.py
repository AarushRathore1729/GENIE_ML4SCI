from __future__ import annotations

import torch
from torch import nn

from .autoencoder import AutoencoderKL
from .scheduler import DiffusionScheduler
from .unet import UNetModel


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        autoencoder: AutoencoderKL,
        denoiser: UNetModel,
        scheduler: DiffusionScheduler,
        latent_scaling_factor: float = 1.0,
        use_mean_latents: bool = True,
        latent_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.autoencoder: AutoencoderKL = autoencoder
        self.denoiser: UNetModel = denoiser
        self.scheduler: DiffusionScheduler = scheduler
        self.latent_scaling_factor: float = float(latent_scaling_factor)
        self.use_mean_latents: bool = use_mean_latents
        self.latent_shift: float = float(latent_shift)

    @torch.no_grad()
    def encode_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.autoencoder.encode(images)
        latents = (
            mean
            if self.use_mean_latents
            else self.autoencoder.reparameterize(mean, logvar)
        )
        normalized_latents = latents - self.latent_shift
        return normalized_latents * self.latent_scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        unscaled_latents = latents / self.latent_scaling_factor
        shifted_latents = unscaled_latents + self.latent_shift
        return self.autoencoder.decode(shifted_latents)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.denoiser(latents, timesteps)
