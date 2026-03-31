from __future__ import annotations

import torch


class DiffusionScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * clean + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def step(self, predicted_noise: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[timestep]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[timestep]

        model_mean = sqrt_recip_alpha_t * (
            sample - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )

        if timestep == 0:
            return model_mean

        noise = torch.randn_like(sample)
        posterior_variance_t = self.posterior_variance[timestep]
        return model_mean + torch.sqrt(posterior_variance_t) * noise
