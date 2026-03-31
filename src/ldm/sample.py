from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ldm.config import ensure_dir, load_config, resolve_device, set_seed
from ldm.models import (
    AutoencoderKL,
    DiffusionScheduler,
    LatentDiffusionModel,
    UNetModel,
)


def infer_latent_size(image_size: int, num_downsamples: int = 3) -> int:
    return image_size // (2**num_downsamples)


def resolve_checkpoint_path(base_path: str | Path, latent_mode: str) -> Path:
    base = Path(base_path)
    if latent_mode == "mean":
        return base

    tagged = base.with_name(f"{base.stem}_{latent_mode}{base.suffix}")
    return tagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample images from the latent diffusion model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--latent-mode",
        type=str,
        choices=("mean", "sample"),
        default=None,
        help=(
            "Latent encoding mode ablation. "
            "If omitted, uses model.use_mean_latents from config."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])

    config_use_mean = bool(config["model"].get("use_mean_latents", True))
    latent_mode = args.latent_mode or ("" if config_use_mean else "sample")
    use_mean_latents = latent_mode == "mean" or latent_mode == ""

    autoencoder = AutoencoderKL(
        in_channels=config["model"]["in_channels"],
        latent_channels=config["model"]["latent_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    autoencoder_checkpoint = torch.load(
        config["autoencoder"]["checkpoint"],
        map_location=device,
    )
    autoencoder.load_state_dict(autoencoder_checkpoint["model_state_dict"])
    autoencoder.eval()

    denoiser = UNetModel(
        in_channels=config["model"]["latent_channels"],
        base_channels=config["model"]["base_channels"],
        channel_multipliers=config["model"]["channel_multipliers"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint_mode = "mean" if use_mean_latents else "sample"
    diffusion_checkpoint_path = resolve_checkpoint_path(
        config["diffusion"]["checkpoint"],
        checkpoint_mode,
    )
    diffusion_checkpoint = torch.load(
        diffusion_checkpoint_path,
        map_location=device,
    )
    denoiser.load_state_dict(diffusion_checkpoint["model_state_dict"])
    denoiser.eval()

    latent_stats = diffusion_checkpoint.get("latent_stats", {})
    latent_scaling_factor = latent_stats.get(
        "scaling_factor",
        diffusion_checkpoint.get(
            "latent_scaling_factor",
            config["model"]["latent_scaling_factor"],
        ),
    )
    latent_shift = latent_stats.get("mean", 0.0)

    scheduler = DiffusionScheduler(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    ldm = LatentDiffusionModel(
        autoencoder=autoencoder,
        denoiser=denoiser,
        scheduler=scheduler,
        latent_scaling_factor=latent_scaling_factor,
        use_mean_latents=use_mean_latents,
        latent_shift=latent_shift,
    ).to(device)
    ldm.eval()

    latent_size = infer_latent_size(config["data"]["image_size"])
    latent_channels = config["model"]["latent_channels"]
    latents = torch.randn(
        args.num_samples,
        latent_channels,
        latent_size,
        latent_size,
        device=device,
    )

    for timestep in reversed(range(config["sampling"]["num_steps"])):
        step_tensor = torch.full(
            (args.num_samples,),
            timestep,
            device=device,
            dtype=torch.long,
        )
        predicted_noise = ldm(latents, step_tensor)
        latents = scheduler.step(predicted_noise, timestep, latents)

    images = ldm.decode_latents(latents)

    output_dir = ensure_dir(config["sampling"]["output_dir"])
    mode_tag = "mean" if use_mean_latents else "sample"
    save_path = Path(output_dir) / f"sample_{mode_tag}_{args.num_samples}.png"
    save_image(
        (images + 1.0) / 2.0,
        save_path,
        nrow=max(1, min(4, args.num_samples)),
    )
    print(
        "saved samples to "
        f"{save_path} using checkpoint={diffusion_checkpoint_path} "
        f"latent_mode={mode_tag} "
        f"latent_mean={latent_shift:.6f} "
        f"latent_scaling_factor={latent_scaling_factor:.6f}"
    )


if __name__ == "__main__":
    main()
