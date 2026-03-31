from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ldm.config import (
    create_grad_scaler,
    ensure_dir,
    load_config,
    resolve_device,
    set_seed,
)
from ldm.data import create_jet_dataloaders
from ldm.models import (
    AutoencoderKL,
    DiffusionScheduler,
    LatentDiffusionModel,
    UNetModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the diffusion stage in latent space."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--latent-mode",
        type=str,
        choices=("mean", "sample"),
        default=None,
        help=(
            "Override latent encoding mode for diffusion training. "
            "If omitted, uses model.use_mean_latents from config."
        ),
    )
    return parser.parse_args()


def estimate_latent_stats(
    autoencoder: AutoencoderKL,
    train_loader: Any,
    device: torch.device,
    num_batches: int,
) -> dict[str, float]:
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    autoencoder.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
        for batch_index, (images, _) in enumerate(train_loader):
            if batch_index >= num_batches:
                break

            images = images.to(device, non_blocking=True)
            mean, _ = autoencoder.encode(images)
            mean = mean.float()

            total_sum += float(mean.sum().item())
            total_sq_sum += float((mean * mean).sum().item())
            total_count += mean.numel()

    if total_count == 0:
        raise RuntimeError(
            "Unable to estimate latent statistics because no training batches were available."
        )

    latent_mean = total_sum / total_count
    latent_var = max(total_sq_sum / total_count - latent_mean * latent_mean, 1e-12)
    latent_std = math.sqrt(latent_var)
    scaling_factor = 1.0 / max(latent_std, 1e-6)

    return {
        "mean": latent_mean,
        "std": latent_std,
        "scaling_factor": scaling_factor,
        "num_values": float(total_count),
        "num_batches": float(num_batches),
    }


def resolve_use_mean_latents(config: dict[str, Any], latent_mode: str | None) -> bool:
    if latent_mode is not None:
        return latent_mode == "mean"
    return bool(config["model"].get("use_mean_latents", True))


def mode_suffix(use_mean_latents: bool) -> str:
    return "mean" if use_mean_latents else "sample"


def tagged_path(base_path: str | Path, suffix: str) -> Path:
    base = Path(base_path)
    if suffix == "mean":
        return base
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def open_csv_writer(path: Path) -> tuple[Any, csv.writer]:
    ensure_dir(path.parent)
    handle = open(path, "w", newline="")
    return handle, csv.writer(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])

    use_mean_latents = resolve_use_mean_latents(config, args.latent_mode)
    latent_mode_name = mode_suffix(use_mean_latents)

    base_checkpoint_path = Path(config["diffusion"]["checkpoint"])
    checkpoint_path = tagged_path(base_checkpoint_path, latent_mode_name)
    ensure_dir(checkpoint_path.parent)

    dataloaders = create_jet_dataloaders(
        config["data"],
        seed=config["experiment"]["seed"],
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

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
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    denoiser = UNetModel(
        in_channels=config["model"]["latent_channels"],
        base_channels=config["model"]["base_channels"],
        channel_multipliers=config["model"]["channel_multipliers"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    scheduler = DiffusionScheduler(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    latent_stats_batches = int(config["diffusion"].get("latent_stats_batches", 32))
    latent_stats = estimate_latent_stats(
        autoencoder=autoencoder,
        train_loader=train_loader,
        device=device,
        num_batches=latent_stats_batches,
    )
    latent_mean = latent_stats["mean"]
    latent_std = latent_stats["std"]
    max_scaling = float(config["diffusion"].get("max_latent_scaling", 10.0))
    latent_scaling_factor = min(latent_stats["scaling_factor"], max_scaling)

    print(
        f"estimated latent stats mean={latent_mean:.6f} std={latent_std:.6f} "
        f"raw_scaling={latent_stats['scaling_factor']:.6f} "
        f"capped_scaling={latent_scaling_factor:.6f} (max={max_scaling}) "
        f"from {int(latent_stats['num_values'])} values"
    )
    print(f"using latent_mode={latent_mode_name} checkpoint={checkpoint_path}")

    ldm = LatentDiffusionModel(
        autoencoder=autoencoder,
        denoiser=denoiser,
        scheduler=scheduler,
        latent_scaling_factor=latent_scaling_factor,
        use_mean_latents=use_mean_latents,
        latent_shift=latent_mean,
    ).to(device)

    optimizer = optim.AdamW(
        ldm.denoiser.parameters(),
        lr=config["diffusion"]["learning_rate"],
    )
    scaler = create_grad_scaler(device)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    epochs = config["diffusion"]["epochs"]
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float("inf")

    losses_csv_path = output_dir / f"diffusion_losses_{latent_mode_name}.csv"
    losses_handle, losses_writer = open_csv_writer(losses_csv_path)
    losses_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    latent_stats_csv_path = (
        output_dir / f"diffusion_latent_stats_{latent_mode_name}.csv"
    )
    latent_stats_handle, latent_stats_writer = open_csv_writer(latent_stats_csv_path)
    latent_stats_writer.writerow(
        [
            "epoch",
            "latent_mode",
            "latent_mean",
            "latent_std",
            "latent_scaling_factor",
            "latent_stats_num_values",
            "latent_stats_num_batches",
        ]
    )

    try:
        for epoch in range(epochs):
            ldm.train()
            progress = tqdm(train_loader, desc=f"Diffusion Epoch {epoch + 1}/{epochs}")
            running_loss = 0.0

            for images, _ in progress:
                images = images.to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
                    latents = ldm.encode_to_latents(images)

                noise = torch.randn_like(latents)
                timesteps = scheduler.sample_timesteps(latents.shape[0])
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype):
                    predicted_noise = ldm(noisy_latents, timesteps)
                    loss = F.mse_loss(predicted_noise, noise)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ldm.denoiser.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.detach().item())
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")

            average_loss = running_loss / len(train_loader)

            ldm.eval()
            validation_loss = 0.0
            with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
                for images, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    latents = ldm.encode_to_latents(images)
                    noise = torch.randn_like(latents)
                    timesteps = scheduler.sample_timesteps(latents.shape[0])
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    predicted_noise = ldm(noisy_latents, timesteps)
                    validation_loss += float(F.mse_loss(predicted_noise, noise).item())

            validation_loss /= max(len(val_loader), 1)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch + 1} diffusion_train_loss={average_loss:.6f} "
                f"diffusion_val_loss={validation_loss:.6f} lr={current_lr:.6f} "
                f"latent_mode={latent_mode_name} latent_mean={latent_mean:.6f} "
                f"latent_std={latent_std:.6f} "
                f"latent_scaling_factor={latent_scaling_factor:.6f}"
            )

            losses_writer.writerow(
                [
                    epoch + 1,
                    f"{average_loss:.6f}",
                    f"{validation_loss:.6f}",
                    f"{current_lr:.6f}",
                ]
            )
            losses_handle.flush()

            latent_stats_writer.writerow(
                [
                    epoch + 1,
                    latent_mode_name,
                    f"{latent_mean:.6f}",
                    f"{latent_std:.6f}",
                    f"{latent_scaling_factor:.6f}",
                    int(latent_stats["num_values"]),
                    int(latent_stats["num_batches"]),
                ]
            )
            latent_stats_handle.flush()

            lr_scheduler.step()

            if validation_loss <= best_val_loss:
                best_val_loss = validation_loss
                torch.save(
                    {
                        "model_state_dict": ldm.denoiser.state_dict(),
                        "config": config,
                        "epoch": epoch + 1,
                        "best_val_loss": best_val_loss,
                        "latent_mode": latent_mode_name,
                        "use_mean_latents": use_mean_latents,
                        "latent_scaling_factor": latent_scaling_factor,
                        "latent_stats": {
                            "mean": latent_mean,
                            "std": latent_std,
                            "scaling_factor": latent_scaling_factor,
                            "num_values": int(latent_stats["num_values"]),
                            "num_batches": int(latent_stats["num_batches"]),
                        },
                    },
                    checkpoint_path,
                )
    finally:
        losses_handle.close()
        latent_stats_handle.close()


if __name__ == "__main__":
    main()
