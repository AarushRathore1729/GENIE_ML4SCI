from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ldm.config import ensure_dir, load_config, resolve_device, set_seed
from ldm.data import create_jet_dataloaders, denormalize_event
from ldm.metrics import (
    accumulate_metrics,
    channel_w1_distances,
    compute_fid,
    jet_observables_w1,
    sparsity_comparison,
)
from ldm.models import (
    AutoencoderKL,
    DiffusionScheduler,
    LatentDiffusionModel,
    UNetModel,
)
from ldm.sample import infer_latent_size, resolve_checkpoint_path

ConfigDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion denoising loss on jet latents."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=None,
        help="Optional number of test batches to evaluate.",
    )
    parser.add_argument(
        "--num-fid-samples",
        type=int,
        default=1024,
        help="Number of generated samples for FID.",
    )
    parser.add_argument(
        "--latent-mode",
        type=str,
        choices=["config", "mean", "sample"],
        default="config",
        help=(
            "Latent encoding mode to evaluate. "
            "'config' uses model.use_mean_latents from the YAML config."
        ),
    )
    return parser.parse_args()


def _resolve_use_mean_latents(config: ConfigDict, latent_mode: str) -> bool:
    if latent_mode == "mean":
        return True
    if latent_mode == "sample":
        return False
    return bool(cast(dict[str, Any], config["model"]).get("use_mean_latents", True))


def _mode_suffix(use_mean_latents: bool) -> str:
    return "mean" if use_mean_latents else "sample"


@torch.no_grad()
def _generate_samples(
    ldm: LatentDiffusionModel,
    scheduler: DiffusionScheduler,
    config: ConfigDict,
    device: torch.device,
    num_samples: int = 1024,
    batch_size: int = 64,
) -> torch.Tensor:
    """Generate samples from the diffusion model."""
    data_config = cast(dict[str, Any], config["data"])
    model_config = cast(dict[str, Any], config["model"])
    sampling_config = cast(dict[str, Any], config["sampling"])

    latent_size = infer_latent_size(int(data_config["image_size"]))
    latent_channels = int(model_config["latent_channels"])
    num_steps = int(sampling_config["num_steps"])
    all_images: list[torch.Tensor] = []

    for start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - start)
        latents = torch.randn(
            n,
            latent_channels,
            latent_size,
            latent_size,
            device=device,
        )

        for timestep in reversed(range(num_steps)):
            step_tensor = torch.full((n,), timestep, device=device, dtype=torch.long)
            predicted_noise = ldm(latents, step_tensor)
            latents = scheduler.step(predicted_noise, timestep, latents)

        images = ldm.decode_latents(latents)
        all_images.append(images.float().cpu())
        print(
            f"  generated {min(start + n, num_samples)}/{num_samples} samples",
            end="\r",
        )

    print()
    return torch.cat(all_images, dim=0)


def _save_channel_histograms(
    real: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path,
    channel_names: tuple[str, ...] = (
        "Channel 0 (pT)",
        "Channel 1 (η)",
        "Channel 2 (φ)",
    ),
) -> None:
    """Plot per-channel intensity histograms comparing real vs generated."""
    real = real.float().cpu()
    generated = generated.float().cpu()
    channels = real.shape[1]

    fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 4))
    if channels == 1:
        axes = [axes]

    for ch in range(channels):
        real_flat = real[:, ch].flatten().numpy()
        gen_flat = generated[:, ch].flatten().numpy()

        real_nonzero = real_flat[real_flat > 0.01]
        gen_nonzero = gen_flat[gen_flat > 0.01]

        ax = axes[ch]
        hi = max(real_flat.max(), gen_flat.max()) if real_flat.size > 0 else 1.0
        bins = np.linspace(0.0, hi, 100)

        ax.hist(
            real_nonzero,
            bins=bins,
            alpha=0.6,
            density=True,
            label="Real",
            color="#4F46E5",
        )
        ax.hist(
            gen_nonzero,
            bins=bins,
            alpha=0.6,
            density=True,
            label="Generated",
            color="#EC4899",
        )
        ax.set_title(
            channel_names[ch] if ch < len(channel_names) else f"Channel {ch}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Intensity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Per-Channel Intensity Distributions: Real vs Generated",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_sample_grid(images: torch.Tensor, output_path: Path, nrow: int = 8) -> None:
    """Save a grid of generated samples."""
    from torchvision.utils import save_image

    save_image((images + 1.0) / 2.0, output_path, nrow=nrow)


def _save_real_vs_generated(
    real: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path,
    n: int = 8,
    render_mode: str = "gamma",
) -> None:
    """Side-by-side grid: top row = real jets, bottom row = generated jets.

    Uses gamma correction (^0.3) so sparse jet deposits are visible.
    """
    def _to_rgb(batch: torch.Tensor) -> np.ndarray:
        imgs = ((batch.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)

        if render_mode == "adaptive":
            flat = imgs.flatten(2)
            q995 = torch.quantile(flat, 0.995, dim=2, keepdim=True).clamp_min(1e-6)
            q995 = q995.view(imgs.shape[0], imgs.shape[1], 1, 1)
            imgs = (imgs / q995).clamp(0.0, 1.0)
            log_base = torch.log1p(torch.tensor(50.0, dtype=imgs.dtype, device=imgs.device))
            imgs = torch.log1p(50.0 * imgs) / log_base
        elif render_mode == "gamma":
            imgs = imgs.pow(0.3)                 # gamma: brightens sparse signal
        else:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        return imgs[:n].permute(0, 2, 3, 1).numpy()   # (n, H, W, 3)

    real_rgb = _to_rgb(real)
    gen_rgb  = _to_rgb(generated)
    n_actual = real_rgb.shape[0]

    fig, axes = plt.subplots(2, n_actual, figsize=(2.5 * n_actual, 5))
    if n_actual == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_actual):
        axes[0, col].imshow(real_rgb[col])
        axes[0, col].set_title(f"Real {col + 1}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(gen_rgb[col])
        axes[1, col].set_title(f"Gen {col + 1}", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle(
        "Jet Events: Real vs Generated",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_jet_observable_histograms(
    real: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path,
    threshold: float = 0.01,
    channel_names: tuple[str, ...] = ("pT", "η", "φ"),
) -> None:
    """Plot per-jet observable distributions: total deposit, multiplicity, max pixel,
    and energy-weighted centroid (y, x) for channel 0."""
    real = real.float().cpu()
    gen = generated.float().cpu()
    b = real.shape[0]

    ch0_r = real[:, 0]
    ch0_g = gen[:, 0]
    h, w = ch0_r.shape[-2], ch0_r.shape[-1]
    ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    total_r = ch0_r.sum(dim=(1, 2)).clamp(min=1e-8)
    total_g = ch0_g.sum(dim=(1, 2)).clamp(min=1e-8)

    observables = {
        f"Total deposit ({channel_names[0]})": (
            ch0_r.view(b, -1).sum(1).numpy(),
            ch0_g.view(b, -1).sum(1).numpy(),
        ),
        f"Multiplicity ({channel_names[0]} > {threshold:.2f})": (
            (ch0_r.view(b, -1) > threshold).float().sum(1).numpy(),
            (ch0_g.view(b, -1) > threshold).float().sum(1).numpy(),
        ),
        f"Max pixel ({channel_names[0]})": (
            ch0_r.view(b, -1).max(1).values.numpy(),
            ch0_g.view(b, -1).max(1).values.numpy(),
        ),
        "Energy centroid η (px)": (
            ((ch0_r * ys).sum(dim=(1, 2)) / total_r).numpy(),
            ((ch0_g * ys).sum(dim=(1, 2)) / total_g).numpy(),
        ),
        "Energy centroid φ (px)": (
            ((ch0_r * xs).sum(dim=(1, 2)) / total_r).numpy(),
            ((ch0_g * xs).sum(dim=(1, 2)) / total_g).numpy(),
        ),
        f"Sparsity ({channel_names[0]} ≤ {threshold:.2f})": (
            (ch0_r.view(b, -1) <= threshold).float().mean(1).numpy(),
            (ch0_g.view(b, -1) <= threshold).float().mean(1).numpy(),
        ),
    }

    n_plots = len(observables)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for idx, (title, (r_vals, g_vals)) in enumerate(observables.items()):
        ax = axes_flat[idx]
        all_vals = np.concatenate([r_vals, g_vals])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        bins = np.linspace(lo, hi, 60) if hi > lo else 10
        ax.hist(r_vals, bins=bins, alpha=0.6, density=True, label="Real", color="#4F46E5")
        ax.hist(g_vals, bins=bins, alpha=0.6, density=True, label="Generated", color="#EC4899")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Jet-Level Observable Distributions: Real vs Generated",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_config = cast(dict[str, Any], config["experiment"])
    model_config = cast(dict[str, Any], config["model"])
    data_config = cast(dict[str, Any], config["data"])
    diffusion_config = cast(dict[str, Any], config["diffusion"])
    evaluation_config = cast(dict[str, Any], config["evaluation"])
    autoencoder_config = cast(dict[str, Any], config["autoencoder"])

    set_seed(int(experiment_config["seed"]))
    device = resolve_device(str(experiment_config["device"]))

    dataloaders = create_jet_dataloaders(
        data_config,
        seed=int(experiment_config["seed"]),
    )
    test_loader = dataloaders["test"]

    use_mean_latents = _resolve_use_mean_latents(config, args.latent_mode)
    mode_tag = _mode_suffix(use_mean_latents)

    autoencoder = AutoencoderKL(
        in_channels=int(model_config["in_channels"]),
        latent_channels=int(model_config["latent_channels"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device)
    autoencoder_checkpoint = torch.load(
        str(autoencoder_config["checkpoint"]),
        map_location=device,
    )
    autoencoder.load_state_dict(autoencoder_checkpoint["model_state_dict"])
    autoencoder.eval()

    denoiser = UNetModel(
        in_channels=int(model_config["latent_channels"]),
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=list(cast(list[int], model_config["channel_multipliers"])),
        time_embedding_dim=int(model_config["time_embedding_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)

    diffusion_checkpoint_path = resolve_checkpoint_path(
        diffusion_config["checkpoint"],
        mode_tag,
    )
    diffusion_checkpoint = torch.load(
        diffusion_checkpoint_path,
        map_location=device,
    )
    denoiser.load_state_dict(diffusion_checkpoint["model_state_dict"])
    denoiser.eval()

    scheduler = DiffusionScheduler(
        num_timesteps=int(diffusion_config["num_timesteps"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
        device=device,
    )

    latent_stats = cast(dict[str, Any], diffusion_checkpoint.get("latent_stats", {}))
    latent_scaling_factor = float(
        latent_stats.get(
            "scaling_factor",
            diffusion_checkpoint.get(
                "latent_scaling_factor",
                float(model_config["latent_scaling_factor"]),
            ),
        )
    )
    latent_shift = float(latent_stats.get("mean", 0.0))

    ldm = LatentDiffusionModel(
        autoencoder=autoencoder,
        denoiser=denoiser,
        scheduler=scheduler,
        latent_scaling_factor=latent_scaling_factor,
        use_mean_latents=use_mean_latents,
        latent_shift=latent_shift,
    ).to(device)
    ldm.eval()

    output_dir = ensure_dir(evaluation_config["output_dir"])
    batch_limit = (
        args.batch_limit
        if args.batch_limit is not None
        else int(evaluation_config["batch_limit"])
    )

    metric_history: list[dict[str, float]] = []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    print(
        "Loaded diffusion checkpoint "
        f"{diffusion_checkpoint_path} "
        f"with latent_mode={mode_tag} "
        f"latent_mean={latent_shift:.6f} "
        f"latent_scaling_factor={latent_scaling_factor:.6f}"
    )

    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
        for batch_index, (images, _) in enumerate(test_loader):
            if batch_index >= batch_limit:
                break

            images = images.to(device, non_blocking=True)
            latents = ldm.encode_to_latents(images)
            noise = torch.randn_like(latents)
            timesteps = scheduler.sample_timesteps(latents.shape[0])
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            predicted_noise = ldm(noisy_latents, timesteps)

            metric_history.append(
                {"denoising_mse": float(F.mse_loss(predicted_noise, noise).item())}
            )

    print("Generating samples for evaluation...")
    generated = _generate_samples(
        ldm=ldm,
        scheduler=scheduler,
        config=config,
        device=device,
        num_samples=args.num_fid_samples,
    )

    grid_path = Path(output_dir) / f"diffusion_sample_grid_{mode_tag}.png"
    _save_sample_grid(generated[:64], grid_path, nrow=8)
    print(f"saved sample grid to {grid_path}")

    # Collect a few real images for the side-by-side comparison
    with torch.no_grad():
        real_for_vis, _ = next(iter(test_loader))
    rvg_path = Path(output_dir) / f"diffusion_real_vs_generated_{mode_tag}.png"
    _save_real_vs_generated(real_for_vis, generated, rvg_path, n=8)
    print(f"saved real-vs-generated comparison to {rvg_path}")
    rvg_enhanced_path = Path(output_dir) / f"diffusion_real_vs_generated_{mode_tag}_enhanced.png"
    _save_real_vs_generated(real_for_vis, generated, rvg_enhanced_path, n=8, render_mode="adaptive")
    print(f"saved enhanced real-vs-generated comparison to {rvg_enhanced_path}")

    real_images: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in test_loader:
            real_images.append(images.cpu())
            if sum(batch.shape[0] for batch in real_images) >= args.num_fid_samples:
                break
    real_cat = torch.cat(real_images, dim=0)[: args.num_fid_samples]

    denorm_real = denormalize_event(
        real_cat,
        normalization=str(data_config["normalization"]),
        channel_scales=list(cast(list[float], data_config["channel_scales"])),
        crop_size=int(data_config["crop_size"]),
    )
    denorm_gen = denormalize_event(
        generated[: len(denorm_real)],
        normalization=str(data_config["normalization"]),
        channel_scales=list(cast(list[float], data_config["channel_scales"])),
        crop_size=int(data_config["crop_size"]),
    )

    # FID on denormalized (physical) images — consistent with W1
    fid = compute_fid(denorm_real.float(), denorm_gen.float())
    print(f"FID: {fid:.4f}")

    w1 = channel_w1_distances(denorm_real, denorm_gen)
    print(f"Pixel W1 distances: {w1}")

    hist_path = Path(output_dir) / f"diffusion_histograms_{mode_tag}.png"
    _save_channel_histograms(denorm_real, denorm_gen, hist_path)
    print(f"saved channel histograms to {hist_path}")

    jet_w1 = jet_observables_w1(denorm_real, denorm_gen)
    print(f"Jet observable W1 distances: {jet_w1}")

    sparsity = sparsity_comparison(denorm_real, denorm_gen)
    print(f"Sparsity: {sparsity}")

    obs_hist_path = Path(output_dir) / f"diffusion_jet_observables_{mode_tag}.png"
    _save_jet_observable_histograms(denorm_real, denorm_gen, obs_hist_path)
    print(f"saved jet observable histograms to {obs_hist_path}")

    summary = accumulate_metrics(metric_history)
    summary["fid"] = fid
    summary["latent_mean"] = latent_shift
    summary["latent_scaling_factor"] = latent_scaling_factor
    summary["latent_mode"] = mode_tag
    summary["use_mean_latents"] = use_mean_latents
    summary["checkpoint_path"] = str(diffusion_checkpoint_path)
    summary.update(w1)
    summary.update(jet_w1)
    summary.update(sparsity)

    metrics_path = Path(output_dir) / f"diffusion_metrics_{mode_tag}.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
