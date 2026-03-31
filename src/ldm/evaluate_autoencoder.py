from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ldm.config import ensure_dir, load_config, resolve_device, set_seed
from ldm.data import create_jet_dataloaders, denormalize_event
from ldm.metrics import accumulate_metrics, reconstruction_metrics
from ldm.models import AutoencoderKL
from ldm.visualize import save_side_by_side_reconstructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate jet event reconstructions from the autoencoder.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--batch-limit", type=int, default=None, help="Optional number of test batches to evaluate.")
    return parser.parse_args()


def _save_channel_histograms(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: Path,
    channel_names: tuple[str, ...] = ("Channel 0 (pT)", "Channel 1 (η)", "Channel 2 (φ)"),
) -> None:
    """Plot per-channel intensity histograms comparing real vs reconstructed."""
    originals = originals.float().cpu()
    reconstructions = reconstructions.float().cpu()
    channels = originals.shape[1]

    fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 4))
    if channels == 1:
        axes = [axes]

    for ch in range(channels):
        real_flat = originals[:, ch].flatten().numpy()
        recon_flat = reconstructions[:, ch].flatten().numpy()

        # Filter out zeros for better visualization (jet images are very sparse)
        real_nonzero = real_flat[real_flat > 0.01]
        recon_nonzero = recon_flat[recon_flat > 0.01]

        ax = axes[ch]
        bins = np.linspace(0, max(real_flat.max(), recon_flat.max()), 100)
        ax.hist(real_nonzero, bins=bins, alpha=0.6, density=True, label="Original", color="#4F46E5")
        ax.hist(recon_nonzero, bins=bins, alpha=0.6, density=True, label="Reconstructed", color="#EC4899")
        ax.set_title(channel_names[ch] if ch < len(channel_names) else f"Channel {ch}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Intensity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Channel Intensity Distributions: Original vs Reconstructed", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_latent_tsne(
    model: AutoencoderKL,
    dataloader,
    device: torch.device,
    output_path: Path,
    max_samples: int = 3000,
) -> None:
    """Create t-SNE visualization of latent space colored by quark/gluon labels."""
    from sklearn.manifold import TSNE

    latents_list = []
    labels_list = []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    count = 0

    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
        for images, labels in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)
            mean, _ = model.encode(images)
            flat = mean.float().cpu().view(mean.shape[0], -1)
            latents_list.append(flat)
            labels_list.append(labels)
            count += flat.shape[0]

    all_latents = torch.cat(latents_list, dim=0)[:max_samples].numpy()
    all_labels = torch.cat(labels_list, dim=0)[:max_samples].numpy()

    print(f"Running t-SNE on {all_latents.shape[0]} samples with {all_latents.shape[1]} dims...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(all_latents)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        embedded[:, 0], embedded[:, 1],
        c=all_labels, cmap="coolwarm", alpha=0.5, s=8, edgecolors="none",
    )
    ax.set_title("Latent Space t-SNE (Quark=0, Gluon=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax, label="Label")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Quark", "Gluon"])
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE to {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])

    dataloaders = create_jet_dataloaders(config["data"], seed=config["experiment"]["seed"])
    test_loader = dataloaders["test"]

    model = AutoencoderKL(
        in_channels=config["model"]["in_channels"],
        latent_channels=config["model"]["latent_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(config["autoencoder"]["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = ensure_dir(config["evaluation"]["output_dir"])
    metrics_path = Path(output_dir) / "autoencoder_metrics.json"
    figure_path = Path(output_dir) / "autoencoder_reconstructions.png"
    enhanced_figure_path = Path(output_dir) / "autoencoder_reconstructions_enhanced.png"
    histogram_path = Path(output_dir) / "autoencoder_histograms.png"
    tsne_path = Path(output_dir) / "latent_tsne.png"

    metric_history: list[dict[str, float]] = []
    all_originals: list[torch.Tensor] = []
    all_reconstructions: list[torch.Tensor] = []
    first_batch_saved = False
    batch_limit = args.batch_limit or config["evaluation"]["batch_limit"]
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
        for batch_index, (images, _) in enumerate(test_loader):
            if batch_index >= batch_limit:
                break

            images = images.to(device)
            reconstructions, _, _ = model(images)

            denorm_original = denormalize_event(
                images.float(),
                normalization=config["data"]["normalization"],
                channel_scales=config["data"]["channel_scales"],
                crop_size=config["data"]["crop_size"],
            )
            denorm_reconstruction = denormalize_event(
                reconstructions.float(),
                normalization=config["data"]["normalization"],
                channel_scales=config["data"]["channel_scales"],
                crop_size=config["data"]["crop_size"],
            )
            metric_history.append(reconstruction_metrics(denorm_original, denorm_reconstruction))

            # Collect for histograms
            all_originals.append(denorm_original.cpu())
            all_reconstructions.append(denorm_reconstruction.cpu())

            if not first_batch_saved:
                save_side_by_side_reconstructions(
                    images,
                    reconstructions,
                    figure_path,
                    max_items=min(config["evaluation"]["num_visualizations"], images.shape[0]),
                )
                save_side_by_side_reconstructions(
                    images,
                    reconstructions,
                    enhanced_figure_path,
                    max_items=min(config["evaluation"]["num_visualizations"], images.shape[0]),
                    render_mode="adaptive",
                )
                first_batch_saved = True

    # Metrics summary
    summary = accumulate_metrics(metric_history)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved metrics to {metrics_path}")
    print(f"saved reconstruction figure to {figure_path}")
    print(f"saved enhanced reconstruction figure to {enhanced_figure_path}")

    # Per-channel histograms
    if all_originals:
        cat_orig = torch.cat(all_originals, dim=0)
        cat_recon = torch.cat(all_reconstructions, dim=0)
        _save_channel_histograms(cat_orig, cat_recon, histogram_path)
        print(f"saved histograms to {histogram_path}")

    # Latent space t-SNE
    _save_latent_tsne(model, test_loader, device, tsne_path)


if __name__ == "__main__":
    main()
