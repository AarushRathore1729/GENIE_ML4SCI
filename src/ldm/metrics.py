from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruction_metrics(original: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    """Compute MSE, MAE, PSNR, and SSIM between original and reconstruction."""
    mse = F.mse_loss(reconstruction, original).item()
    mae = F.l1_loss(reconstruction, original).item()
    max_value = float(torch.max(original).item()) if torch.numel(original) else 1.0
    max_value = max(max_value, 1e-8)
    psnr = 20.0 * math.log10(max_value) - 10.0 * math.log10(max(mse, 1e-12))
    ssim = compute_ssim(original.float(), reconstruction.float())
    return {
        "mse": float(mse),
        "mae": float(mae),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def accumulate_metrics(metric_history: list[dict[str, float]]) -> dict[str, float]:
    if not metric_history:
        return {}
    keys = metric_history[0].keys()
    return {key: float(sum(item[key] for item in metric_history) / len(metric_history)) for key in keys}


# ---------------------------------------------------------------------------
# SSIM (Structural Similarity Index)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    kernel = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


@torch.no_grad()
def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute SSIM between two image batches (B, C, H, W)."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, channels).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(images: torch.Tensor) -> torch.Tensor:
    """Extract simple statistical features from jet images for FID.

    For jet images (not natural images), InceptionV3 features are meaningless.
    Instead we use per-channel statistics: mean, std, max, sum, and spatial
    moments — giving a compact feature vector per image.
    """
    b, c, h, w = images.shape
    features = []

    for ch in range(c):
        channel = images[:, ch, :, :]  # (B, H, W)
        flat = channel.view(b, -1)  # (B, H*W)

        features.append(flat.mean(dim=1, keepdim=True))
        features.append(flat.std(dim=1, keepdim=True))
        features.append(flat.max(dim=1, keepdim=True).values)
        features.append(flat.sum(dim=1, keepdim=True))

        # Spatial moments (center of energy)
        ys = torch.arange(h, device=images.device, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, device=images.device, dtype=torch.float32).view(1, 1, w)
        total = channel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        cy = (channel * ys).sum(dim=(1, 2), keepdim=True) / total
        cx = (channel * xs).sum(dim=(1, 2), keepdim=True) / total
        features.append(cy.view(b, 1))
        features.append(cx.view(b, 1))

        # Spread (std of energy position)
        spread_y = ((channel * (ys - cy) ** 2).sum(dim=(1, 2)) / total.squeeze()).sqrt()
        spread_x = ((channel * (xs - cx) ** 2).sum(dim=(1, 2)) / total.squeeze()).sqrt()
        features.append(spread_y.unsqueeze(1))
        features.append(spread_x.unsqueeze(1))

    return torch.cat(features, dim=1)  # (B, C*8)


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
    """Compute FID between real and generated image sets using physics-aware features."""
    real_feat = _extract_features(real_images.float()).cpu().numpy()
    gen_feat = _extract_features(generated_images.float()).cpu().numpy()

    mu_real, sigma_real = real_feat.mean(axis=0), np.cov(real_feat, rowvar=False)
    mu_gen, sigma_gen = gen_feat.mean(axis=0), np.cov(gen_feat, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_real + sigma_gen - 2.0 * covmean))
    return max(fid, 0.0)


# ---------------------------------------------------------------------------
# Histogram comparison (Wasserstein-1 / Earth Mover's Distance)
# ---------------------------------------------------------------------------

def channel_w1_distances(
    real: torch.Tensor,
    generated: torch.Tensor,
    n_bins: int = 200,
) -> dict[str, float]:
    """Compute per-channel Wasserstein-1 distance between real and generated."""
    real = real.float().cpu()
    generated = generated.float().cpu()
    channels = real.shape[1]
    distances = {}

    for ch in range(channels):
        real_flat = real[:, ch].flatten().numpy()
        gen_flat = generated[:, ch].flatten().numpy()

        # Compute W1 via sorted CDF difference
        all_vals = np.concatenate([real_flat, gen_flat])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        if hi - lo < 1e-10:
            distances[f"w1_ch{ch}"] = 0.0
            continue

        bins = np.linspace(lo, hi, n_bins + 1)
        real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        cdf_diff = np.abs(np.cumsum(real_hist - gen_hist)) * bin_width
        distances[f"w1_ch{ch}"] = float(cdf_diff.sum() * bin_width)

    return distances


# ---------------------------------------------------------------------------
# Jet-level physics observables
# ---------------------------------------------------------------------------

def _scalar_w1(a: np.ndarray, b: np.ndarray, n_bins: int = 200) -> float:
    """W1 distance between two 1-D scalar distributions via CDF integration."""
    all_vals = np.concatenate([a, b])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if hi - lo < 1e-10:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    bin_width = bins[1] - bins[0]
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.abs(np.cumsum(ha - hb)).sum() * bin_width ** 2)


@torch.no_grad()
def jet_observables_w1(
    real: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = 0.01,
    n_bins: int = 200,
) -> dict[str, float]:
    """W1 distances on per-jet scalar observables.

    Computes W1 for total deposit, hit multiplicity, and max pixel per channel,
    plus energy-weighted spatial centroid (y, x) using channel 0 as weights.

    Args:
        real: Denormalized tensor (B, C, H, W).
        generated: Denormalized tensor (B, C, H, W).
        threshold: Pixel value below which a pixel is considered empty (for multiplicity).
    """
    real = real.float().cpu()
    gen = generated.float().cpu()
    results: dict[str, float] = {}

    for ch in range(real.shape[1]):
        r_flat = real[:, ch].view(real.shape[0], -1)
        g_flat = gen[:, ch].view(gen.shape[0], -1)

        results[f"w1_obs_total_ch{ch}"] = _scalar_w1(
            r_flat.sum(1).numpy(), g_flat.sum(1).numpy(), n_bins
        )
        results[f"w1_obs_mult_ch{ch}"] = _scalar_w1(
            (r_flat > threshold).float().sum(1).numpy(),
            (g_flat > threshold).float().sum(1).numpy(),
            n_bins,
        )
        results[f"w1_obs_max_ch{ch}"] = _scalar_w1(
            r_flat.max(1).values.numpy(), g_flat.max(1).values.numpy(), n_bins
        )

    # Energy-weighted spatial centroid using channel 0
    ch0_r = real[:, 0]   # (B, H, W)
    ch0_g = gen[:, 0]
    h, w = ch0_r.shape[-2], ch0_r.shape[-1]
    ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    total_r = ch0_r.sum(dim=(1, 2)).clamp(min=1e-8)
    total_g = ch0_g.sum(dim=(1, 2)).clamp(min=1e-8)
    cy_r = ((ch0_r * ys).sum(dim=(1, 2)) / total_r).numpy()
    cx_r = ((ch0_r * xs).sum(dim=(1, 2)) / total_r).numpy()
    cy_g = ((ch0_g * ys).sum(dim=(1, 2)) / total_g).numpy()
    cx_g = ((ch0_g * xs).sum(dim=(1, 2)) / total_g).numpy()
    results["w1_obs_centroid_y"] = _scalar_w1(cy_r, cy_g, n_bins)
    results["w1_obs_centroid_x"] = _scalar_w1(cx_r, cx_g, n_bins)

    return results


@torch.no_grad()
def sparsity_comparison(
    real: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Mean fraction of near-zero pixels per channel for real and generated."""
    real = real.float().cpu()
    gen = generated.float().cpu()
    result: dict[str, float] = {}
    for ch in range(real.shape[1]):
        result[f"sparsity_real_ch{ch}"] = float(
            (real[:, ch] <= threshold).float().mean().item()
        )
        result[f"sparsity_gen_ch{ch}"] = float(
            (gen[:, ch] <= threshold).float().mean().item()
        )
    return result
