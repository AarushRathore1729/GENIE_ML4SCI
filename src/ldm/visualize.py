from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _adaptive_stretch(images: torch.Tensor) -> torch.Tensor:
    flat = images.flatten(2)
    q995 = torch.quantile(flat, 0.995, dim=2, keepdim=True).clamp_min(1e-6)
    q995 = q995.view(images.shape[0], images.shape[1], 1, 1)
    stretched = (images / q995).clamp(0.0, 1.0)
    log_base = torch.log1p(torch.tensor(50.0, dtype=stretched.dtype, device=stretched.device))
    return torch.log1p(50.0 * stretched) / log_base


def _to_rgb_composite(batch: torch.Tensor, render_mode: str = "gamma") -> torch.Tensor:
    images = ((batch.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)

    if render_mode == "adaptive":
        images = _adaptive_stretch(images)
    elif render_mode == "gamma":
        # Gamma correction: brightens the sparse signal (most pixels near 0) so jet
        # deposits are visible; without this jet images appear uniformly black.
        images = images.pow(0.3)
    else:
        raise ValueError(f"Unsupported render_mode: {render_mode}")

    return images.permute(0, 2, 3, 1)


def save_side_by_side_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: str | Path,
    max_items: int = 8,
    render_mode: str = "gamma",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    originals_rgb = _to_rgb_composite(originals[:max_items], render_mode=render_mode)
    reconstructions_rgb = _to_rgb_composite(reconstructions[:max_items], render_mode=render_mode)
    num_items = originals_rgb.shape[0]

    fig, axes = plt.subplots(2, num_items, figsize=(3 * num_items, 6))
    if num_items == 1:
        axes = axes.reshape(2, 1)

    for index in range(num_items):
        axes[0, index].imshow(originals_rgb[index].numpy())
        axes[0, index].set_title(f"Original {index + 1}")
        axes[1, index].imshow(reconstructions_rgb[index].numpy())
        axes[1, index].set_title(f"Reconstruction {index + 1}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
