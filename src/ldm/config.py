from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import torch
import torch.backends.cudnn
import yaml

ConfigDict = Dict[str, Any]


def load_config(config_path: str | Path) -> ConfigDict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def create_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Create a GradScaler for mixed-precision training (enabled only on CUDA)."""
    return torch.amp.GradScaler(enabled=device.type == "cuda")


def latent_mode_tag(use_mean_latents: bool) -> str:
    return "mean" if use_mean_latents else "sample"


def resolve_ablation_checkpoint_path(
    checkpoint_path: str | Path,
    use_mean_latents: bool,
) -> Path:
    checkpoint = Path(checkpoint_path)
    if use_mean_latents:
        return checkpoint
    return checkpoint.with_name(
        f"{checkpoint.stem}_{latent_mode_tag(use_mean_latents)}{checkpoint.suffix}"
    )


def mode_tagged_output_path(path: str | Path, use_mean_latents: bool) -> Path:
    output_path = Path(path)
    tag = latent_mode_tag(use_mean_latents)
    return output_path.with_name(f"{output_path.stem}_{tag}{output_path.suffix}")
