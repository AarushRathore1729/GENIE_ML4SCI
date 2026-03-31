from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def _as_scale_tensor(channel_scales: list[float] | tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(channel_scales, dtype=torch.float32).view(-1, 1, 1)


def _center_pad(image: torch.Tensor, image_size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    if height > image_size or width > image_size:
        raise ValueError(
            f"Image size {height}x{width} is larger than target padded size {image_size}."
        )

    pad_h = image_size - height
    pad_w = image_size - width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom))


def center_crop(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    start_h = (height - crop_size) // 2
    start_w = (width - crop_size) // 2
    return image[..., start_h : start_h + crop_size, start_w : start_w + crop_size]


def preprocess_event(
    event: np.ndarray,
    image_size: int,
    normalization: str,
    channel_scales: list[float] | tuple[float, ...],
) -> torch.Tensor:
    tensor = torch.from_numpy(event).permute(2, 0, 1).float().clamp_min(0.0)
    scales = _as_scale_tensor(channel_scales)

    if normalization == "log_scale":
        tensor = torch.log1p(tensor) / torch.log1p(scales)
    elif normalization == "max_scale":
        tensor = tensor / scales
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor * 2.0 - 1.0
    return _center_pad(tensor, image_size)


def denormalize_event(
    event: torch.Tensor,
    normalization: str,
    channel_scales: list[float] | tuple[float, ...],
    crop_size: int | None = None,
) -> torch.Tensor:
    tensor = ((event.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    if crop_size is not None:
        tensor = center_crop(tensor, crop_size)

    scales = _as_scale_tensor(channel_scales)
    if normalization == "log_scale":
        tensor = torch.expm1(tensor * torch.log1p(scales))
    elif normalization == "max_scale":
        tensor = tensor * scales
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    return tensor


class JetHDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        image_key: str,
        label_key: str,
        image_size: int,
        normalization: str,
        channel_scales: list[float] | tuple[float, ...],
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        print(f"Loading dataset into RAM from {self.file_path}...")
        with h5py.File(self.file_path, "r") as handle:
            if image_key not in handle:
                raise KeyError(f"Missing image dataset key: {image_key}")
            if label_key not in handle:
                raise KeyError(f"Missing label dataset key: {label_key}")

            self.labels = handle[label_key][:].astype(int)
            n = handle[image_key].shape[0]
            self.images = torch.empty(n, len(channel_scales), image_size, image_size)
            scales = _as_scale_tensor(channel_scales)

            # Process in chunks to limit peak RAM usage
            chunk_size = 5000
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                raw = handle[image_key][start:end]
                chunk = torch.from_numpy(raw).permute(0, 3, 1, 2).float().clamp_min(0.0)
                del raw

                if normalization == "log_scale":
                    chunk = torch.log1p(chunk) / torch.log1p(scales)
                elif normalization == "max_scale":
                    chunk = chunk / scales
                else:
                    raise ValueError(f"Unsupported normalization: {normalization}")

                chunk = chunk.clamp(0.0, 1.0) * 2.0 - 1.0
                self.images[start:end] = _center_pad(chunk, image_size)
                del chunk

                loaded = min(end, n)
                print(f"  loaded {loaded}/{n} ({100 * loaded // n}%)", end="\r")

        self.length = len(self.images)
        print(f"\nDataset loaded: {self.length} samples, "
              f"tensor shape {tuple(self.images.shape)}, "
              f"~{self.images.nbytes / 1e9:.1f} GB RAM")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def build_splits(length: int, train_split: float, val_split: float, test_split: float, seed: int) -> dict[str, list[int]]:
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train/val/test splits must sum to 1.0")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()

    train_end = int(length * train_split)
    val_end = train_end + int(length * val_split)
    return {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }


def create_jet_dataloaders(data_config: dict[str, Any], seed: int) -> dict[str, DataLoader]:
    dataset = JetHDF5Dataset(
        file_path=data_config["file_path"],
        image_key=data_config["image_key"],
        label_key=data_config["label_key"],
        image_size=data_config["image_size"],
        normalization=data_config["normalization"],
        channel_scales=data_config["channel_scales"],
    )
    splits = build_splits(
        length=len(dataset),
        train_split=data_config["train_split"],
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
        seed=seed,
    )
    subsets = {name: Subset(dataset, indices) for name, indices in splits.items()}
    num_workers = data_config["num_workers"]
    pin_memory = data_config.get("pin_memory", True) and torch.cuda.is_available()
    common = {
        "batch_size": data_config["batch_size"],
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    return {
        "train": DataLoader(subsets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(subsets["val"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(subsets["test"], shuffle=False, drop_last=False, **common),
    }
