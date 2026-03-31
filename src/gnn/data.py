"""Point-cloud graph dataset for quark/gluon jet classification.

Each non-zero pixel in the 3-channel calorimeter image becomes a node:
    features = [η_coord, φ_coord, ECAL_norm, HCAL_norm, Track_norm]   (5 dims)

Nodes are connected by a symmetric k-NN graph in (η, φ) space.

All graphs are precomputed in __init__ and stored as padded tensors so
__getitem__ is O(1) — avoids slow per-sample HDF5 I/O during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Graph construction helper
# ---------------------------------------------------------------------------

def _build_knn_adj(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Return a symmetric k-NN binary adjacency matrix.

    Args:
        coords: (N, 2) node positions in [0, 1]^2.
        k: number of nearest neighbours (excluding self).

    Returns:
        adj: (N, N) float32 with 0/1 entries.
    """
    N = coords.shape[0]
    if N <= 1:
        return torch.ones(max(N, 1), max(N, 1), dtype=torch.float32)

    k_actual = min(k, N - 1)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)   # (N, N, 2)
    dist2 = (diff ** 2).sum(-1)                         # (N, N)

    # Indices of k+1 nearest (first is self with dist 0)
    _, idx = dist2.topk(k_actual + 1, largest=False, dim=1)
    idx = idx[:, 1:]   # (N, k_actual) — drop self

    adj = torch.zeros(N, N, dtype=torch.float32)
    row = torch.arange(N, dtype=torch.long).unsqueeze(1).expand_as(idx)
    adj[row, idx] = 1.0
    adj[idx, row] = 1.0   # symmetric
    return adj


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JetGraphDataset(Dataset):
    """Precomputed point-cloud graph dataset loaded from an HDF5 jet image file.

    Preprocessing (done once in __init__):
      - Log-normalise each channel: norm = log1p(raw) / log1p(scale)
      - Keep pixels where any channel > threshold
      - If a jet has more than max_nodes active pixels, keep the top-N by
        summed channel energy
      - Build a symmetric k-NN adjacency matrix in (η, φ) space

    Memory layout (float16 / uint8 to minimise RAM):
      _node_feats : (n_samples, max_nodes, 5)  float16  ~209 MB
      _adj        : (n_samples, max_nodes, max_nodes) uint8  ~3.1 GB
      _num_nodes  : (n_samples,)  int16
      labels      : (n_samples,)  int8
    """

    def __init__(
        self,
        file_path: str | Path,
        image_key: str = "X_jets",
        label_key: str = "y",
        k_neighbors: int = 8,
        max_nodes: int = 150,
        threshold: float = 0.01,
        channel_scales: list[float] | tuple[float, ...] = (25.0, 2.0, 0.1),
    ) -> None:
        self.k = k_neighbors
        self.max_nodes = max_nodes

        log_scales = torch.log1p(torch.tensor(channel_scales, dtype=torch.float32))  # (C,)

        print(f"Loading raw jet images from {file_path} ...")
        with h5py.File(str(file_path), "r") as f:
            raw: np.ndarray = f[image_key][:]        # (N, H, W, C) float32
            labels_np: np.ndarray = f[label_key][:].astype(np.int8)

        n_total, H, W, C = raw.shape
        print(f"  {n_total} samples, image shape ({H}, {W}, {C})")

        # Pre-allocate storage tensors
        self._node_feats = torch.zeros(n_total, max_nodes, 5, dtype=torch.float16)
        self._adj        = torch.zeros(n_total, max_nodes, max_nodes, dtype=torch.uint8)
        self._num_nodes  = torch.zeros(n_total, dtype=torch.int16)
        self.labels      = torch.from_numpy(labels_np)

        print("Precomputing point-cloud graphs (runs once) ...")
        for i in tqdm(range(n_total)):
            img = torch.from_numpy(raw[i]).float()   # (H, W, C)

            # Log-normalise to [0, 1]
            norm = torch.log1p(img) / log_scales.view(1, 1, C)   # (H, W, C)

            # Active pixels: any channel above threshold
            active = (norm > threshold).any(dim=-1)   # (H, W)
            ys, xs = active.nonzero(as_tuple=True)
            N = ys.shape[0]

            if N == 0:
                self._num_nodes[i] = 1
                continue   # single dummy zero node already in _node_feats

            # Coordinates normalised to [0, 1]
            eta = ys.float() / (H - 1)
            phi = xs.float() / (W - 1)

            feats = norm[ys, xs, :]            # (N, C)
            node_feats = torch.cat(
                [eta.unsqueeze(1), phi.unsqueeze(1), feats], dim=1
            )   # (N, 5)

            # Trim to max_nodes by highest total energy
            if N > max_nodes:
                topk = feats.sum(1).topk(max_nodes).indices
                node_feats = node_feats[topk]
                N = max_nodes

            # k-NN adjacency
            coords = node_feats[:, :2]   # (N, 2)
            adj = _build_knn_adj(coords, self.k)   # (N, N)

            # Store (pad to max_nodes)
            self._node_feats[i, :N] = node_feats.half()
            self._adj[i, :N, :N]   = adj.to(torch.uint8)
            self._num_nodes[i]     = N

        print("Graph precomputation complete.")

    def __len__(self) -> int:
        return self._num_nodes.shape[0]

    def __getitem__(self, idx: int):
        node_feats = self._node_feats[idx].float()     # (max_nodes, 5)
        adj        = self._adj[idx].float()            # (max_nodes, max_nodes)
        num_nodes  = int(self._num_nodes[idx])
        label      = int(self.labels[idx])
        return node_feats, adj, num_nodes, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_jet_graph_dataloaders(
    data_config: dict[str, Any],
    gnn_config: dict[str, Any],
    seed: int,
) -> dict[str, DataLoader]:
    dataset = JetGraphDataset(
        file_path=data_config["file_path"],
        image_key=data_config["image_key"],
        label_key=data_config["label_key"],
        k_neighbors=int(gnn_config["k_neighbors"]),
        max_nodes=int(gnn_config["max_nodes"]),
        threshold=float(gnn_config.get("threshold", 0.01)),
        channel_scales=list(data_config["channel_scales"]),
    )

    n = len(dataset)
    train_frac = float(data_config["train_split"])
    val_frac   = float(data_config["val_split"])
    test_frac  = float(data_config["test_split"])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    train_end = int(n * train_frac)
    val_end   = train_end + int(n * val_frac)

    subsets = {
        "train": Subset(dataset, perm[:train_end]),
        "val":   Subset(dataset, perm[train_end:val_end]),
        "test":  Subset(dataset, perm[val_end:]),
    }

    batch_size = int(gnn_config["batch_size"])
    # num_workers=0: all data is already in RAM tensors, no benefit from workers
    common: dict[str, Any] = {"batch_size": batch_size, "num_workers": 0}

    return {
        "train": DataLoader(subsets["train"], shuffle=True,  **common),
        "val":   DataLoader(subsets["val"],   shuffle=False, **common),
        "test":  DataLoader(subsets["test"],  shuffle=False, **common),
    }
