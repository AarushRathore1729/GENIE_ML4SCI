"""GraphSAGE-based classifier for quark/gluon jet images represented as graphs.

Architecture
------------
Input  : node_feats (B, N, 5), adj (B, N, N), num_nodes (B,)
         N = max_nodes (padded), 5 features = [η, φ, ECAL, HCAL, Track]

Encoder: 3 × SAGEConv  5 → 64 → 128 → 128
Pool   : global mean + max over real (non-padded) nodes  →  (B, 256)
Head   : Linear(256, 128) → ReLU → Dropout → Linear(128, 1)
Output : raw logit  (pass through sigmoid for probability)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    """GraphSAGE convolution with masked mean-neighbour aggregation.

    h_v' = LayerNorm( W_self * h_v  +  W_neigh * mean_{u ∈ N(v)} h_u )

    Padded nodes (mask=False) are zeroed out before and after the op so they
    never contribute to or receive messages.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin_self  = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        self.norm      = nn.LayerNorm(out_features)

    def forward(
        self,
        x:    torch.Tensor,   # (B, N, F)
        adj:  torch.Tensor,   # (B, N, N)  float
        mask: torch.Tensor,   # (B, N)     bool  — True for real nodes
    ) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)                             # zero padded

        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, N, 1)
        neigh  = torch.bmm(adj, x) / degree                    # (B, N, F)

        h = self.lin_self(x) + self.lin_neigh(neigh)
        h = F.relu(self.norm(h))
        return h * mask.unsqueeze(-1)                           # re-zero padded


def _masked_global_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean-pool and max-pool over nodes.

    Args:
        x   : (B, N, F)
        mask: (B, N) bool

    Returns:
        (B, 2*F)
    """
    mask_f = mask.unsqueeze(-1).float()                          # (B, N, 1)
    mean = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)   # (B, F)

    x_inf = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    max_  = x_inf.max(dim=1).values                              # (B, F)

    return torch.cat([mean, max_], dim=-1)                       # (B, 2F)


class GNNClassifier(nn.Module):
    """GraphSAGE encoder + global pool + MLP for binary jet classification."""

    def __init__(
        self,
        in_features: int = 5,
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden * 2)
        self.conv3 = SAGEConv(hidden * 2, hidden * 2)

        pool_dim = hidden * 2 * 2   # mean + max concat
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # (B, N, 5)
        adj:        torch.Tensor,   # (B, N, N)
        num_nodes:  torch.Tensor,   # (B,)  int
    ) -> torch.Tensor:              # (B, 1) logit
        B, N, _ = node_feats.shape
        mask = (
            torch.arange(N, device=num_nodes.device).unsqueeze(0)
            < num_nodes.unsqueeze(1)
        )   # (B, N)

        h = self.conv1(node_feats, adj, mask)
        h = self.conv2(h, adj, mask)
        h = self.conv3(h, adj, mask)

        pooled = _masked_global_pool(h, mask)   # (B, pool_dim)
        return self.head(pooled)                 # (B, 1)
