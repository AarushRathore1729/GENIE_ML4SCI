"""Train the GNN quark/gluon classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from tqdm import tqdm

from ldm.config import ensure_dir, load_config, resolve_device, set_seed
from gnn.data import create_jet_graph_dataloaders
from gnn.model import GNNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN quark/gluon classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


@torch.no_grad()
def _evaluate(
    model: GNNClassifier,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Returns (avg_loss, roc_auc, accuracy)."""
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs:  list[float] = []

    for node_feats, adj, num_nodes, labels in loader:
        node_feats = node_feats.to(device)
        adj        = adj.to(device)
        num_nodes  = num_nodes.to(device)
        labels_f   = labels.float().to(device)

        logits = model(node_feats, adj, num_nodes).squeeze(1)
        total_loss += criterion(logits, labels_f).item()
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.tolist())

    avg_loss = total_loss / max(len(loader), 1)
    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    return avg_loss, auc, acc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gnn_cfg  = config["gnn"]
    exp_cfg  = config["experiment"]
    data_cfg = config["data"]

    set_seed(int(exp_cfg["seed"]))
    device = resolve_device(str(exp_cfg["device"]))
    output_dir = ensure_dir(exp_cfg["output_dir"])

    checkpoint_path = Path(gnn_cfg["checkpoint"])
    ensure_dir(checkpoint_path.parent)

    # ------------------------------------------------------------------ data
    loaders = create_jet_graph_dataloaders(data_cfg, gnn_cfg, int(exp_cfg["seed"]))
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ----------------------------------------------------------------- model
    model = GNNClassifier(
        in_features=5,
        hidden=int(gnn_cfg["hidden"]),
        dropout=float(gnn_cfg.get("dropout", 0.3)),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GNNClassifier: {total_params:,} trainable parameters")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(gnn_cfg["learning_rate"]))

    epochs = int(gnn_cfg["epochs"])
    best_val_auc = 0.0

    log_path = output_dir / "gnn_losses.csv"
    log_handle = open(log_path, "w", newline="")
    log_writer  = csv.writer(log_handle)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_acc"])

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for node_feats, adj, num_nodes, labels in tqdm(
                train_loader, desc=f"GNN Epoch {epoch + 1}/{epochs}"
            ):
                node_feats = node_feats.to(device)
                adj        = adj.to(device)
                num_nodes  = num_nodes.to(device)
                labels_f   = labels.float().to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(node_feats, adj, num_nodes).squeeze(1)
                loss = criterion(logits, labels_f)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_loss, val_auc, val_acc = _evaluate(model, val_loader, device, criterion)

            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} "
                f"val_acc={val_acc:.4f}"
            )
            log_writer.writerow(
                [epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{val_auc:.6f}", f"{val_acc:.4f}"]
            )
            log_handle.flush()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_val_auc": best_val_auc,
                        "config": config,
                    },
                    checkpoint_path,
                )
    finally:
        log_handle.close()

    print(f"Training complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
