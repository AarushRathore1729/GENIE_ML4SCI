"""Evaluate the trained GNN on the test set and produce submission-ready outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ldm.config import ensure_dir, load_config, resolve_device, set_seed
from gnn.data import create_jet_graph_dataloaders
from gnn.model import GNNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GNN quark/gluon classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config   = load_config(args.config)
    gnn_cfg  = config["gnn"]
    exp_cfg  = config["experiment"]
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    set_seed(int(exp_cfg["seed"]))
    device     = resolve_device(str(exp_cfg["device"]))
    output_dir = ensure_dir(eval_cfg["output_dir"])

    # -------------------------------------------------------------- data
    loaders  = create_jet_graph_dataloaders(data_cfg, gnn_cfg, int(exp_cfg["seed"]))
    test_loader = loaders["test"]

    # ------------------------------------------------------------- model
    model = GNNClassifier(
        in_features=5,
        hidden=int(gnn_cfg["hidden"]),
        dropout=float(gnn_cfg.get("dropout", 0.3)),
    ).to(device)

    checkpoint_path = Path(gnn_cfg["checkpoint"])
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {checkpoint['epoch']}, best_val_auc={checkpoint['best_val_auc']:.4f})"
    )

    # -------------------------------------------------- collect predictions
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for node_feats, adj, num_nodes, labels in test_loader:
            node_feats = node_feats.to(device)
            adj        = adj.to(device)
            num_nodes  = num_nodes.to(device)

            logits = model(node_feats, adj, num_nodes).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds = [1 if p > 0.5 else 0 for p in all_probs]

    # -------------------------------------------------------------- metrics
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)
    cm  = confusion_matrix(all_labels, preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    print(f"Test ROC-AUC : {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # ---------------------------------------------------------- ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4F46E5", lw=2.5, label=f"GNN  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Quark/Gluon Classification", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = Path(output_dir) / "gnn_roc_curve.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve to {roc_path}")

    # ----------------------------------------------- score distribution plot
    quark_probs  = [p for p, l in zip(all_probs, all_labels) if l == 0]
    gluon_probs  = [p for p, l in zip(all_probs, all_labels) if l == 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 50
    ax.hist(quark_probs, bins=bins, alpha=0.6, density=True, label="Quark", color="#4F46E5")
    ax.hist(gluon_probs, bins=bins, alpha=0.6, density=True, label="Gluon", color="#EC4899")
    ax.set_xlabel("GNN output (P(gluon))", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Classifier Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    score_path = Path(output_dir) / "gnn_score_distribution.png"
    fig.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved score distribution to {score_path}")

    # --------------------------------------------------------- metrics JSON
    summary = {
        "roc_auc":          float(auc),
        "accuracy":         float(acc),
        "num_test_samples": len(all_labels),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_val_auc":     float(checkpoint["best_val_auc"]),
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = Path(output_dir) / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
