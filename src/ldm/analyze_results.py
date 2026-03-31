from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        values.append(float(raw) if raw and raw.lower() != "nan" else np.nan)
    return np.asarray(values, dtype=float)


def _load_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_autoencoder_losses(output_dir: Path) -> None:
    rows = _read_csv_rows(ROOT / "outputs_tuned" / "autoencoder_losses.csv")
    epochs = _float_series(rows, "epoch")
    train = _float_series(rows, "train_loss")
    val = _float_series(rows, "val_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train loss", color="#1D4ED8", lw=2)
    ax.plot(epochs, val, label="Val loss", color="#DC2626", lw=2)

    nan_mask = np.isnan(train) | np.isnan(val)
    if nan_mask.any():
        nan_epochs = epochs[nan_mask]
        ax.scatter(
            nan_epochs,
            np.full_like(nan_epochs, np.nanmax(np.nan_to_num(val, nan=0.0))),
            color="#111827",
            marker="x",
            s=60,
            label="NaN epochs",
            zorder=5,
        )

    ax.set_title("Tuned Autoencoder Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "autoencoder_loss_plot_tuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diffusion_losses(output_dir: Path) -> None:
    tuned_rows = _read_csv_rows(ROOT / "outputs_tuned" / "diffusion_losses_mean.csv")
    base_rows = _read_csv_rows(ROOT / "outputs" / "diffusion_losses_mean.csv")

    tuned_epochs = _float_series(tuned_rows, "epoch")
    tuned_train = _float_series(tuned_rows, "train_loss")
    tuned_val = _float_series(tuned_rows, "val_loss")
    base_epochs = _float_series(base_rows, "epoch")
    base_val = _float_series(base_rows, "val_loss")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(tuned_epochs, tuned_train, label="Train loss", color="#1D4ED8", lw=2)
    axes[0].plot(tuned_epochs, tuned_val, label="Val loss", color="#DC2626", lw=2)
    axes[0].set_title("Tuned Diffusion Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(base_epochs, base_val, label="Baseline val", color="#6B7280", lw=2)
    axes[1].plot(tuned_epochs, tuned_val, label="Tuned val", color="#059669", lw=2)
    axes[1].set_title("Baseline vs Tuned Diffusion Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "diffusion_loss_analysis_tuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_comparison(output_dir: Path) -> None:
    baseline = _load_json(ROOT / "outputs" / "evaluation" / "diffusion_metrics_mean.json")
    tuned = _load_json(ROOT / "outputs_tuned" / "evaluation" / "diffusion_metrics_mean.json")

    labels = [
        "FID",
        "Denoising MSE",
        "W1 total ch0",
        "W1 mult ch0",
        "Centroid Y W1",
        "Sparsity gap ch0",
    ]
    baseline_vals = np.asarray(
        [
            baseline["fid"],
            baseline["denoising_mse"],
            baseline["w1_obs_total_ch0"],
            baseline["w1_obs_mult_ch0"],
            baseline["w1_obs_centroid_y"],
            abs(baseline["sparsity_real_ch0"] - baseline["sparsity_gen_ch0"]),
        ],
        dtype=float,
    )
    tuned_vals = np.asarray(
        [
            tuned["fid"],
            tuned["denoising_mse"],
            tuned["w1_obs_total_ch0"],
            tuned["w1_obs_mult_ch0"],
            tuned["w1_obs_centroid_y"],
            abs(tuned["sparsity_real_ch0"] - tuned["sparsity_gen_ch0"]),
        ],
        dtype=float,
    )

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#94A3B8")
    ax.bar(x + width / 2, tuned_vals, width, label="Tuned", color="#0F766E")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Diffusion Metric Comparison: Baseline vs Tuned")
    ax.set_ylabel("Metric value (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "diffusion_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary(output_dir: Path) -> None:
    auto = _load_json(ROOT / "outputs_tuned" / "evaluation" / "autoencoder_metrics.json")
    base = _load_json(ROOT / "outputs" / "evaluation" / "diffusion_metrics_mean.json")
    tuned = _load_json(ROOT / "outputs_tuned" / "evaluation" / "diffusion_metrics_mean.json")

    summary = f"""# Tuned Results Summary

## Autoencoder

- MSE: {auto["mse"]:.6g}
- MAE: {auto["mae"]:.6g}
- PSNR: {auto["psnr"]:.4f}
- SSIM: {auto["ssim"]:.6f}

## Diffusion

- Baseline FID: {base["fid"]:.4f}
- Tuned FID: {tuned["fid"]:.4f}
- Baseline denoising MSE: {base["denoising_mse"]:.6f}
- Tuned denoising MSE: {tuned["denoising_mse"]:.6f}
- Baseline W1 total ch0: {base["w1_obs_total_ch0"]:.4f}
- Tuned W1 total ch0: {tuned["w1_obs_total_ch0"]:.4f}
- Baseline W1 multiplicity ch0: {base["w1_obs_mult_ch0"]:.4f}
- Tuned W1 multiplicity ch0: {tuned["w1_obs_mult_ch0"]:.4f}

## Notes

- The tuned run improved FID substantially relative to baseline.
- The tuned autoencoder training became unstable late in training because the loss log contains NaNs from epoch 26 onward.
- Generated samples remain too sparse, especially in channels 1 and 2.
"""
    (output_dir / "results_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "outputs_tuned" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_autoencoder_losses(output_dir)
    _plot_diffusion_losses(output_dir)
    _plot_metric_comparison(output_dir)
    _write_summary(output_dir)
    print(f"Saved analysis plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
