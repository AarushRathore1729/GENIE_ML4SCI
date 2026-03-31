# ML4SCI GENIE 

PyTorch implementation of all three tasks from the ML4SCI GENIE evaluation:

| Task | Description |
|------|-------------|
| **Common Task 1** | Variational autoencoder on 3-channel quark/gluon jet images |
| **Common Task 2** | Graph neural network (GNN) for quark/gluon jet classification |
| **Specific Task 3** | Latent diffusion model on the VAE's learned latent space |

Dataset: `quark-gluon_data-set_n139306.hdf5` — 139 306 jet events, 3 channels (ECAL, HCAL, Tracks), 125 × 125 pixels each.

---

## Project layout

```
configs/
  latent_diffusion.yaml       — baseline configuration for all stages
  latent_diffusion_tuned.yaml — tuned configuration (wider latent, dropout, fewer timesteps)
  latent_diffusion_fixed.yaml — ablation configuration (KL warmup, scaling cap)
src/
  ldm/                      — autoencoder + diffusion pipeline
    data.py                 — HDF5 loader, log-scale normalisation, denormalisation
    metrics.py              — MSE, MAE, PSNR, SSIM, FID, pixel W1, jet-observable W1, sparsity
    visualize.py            — side-by-side reconstruction plots (gamma-corrected for sparse jets)
    config.py               — seed, device, AMP helpers
    models/
      autoencoder.py        — convolutional VAE (encoder + decoder with Tanh output)
      unet.py               — UNet denoiser with sinusoidal time embeddings and mid-block self-attention
      scheduler.py          — DDPM forward diffusion and DDPM reverse sampling
      ldm.py                — wrapper combining VAE and UNet
    train_autoencoder.py    — Stage 1 training (MSE + KL loss, AMP, GradScaler)
    train_diffusion.py      — Stage 2 training (noise prediction MSE, latent normalisation)
    evaluate_autoencoder.py — MSE/PSNR/SSIM metrics, reconstruction grid, histograms, t-SNE
    evaluate_diffusion.py   — denoising MSE, FID, pixel W1, jet-observable W1, sparsity,
                              real-vs-generated comparison, jet observable histograms
    sample.py               — unconditional image generation
    run_all.py              — single-command full pipeline runner
  gnn/                      — graph neural network classifier (Common Task 2)
    data.py                 — converts jet images to point-cloud graphs; precomputes all
                              graphs in __init__ so __getitem__ is O(1)
    model.py                — GNNClassifier: 3 × SAGEConv + global mean/max pool + MLP head
    train.py                — BCE training, saves best checkpoint by val ROC-AUC
    evaluate.py             — ROC curve, score distribution plot, metrics JSON
pyproject.toml              — dependencies and CLI entry-points
```

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Or prefix every command with `uv run` to avoid activating the environment.

---

## Experiment configurations

| Config | Key changes | Output directory |
|--------|-------------|------------------|
| `latent_diffusion.yaml` | Baseline (4-ch latent, KL 0.0001, 1000 timesteps) | `outputs/` |
| `latent_diffusion_tuned.yaml` | 8-ch latent, KL 0.001, 500 timesteps, dropout 0.1 | `outputs_tuned/` |
| `latent_diffusion_fixed.yaml` | KL 0.05, KL warmup 10 epochs, scaling cap 5.0 | (ablation) |

The **tuned** config gives the best diffusion results (FID 1078 vs baseline 2740).

---

## Common Task 1 — Autoencoder

### Train

```bash
uv run ldm-train-autoencoder --config configs/latent_diffusion.yaml
```

### Evaluate

```bash
uv run ldm-eval-autoencoder --config configs/latent_diffusion.yaml
```

**Outputs** in `outputs/evaluation/`:

| File | Content |
|------|---------|
| `autoencoder_reconstructions.png` | Side-by-side: original events vs reconstructions |
| `autoencoder_histograms.png` | Per-channel intensity distributions |
| `latent_tsne.png` | t-SNE of latent space coloured by quark / gluon label |
| `autoencoder_metrics.json` | MSE, MAE, PSNR, SSIM (averaged over test set) |

---

## Common Task 2 — GNN Classifier

Each non-zero pixel in the calorimeter image becomes a graph node with five features:

```
[η_coord, φ_coord, ECAL_norm, HCAL_norm, Track_norm]
```

Nodes are connected by a symmetric k-NN graph (k = 8) in (η, φ) space.

**Architecture**: 3 × GraphSAGE convolution layers → global mean + max pooling → 2-layer MLP → binary logit.

### Train

```bash
uv run gnn-train --config configs/latent_diffusion.yaml
```

Graph precomputation runs once at startup (~3–5 min, ~3.5 GB RAM). Training itself is fast after that.

### Evaluate

```bash
uv run gnn-eval --config configs/latent_diffusion.yaml
```

**Outputs** in `outputs/evaluation/`:

| File | Content |
|------|---------|
| `gnn_roc_curve.png` | ROC curve with AUC score |
| `gnn_score_distribution.png` | Classifier output distribution for quarks vs gluons |
| `gnn_metrics.json` | ROC-AUC, accuracy, confusion matrix |
| `gnn_losses.csv` | Per-epoch train/val loss and AUC |

---

## Specific Task 3 — Latent Diffusion Model

Trains a DDPM-style denoiser in the VAE latent space (4 channels, 16 × 16).
The UNet denoiser includes a self-attention block at the 4 × 4 bottleneck for global context.

### Train (requires autoencoder checkpoint first)

```bash
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml
```

### Sample

```bash
uv run ldm-sample --config configs/latent_diffusion.yaml --num-samples 16
```

### Evaluate

```bash
uv run ldm-eval-diffusion --config configs/latent_diffusion.yaml
```

**Outputs** in `outputs/evaluation/`:

| File | Content |
|------|---------|
| `diffusion_real_vs_generated_mean.png` | Side-by-side: real jets vs generated jets |
| `diffusion_sample_grid_mean.png` | Grid of 64 generated jet events |
| `diffusion_histograms_mean.png` | Per-channel pixel intensity: real vs generated |
| `diffusion_jet_observables_mean.png` | Jet-level observable distributions (total pT, multiplicity, max pixel, centroid) |
| `diffusion_metrics_mean.json` | Denoising MSE, FID, pixel W1, jet-observable W1, sparsity |

### Full LDM pipeline in one command

```bash
uv run ldm-run-all --config configs/latent_diffusion.yaml
```

Stages: `train-ae` → `eval-ae` → `train-diff` → `sample` → `eval-diff`.
Run a subset with `--stages train-diff sample eval-diff`.

### Latent-mode ablation

```bash
# Default (deterministic encoder means)
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml --latent-mode mean

# Ablation (sampled VAE latents)
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml --latent-mode sample
uv run ldm-eval-diffusion  --config configs/latent_diffusion.yaml --latent-mode sample
```

---

## Recommended run order (first time)

```bash
uv sync

# Common Task 2 — GNN (independent of autoencoder)
uv run gnn-train --config configs/latent_diffusion.yaml
uv run gnn-eval  --config configs/latent_diffusion.yaml

# Common Task 1 — Autoencoder
uv run ldm-train-autoencoder --config configs/latent_diffusion.yaml
uv run ldm-eval-autoencoder  --config configs/latent_diffusion.yaml

# Specific Task 3 — Diffusion (needs autoencoder checkpoint)
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml
uv run ldm-sample          --config configs/latent_diffusion.yaml --num-samples 16
uv run ldm-eval-diffusion  --config configs/latent_diffusion.yaml
```

### Running the tuned configuration

```bash
uv run ldm-run-all --config configs/latent_diffusion_tuned.yaml
```

---

## Results summary

| Task | Metric | Baseline | Tuned |
|------|--------|----------|-------|
| VAE (Common Task 1) | MSE | 2.13e-5 | 2.82e-5 |
| | PSNR (dB) | 60.87 | 59.21 |
| | SSIM | 0.996 | 0.998 |
| GNN (Common Task 2) | ROC-AUC | 0.785 | — |
| | Accuracy | 0.716 | — |
| Diffusion (Specific Task 3) | FID | 2739.6 | 1078.0 |
| | W1 total intensity (ECAL) | 7.95 | 1.63 |
| | Denoising MSE | 0.00454 | 0.00681 |

I also ran ablation experiments (KL warmup, latent scaling cap, stochastic latent mode, lower KL weight) that improved training stability but didn't beat the tuned config on FID. Details in `RUN_SUMMARY.md`.

---

## GPU optimisations (LDM pipeline)

| Optimisation | Effect |
|---|---|
| AMP mixed precision (`autocast` + `GradScaler`) | ~2× speedup, ~50 % VRAM saving |
| `cudnn.benchmark = True` | Faster convolutions for fixed-size inputs |
| `persistent_workers` | Avoids DataLoader worker respawn overhead |
| `non_blocking` transfers | Overlaps CPU→GPU copies with compute |

These kick in automatically on CUDA and fall back to CPU.

---

## Hardware requirements

| Setup | Spec |
|-------|------|
| Minimum (CPU only) | 8 GB RAM, 4-core CPU — very slow, for testing only |
| Recommended | NVIDIA T4 (16 GB VRAM, free on Colab/Kaggle), 16 GB system RAM |
| Comfortable | RTX 3090/4090 or A100, 32 GB RAM |

**GNN note**: graph precomputation allocates ~3.5 GB RAM. This is a one-time cost per run.

If you run out of GPU VRAM, reduce `data.batch_size` in `configs/latent_diffusion.yaml`.
