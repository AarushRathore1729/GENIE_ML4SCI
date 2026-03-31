# Run Summary

## What this does

All three tasks on the quark/gluon jet dataset:

1. **Common Task 1** — Variational autoencoder on 3-channel jet images (ECAL, HCAL, Tracks)
2. **Common Task 2** — Graph neural network for quark/gluon classification
3. **Specific Task 3** — Latent diffusion model on the VAE's learned latent space

Dataset:

- File: `quark-gluon_data-set_n139306.hdf5` (~669 MB)
- Images: `X_jets` shape `(139306, 125, 125, 3)`, padded to `128 × 128` for the LDM pipeline
- Labels: `y` — 0 = quark, 1 = gluon

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

## Commands

### Common Task 2 — GNN classifier

```bash
# Train (precomputes graphs once, then trains 30 epochs)
uv run gnn-train --config configs/latent_diffusion.yaml

# Evaluate on test set → ROC-AUC, ROC curve, score distribution
uv run gnn-eval  --config configs/latent_diffusion.yaml
```

Outputs:
- `outputs/evaluation/gnn_roc_curve.png`
- `outputs/evaluation/gnn_score_distribution.png`
- `outputs/evaluation/gnn_metrics.json`
- `outputs/gnn_losses.csv`

### Common Task 1 — Autoencoder

```bash
uv run ldm-train-autoencoder --config configs/latent_diffusion.yaml
uv run ldm-eval-autoencoder  --config configs/latent_diffusion.yaml
```

Outputs:
- `outputs/evaluation/autoencoder_reconstructions.png` — side-by-side original vs reconstructed
- `outputs/evaluation/autoencoder_histograms.png`
- `outputs/evaluation/latent_tsne.png`
- `outputs/evaluation/autoencoder_metrics.json` — MSE, MAE, PSNR, SSIM

### Specific Task 3 — Latent diffusion model

```bash
# Full pipeline in one command
uv run ldm-run-all --config configs/latent_diffusion.yaml

# Or step by step
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml
uv run ldm-sample          --config configs/latent_diffusion.yaml --num-samples 16
uv run ldm-eval-diffusion  --config configs/latent_diffusion.yaml
```

Outputs:
- `outputs/evaluation/diffusion_real_vs_generated_mean.png` — real vs generated side-by-side
- `outputs/evaluation/diffusion_sample_grid_mean.png`
- `outputs/evaluation/diffusion_histograms_mean.png`
- `outputs/evaluation/diffusion_jet_observables_mean.png`
- `outputs/evaluation/diffusion_metrics_mean.json` — FID, pixel W1, jet-observable W1, sparsity

### Latent-mode ablation (diffusion)

The diffusion stage supports two latent encoding modes:

```bash
# Mean mode (default — deterministic encoder mean)
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml --latent-mode mean

# Sample mode — uses reparameterised VAE samples
uv run ldm-train-diffusion --config configs/latent_diffusion.yaml --latent-mode sample
uv run ldm-eval-diffusion  --config configs/latent_diffusion.yaml --latent-mode sample
```

Checkpoints and output files are tagged with the mode name (`_mean` or `_sample`).

---

## Important output paths

| Artefact | Path |
|----------|------|
| Autoencoder checkpoint | `outputs/checkpoints/autoencoder.pt` |
| Diffusion checkpoint | `outputs/checkpoints/diffusion.pt` |
| GNN checkpoint | `outputs/checkpoints/gnn.pt` |
| Reconstruction figures | `outputs/reconstructions/` |
| Evaluation outputs | `outputs/evaluation/` |
| Generated samples | `outputs/samples/` |
| Loss CSVs | `outputs/` |

---

## Suggested run order (fresh start)

```bash
uv sync

# GNN (independent — run first or in parallel)
uv run gnn-train --config configs/latent_diffusion.yaml
uv run gnn-eval  --config configs/latent_diffusion.yaml

# Autoencoder then diffusion (sequential dependency)
uv run ldm-run-all --config configs/latent_diffusion.yaml
```

---

## GPU optimisations

| Optimisation | Effect |
|---|---|
| AMP mixed precision | ~2× speedup, ~50 % VRAM saving |
| `cudnn.benchmark = True` | Faster fixed-size convolutions |
| `persistent_workers` | No DataLoader worker respawn overhead |
| `non_blocking` device transfers | Overlaps CPU→GPU copies with compute |
| Smart `pin_memory` | Enabled only when CUDA is available |

---

## Hardware requirements

| Setup | Notes |
|-------|-------|
| CPU only | Works, but LDM training is very slow |
| NVIDIA T4 (16 GB) | Recommended free option (Colab / Kaggle) |
| RTX 3090/4090 or A100 | Comfortable for full 100-epoch diffusion run |

**RAM**: GNN graph precomputation uses ~3.5 GB. LDM dataset loading uses ~27 GB.
Minimum recommended system RAM: **16 GB** (GNN only) / **32 GB** (full pipeline).

Reduce `data.batch_size` in the config if you run out of GPU VRAM.

---

## Performance discussion and validity notes

### Common Task 1 (Autoencoder)

- `MSE = 2.13e-05`, `PSNR = 60.87 dB`, `SSIM = 0.996` on test set.
- These images are extremely sparse (most pixels ~0), so reconstructions look dark — that's expected for calorimeter data. Gamma correction is applied for visualization.

### Common Task 2 (GNN classification)

- `ROC-AUC = 0.785`, `accuracy = 0.716` on 13,932 test samples.
- Not state-of-the-art, but shows meaningful quark/gluon separation from the sparse graph structure.

### Specific Task 3 (Latent diffusion)

- Tuned run: `FID = 1078.0` (down from 2739.6 baseline). Real improvement.
- But generated samples are still too sparse in HCAL and Tracks channels (`sparsity = 1.0`).
- The model captures overall energy distributions better but doesn't yet activate the sparser channels.

### Honest assessment

- All three tasks produce end-to-end results: VAE with reconstructions, GNN with classification metrics, diffusion with generated samples and evaluation.
- The main weakness is that diffusion samples are too sparse — the model hasn't learned to generate the sparser channels (HCAL, Tracks). This is the primary thing to fix.

---

## Ablation experiments

Three additional ablation experiments were conducted to investigate latent space configuration effects on diffusion generation quality:

| Configuration | FID (↓) | Denoising MSE | Key Change |
|---------------|---------|---------------|------------|
| Baseline | 2739.6 | 0.00454 | — (reference) |
| **Tuned (best)** | **1078.0** | **0.00681** | **8-ch latent, 500 timesteps, KL 0.001** |
| Fixed (KL 0.05, warmup) | 2843.5 | 8.5e-6 | KL warmup 10 epochs, scaling cap 5.0 |
| Ablation A (stochastic latents) | 2505.2 | 0.560 | sample mode instead of mean |
| Ablation B (KL 0.01, warmup) | 1905.1 | 2.66e-5 | lower KL + warmup + scaling cap |

**Key findings:**
- The tuned configuration remains the best overall (lowest FID by a wide margin).
- Increasing KL regularization weight did not improve generation; it made the latent space too narrow.
- Stochastic latent encoding (sample mode) produced very high denoising MSE, indicating the diffusion model struggled with added noise.
- The ablation with KL=0.01 + warmup showed the best FID among ablations (1905.1), but still suffered from collapsed sparsity in HCAL/Tracks channels.
- All ablation runs produce `sparsity_gen_ch1` and `sparsity_gen_ch2` of 1.0 (all-zero), confirming that learning to generate sparse detector channels remains the primary open challenge.
