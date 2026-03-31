from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from ldm.config import create_grad_scaler, ensure_dir, load_config, resolve_device, set_seed
from ldm.data import create_jet_dataloaders
from ldm.visualize import save_side_by_side_reconstructions
from ldm.models import AutoencoderKL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder stage for latent diffusion.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    checkpoint_path = Path(config["autoencoder"]["checkpoint"])
    ensure_dir(checkpoint_path.parent)
    recon_dir = ensure_dir(output_dir / "reconstructions")

    dataloaders = create_jet_dataloaders(config["data"], seed=config["experiment"]["seed"])
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = AutoencoderKL(
        in_channels=config["model"]["in_channels"],
        latent_channels=config["model"]["latent_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["autoencoder"]["learning_rate"])
    scaler = create_grad_scaler(device)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    epochs = config["autoencoder"]["epochs"]
    kl_weight = config["autoencoder"]["kl_weight"]
    kl_warmup_epochs = int(config["autoencoder"].get("kl_warmup_epochs", 0))
    best_val_loss = float("inf")

    # CSV loss log
    log_path = output_dir / "autoencoder_losses.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(epochs):
        if kl_warmup_epochs > 0 and epoch < kl_warmup_epochs:
            effective_kl = kl_weight * (epoch + 1) / kl_warmup_epochs
        else:
            effective_kl = kl_weight

        model.train()
        progress = tqdm(train_loader, desc=f"Autoencoder Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype):
                reconstruction, mean, logvar = model(images)
                loss, metrics = model.loss_function(reconstruction, images, mean, logvar, effective_kl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += metrics["loss"]
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", recon=f"{metrics['recon_loss']:.4f}", kl_w=f"{effective_kl:.4f}")

        average_loss = running_loss / len(train_loader)

        model.eval()
        validation_loss = 0.0
        first_val_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
            for batch_index, (images, _) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                reconstruction, mean, logvar = model(images)
                loss, _ = model.loss_function(reconstruction, images, mean, logvar, effective_kl)
                validation_loss += float(loss.item())
                if batch_index == 0:
                    first_val_batch = (images, reconstruction)

        validation_loss /= max(len(val_loader), 1)
        print(
            f"epoch={epoch + 1} autoencoder_train_loss={average_loss:.6f} "
            f"autoencoder_val_loss={validation_loss:.6f}"
        )
        log_writer.writerow([epoch + 1, f"{average_loss:.6f}", f"{validation_loss:.6f}"])
        log_file.flush()

        if first_val_batch is not None:
            original_batch, reconstructed_batch = first_val_batch
            save_side_by_side_reconstructions(
                original_batch,
                reconstructed_batch,
                recon_dir / f"epoch_{epoch + 1:03d}.png",
                max_items=min(8, original_batch.shape[0]),
            )

        if validation_loss <= best_val_loss:
            best_val_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
