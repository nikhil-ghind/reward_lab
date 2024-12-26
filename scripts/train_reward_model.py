"""
Train the RewardModel using Bradley-Terry pairwise preference loss.

Usage:
    python -m scripts.train_reward_model --config configs/config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from models.reward_model import RewardModel
from eval.metrics import RewardModelAccuracy


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(
    model: RewardModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        optimizer.zero_grad()
        loss, metrics = model.compute_reward_loss(
            chosen_ids, chosen_mask, rejected_ids, rejected_mask
        )
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]
        total_margin += metrics["reward_margin"]
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "acc": f"{metrics['accuracy']:.3f}",
        })

    return {
        "train_loss": total_loss / n_batches,
        "train_accuracy": total_accuracy / n_batches,
        "train_margin": total_margin / n_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: RewardModel,
    loader,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    for batch in pbar:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        loss, metrics = model.compute_reward_loss(
            chosen_ids, chosen_mask, rejected_ids, rejected_mask
        )
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]
        n_batches += 1

    return {
        "val_loss": total_loss / n_batches,
        "val_accuracy": total_accuracy / n_batches,
    }


def train(config_path: str, data_path: Optional[str] = None):
    cfg = load_config(config_path)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    torch.manual_seed(cfg["training"]["seed"])

    # Optional wandb logging
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["wandb_project"], config=cfg)
        except ImportError:
            print("wandb not installed, skipping")
            use_wandb = False

    # Data
    data_file = data_path or cfg["data"]["preferences_path"]
    print(f"Loading data from {data_file}")
    train_loader, val_loader = create_dataloaders(
        data_path=data_file,
        tokenizer_name=cfg["model"]["tokenizer"],
        max_length=cfg["model"]["max_seq_len"],
        batch_size=cfg["training"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["training"]["seed"],
    )

    # Model
    print(f"Initializing RewardModel from '{cfg['model']['base_model']}'")
    model = RewardModel(
        model_name=cfg["model"]["base_model"],
        dropout=cfg["reward_model"]["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["reward_model"]["lr"],
        weight_decay=cfg["reward_model"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["reward_model"]["epochs"],
        eta_min=cfg["reward_model"]["lr"] / 10,
    )

    # Training loop
    save_dir = Path(cfg["checkpoints"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    best_epoch = 0
    history = []

    print("\n" + "=" * 60)
    print("Starting Reward Model Training")
    print("=" * 60)

    for epoch in range(1, cfg["reward_model"]["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = eval_epoch(model, val_loader, device, epoch)
        scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        history.append(epoch_metrics)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"train_acc={train_metrics['train_accuracy']:.3f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_acc={val_metrics['val_accuracy']:.3f} | "
            f"time={elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log(epoch_metrics)

        # Save best model
        if val_metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["val_accuracy"]
            best_epoch = epoch
            checkpoint_path = str(save_dir / "reward_model_best.pt")
            model.save(checkpoint_path)
            print(f"  -> New best val accuracy: {best_val_accuracy:.3f} (saved)")

        # Save latest
        model.save(str(save_dir / "reward_model_latest.pt"))

    print("\n" + "=" * 60)
    print(f"Training complete. Best val accuracy: {best_val_accuracy:.3f} at epoch {best_epoch}")
    print("=" * 60)

    if use_wandb:
        import wandb
        wandb.finish()

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train reward model for RLHF")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    args = parser.parse_args()

    train(args.config, args.data)


if __name__ == "__main__":
    main()
