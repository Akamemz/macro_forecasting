"""
Training script for the macro forecasting pipeline.

Models
------
  dlinear        : DLinear (unimodal numerical baseline)
  numerical_gru  : GRU numerical-only (unimodal, same encoder as GR-Add)
  gr_add         : GRAddMMF (full multimodal: GRU + RecAvg + GR-Add)

Usage
-----
  python -m src.train --model gr_add
  python -m src.train --model dlinear --epochs 50
  python -m src.train --model numerical_gru
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HIDDEN_DIM, N_TARGETS, N_INDICATORS, SIGMA, CONTEXT_WINDOW,
    LR, EPOCHS, EARLY_STOPPING, BATCH_SIZE, ROOT,
)
from src.data.prealign import build_all_splits
from src.data.dataset import make_loaders
from src.models.dlinear import DLinear
from src.models.mmf import GRAddMMF, NumericalOnlyGRU


CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"


def build_model(name: str) -> nn.Module:
    if name == "dlinear":
        return DLinear(seq_len=CONTEXT_WINDOW, n_indicators=N_INDICATORS, n_targets=N_TARGETS)
    if name == "numerical_gru":
        return NumericalOnlyGRU(n_indicators=N_INDICATORS, hidden_dim=HIDDEN_DIM, n_targets=N_TARGETS)
    if name == "gr_add":
        return GRAddMMF(n_indicators=N_INDICATORS, hidden_dim=HIDDEN_DIM,
                        n_targets=N_TARGETS, sigma=SIGMA)
    raise ValueError(f"Unknown model: {name}")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred   = model(batch)
            loss   = mse_loss(pred, batch["target"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


def train(model_name: str, epochs: int, batch_size: int, lr: float, device_str: str):
    device = torch.device(device_str)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Model: {model_name}")
    print(f" Device: {device} | LR: {lr} | Epochs: {epochs}")
    print(f"{'='*60}\n")

    # ── data ────────────────────────────────────────────────────────────────
    train_s, val_s, test_s, means, stds = build_all_splits()
    train_loader, val_loader, test_loader = make_loaders(train_s, val_s, test_s, batch_size)

    # ── model ────────────────────────────────────────────────────────────────
    model     = build_model(model_name).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── training loop ────────────────────────────────────────────────────────
    history = {"train_mse": [], "val_mse": []}
    best_val_mse = float("inf")
    patience_cnt = 0
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_mse = run_epoch(model, train_loader, optimizer, device, train=True)
        val_mse   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step(val_mse)

        history["train_mse"].append(round(train_mse, 6))
        history["val_mse"].append(round(val_mse, 6))

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"Ep {ep:03d}/{epochs}  train={train_mse:.4f}  val={val_mse:.4f}"
                  f"  ({elapsed:.0f}s)")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING:
                print(f"\nEarly stopping at epoch {ep} (patience={EARLY_STOPPING})")
                break

    # ── test evaluation ──────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val MSE={best_val_mse:.4f}) …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_mse = run_epoch(model, test_loader, optimizer, device, train=False)
    print(f"Test MSE: {test_mse:.4f}")

    # ── save results ─────────────────────────────────────────────────────────
    results = {
        "model":      model_name,
        "n_params":   n_params,
        "best_val_mse": round(best_val_mse, 6),
        "test_mse":   round(test_mse, 6),
        "history":    history,
    }
    res_path = RESULTS_DIR / f"{model_name}_results.json"
    res_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {res_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gr_add",
                        choices=["dlinear", "numerical_gru", "gr_add"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch, args.lr, args.device)
