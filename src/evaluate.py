"""
Evaluation script: load trained checkpoints and produce a comparison table.

Metrics computed per model × per target × overall:
  - MSE on test split (2022-2023)
  - MAE on test split
  - MSE gain of gr_add over dlinear (%)

Secondary analysis: stratify MSE gain by reporting irregularity severity
  (mean missingness rate across context window).

Usage:
  python -m src.evaluate
  python -m src.evaluate --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HIDDEN_DIM, N_TARGETS, N_INDICATORS, SIGMA, CONTEXT_WINDOW,
    TARGET_INDICATORS, BATCH_SIZE, ROOT,
)
from src.data.prealign import build_all_splits
from src.data.dataset import MacroDataset, macro_collate, DataLoader
from src.models.dlinear import DLinear
from src.models.mmf import GRAddMMF, NumericalOnlyGRU
from src.train import build_model

CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"


def collect_preds(model, loader, device):
    """Run model on loader; return (preds, targets) as numpy arrays."""
    model.eval()
    all_preds, all_targets, all_miss_rates = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch).cpu().numpy()
            tgt  = batch["target"].cpu().numpy()
            # missingness rate per sample = fraction of masked entries
            miss = 1.0 - batch["mask"].cpu().numpy().mean(axis=(1, 2))  # (B,)

            all_preds.append(pred)
            all_targets.append(tgt)
            all_miss_rates.append(miss)

    return (np.vstack(all_preds),
            np.vstack(all_targets),
            np.concatenate(all_miss_rates))


def evaluate_model(model_name: str, samples: list, device: torch.device):
    ckpt = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not ckpt.exists():
        print(f"  [skip] {model_name}: no checkpoint at {ckpt}")
        return None

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    loader = DataLoader(
        MacroDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=macro_collate,
    )
    preds, targets, miss_rates = collect_preds(model, loader, device)

    mse_per_target = ((preds - targets) ** 2).mean(axis=0)   # (n_targets,)
    mae_per_target = np.abs(preds - targets).mean(axis=0)
    mse_overall    = ((preds - targets) ** 2).mean()

    return {
        "preds":      preds,
        "targets":    targets,
        "miss_rates": miss_rates,
        "mse":        mse_per_target,
        "mae":        mae_per_target,
        "mse_overall": mse_overall,
    }


def print_comparison_table(results: dict[str, dict]):
    print("\n" + "=" * 60)
    print(" MODEL COMPARISON — TEST SET MSE")
    print("=" * 60)

    header = f"{'Model':<20}" + "".join(f"{t:>14}" for t in TARGET_INDICATORS) + f"{'Overall':>14}"
    print(header)
    print("-" * 60)

    for model_name, res in results.items():
        if res is None:
            print(f"{model_name:<20}  [not trained]")
            continue
        cols = "".join(f"{res['mse'][i]:>14.4f}" for i in range(N_TARGETS))
        print(f"{model_name:<20}{cols}{res['mse_overall']:>14.4f}")

    print("=" * 60)

    # MSE improvement of gr_add over dlinear
    if "gr_add" in results and "dlinear" in results:
        if results["gr_add"] and results["dlinear"]:
            gain = (results["dlinear"]["mse_overall"] - results["gr_add"]["mse_overall"])
            gain_pct = 100 * gain / results["dlinear"]["mse_overall"]
            print(f"\nGR-Add vs DLinear MSE reduction: {gain:.4f} ({gain_pct:+.1f}%)")


def stratified_analysis(results: dict[str, dict]):
    """
    Split test samples into LOW / HIGH missingness and compare GR-Add vs DLinear.
    Hypothesis: text helps more when numerical data is gappy.
    """
    if not (results.get("gr_add") and results.get("dlinear")):
        return

    miss = results["gr_add"]["miss_rates"]
    threshold = np.median(miss)

    for label, idx in [("LOW miss  (≤median)", miss <= threshold),
                        ("HIGH miss (>median)", miss > threshold)]:
        if idx.sum() == 0:
            continue
        gr_mse  = ((results["gr_add"]["preds"][idx]  - results["gr_add"]["targets"][idx])  ** 2).mean()
        dl_mse  = ((results["dlinear"]["preds"][idx] - results["dlinear"]["targets"][idx]) ** 2).mean()
        gain_pct = 100 * (dl_mse - gr_mse) / (dl_mse + 1e-8)
        print(f"  {label}: DLinear={dl_mse:.4f}  GR-Add={gr_mse:.4f}  "
              f"improvement={gain_pct:+.1f}%")


def run(split: str = "test", device_str: str = "cpu"):
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data splits …")
    train_s, val_s, test_s, _, _ = build_all_splits()
    samples = {"train": train_s, "val": val_s, "test": test_s}[split]
    print(f"Evaluating {len(samples)} samples from [{split}] split\n")

    model_names = ["dlinear", "numerical_gru", "gr_add"]
    all_results = {}

    for name in model_names:
        print(f"Evaluating {name} …")
        all_results[name] = evaluate_model(name, samples, device)

    print_comparison_table(all_results)

    print("\n--- Stratified Analysis (text benefit vs missingness) ---")
    stratified_analysis(all_results)

    # Save summary JSON
    summary = {}
    for name, res in all_results.items():
        if res:
            summary[name] = {
                "mse_gdp":      float(res["mse"][0]),
                "mse_inf":      float(res["mse"][1]),
                "mae_gdp":      float(res["mae"][0]),
                "mae_inf":      float(res["mae"][1]),
                "mse_overall":  float(res["mse_overall"]),
            }
    out = RESULTS_DIR / f"eval_{split}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.split, args.device)
