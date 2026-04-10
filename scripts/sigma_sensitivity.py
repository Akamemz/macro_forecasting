"""
Sigma sensitivity analysis for the RecAvg TTF Gaussian kernel.

Trains GR-Add MMF for each sigma in {0.5, 1.0, 2.0, 5.0} and compares
test MSE to isolate whether temporal decay in the text fusion matters.

Results saved to:
  experiments/03_sigma_sensitivity/sigma_{value}/

Usage:
    python scripts/sigma_sensitivity.py
    python scripts/sigma_sensitivity.py --sigmas 0.5 1.0 2.0 5.0 --device cuda
    python scripts/sigma_sensitivity.py --epochs 100 --device cuda
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.data.prealign import build_all_splits
from src.data.dataset import MacroDataset, macro_collate
from src.models.mmf import GRAddMMF
from src.evaluate import evaluate_model, print_comparison_table

EXP_DIR = ROOT / "experiments" / "03_sigma_sensitivity"
CKPT_DIR = ROOT / "checkpoints"


def train_one(sigma: float, epochs: int, device: torch.device,
              train_samples, val_samples) -> dict:
    """Train GR-Add MMF with a specific sigma. Returns best val MSE and history."""

    # patch sigma at runtime without touching config.py
    model = GRAddMMF(
        n_indicators=cfg.N_INDICATORS,
        hidden_dim=cfg.HIDDEN_DIM,
        n_targets=cfg.N_TARGETS,
        sigma=sigma,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.MSELoss()

    train_loader = DataLoader(MacroDataset(train_samples), batch_size=cfg.BATCH_SIZE,
                              shuffle=True, collate_fn=macro_collate)
    val_loader   = DataLoader(MacroDataset(val_samples),  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=macro_collate)

    best_val, patience_count = float("inf"), 0
    best_state = None
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimiser.zero_grad()
            loss = criterion(model(batch), batch["target"])
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                val_losses.append(criterion(model(batch), batch["target"]).item())

        train_mse = float(np.mean(train_losses))
        val_mse   = float(np.mean(val_losses))
        train_hist.append(round(train_mse, 6))
        val_hist.append(round(val_mse, 6))

        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.EARLY_STOPPING:
                print(f"    Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  train={train_mse:.4f}  val={val_mse:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    return model, best_val, train_hist, val_hist


def evaluate_gr_add(model, test_samples, device):
    """Return per-target MSE/MAE and overall MSE."""
    loader = DataLoader(MacroDataset(test_samples), batch_size=cfg.BATCH_SIZE,
                        shuffle=False, collate_fn=macro_collate)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            preds.append(model(batch).cpu().numpy())
            targets.append(batch["target"].cpu().numpy())

    preds   = np.vstack(preds)
    targets = np.vstack(targets)
    mse_per = ((preds - targets) ** 2).mean(axis=0)
    mae_per = np.abs(preds - targets).mean(axis=0)
    return {
        "mse_gdp":     float(mse_per[0]),
        "mse_inf":     float(mse_per[1]),
        "mae_gdp":     float(mae_per[0]),
        "mae_inf":     float(mae_per[1]),
        "mse_overall": float(((preds - targets) ** 2).mean()),
    }


def plot_sigma_comparison(summary: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        NAVY = "#1a1a2e"; WHITE = "#ffffff"; YELLOW = "#f1c40f"; GREEN = "#2ecc71"
        plt.rcParams.update({
            "figure.facecolor": NAVY, "axes.facecolor": NAVY,
            "axes.edgecolor": "#444466", "axes.labelcolor": WHITE,
            "xtick.color": WHITE, "ytick.color": WHITE, "text.color": WHITE,
            "grid.color": "#333355", "grid.linestyle": "--", "grid.alpha": 0.4,
        })

        sigmas  = [s for s in sorted(summary.keys())]
        overall = [summary[s]["mse_overall"] for s in sigmas]
        gdp     = [summary[s]["mse_gdp"]     for s in sigmas]
        inf_    = [summary[s]["mse_inf"]      for s in sigmas]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(NAVY)

        x = np.arange(len(sigmas))
        ax.plot(x, overall, "o-", color=YELLOW, linewidth=2.5, markersize=9,
                label="Overall MSE", zorder=3)
        ax.plot(x, gdp,     "s--", color=GREEN,  linewidth=1.8, markersize=7,
                label="GDP MSE",     alpha=0.8)
        ax.plot(x, inf_,    "^--", color="#e74c3c", linewidth=1.8, markersize=7,
                label="Inflation MSE", alpha=0.8)

        for xi, (s, v) in enumerate(zip(sigmas, overall)):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=YELLOW)

        ax.set_xticks(x)
        ax.set_xticklabels([f"σ = {s}" for s in sigmas], fontsize=11)
        ax.set_ylabel("Test MSE", fontsize=11)
        ax.set_title("GR-Add MMF: Sigma Sensitivity Analysis\n"
                     "RecAvg TTF Gaussian kernel width  ·  Test 2022–2023",
                     fontsize=12, fontweight="bold", color=WHITE, pad=10)
        ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#444466")
        ax.grid(axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        plt.tight_layout()
        out = EXP_DIR / "sigma_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close()
        print(f"  Plot saved: {out}")
    except Exception as e:
        print(f"  Plot skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmas",  nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data splits …")
    train_s, val_s, test_s, _, _ = build_all_splits()
    print(f"  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}\n")

    summary = {}

    for sigma in args.sigmas:
        tag = f"sigma_{sigma}".replace(".", "_")
        run_dir = EXP_DIR / tag
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*55}")
        print(f"  σ = {sigma}")
        print(f"{'='*55}")

        model, best_val, train_hist, val_hist = train_one(
            sigma, args.epochs, device, train_s, val_s
        )

        results = evaluate_gr_add(model, test_s, device)
        results["sigma"]        = sigma
        results["best_val_mse"] = round(best_val, 6)
        results["history"]      = {"train_mse": train_hist, "val_mse": val_hist}

        # save checkpoint and results
        torch.save(model.state_dict(), run_dir / "gr_add_best.pt")
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))

        summary[sigma] = results
        print(f"  Test MSE: overall={results['mse_overall']:.4f}  "
              f"gdp={results['mse_gdp']:.4f}  inf={results['mse_inf']:.4f}")

    # summary table
    print(f"\n{'='*55}")
    print(f"  SIGMA SENSITIVITY SUMMARY — GR-Add MMF test MSE")
    print(f"{'='*55}")
    print(f"  {'σ':>6}  {'GDP MSE':>9}  {'Inf MSE':>9}  {'Overall':>9}")
    print(f"  {'-'*45}")
    for sigma in sorted(summary.keys()):
        r = summary[sigma]
        marker = " ← default" if sigma == 1.0 else ""
        print(f"  {sigma:>6}  {r['mse_gdp']:>9.4f}  {r['mse_inf']:>9.4f}  {r['mse_overall']:>9.4f}{marker}")

    # save overall summary
    out = EXP_DIR / "summary.json"
    out.write_text(json.dumps(
        {str(s): v for s, v in summary.items()}, indent=2
    ))
    print(f"\nSummary → {out}")

    print("\nGenerating comparison plot …")
    plot_sigma_comparison(summary)


if __name__ == "__main__":
    main()
