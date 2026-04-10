"""
Generate result visualizations:
  1. Stratified analysis bar chart (gain% per country, colored by missingness)
  2. Learning curves (train vs val MSE for all three models)
  3. Bootstrap confidence intervals on test MSE

Saved to research/figures/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "research" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY   = "#1a1a2e"
ACCENT = "#2d4a7a"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
YELLOW = "#f1c40f"
GREY   = "#95a5a6"
WHITE  = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor":   NAVY,
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  WHITE,
    "xtick.color":      WHITE,
    "ytick.color":      WHITE,
    "text.color":       WHITE,
    "grid.color":       "#333355",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
})


# ── 1. Stratified bar chart ───────────────────────────────────────────────────

def plot_stratified():
    path = ROOT / "results" / "stratified_analysis.json"
    data = json.loads(path.read_text())
    rows = data["per_country"]
    median = data["median_text_miss_rate"]

    # sort by missingness
    rows = sorted(rows, key=lambda r: r["text_miss_rate"])

    countries  = [r["country"]      for r in rows]
    gain_pcts  = [r["gain_pct"]     for r in rows]
    miss_rates = [r["text_miss_rate"] for r in rows]

    colors = [GREEN if g > 0 else RED for g in gain_pcts]
    alphas = [0.65 + 0.35 * m for m in miss_rates]   # gappier = more opaque

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(NAVY)

    bars = ax.bar(countries, gain_pcts, color=colors, alpha=0.85,
                  edgecolor="#ffffff22", linewidth=0.5)

    # color bars by actual alpha
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(min(alpha, 1.0))

    ax.axhline(0, color=WHITE, linewidth=0.8, alpha=0.6)

    # median missingness divider
    median_idx = sum(1 for r in rows if r["text_miss_rate"] <= median) - 0.5
    ax.axvline(median_idx, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.text(median_idx - 0.3, ax.get_ylim()[1] * 0.88,
            f"median\nmiss={median*100:.0f}%",
            color=YELLOW, fontsize=8, ha="right", va="top")

    # labels on bars
    for bar, val in zip(bars, gain_pcts):
        y = val + (2 if val >= 0 else -4)
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"{val:+.0f}%", ha="center", va="bottom",
                fontsize=7.5, color=WHITE, fontweight="bold")

    ax.set_xlabel("Country  (sorted by text missingness, low → high)", fontsize=11)
    ax.set_ylabel("MSE Gain%  (GR-Add vs NumericalGRU)\npositive = text helps", fontsize=10)
    ax.set_title("Stratified Analysis: Text Benefit by Country\n"
                 "OpenAI text-embedding-3-small  ·  Test set 2022–2023",
                 fontsize=13, fontweight="bold", color=WHITE, pad=12)
    ax.grid(axis="y")
    ax.set_xlim(-0.6, len(countries) - 0.4)

    low_patch  = mpatches.Patch(color=GREEN, alpha=0.85, label="GR-Add better")
    high_patch = mpatches.Patch(color=RED,   alpha=0.85, label="NumericalGRU better")
    ax.legend(handles=[low_patch, high_patch], loc="upper left",
              facecolor="#16213e", edgecolor="#444466", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "stratified_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved: {out}")


# ── 2. Learning curves ────────────────────────────────────────────────────────

def plot_learning_curves():
    model_files = {
        "DLinear":       ROOT / "results" / "dlinear_results.json",
        "NumericalGRU":  ROOT / "results" / "numerical_gru_results.json",
        "GR-Add MMF":    ROOT / "results" / "gr_add_results.json",
    }
    colors_map = {
        "DLinear":      GREY,
        "NumericalGRU": YELLOW,
        "GR-Add MMF":   GREEN,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.patch.set_facecolor(NAVY)
    fig.suptitle("Training & Validation Learning Curves\nOpenAI text-embedding-3-small",
                 fontsize=13, fontweight="bold", color=WHITE, y=1.02)

    for ax, (name, path) in zip(axes, model_files.items()):
        if not path.exists():
            ax.set_title(f"{name}\n(no data)", color=WHITE)
            continue
        data = json.loads(path.read_text())
        hist = data.get("history", {})
        train = hist.get("train_mse", [])
        val   = hist.get("val_mse",   [])
        col   = colors_map[name]

        epochs = list(range(1, len(train) + 1))
        ax.plot(epochs, train, color=col,   linewidth=2,   label="Train MSE", alpha=0.9)
        ax.plot(epochs, val,   color=WHITE, linewidth=1.5, label="Val MSE",
                linestyle="--", alpha=0.7)

        best_val = min(val)
        best_ep  = val.index(best_val) + 1
        ax.axvline(best_ep, color=col, linewidth=1, linestyle=":", alpha=0.6)
        ax.scatter([best_ep], [best_val], color=col, s=60, zorder=5)
        ax.text(best_ep + 0.3, best_val, f" best={best_val:.3f}\n ep={best_ep}",
                color=col, fontsize=7.5, va="center")

        ax.set_title(f"{name}", fontsize=11, color=col, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.legend(fontsize=8, facecolor="#16213e", edgecolor="#444466")
        ax.grid(True)
        ax.set_facecolor(NAVY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    plt.tight_layout()
    out = FIG_DIR / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved: {out}")


# ── 3. Bootstrap confidence intervals ─────────────────────────────────────────

def bootstrap_ci(preds, targets, n_boot=2000, ci=95):
    """Return (mean_mse, lower, upper) via percentile bootstrap."""
    rng = np.random.default_rng(42)
    n = len(preds)
    boot_mses = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mse = ((preds[idx] - targets[idx]) ** 2).mean()
        boot_mses.append(mse)
    lo = np.percentile(boot_mses, (100 - ci) / 2)
    hi = np.percentile(boot_mses, 100 - (100 - ci) / 2)
    return float(((preds - targets) ** 2).mean()), lo, hi


def plot_bootstrap():
    import torch
    from src.config import BATCH_SIZE
    from src.data.prealign import build_all_splits
    from src.data.dataset import MacroDataset, macro_collate, DataLoader
    from src.train import build_model

    CKPT = ROOT / "checkpoints"

    _, _, test_samples, _, _ = build_all_splits()

    model_names  = ["dlinear", "numerical_gru", "gr_add"]
    display_names = ["DLinear", "NumericalGRU", "GR-Add MMF"]
    colors_list  = [GREY, YELLOW, GREEN]

    all_preds, all_targets = {}, {}
    for name in model_names:
        ckpt = CKPT / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        model = build_model(name)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        model.eval()
        loader = DataLoader(MacroDataset(test_samples), batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=macro_collate)
        preds, targets = [], []
        try:
            with torch.no_grad():
                for batch in loader:
                    preds.append(model(batch).numpy())
                    targets.append(batch["target"].numpy())
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        all_preds[name]   = np.vstack(preds)
        all_targets[name] = np.vstack(targets)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)

    x = np.arange(len(model_names))
    width = 0.55

    for i, (name, dname, col) in enumerate(zip(model_names, display_names, colors_list)):
        if name not in all_preds:
            continue
        mean, lo, hi = bootstrap_ci(all_preds[name], all_targets[name])
        bar = ax.bar(i, mean, width, color=col, alpha=0.85,
                     edgecolor="#ffffff22", linewidth=0.5)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt="none", color=WHITE, linewidth=2, capsize=8, capthick=2)
        ax.text(i, hi + 0.03, f"{mean:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", va="bottom", fontsize=8.5, color=WHITE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Overall MSE", fontsize=11)
    ax.set_title("Test Set MSE with 95% Bootstrap Confidence Intervals\n"
                 "OpenAI text-embedding-3-small  ·  Test 2022–2023  ·  n=2000 bootstrap samples",
                 fontsize=12, fontweight="bold", color=WHITE, pad=10)
    ax.grid(axis="y")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    plt.tight_layout()
    out = FIG_DIR / "bootstrap_ci.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved: {out}")


# ── 4. Benchmark comparison bar chart ─────────────────────────────────────────

def plot_benchmark():
    """
    Bar chart comparing persistence baseline vs our three models on test MSE.
    """
    bench_path = ROOT / "experiments" / "05_weo_comparison" / "benchmark_comparison.json"
    res_path   = ROOT / "results" / "eval_test.json"
    if not bench_path.exists() or not res_path.exists():
        print("  [skip] benchmark_comparison.json or eval_test.json not found")
        return

    bench = json.loads(bench_path.read_text())
    res   = json.loads(res_path.read_text())

    persist_overall = bench["persistence"]["mse_overall"]
    persist_gdp     = bench["persistence"]["mse_gdp"]
    persist_inf     = bench["persistence"]["mse_inf"]

    models = [
        ("Persistence\n(random walk)", persist_gdp,     persist_inf,     persist_overall, RED),
        ("DLinear",                    res["dlinear"]["mse_gdp"],    res["dlinear"]["mse_inf"],    res["dlinear"]["mse_overall"],    GREY),
        ("NumericalGRU",               res["numerical_gru"]["mse_gdp"], res["numerical_gru"]["mse_inf"], res["numerical_gru"]["mse_overall"], YELLOW),
        ("GR-Add MMF\n(ours)",         res["gr_add"]["mse_gdp"],     res["gr_add"]["mse_inf"],     res["gr_add"]["mse_overall"],     GREEN),
    ]

    labels    = [m[0] for m in models]
    gdp_vals  = [m[1] for m in models]
    inf_vals  = [m[2] for m in models]
    total_vals = [m[3] for m in models]
    colors    = [m[4] for m in models]

    x = np.arange(len(models))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)

    bars1 = ax.bar(x - width, gdp_vals,  width, label="GDP MSE",       color=[c + "bb" for c in colors])
    bars2 = ax.bar(x,          inf_vals,  width, label="Inflation MSE", color=colors,   alpha=0.75)
    bars3 = ax.bar(x + width,  total_vals, width, label="Overall MSE",  color=colors,   alpha=0.5,
                   edgecolor=WHITE, linewidth=0.8)

    # annotate overall MSE on top of each group
    for i, (val, col) in enumerate(zip(total_vals, colors)):
        ax.text(x[i] + width, val + 0.5, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, color=WHITE, fontweight="bold")

    # reduction vs persistence annotation for GR-Add
    reduction = (1 - total_vals[-1] / total_vals[0]) * 100
    ax.annotate(
        f"−{reduction:.0f}% vs\npersistence",
        xy=(x[-1] + width, total_vals[-1]),
        xytext=(x[-1] + width + 0.3, total_vals[-1] + 8),
        fontsize=9, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test MSE", fontsize=11)
    ax.set_title(
        "Benchmark Comparison — Test MSE (2022–2023)\n"
        "GR-Add MMF vs persistence baseline and ablation models",
        fontsize=12, fontweight="bold", color=WHITE, pad=10,
    )
    ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#444466", loc="upper right")
    ax.grid(axis="y")
    ax.set_ylim(0, max(total_vals) * 1.25)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    plt.tight_layout()
    out = FIG_DIR / "benchmark_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating visualizations …\n")
    plot_stratified()
    plot_learning_curves()
    print("\nRunning bootstrap CI (loads models + data) …")
    plot_bootstrap()
    print("\nGenerating benchmark comparison …")
    plot_benchmark()
    print("\nAll figures saved to research/figures/")
