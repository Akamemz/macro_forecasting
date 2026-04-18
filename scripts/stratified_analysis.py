"""
Secondary analysis: does text help more for countries with less Article IV coverage?

Compares GR-Add MMF vs NumericalGRU (isolates the text signal effect)
stratified by per-country text missingness rate on the test split (2022-2023).

Usage:
    python scripts/stratified_analysis.py
    python scripts/stratified_analysis.py --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    COUNTRIES, YEARS, TEST_QUERY_YEARS, TEXT_DIR,
    HIDDEN_DIM, N_TARGETS, BATCH_SIZE,
)
from src.data.prealign import build_all_splits
from src.data.dataset import MacroDataset, macro_collate, DataLoader
from src.train import build_model

CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"


# ── text coverage per country ─────────────────────────────────────────────────

def text_missingness_rate(country: str) -> float:
    """Fraction of years (across full dataset) with no Article IV text."""
    missing = sum(1 for y in YEARS if not (TEXT_DIR / f"{country}_{y}.txt").exists())
    return missing / len(YEARS)


# ── collect per-sample predictions ───────────────────────────────────────────

def get_preds(model_name: str, samples: list, device: torch.device):
    ckpt = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not ckpt.exists():
        print(f"  [skip] no checkpoint: {ckpt}")
        return None

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()

    loader = DataLoader(
        MacroDataset(samples), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=macro_collate,
    )

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            preds.append(model(batch).cpu().numpy())
            targets.append(batch["target"].cpu().numpy())

    # country / query_year come from the sample list directly (same order, no shuffle)
    countries = [s["country"]    for s in samples]
    years     = [s["query_year"] for s in samples]

    return {
        "preds":     np.vstack(preds),
        "targets":   np.vstack(targets),
        "countries": countries,
        "years":     years,
    }


# ── main analysis ─────────────────────────────────────────────────────────────

def run(device_str: str = "cpu"):
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test split …")
    _, _, test_samples, _, _ = build_all_splits()
    print(f"  {len(test_samples)} test samples\n")

    print("Running models …")
    gru = get_preds("numerical_gru", test_samples, device)
    mmf = get_preds("gr_add",        test_samples, device)
    if gru is None or mmf is None:
        print("Missing checkpoints — run training first.")
        return

    # ── per-country MSE gain ──────────────────────────────────────────────────
    countries_in_test = sorted(set(gru["countries"]))
    rows = []
    for c in countries_in_test:
        idx = [i for i, co in enumerate(gru["countries"]) if co == c]
        gru_mse = ((gru["preds"][idx] - gru["targets"][idx]) ** 2).mean()
        mmf_mse = ((mmf["preds"][idx] - mmf["targets"][idx]) ** 2).mean()
        gain    = gru_mse - mmf_mse          # positive = MMF better
        gain_pct = 100 * gain / (gru_mse + 1e-8)
        miss_rate = text_missingness_rate(c)
        n_texts = sum(1 for y in YEARS if (TEXT_DIR / f"{c}_{y}.txt").exists())
        rows.append({
            "country":    c,
            "text_miss_rate": round(miss_rate, 3),
            "n_texts":    n_texts,
            "gru_mse":    round(float(gru_mse), 4),
            "mmf_mse":    round(float(mmf_mse), 4),
            "gain":       round(float(gain), 4),
            "gain_pct":   round(float(gain_pct), 1),
        })

    rows.sort(key=lambda r: r["text_miss_rate"])

    # ── print per-country table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" PER-COUNTRY: GR-Add MMF vs NumericalGRU  (test set 2022-2023)")
    print(" Sorted by text missingness (low → high)")
    print("=" * 72)
    print(f"{'Country':<8} {'Texts':>6} {'Miss%':>7} {'GRU MSE':>10} {'MMF MSE':>10} {'Gain':>8} {'Gain%':>7}")
    print("-" * 72)
    for r in rows:
        marker = " ◀" if r["gain_pct"] > 5 else ""
        print(f"{r['country']:<8} {r['n_texts']:>6} {r['text_miss_rate']*100:>6.0f}%"
              f" {r['gru_mse']:>10.4f} {r['mmf_mse']:>10.4f}"
              f" {r['gain']:>+8.4f} {r['gain_pct']:>+6.1f}%{marker}")

    # ── stratified summary ────────────────────────────────────────────────────
    miss_rates = [r["text_miss_rate"] for r in rows]
    median_miss = np.median(miss_rates)

    print(f"\nMedian text missingness: {median_miss*100:.0f}%")
    print("\n--- Stratified summary ---")
    for label, subset in [
        (f"LOW  missingness (≤{median_miss*100:.0f}%)", [r for r in rows if r["text_miss_rate"] <= median_miss]),
        (f"HIGH missingness (>{median_miss*100:.0f}%)",  [r for r in rows if r["text_miss_rate"] >  median_miss]),
    ]:
        avg_gain     = np.mean([r["gain"]     for r in subset])
        avg_gain_pct = np.mean([r["gain_pct"] for r in subset])
        avg_miss     = np.mean([r["text_miss_rate"] for r in subset]) * 100
        print(f"  {label}: {len(subset)} countries, avg text miss={avg_miss:.0f}%, "
              f"avg MSE gain={avg_gain:+.4f} ({avg_gain_pct:+.1f}%)")

    # ── save results ──────────────────────────────────────────────────────────
    out = RESULTS_DIR / "stratified_analysis.json"
    out.write_text(json.dumps({
        "median_text_miss_rate": median_miss,
        "per_country": rows,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.device)
