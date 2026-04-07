"""
Pre-alignment: convert per-country CSVs into fixed-length tensors following
Appendix H of TIME-IMM.

Each (country, query_year) sample has:
  values      : (T, D) float32  — observed indicator values (0.0 where missing)
  mask        : (T, D) float32  — 1.0 observed, 0.0 missing
  timestamps  : (T,)   float32  — normalised context years in [0, 1]
  query_ts    : scalar float32  — normalised query year
  text_embs   : (N, 768) float32 — embeddings of available Article IV reports
  text_ts     : (N,)    float32  — normalised publication dates
  target      : (2,)    float32  — [gdp_growth, inflation] for query_year

Normalisation of indicators: each column is z-scored using training-set stats
(mean/std computed only over TRAIN_QUERY_YEARS context windows to avoid leakage).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    COUNTRIES, INDICATORS, YEARS, CONTEXT_WINDOW,
    TRAIN_QUERY_YEARS, VAL_QUERY_YEARS, TEST_QUERY_YEARS,
    TARGET_COLS, TEXT_EMB_DIM,
    MACRO_DIR, TEXT_DIR, EMB_DIR,
    norm_year,
)


# ── load raw macro panel ──────────────────────────────────────────────────────

def load_macro_panel() -> dict[str, pd.DataFrame]:
    """Returns {country: DataFrame with columns [year] + INDICATORS}."""
    panel = {}
    for country in COUNTRIES:
        csv = MACRO_DIR / f"{country}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["year"] = df["year"].astype(int)
            df = df.set_index("year")
            panel[country] = df
    return panel


def load_text_metadata() -> pd.DataFrame:
    """Returns metadata DataFrame with pub_date parsed."""
    meta_path = TEXT_DIR / "metadata.csv"
    meta = pd.read_csv(meta_path)
    meta["pub_date"] = pd.to_datetime(meta["pub_date"])
    meta["pub_year_frac"] = meta["pub_date"].dt.year + (meta["pub_date"].dt.dayofyear - 1) / 365.0
    return meta


# ── compute normalisation stats ───────────────────────────────────────────────

def compute_norm_stats(panel: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-indicator mean and std using only training-split context windows
    (years that appear in context of TRAIN_QUERY_YEARS).
    Returns (mean, std) each of shape (D,).
    """
    context_years = set()
    for qy in TRAIN_QUERY_YEARS:
        for y in range(qy - CONTEXT_WINDOW, qy):
            context_years.add(y)

    all_vals = {col: [] for col in INDICATORS}
    for df in panel.values():
        sub = df[df.index.isin(context_years)]
        for col in INDICATORS:
            if col in sub.columns:
                vals = sub[col].dropna().values
                all_vals[col].extend(vals.tolist())

    means = np.array([np.mean(all_vals[c]) if all_vals[c] else 0.0 for c in INDICATORS], dtype=np.float32)
    stds  = np.array([np.std(all_vals[c])  if all_vals[c] else 1.0 for c in INDICATORS], dtype=np.float32)
    stds  = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


# ── build one sample ──────────────────────────────────────────────────────────

def build_sample(
    country: str,
    query_year: int,
    panel: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
) -> dict | None:
    """
    Build the tensor dict for one (country, query_year) pair.
    Returns None if the target is not available.
    """
    df = panel.get(country)
    if df is None:
        return None

    context_start = query_year - CONTEXT_WINDOW
    context_years = list(range(context_start, query_year))  # T years

    # ── numerical tensors ────────────────────────────────────────────────────
    D = len(INDICATORS)
    T = CONTEXT_WINDOW
    values = np.zeros((T, D), dtype=np.float32)
    mask   = np.zeros((T, D), dtype=np.float32)

    for t, yr in enumerate(context_years):
        if yr in df.index:
            row = df.loc[yr]
            for d, col in enumerate(INDICATORS):
                val = row[col]
                if not pd.isna(val):
                    values[t, d] = (val - means[d]) / stds[d]
                    mask[t, d]   = 1.0
                # else: leave as 0.0 with mask=0

    timestamps = np.array([norm_year(y) for y in context_years], dtype=np.float32)
    query_ts   = np.float32(norm_year(query_year))

    # ── target ───────────────────────────────────────────────────────────────
    if query_year not in df.index:
        return None
    tgt_row = df.loc[query_year]
    target  = np.array(
        [(tgt_row[INDICATORS[c]] - means[c]) / stds[c] for c in TARGET_COLS],
        dtype=np.float32,
    )
    if np.any(np.isnan(target)):
        return None

    # ── text embeddings (all reports published before Jan 1 of query_year) ───
    cutoff = float(query_year)  # pub_year_frac < query_year → published before it
    country_meta = meta[
        (meta["country"] == country) & (meta["pub_year_frac"] < cutoff)
    ].sort_values("pub_year_frac")

    text_embs = []
    text_ts   = []
    for _, row in country_meta.iterrows():
        emb_file = EMB_DIR / f"{country}_{int(row['ref_year'])}.npy"
        if emb_file.exists():
            text_embs.append(np.load(emb_file))
            text_ts.append(norm_year(row["pub_year_frac"]))

    if text_embs:
        text_embs_arr = np.stack(text_embs, axis=0).astype(np.float32)  # (N, 768)
        text_ts_arr   = np.array(text_ts, dtype=np.float32)              # (N,)
    else:
        text_embs_arr = np.zeros((1, TEXT_EMB_DIM), dtype=np.float32)
        text_ts_arr   = np.array([0.0], dtype=np.float32)

    return {
        "country":    country,
        "query_year": query_year,
        "values":     values,          # (T, D)
        "mask":       mask,            # (T, D)
        "timestamps": timestamps,      # (T,)
        "query_ts":   query_ts,        # scalar
        "text_embs":  text_embs_arr,   # (N, 768)
        "text_ts":    text_ts_arr,     # (N,)
        "target":     target,          # (2,)
    }


# ── build full split ──────────────────────────────────────────────────────────

def build_split(
    split: str,
    panel: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
) -> list[dict]:
    query_years = {
        "train": TRAIN_QUERY_YEARS,
        "val":   VAL_QUERY_YEARS,
        "test":  TEST_QUERY_YEARS,
    }[split]

    samples = []
    for country in COUNTRIES:
        for qy in query_years:
            s = build_sample(country, qy, panel, meta, means, stds)
            if s is not None:
                samples.append(s)

    print(f"  [{split}] {len(samples)} samples "
          f"({len(COUNTRIES)} countries × {len(query_years)} query years)")
    return samples


def build_all_splits() -> tuple[list, list, list, np.ndarray, np.ndarray]:
    """
    Main entry point. Returns (train_samples, val_samples, test_samples, means, stds).
    """
    print("Loading macro panel …")
    panel = load_macro_panel()
    print(f"  {len(panel)} countries loaded")

    print("Loading text metadata …")
    meta = load_text_metadata()

    print("Computing normalisation stats from training split …")
    means, stds = compute_norm_stats(panel)

    train = build_split("train", panel, meta, means, stds)
    val   = build_split("val",   panel, meta, means, stds)
    test  = build_split("test",  panel, meta, means, stds)

    return train, val, test, means, stds


if __name__ == "__main__":
    train, val, test, _, _ = build_all_splits()
    s = train[0]
    print("\nSample keys:", list(s.keys()))
    print("values shape:", s["values"].shape)
    print("mask shape:  ", s["mask"].shape)
    print("text_embs:   ", s["text_embs"].shape)
    print("target:      ", s["target"])
