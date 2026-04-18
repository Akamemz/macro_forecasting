"""
One-time setup: prepare real data for the pipeline.

Does two things:
  1. Splits data/macro/wdi_data.csv → data/macro/{COUNTRY}.csv (one per country)
  2. Scans data/text/*.txt and writes data/text/metadata.csv

Publication dates are estimated as July 1 of the year AFTER the reference year,
which approximates the typical 3–8 month IMF publication lag.

Usage:
    python scripts/prepare_real_data.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, INDICATORS, MACRO_DIR, TEXT_DIR


# ── Step 1: split wdi_data.csv into per-country CSVs ─────────────────────────

def split_wdi():
    csv_path = MACRO_DIR / "wdi_data.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    written = 0
    for country in COUNTRIES:
        sub = df[df["country_code"] == country][["year"] + INDICATORS].copy()
        if sub.empty:
            print(f"  WARNING: no WDI rows for {country}")
            continue
        sub = sub.sort_values("year").reset_index(drop=True)
        out = MACRO_DIR / f"{country}.csv"
        sub.to_csv(out, index=False)
        written += 1

    print(f"[1/2] Split wdi_data.csv → {written} per-country CSVs in {MACRO_DIR}")


# ── Step 2: build metadata.csv from data/text/*.txt ──────────────────────────

def build_metadata():
    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    rows = []
    for f in txt_files:
        stem = f.stem          # e.g. "KAZ_2022"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        country, year_str = parts
        if country not in COUNTRIES:
            continue
        try:
            ref_year = int(year_str)
        except ValueError:
            continue

        # Approximate pub_date: July 1 of the year after the reference year
        pub_date = f"{ref_year + 1}-07-01"
        summary_path = f"text/{f.name}"   # relative to data/

        rows.append({
            "country":      country,
            "ref_year":     ref_year,
            "pub_date":     pub_date,
            "summary_path": summary_path,
        })

    meta = pd.DataFrame(rows).sort_values(["country", "ref_year"]).reset_index(drop=True)
    out = TEXT_DIR / "metadata.csv"
    meta.to_csv(out, index=False)
    print(f"[2/2] Written metadata.csv with {len(meta)} entries → {out}")
    return meta


if __name__ == "__main__":
    split_wdi()
    build_metadata()
    print("\nDone. Next steps:")
    print("  python -m src.pipeline.encode_text --mode bert")
    print("  python run_pipeline.py --emb bert --skip-gen")
