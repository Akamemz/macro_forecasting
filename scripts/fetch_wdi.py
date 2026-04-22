"""
Pull World Bank WDI data for missing countries and append to wdi_data.csv.

Usage:
    python scripts/fetch_wdi.py                        # auto-detects missing countries
    python scripts/fetch_wdi.py --countries USA JPN MNG
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, INDICATORS, MACRO_DIR, YEARS

# World Bank indicator codes matching src/config.py INDICATORS order
WB_CODES = {
    "gdp_growth":      "NY.GDP.MKTP.KD.ZG",
    "inflation":       "FP.CPI.TOTL.ZG",
    "fiscal_balance":  "GC.BAL.CASH.GD.ZS",
    "current_account": "BN.CAB.XOKA.GD.ZS",
    "remittances":     "BX.TRF.PWKR.DT.GD.ZS",
    "unemployment":    "SL.UEM.TOTL.ZS",
    "govt_debt":       "GC.DOD.TOTL.GD.ZS",
}

WB_API = "https://api.worldbank.org/v2/country/{iso}/indicator/{ind}"


def fetch_series(iso3: str, indicator: str, year_from: int, year_to: int,
                 retries: int = 3) -> dict[int, float]:
    url = WB_API.format(iso=iso3, ind=indicator)
    params = {"date": f"{year_from}:{year_to}", "format": "json", "per_page": 100}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if len(data) < 2 or not data[1]:
                return {}
            return {
                int(entry["date"]): entry["value"]
                for entry in data[1]
                if entry.get("value") is not None and entry.get("date")
            }
        except Exception as e:
            if attempt < retries:
                wait = 5 * attempt
                print(f"    attempt {attempt} failed ({e}), retrying in {wait}s …", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"    WARNING: {iso3} {indicator}: {e}")
                return {}
    return {}


def fetch_country(iso3: str) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        row = {"country_code": iso3, "year": year}
        for ind_name, wb_code in WB_CODES.items():
            row[ind_name] = None
        rows.append(row)

    year_index = {y: i for i, y in enumerate(YEARS)}

    for ind_name, wb_code in WB_CODES.items():
        print(f"  {ind_name} ...", end=" ", flush=True)
        series = fetch_series(iso3, wb_code, min(YEARS), max(YEARS))
        for year, value in series.items():
            if year in year_index:
                rows[year_index[year]][ind_name] = value
        print(f"{len(series)} values")
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    # add country name column (blank for now — will be filled from existing data pattern)
    df.insert(2, "country", iso3)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=None)
    args = parser.parse_args()

    csv_path = MACRO_DIR / "wdi_data.csv"
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    already_have = set(existing["country_code"].unique()) if not existing.empty else set()

    if args.countries:
        targets = args.countries
    else:
        targets = [c for c in COUNTRIES if c not in already_have]

    if not targets:
        print("No missing countries — nothing to fetch.")
        return

    print(f"Fetching WDI for: {targets}")

    new_frames = []
    for iso3 in targets:
        print(f"\n{iso3}")
        df = fetch_country(iso3)
        new_frames.append(df)
        time.sleep(0.5)

    if not new_frames:
        return

    # drop existing rows for countries we just re-fetched (avoid duplicates)
    existing = existing[~existing["country_code"].isin(targets)]
    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined = combined.sort_values(["country_code", "year"]).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
