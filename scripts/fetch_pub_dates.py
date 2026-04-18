"""
Fetch real publication dates from Coveo for all Article IV texts we have,
and update data/text/metadata.csv with actual dates instead of estimates.

No PDFs are downloaded — this only queries the Coveo search API.

Results saved to:
  experiments/04_real_pub_dates/metadata_real_dates.csv
  (also overwrites data/text/metadata.csv)

Usage:
    python scripts/fetch_pub_dates.py
    python scripts/fetch_pub_dates.py --token <new_token>
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, TEXT_DIR, YEARS
from scripts.fetch_article_iv import (
    COVEO_ENDPOINT, COVEO_ORG, COVEO_TOKEN,
    get_all_results, title_matches_country,
    COUNTRY_QUERY,
)
import re

def parse_year(text):
    m = re.search(r"\b(20\d{2}|199\d)\b", str(text))
    return int(m.group(1)) if m else None

EXP_DIR = ROOT / "experiments" / "04_real_pub_dates"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def parse_coveo_date(raw_date):
    """
    Coveo date field is a Unix timestamp in milliseconds.
    Returns ISO date string YYYY-MM-DD or None.
    """
    if not raw_date:
        return None
    try:
        ts_ms = int(raw_date)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # sometimes it's already a string like "2023-07-15"
        try:
            dt = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None


def fetch_dates_for_country(iso3: str, token: str) -> dict[int, str]:
    """
    Returns {ref_year: pub_date_str} for all matched Article IV results.
    """
    try:
        results = get_all_results(iso3, token)
    except Exception as e:
        print(f"  ERROR fetching {iso3}: {e}")
        return {}

    year_dates = {}
    for result in results:
        title    = result.get("title", "")
        raw      = result.get("raw", {})
        date_raw = raw.get("date")
        date_str = str(raw.get("date", ""))

        year = parse_year(title) or parse_year(date_str)
        if year is None or year not in YEARS:
            continue
        if not title_matches_country(title, iso3):
            continue
        if "article iv" not in title.lower() and "consultation" not in title.lower():
            continue
        if year in year_dates:
            continue

        pub_date = parse_coveo_date(date_raw)
        if pub_date:
            year_dates[year] = pub_date

    return year_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=COVEO_TOKEN)
    args = parser.parse_args()

    meta_path = TEXT_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"[ERROR] {meta_path} not found. Run prepare_real_data.py first.")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata.csv: {len(meta)} entries")
    print(f"Current pub_date sample: {meta['pub_date'].head(3).tolist()}\n")

    found, estimated = 0, 0
    new_dates = {}   # (country, ref_year) → real pub_date

    for iso3 in COUNTRIES:
        country_rows = meta[meta["country"] == iso3]
        if country_rows.empty:
            continue

        print(f"{iso3} ...", end=" ", flush=True)
        year_dates = fetch_dates_for_country(iso3, args.token)

        hits = 0
        for _, row in country_rows.iterrows():
            ref_year = int(row["ref_year"])
            if ref_year in year_dates:
                new_dates[(iso3, ref_year)] = year_dates[ref_year]
                hits += 1
            else:
                # keep existing estimate
                new_dates[(iso3, ref_year)] = row["pub_date"]

        print(f"{hits}/{len(country_rows)} real dates found")
        found     += hits
        estimated += len(country_rows) - hits
        time.sleep(0.5)

    # update metadata
    meta["pub_date_original"] = meta["pub_date"]   # keep estimate as backup
    meta["pub_date"] = meta.apply(
        lambda r: new_dates.get((r["country"], int(r["ref_year"])), r["pub_date"]),
        axis=1
    )

    # save to experiment dir and overwrite metadata
    out_exp = EXP_DIR / "metadata_real_dates.csv"
    meta.to_csv(out_exp, index=False)
    meta.to_csv(meta_path, index=False)

    print(f"\nDone.")
    print(f"  Real dates found:  {found}")
    print(f"  Still estimated:   {estimated}")
    print(f"  Saved → {out_exp}")
    print(f"  Updated → {meta_path}")
    print(f"\nNext steps:")
    print(f"  python -m src.pipeline.encode_text --mode openai --overwrite")
    print(f"  python run_pipeline.py --skip-gen --epochs 100")
    print(f"  (then copy results to experiments/04_real_pub_dates/)")


if __name__ == "__main__":
    main()
