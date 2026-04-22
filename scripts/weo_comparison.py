"""
WEO Benchmark Comparison.

Compares our model results against two benchmarks on the test set (2022-2023):

1. NAIVE PERSISTENCE: uses the last observed value before the forecast year.
   - 2022 forecast → 2021 observed value
   - 2023 forecast → 2022 observed value
   This is a standard forecasting baseline (random-walk benchmark).

2. WEO (DataMapper, current vintage): IMF World Economic Outlook values for
   2022 and 2023, fetched from the IMF DataMapper API. These reflect current
   WEO estimates (which for 2022/2023 are now revised/actual values) and are
   used here to quantify how IMF's published numbers compare against WDI actuals.
   NOTE: Because vintage WEO forecasts (Oct 2021 / Oct 2022) are not accessible
   via public API, we use the DataMapper's latest data as an approximate benchmark.

WEO DataMapper indicators:
  NGDP_RPCH  — Real GDP growth (% change)
  PCPIPCH    — CPI inflation (% change)

Results saved to:
  experiments/05_weo_comparison/

Usage:
    python scripts/weo_comparison.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, MACRO_DIR

EXP_DIR = ROOT / "experiments" / "05_weo_comparison"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# IMF DataMapper API (public, no auth required)
# Note: no ?periods= param — Akamai WAF blocks parameterised requests; filter locally
DATAMAPPER_BASE = "https://www.imf.org/external/datamapper/api/v1/{indicator}/{country}"

WEO_INDICATORS = {
    "gdp_growth": "NGDP_RPCH",   # Real GDP growth (% change)
    "inflation":  "PCPIPCH",     # CPI inflation (% change)
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ── fetch WEO via DataMapper API ──────────────────────────────────────────────

def fetch_datamapper(var_name, indicator_code, years, retries=3):
    """
    Fetch indicator for all 22 countries via IMF DataMapper API.
    Fetches country by country to avoid URL-length 403s.
    Returns {iso3: {year_str: value}} nested dict.
    """
    cache = EXP_DIR / f"datamapper_{indicator_code}.json"
    if cache.exists():
        print(f"  Using cached: {cache.name}")
        return json.loads(cache.read_text())

    print(f"  Fetching {indicator_code} via DataMapper (per country) …")
    result = {}
    yr_set = set(str(y) for y in years)

    for iso3 in COUNTRIES:
        url = DATAMAPPER_BASE.format(indicator=indicator_code, country=iso3)
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # DataMapper format: {"values": {"NGDP_RPCH": {"USA": {"2022": 2.5, ...}}}}
                inner = data.get("values", {}).get(indicator_code, {})
                if iso3 in inner:
                    result[iso3] = {
                        yr: float(v)
                        for yr, v in inner[iso3].items()
                        if yr in yr_set
                    }
                break
            except Exception as e:
                if attempt < retries:
                    time.sleep(3 * attempt)
                else:
                    print(f"    {iso3}: failed ({e})")
        time.sleep(0.3)

    cache.write_text(json.dumps(result, indent=2))
    print(f"  Got data for {len(result)}/{len(COUNTRIES)} countries")
    return result


# ── load actual values from WDI ───────────────────────────────────────────────

def load_actuals():
    """Load actual GDP growth and inflation from WDI CSVs (ground truth)."""
    actuals = {}
    for iso3 in COUNTRIES:
        csv = MACRO_DIR / f"{iso3}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["year"] = df["year"].astype(int)
        df = df.set_index("year")
        for year in range(2020, 2024):   # need 2021, 2022, 2023
            if year not in df.index:
                continue
            row = df.loc[year]
            gdp = row.get("gdp_growth")
            inf = row.get("inflation")
            actuals[(iso3, year)] = {
                "gdp_growth": float(gdp) if pd.notna(gdp) else None,
                "inflation":  float(inf) if pd.notna(inf) else None,
            }
    return actuals


# ── compute MSE for a baseline ────────────────────────────────────────────────

def compute_mse(forecasts_by_year, actuals, test_years=(2022, 2023)):
    """
    forecasts_by_year: {year: {iso3: {gdp_growth, inflation}}}
    actuals:           {(iso3, year): {gdp_growth, inflation}}
    Returns per-target and overall MSE.
    """
    gdp_errors, inf_errors = [], []
    per_country = {}

    for year in test_years:
        for iso3 in COUNTRIES:
            actual = actuals.get((iso3, year))
            if actual is None:
                continue
            if actual["gdp_growth"] is None or actual["inflation"] is None:
                continue

            fcast = forecasts_by_year.get(year, {}).get(iso3, {})
            gdp_f = fcast.get("gdp_growth")
            inf_f = fcast.get("inflation")
            if gdp_f is None or inf_f is None:
                continue

            gdp_se = (gdp_f - actual["gdp_growth"]) ** 2
            inf_se = (inf_f - actual["inflation"])  ** 2
            gdp_errors.append(gdp_se)
            inf_errors.append(inf_se)

            if iso3 not in per_country:
                per_country[iso3] = {"gdp_se": [], "inf_se": []}
            per_country[iso3]["gdp_se"].append(gdp_se)
            per_country[iso3]["inf_se"].append(inf_se)

    if not gdp_errors:
        return None

    return {
        "mse_gdp":     round(float(np.mean(gdp_errors)),     4),
        "mse_inf":     round(float(np.mean(inf_errors)),      4),
        "mse_overall": round(float(np.mean(gdp_errors + inf_errors)), 4),
        "n_pairs":     len(gdp_errors),
        "per_country": {
            iso3: {
                "mse_gdp": round(float(np.mean(v["gdp_se"])), 4),
                "mse_inf": round(float(np.mean(v["inf_se"])), 4),
            }
            for iso3, v in per_country.items()
        }
    }


# ── build baselines ───────────────────────────────────────────────────────────

def build_persistence_forecasts(actuals):
    """
    Persistence baseline: forecast for year t = observed value in year t-1.
    Returns {year: {iso3: {gdp_growth, inflation}}}.
    """
    forecasts = {}
    for year in [2022, 2023]:
        forecasts[year] = {}
        for iso3 in COUNTRIES:
            prev = actuals.get((iso3, year - 1))
            if prev and prev["gdp_growth"] is not None and prev["inflation"] is not None:
                forecasts[year][iso3] = {
                    "gdp_growth": prev["gdp_growth"],
                    "inflation":  prev["inflation"],
                }
    return forecasts


def build_weo_forecasts(gdp_data, inf_data):
    """
    Build {year: {iso3: {gdp_growth, inflation}}} from DataMapper data.
    Uses the DataMapper value for each year directly (current vintage).
    """
    forecasts = {}
    for year in [2022, 2023]:
        yr = str(year)
        forecasts[year] = {}
        for iso3 in COUNTRIES:
            gdp_val = gdp_data.get(iso3, {}).get(yr)
            inf_val = inf_data.get(iso3, {}).get(yr)
            if gdp_val is not None and inf_val is not None:
                forecasts[year][iso3] = {
                    "gdp_growth": float(gdp_val),
                    "inflation":  float(inf_val),
                }
    return forecasts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Benchmark Comparison — Test MSE (2022-2023)")
    print("  Baselines: Persistence + WEO (DataMapper)")
    print("=" * 60)

    # load actual values (ground truth from WDI)
    print("\nLoading actual values from WDI …")
    actuals = load_actuals()
    test_actuals = {k: v for k, v in actuals.items()
                    if k[1] in [2022, 2023]
                    and v["gdp_growth"] is not None and v["inflation"] is not None}
    print(f"  {len(test_actuals)} (country, year) pairs with complete targets")

    # ── persistence baseline ──────────────────────────────────────────────────
    print("\nBuilding persistence baseline …")
    persist_forecasts = build_persistence_forecasts(actuals)
    persist_found = sum(
        1 for y in [2022, 2023] for iso3 in COUNTRIES
        if iso3 in persist_forecasts.get(y, {})
    )
    print(f"  Persistence forecasts built for {persist_found} (country, year) pairs")
    persist_results = compute_mse(persist_forecasts, actuals)

    # ── WEO DataMapper ────────────────────────────────────────────────────────
    print("\nFetching WEO data from IMF DataMapper API …")
    gdp_data = fetch_datamapper("gdp_growth", WEO_INDICATORS["gdp_growth"],
                                years=[2022, 2023])
    time.sleep(1)
    inf_data = fetch_datamapper("inflation",  WEO_INDICATORS["inflation"],
                                years=[2022, 2023])

    weo_forecasts = build_weo_forecasts(gdp_data, inf_data)
    for year in [2022, 2023]:
        found = len(weo_forecasts.get(year, {}))
        print(f"  WEO {year}: data for {found}/{len(COUNTRIES)} countries")

    weo_results = compute_mse(weo_forecasts, actuals)

    # ── load our model results ────────────────────────────────────────────────
    our_results_path = ROOT / "results" / "eval_test.json"
    our_results = json.loads(our_results_path.read_text()) if our_results_path.exists() else {}

    # ── print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BENCHMARK COMPARISON — Test MSE (2022-2023)")
    print("=" * 65)
    print(f"  {'Model':<32} {'GDP MSE':>9}  {'Inf MSE':>9}  {'Overall':>9}")
    print(f"  {'-'*60}")

    if persist_results:
        print(f"  {'Persistence (last obs)':<32} {persist_results['mse_gdp']:>9.4f}  "
              f"{persist_results['mse_inf']:>9.4f}  {persist_results['mse_overall']:>9.4f}  "
              f"(n={persist_results['n_pairs']})")
    else:
        print(f"  {'Persistence (last obs)':<32} {'N/A':>9}")

    if weo_results:
        print(f"  {'WEO DataMapper (current vintage)':<32} {weo_results['mse_gdp']:>9.4f}  "
              f"{weo_results['mse_inf']:>9.4f}  {weo_results['mse_overall']:>9.4f}  "
              f"(n={weo_results['n_pairs']})")
    else:
        print(f"  {'WEO DataMapper (current vintage)':<32} {'N/A':>9}")

    print(f"  {'-'*60}")
    for name, label in [
        ("dlinear",       "DLinear"),
        ("numerical_gru", "NumericalGRU"),
        ("gr_add",        "GR-Add MMF"),
    ]:
        if name in our_results:
            r = our_results[name]
            print(f"  {label:<32} {r['mse_gdp']:>9.4f}  "
                  f"{r['mse_inf']:>9.4f}  {r['mse_overall']:>9.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    output = {
        "note": (
            "WEO uses IMF DataMapper current vintage (revised/actual for 2022-2023). "
            "Persistence uses last observed WDI value as forecast. "
            "Our models trained on WDI data up to 2021."
        ),
        "persistence": persist_results,
        "weo_datamapper": weo_results,
        "our_models": our_results,
        "weo_forecasts_by_year": {
            str(yr): fcast for yr, fcast in weo_forecasts.items()
        },
        "persistence_forecasts_by_year": {
            str(yr): fcast for yr, fcast in persist_forecasts.items()
        },
    }
    out_path = EXP_DIR / "benchmark_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")

    # ── per-country table ─────────────────────────────────────────────────────
    if weo_results and persist_results:
        print(f"\n{'='*65}")
        print("  PER-COUNTRY MSE — Test 2022-2023 average")
        print(f"{'='*65}")
        print(f"  {'Country':<8}  {'Persist GDP':>12}  {'Persist Inf':>12}  "
              f"{'WEO GDP':>9}  {'WEO Inf':>9}")
        print(f"  {'-'*58}")
        all_countries = sorted(set(
            list(persist_results["per_country"].keys()) +
            list(weo_results["per_country"].keys())
        ))
        for iso3 in all_countries:
            p = persist_results["per_country"].get(iso3, {})
            w = weo_results["per_country"].get(iso3, {})
            print(f"  {iso3:<8}  "
                  f"{p.get('mse_gdp', float('nan')):>12.4f}  "
                  f"{p.get('mse_inf', float('nan')):>12.4f}  "
                  f"{w.get('mse_gdp', float('nan')):>9.4f}  "
                  f"{w.get('mse_inf', float('nan')):>9.4f}")


if __name__ == "__main__":
    main()
