"""
Generate synthetic macro data and text summaries for pipeline development.

Produces:
  data/macro/{COUNTRY}.csv          — 7 WDI-style indicators, 2005-2023
  data/text/{COUNTRY}_{YEAR}.txt    — 5-sentence Article IV dummy summary
  data/text/metadata.csv            — country, ref_year, pub_date, summary_path

Country profiles simulate realistic heterogeneity:
  - Central Asian / ECA developing: higher volatility + endemic data gaps
  - EU-adjacent / developed: stable, nearly complete data
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COUNTRIES, INDICATORS, YEARS, MACRO_DIR, TEXT_DIR

random.seed(42)
np.random.seed(42)


# ── country profiles ──────────────────────────────────────────────────────────

@dataclass
class CountryProfile:
    # GDP growth (mean, std)
    gdp_mean: float = 3.0
    gdp_std: float = 2.0
    # Inflation (mean, std)
    inf_mean: float = 4.0
    inf_std: float = 2.0
    # Fiscal balance % GDP (mean, std)
    fisc_mean: float = -2.5
    fisc_std: float = 2.0
    # Current account % GDP
    ca_mean: float = -2.0
    ca_std: float = 3.0
    # Remittances % GDP
    rem_mean: float = 3.0
    rem_std: float = 1.5
    # Unemployment %
    unemp_mean: float = 8.0
    unemp_std: float = 2.0
    # Govt debt % GDP
    debt_mean: float = 45.0
    debt_std: float = 8.0
    # Probability a given year-indicator cell is missing (reporting gap)
    miss_prob: float = 0.08
    # Publication lag: months after reference year-end (uniform range)
    lag_min: int = 3
    lag_max: int = 8
    # Region tag (for summary templates)
    region: str = "ECA"


PROFILES: dict[str, CountryProfile] = {
    # ── Central Asia ──────────────────────────────────────────────────────────
    "KAZ": CountryProfile(gdp_mean=4.5, gdp_std=4.0, inf_mean=7.0, inf_std=4.0,
                          fisc_mean=-1.5, ca_mean=0.5, rem_mean=0.5, unemp_mean=5.0,
                          debt_mean=22.0, miss_prob=0.15, region="Central Asia"),
    "KGZ": CountryProfile(gdp_mean=4.0, gdp_std=5.0, inf_mean=9.0, inf_std=5.0,
                          fisc_mean=-4.0, ca_mean=-10.0, rem_mean=28.0, unemp_mean=7.0,
                          debt_mean=55.0, miss_prob=0.25, region="Central Asia"),
    "TJK": CountryProfile(gdp_mean=6.0, gdp_std=3.0, inf_mean=8.0, inf_std=5.0,
                          fisc_mean=-3.0, ca_mean=-5.0, rem_mean=35.0, unemp_mean=10.0,
                          debt_mean=85.0, miss_prob=0.30, region="Central Asia"),
    "TKM": CountryProfile(gdp_mean=7.0, gdp_std=5.0, inf_mean=7.0, inf_std=4.0,
                          fisc_mean=1.0, ca_mean=2.0, rem_mean=0.2, unemp_mean=4.0,
                          debt_mean=20.0, miss_prob=0.35, lag_min=5, lag_max=12,
                          region="Central Asia"),
    "UZB": CountryProfile(gdp_mean=6.5, gdp_std=3.0, inf_mean=12.0, inf_std=5.0,
                          fisc_mean=-2.0, ca_mean=-3.0, rem_mean=15.0, unemp_mean=9.0,
                          debt_mean=30.0, miss_prob=0.20, region="Central Asia"),
    # ── ECA ───────────────────────────────────────────────────────────────────
    "ARM": CountryProfile(gdp_mean=5.5, gdp_std=5.0, inf_mean=4.5, inf_std=3.0,
                          fisc_mean=-3.5, ca_mean=-8.0, rem_mean=12.0, unemp_mean=17.0,
                          debt_mean=50.0, miss_prob=0.10, region="ECA"),
    "AZE": CountryProfile(gdp_mean=5.0, gdp_std=7.0, inf_mean=5.0, inf_std=3.5,
                          fisc_mean=2.0, ca_mean=8.0, rem_mean=2.0, unemp_mean=5.5,
                          debt_mean=20.0, miss_prob=0.10, region="ECA"),
    "GEO": CountryProfile(gdp_mean=5.0, gdp_std=4.0, inf_mean=4.0, inf_std=3.0,
                          fisc_mean=-3.0, ca_mean=-7.0, rem_mean=10.0, unemp_mean=15.0,
                          debt_mean=42.0, miss_prob=0.08, region="ECA"),
    "MDA": CountryProfile(gdp_mean=4.0, gdp_std=5.0, inf_mean=6.0, inf_std=4.0,
                          fisc_mean=-4.0, ca_mean=-8.0, rem_mean=18.0, unemp_mean=4.5,
                          debt_mean=35.0, miss_prob=0.12, region="ECA"),
    "BLR": CountryProfile(gdp_mean=2.5, gdp_std=5.0, inf_mean=15.0, inf_std=10.0,
                          fisc_mean=-1.0, ca_mean=-3.0, rem_mean=1.0, unemp_mean=1.0,
                          debt_mean=40.0, miss_prob=0.15, region="ECA"),
    "UKR": CountryProfile(gdp_mean=2.0, gdp_std=8.0, inf_mean=12.0, inf_std=8.0,
                          fisc_mean=-4.0, ca_mean=-2.0, rem_mean=8.0, unemp_mean=9.0,
                          debt_mean=55.0, miss_prob=0.12, region="ECA"),
    "MNG": CountryProfile(gdp_mean=7.0, gdp_std=6.0, inf_mean=8.0, inf_std=5.0,
                          fisc_mean=-3.0, ca_mean=-10.0, rem_mean=1.0, unemp_mean=7.0,
                          debt_mean=70.0, miss_prob=0.18, region="ECA"),
    # ── ECA comparators ───────────────────────────────────────────────────────
    "ALB": CountryProfile(gdp_mean=3.5, gdp_std=3.0, inf_mean=2.5, inf_std=1.5,
                          fisc_mean=-3.5, ca_mean=-8.0, rem_mean=6.0, unemp_mean=14.0,
                          debt_mean=65.0, miss_prob=0.08, region="ECA"),
    "BIH": CountryProfile(gdp_mean=2.5, gdp_std=3.0, inf_mean=1.5, inf_std=2.0,
                          fisc_mean=-2.0, ca_mean=-5.0, rem_mean=9.0, unemp_mean=20.0,
                          debt_mean=35.0, miss_prob=0.10, region="ECA"),
    "MKD": CountryProfile(gdp_mean=2.5, gdp_std=3.0, inf_mean=2.0, inf_std=2.0,
                          fisc_mean=-3.0, ca_mean=-2.0, rem_mean=4.0, unemp_mean=22.0,
                          debt_mean=45.0, miss_prob=0.08, region="ECA"),
    "SRB": CountryProfile(gdp_mean=2.5, gdp_std=3.0, inf_mean=4.0, inf_std=3.0,
                          fisc_mean=-3.5, ca_mean=-5.0, rem_mean=5.0, unemp_mean=15.0,
                          debt_mean=60.0, miss_prob=0.06, region="ECA"),
    "TUR": CountryProfile(gdp_mean=4.5, gdp_std=5.0, inf_mean=12.0, inf_std=10.0,
                          fisc_mean=-3.0, ca_mean=-4.0, rem_mean=0.2, unemp_mean=10.0,
                          debt_mean=32.0, miss_prob=0.05, region="ECA"),
    # ── EU-adjacent ───────────────────────────────────────────────────────────
    "ROU": CountryProfile(gdp_mean=3.5, gdp_std=3.5, inf_mean=4.0, inf_std=3.0,
                          fisc_mean=-3.5, ca_mean=-4.0, rem_mean=3.0, unemp_mean=6.0,
                          debt_mean=38.0, miss_prob=0.05, region="EU"),
    "BGR": CountryProfile(gdp_mean=3.0, gdp_std=3.0, inf_mean=3.0, inf_std=2.5,
                          fisc_mean=-1.5, ca_mean=-1.0, rem_mean=4.0, unemp_mean=8.0,
                          debt_mean=25.0, miss_prob=0.04, region="EU"),
    "HRV": CountryProfile(gdp_mean=2.5, gdp_std=3.5, inf_mean=2.0, inf_std=2.0,
                          fisc_mean=-2.5, ca_mean=2.0, rem_mean=3.0, unemp_mean=12.0,
                          debt_mean=72.0, miss_prob=0.04, region="EU"),
    # ── developed anchors ─────────────────────────────────────────────────────
    "USA": CountryProfile(gdp_mean=2.2, gdp_std=2.0, inf_mean=2.5, inf_std=1.5,
                          fisc_mean=-5.0, ca_mean=-2.5, rem_mean=0.0, unemp_mean=5.5,
                          debt_mean=90.0, miss_prob=0.01, lag_min=3, lag_max=5,
                          region="Developed"),
    "JPN": CountryProfile(gdp_mean=1.0, gdp_std=2.0, inf_mean=0.8, inf_std=0.8,
                          fisc_mean=-5.0, ca_mean=3.0, rem_mean=0.0, unemp_mean=3.5,
                          debt_mean=220.0, miss_prob=0.01, lag_min=3, lag_max=5,
                          region="Developed"),
}

# ── summary templates ─────────────────────────────────────────────────────────

_TEMPLATES = [
    (
        "{country}'s economy grew by approximately {gdp:.1f} percent in {year}, supported by "
        "strong domestic demand and favorable external conditions. "
        "Inflationary pressures remained {inf_level}, with headline CPI averaging {inf:.1f} percent. "
        "The fiscal deficit stood at {fisc:.1f} percent of GDP, reflecting ongoing infrastructure "
        "investment and social spending. "
        "External vulnerabilities persist, with the current account deficit at {ca:.1f} percent "
        "of GDP, partially financed by remittances. "
        "Staff recommend maintaining prudent monetary policy while accelerating structural reforms "
        "to strengthen the medium-term growth outlook."
    ),
    (
        "In {year}, {country} experienced {growth_desc} economic growth of {gdp:.1f} percent amid "
        "{ext_env} external environment. "
        "Inflation reached {inf:.1f} percent, driven by {inf_driver}. "
        "The government maintained a fiscal deficit of {fisc:.1f} percent of GDP, with debt at "
        "{debt:.0f} percent of GDP. "
        "Remittance inflows contributed {rem:.1f} percent of GDP, providing a key buffer for "
        "household consumption. "
        "The IMF urges authorities to strengthen revenue mobilization and reduce dependence on "
        "commodity exports to build resilience."
    ),
    (
        "{country}'s macroeconomic performance in {year} was characterized by {gdp:.1f} percent "
        "GDP growth and {inf:.1f} percent inflation. "
        "The fiscal position showed a balance of {fisc:.1f} percent of GDP, while unemployment "
        "remained elevated at {unemp:.1f} percent. "
        "Current account dynamics were shaped by {ca_desc}, with a balance of {ca:.1f} percent "
        "of GDP. "
        "Government debt reached {debt:.0f} percent of GDP, necessitating careful debt management "
        "to ensure sustainability. "
        "The authorities are advised to pursue exchange rate flexibility and strengthen "
        "financial sector supervision to mitigate systemic risks."
    ),
]


def _describe_growth(gdp: float) -> str:
    if gdp > 6:   return "robust"
    if gdp > 3:   return "moderate"
    if gdp > 0:   return "sluggish"
    return "contractionary"


def _describe_inf(inf: float) -> str:
    if inf > 10:  return "elevated"
    if inf > 5:   return "moderate"
    return "subdued"


def _describe_ca(ca: float) -> str:
    if ca > 2:    return "strong export performance"
    if ca > -3:   return "balanced trade flows"
    return "import pressure and weak export diversification"


def _describe_inf_driver(inf: float) -> str:
    if inf > 10:  return "supply-side shocks and currency depreciation"
    if inf > 5:   return "food and energy price increases"
    return "stable commodity prices and anchored expectations"


def _describe_ext_env() -> str:
    return random.choice(["a challenging", "a supportive", "an uncertain", "a mixed"])


def generate_summary(country: str, year: int, row: dict, region: str) -> str:
    tmpl = _TEMPLATES[year % len(_TEMPLATES)]
    return tmpl.format(
        country=country,
        year=year,
        region=region,
        gdp=row["gdp_growth"],
        inf=row["inflation"],
        fisc=row["fiscal_balance"],
        ca=row["current_account"],
        rem=row["remittances"],
        unemp=row["unemployment"],
        debt=row["govt_debt"],
        growth_desc=_describe_growth(row["gdp_growth"]),
        inf_level=_describe_inf(row["inflation"]),
        inf_driver=_describe_inf_driver(row["inflation"]),
        ca_desc=_describe_ca(row["current_account"]),
        ext_env=_describe_ext_env(),
    )


# ── data generation ───────────────────────────────────────────────────────────

def generate_macro_series(country: str, profile: CountryProfile) -> pd.DataFrame:
    """Simulate annual macro indicators with correlated noise and reporting gaps."""
    rng = np.random.default_rng(abs(hash(country)) % (2**32))

    n = len(YEARS)
    # Correlated shocks (e.g. commodity boom hits GDP + fiscal simultaneously)
    common_shock = rng.normal(0, 1, n)

    gdp_growth    = profile.gdp_mean   + profile.gdp_std   * (0.7 * common_shock + 0.3 * rng.normal(0, 1, n))
    inflation     = profile.inf_mean   + profile.inf_std   * rng.normal(0, 1, n)
    fiscal_bal    = profile.fisc_mean  + profile.fisc_std  * (0.4 * common_shock + 0.6 * rng.normal(0, 1, n))
    current_acct  = profile.ca_mean    + profile.ca_std    * rng.normal(0, 1, n)
    remittances   = np.clip(profile.rem_mean + profile.rem_std * rng.normal(0, 1, n), 0, 50)
    unemployment  = np.clip(profile.unemp_mean + profile.unemp_std * rng.normal(0, 1, n), 0, 40)
    govt_debt     = np.clip(profile.debt_mean + profile.debt_std * rng.normal(0, 1, n), 0, 300)

    # Simulate COVID shock (2020) and recovery (2021)
    if 2020 in YEARS:
        idx_2020 = YEARS.index(2020)
        gdp_growth[idx_2020] -= rng.uniform(3, 10)
        inflation[idx_2020]  += rng.uniform(1, 4)
        if idx_2020 + 1 < n:
            gdp_growth[idx_2020 + 1] += rng.uniform(2, 6)

    df = pd.DataFrame({
        "year":            YEARS,
        "gdp_growth":      np.round(gdp_growth, 2),
        "inflation":       np.round(np.abs(inflation), 2),
        "fiscal_balance":  np.round(fiscal_bal, 2),
        "current_account": np.round(current_acct, 2),
        "remittances":     np.round(remittances, 2),
        "unemployment":    np.round(unemployment, 2),
        "govt_debt":       np.round(govt_debt, 2),
    })

    # Introduce missingness (reporting gaps)
    mask = rng.random((n, len(INDICATORS))) < profile.miss_prob
    for j, col in enumerate(INDICATORS):
        df.loc[mask[:, j], col] = np.nan

    # Targets (GDP growth, inflation) should never be missing for training years
    # — keep them observed (drop NaN only for non-target cols on early years)
    return df


def generate_pub_date(reference_year: int, profile: CountryProfile) -> str:
    """Publication date = reference_year + lag months (YYYY-MM-DD format)."""
    lag_months = random.randint(profile.lag_min, profile.lag_max)
    month = lag_months          # reference year-end = Dec 31; add months
    year  = reference_year + 1 if month > 12 else reference_year + 1
    # Lag is from year-end, so always spills into the next calendar year
    actual_month = (month - 1) % 12 + 1
    return f"{reference_year + 1}-{actual_month:02d}-15"


# ── main entry ────────────────────────────────────────────────────────────────

def run():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    for country in COUNTRIES:
        profile = PROFILES.get(country, CountryProfile())

        # ── macro CSV ────────────────────────────────────────────────────────
        df = generate_macro_series(country, profile)
        csv_path = MACRO_DIR / f"{country}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [macro] {country}: {csv_path.name}  "
              f"(missing_rate={df[INDICATORS].isna().mean().mean():.0%})")

        # ── text summaries ───────────────────────────────────────────────────
        for _, row in df.iterrows():
            year = int(row["year"])
            # Only generate summaries where we have at least GDP and inflation
            if pd.isna(row["gdp_growth"]) or pd.isna(row["inflation"]):
                continue

            summary = generate_summary(country, year, row.to_dict(), profile.region)
            txt_path = TEXT_DIR / f"{country}_{year}.txt"
            txt_path.write_text(summary)

            pub_date = generate_pub_date(year, profile)
            meta_rows.append({
                "country":      country,
                "ref_year":     year,
                "pub_date":     pub_date,
                "summary_path": str(txt_path.relative_to(TEXT_DIR.parent)),
            })

    # ── metadata CSV ─────────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    meta_path = TEXT_DIR / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[meta] {len(meta_rows)} text records → {meta_path}")


if __name__ == "__main__":
    run()
