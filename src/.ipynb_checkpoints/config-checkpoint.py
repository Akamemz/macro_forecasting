"""
Central configuration for the macro forecasting project.

Numerical stream: World Bank WDI (7 indicators, 2005-2023, 22 countries)
Text stream:      IMF Article IV reports → 5-sentence summaries → 768-dim BERT embeddings
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
MACRO_DIR   = DATA_DIR / "macro"
PDF_DIR     = DATA_DIR / "pdf"
TEXT_DIR    = DATA_DIR / "text"
EMB_DIR     = DATA_DIR / "embeddings"

# ── countries ───────────────────────────────────────────────────────────────
# Central Asia + ECA comparators (20 dev. economies) + USA/JPN as anchors
COUNTRIES = [
    "KAZ", "KGZ", "TJK", "TKM", "UZB",          # Central Asia
    "ARM", "AZE", "GEO", "MDA", "BLR", "UKR", "MNG",  # ECA
    "ALB", "BIH", "MKD", "SRB", "TUR",           # ECA comparators
    "ROU", "BGR", "HRV",                          # EU-adjacent
    "USA", "JPN",                                 # developed anchors
]

# ── indicators (World Bank WDI codes mapped to short names) ─────────────────
INDICATORS = [
    "gdp_growth",       # NY.GDP.MKTP.KD.ZG
    "inflation",        # FP.CPI.TOTL.ZG
    "fiscal_balance",   # GC.BAL.CASH.GD.ZS
    "current_account",  # BN.CAB.XOKA.GD.ZS
    "remittances",      # BX.TRF.PWKR.DT.GD.ZS
    "unemployment",     # SL.UEM.TOTL.ZS
    "govt_debt",        # GC.DOD.TOTL.GD.ZS
]
N_INDICATORS = len(INDICATORS)

TARGET_INDICATORS = ["gdp_growth", "inflation"]
TARGET_COLS = [INDICATORS.index(t) for t in TARGET_INDICATORS]

# ── time ─────────────────────────────────────────────────────────────────────
YEARS = list(range(2005, 2024))   # 2005-2023 inclusive
MIN_YEAR, MAX_YEAR = YEARS[0], YEARS[-1]
CONTEXT_WINDOW = 10               # years of historical context per sample

# Chronological split: need CONTEXT_WINDOW years before first query
TRAIN_QUERY_YEARS = list(range(2015, 2020))   # predict 2015-2019 (context 2005-2014)
VAL_QUERY_YEARS   = [2020, 2021]
TEST_QUERY_YEARS  = [2022, 2023]

# ── model / training ─────────────────────────────────────────────────────────
TEXT_EMB_DIM = 768     # BERT-base hidden size
TTF_TYPE     = "T2V-XAttn"  # Time2Vector+Cross attention
SIGMA        = 1.0     # Gaussian kernel width (years) for RecAvg TTF
HIDDEN_DIM   = 64

N_TARGETS    = len(TARGET_INDICATORS)

LR                    = 1e-3
EPOCHS                = 100
EARLY_STOPPING        = 10     # patience (val epochs)
BATCH_SIZE            = 8

# ── normalisation helpers ────────────────────────────────────────────────────
def norm_year(year: float) -> float:
    """Map a calendar year to [0, 1] over the dataset range."""
    return (year - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
