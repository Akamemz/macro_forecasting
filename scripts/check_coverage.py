"""
Check which country-year Article IV text files are present vs missing.

A file is considered REAL if it contains IMF-specific phrases.
A file is DUMMY if it exists but has no recognizable IMF content.

Usage:
    python scripts/check_coverage.py            # full summary table
    python scripts/check_coverage.py --missing  # print only missing pairs
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, TEXT_DIR, YEARS

IMF_MARKERS = ["article iv", "consultation", "imf", "executive board", "staff report"]

def classify(path: Path) -> str:
    """Return 'real', 'dummy', or 'missing'."""
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    if any(m in text for m in IMF_MARKERS):
        return "real"
    return "dummy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", action="store_true", help="Print only missing/dummy pairs")
    args = parser.parse_args()

    total = len(COUNTRIES) * len(YEARS)
    counts = {"real": 0, "dummy": 0, "missing": 0}
    missing_pairs: list[tuple[str, int]] = []

    # header
    if not args.missing:
        year_labels = "".join(f"{y % 100:3d}" for y in YEARS)
        print(f"{'ISO3':6s}{year_labels}")
        print(f"{'':6s}" + "---" * len(YEARS))

    for iso3 in COUNTRIES:
        row = f"{iso3:6s}"
        for year in YEARS:
            path = TEXT_DIR / f"{iso3}_{year}.txt"
            status = classify(path)
            counts[status] += 1
            if status in ("missing", "dummy"):
                missing_pairs.append((iso3, year))
            if not args.missing:
                symbol = {"real": " ✓", "dummy": " ~", "missing": " ·"}[status]
                row += symbol
        if not args.missing:
            print(row)

    if not args.missing:
        print()
        print(f"Legend:  ✓ = real IMF text   ~ = dummy/placeholder   · = missing")
        print()
        print(f"Total pairs : {total}")
        print(f"  Real      : {counts['real']:4d}  ({100*counts['real']/total:.1f}%)")
        print(f"  Dummy     : {counts['dummy']:4d}  ({100*counts['dummy']/total:.1f}%)")
        print(f"  Missing   : {counts['missing']:4d}  ({100*counts['missing']/total:.1f}%)")

    if args.missing or counts["dummy"] + counts["missing"] > 0:
        print(f"\nNot yet fetched ({len(missing_pairs)} pairs):")
        # group by country for readability
        from itertools import groupby
        for iso3, pairs in groupby(missing_pairs, key=lambda x: x[0]):
            years = [str(y) for _, y in pairs]
            print(f"  {iso3}: {', '.join(years)}")


if __name__ == "__main__":
    main()
