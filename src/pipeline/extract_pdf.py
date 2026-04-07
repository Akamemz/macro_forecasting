"""
PDF extraction pipeline for IMF Article IV Consultation reports.

Extracts the Executive Summary and Staff Appraisal sections (pages 1-6)
and returns cleaned text suitable for LLM summarisation.

Usage:
    python -m src.pipeline.extract_pdf data/pdf/1jpnea2026001.pdf
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# Section headings we want to capture
SECTION_PATTERNS = [
    r"executive\s+summary",
    r"staff\s+appraisal",
    r"overview",
    r"context",
]
_SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)

# Boilerplate lines to drop
_BOILERPLATE_RE = re.compile(
    r"(international\s+monetary\s+fund|imf\s+country\s+report"
    r"|©\s*\d{4}|confidential|washington.*d\.?c\.?|approved\s+by)",
    re.IGNORECASE,
)


def extract_report_text(pdf_path: str | Path, max_pages: int = 6) -> str:
    """
    Extract the first ``max_pages`` pages from an IMF Article IV PDF and
    return the cleaned text.  The first ~6 pages typically contain the
    cover, press release, executive summary, and staff appraisal.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages_to_read = min(max_pages, doc.page_count)
    raw_blocks = []
    for page_idx in range(pages_to_read):
        page = doc[page_idx]
        raw_blocks.append(page.get_text("text"))

    doc.close()
    raw = "\n".join(raw_blocks)

    # ── clean ────────────────────────────────────────────────────────────────
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if _BOILERPLATE_RE.search(line):
            continue
        if len(line) < 4:          # stray page numbers / letters
            continue
        lines.append(line)

    return " ".join(lines)


def parse_country_year_from_filename(filename: str) -> tuple[str, int] | tuple[None, None]:
    """
    IMF filenames follow patterns like:
        1jpnea2026001.pdf  → JPN, 2026
        1usaea2026001.pdf  → USA, 2026
    Returns (country_code_upper, year) or (None, None) if not parseable.
    """
    stem = Path(filename).stem.lower()
    # strip leading digit and trailing sequence number
    # pattern: 1<alpha3><alpha2><year4><seq3>
    m = re.match(r"\d([a-z]{3})[a-z]{2}(\d{4})\d+", stem)
    if m:
        iso3, year = m.group(1).upper(), int(m.group(2))
        return iso3, year
    return None, None


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/pdf/1jpnea2026001.pdf"
    country, year = parse_country_year_from_filename(pdf)
    text = extract_report_text(pdf)
    print(f"Country: {country}, Year: {year}")
    print(f"Extracted {len(text)} chars")
    print("\n--- FIRST 800 CHARS ---")
    print(text[:800])
