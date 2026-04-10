"""
IMF Article IV – Coveo API fetcher.

Strategy
--------
1. Query the IMF's Coveo search API directly (no browser needed for search).
   Search per country: ~20-40 hits instead of paginating 7,335 results.
2. Filter results by year range and content type (Country Report / Article IV).
3. For each match, fetch the article page with requests + BeautifulSoup to find
   the PDF download link.
4. Stream PDF bytes into memory, extract first 6 pages with PyMuPDF, save as
   data/text/<ISO3>_<year>.txt.  The full PDF is NEVER written to disk.

If the Coveo token has expired, re-run:
    python scripts/debug_search.py
and paste the new token into COVEO_TOKEN below (or pass --token <tok>).

Requirements
    pip install requests beautifulsoup4 pymupdf playwright
    playwright install chromium   # only needed if HTML parsing fallback is used

Usage
    python scripts/fetch_article_iv.py                          # all 22 countries
    python scripts/fetch_article_iv.py --countries KAZ UZB KGZ
    python scripts/fetch_article_iv.py --year-from 2015 --year-to 2023
    python scripts/fetch_article_iv.py --dry-run                # list URLs only
    python scripts/fetch_article_iv.py --token <new_token>      # override token
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, TEXT_DIR, YEARS

# ── Coveo config ─────────────────────────────────────────────────────────────
COVEO_ENDPOINT  = "https://imfproduction561s308u.org.coveo.com/rest/search/v2"
COVEO_ORG       = "imfproduction561s308u"
# Token captured from browser session — re-run debug_search.py if expired
COVEO_TOKEN     = "xx742a6c66-f427-4f5a-ae1e-770dc7264e8a"

IMF_ROOT = "https://www.imf.org"

# ── country config ────────────────────────────────────────────────────────────
COUNTRY_QUERY = {
    "KAZ": "Kazakhstan",
    "KGZ": "Kyrgyz Republic",
    "TJK": "Tajikistan",
    "TKM": "Turkmenistan",
    "UZB": "Uzbekistan",
    "ARM": "Armenia",
    "AZE": "Azerbaijan",
    "GEO": "Georgia",
    "MDA": "Moldova",
    "BLR": "Belarus",
    "UKR": "Ukraine",
    "MNG": "Mongolia",
    "ALB": "Albania",
    "BIH": "Bosnia and Herzegovina",
    "MKD": "North Macedonia",
    "SRB": "Serbia",
    "TUR": "Turkey",
    "ROU": "Romania",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "USA": "United States",
    "JPN": "Japan",
}

COUNTRY_ALIASES = {
    "KGZ": ["Kyrgyz Republic", "Kyrgyzstan"],
    "MKD": ["North Macedonia", "FYR Macedonia", "Republic of North Macedonia", "Former Yugoslav Republic of Macedonia", "Macedonia"],
    "TUR": ["Turkey", "Türkiye"],
    "MDA": ["Moldova", "Republic of Moldova"],
}

# ── text cleaning (mirrors extract_pdf.py logic) ─────────────────────────────
_BOILERPLATE = re.compile(
    r"(international\s+monetary\s+fund|imf\s+country\s+report"
    r"|©\s*\d{4}|confidential|washington.*d\.?c\.?|approved\s+by)",
    re.IGNORECASE,
)

def extract_text_from_bytes(pdf_bytes, max_pages=6):
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = min(max_pages, doc.page_count)
    raw = "\n".join(doc[i].get_text("text") for i in range(pages))
    doc.close()
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        if _BOILERPLATE.search(line):
            continue
        lines.append(line)
    return " ".join(lines)

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_year(text):
    m = re.search(r"\b(20\d{2}|199\d)\b", text)
    return int(m.group(1)) if m else None

def title_matches_country(title, iso3):
    title_lower = title.lower()
    names = COUNTRY_ALIASES.get(iso3, [COUNTRY_QUERY[iso3]])
    return any(n.lower() in title_lower for n in names)

_IMF_MARKERS = ["article iv", "consultation", "imf", "executive board", "staff report"]

def is_real(iso3, year):
    """True if we already have genuine IMF text (not dummy)."""
    path = TEXT_DIR / f"{iso3}_{year}.txt"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(m in text for m in _IMF_MARKERS)

def save_text(iso3, year, text):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    out = TEXT_DIR / f"{iso3}_{year}.txt"
    out.write_text(text, encoding="utf-8")
    return out

# ── Coveo search ──────────────────────────────────────────────────────────────

def _coveo_post(token, body):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{COVEO_ENDPOINT}?organizationId={COVEO_ORG}",
        headers=headers, json=body, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def coveo_search(iso3, token, page=0, page_size=100):
    """
    Query Coveo for Article IV country reports using imfisocode (ISO3).
    - @imfisocode for exact country match (works for all countries)
    - q="Article IV consultation" for relevance ranking
    - No content type filter: some countries tag as ARTICLE4, others as COUNTRYREPS only
    """
    body = {
        "locale": "en-US",
        "tab": "default",
        "q": "Article IV consultation",
        "aq": f'(@imfisocode=="{iso3}") AND (@imfcontenttype="PUBS|COUNTRYREPS")',
        "numberOfResults": page_size,
        "firstResult": page * page_size,
        "fieldsToInclude": ["title", "date", "sysuri", "imfcontenttype"],
        "context": {"applicationExperience": "modern"},
    }
    return _coveo_post(token, body)

def get_all_results(iso3, token):
    """Page through all Coveo results for this ISO3 country code."""
    all_results = []
    page = 0
    while True:
        data = coveo_search(iso3, token, page=page, page_size=100)
        results = data.get("results", [])
        all_results.extend(results)
        total = data.get("totalCountFiltered", 0)
        if len(all_results) >= total or not results:
            break
        page += 1
        time.sleep(0.5)
    return all_results

# ── PDF retrieval via Playwright ──────────────────────────────────────────────

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

def normalize_imf_url(url):
    """Replace internal IMF staging hostnames with the public www.imf.org."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc and parsed.netloc != "www.imf.org" and "imf.org" in parsed.netloc:
        url = parsed._replace(netloc="www.imf.org").geturl()
    return url


def get_pdf_bytes(ctx, article_url):
    """
    Open the article page in a browser, find the PDF, return bytes.
    Never writes to disk.

    Strategy:
    1. Parse DOM for a direct .pdf / .ashx href.
    2. If found, download via requests (reusing browser cookies).
    3. If not, click the Download button and intercept via expect_download().
    """
    import tempfile, os

    article_url = normalize_imf_url(article_url)
    page = ctx.new_page()
    try:
        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=45_000)
        except Exception as e:
            print(f"             → page load error: {e}")
            return None
        time.sleep(3)

        # ── Strategy 1: find direct PDF link in rendered DOM ─────────────────
        pdf_url = None
        for a in page.query_selector_all("a[href]"):
            href = a.get_attribute("href") or ""
            text = a.inner_text().lower()
            if not (href.endswith(".pdf") or ".ashx" in href):
                continue
            if any(kw in text for kw in ["staff report", "country report", "download", "pdf", "full text"]):
                pdf_url = href if href.startswith("http") else IMF_ROOT + href
                break
        if pdf_url is None:
            for a in page.query_selector_all("a[href$='.pdf'], a[href*='.ashx']"):
                href = a.get_attribute("href") or ""
                pdf_url = href if href.startswith("http") else IMF_ROOT + href
                break

        if pdf_url:
            print(f"             → direct PDF link found")
            # download using browser cookies so we're authenticated
            cookies = {c["name"]: c["value"] for c in ctx.cookies()}
            resp = requests.get(pdf_url, cookies=cookies, timeout=120,
                                headers={"User-Agent": UA})
            if resp.ok:
                return resp.content
            print(f"             → direct download failed ({resp.status_code}), trying click …")

        # ── Strategy 2: click Download button, intercept the download ─────────
        download_btn = None
        for btn in page.query_selector_all("a, button"):
            text = btn.inner_text().lower()
            if any(kw in text for kw in ["download", "pdf", "staff report", "full text"]):
                download_btn = btn
                break

        if download_btn is None:
            print(f"             → no download button found")
            return None

        print(f"             → clicking download button …")
        tmp_dir = tempfile.mkdtemp()
        try:
            with page.expect_download(timeout=60_000) as dl_info:
                download_btn.click()
            dl = dl_info.value
            tmp_path = os.path.join(tmp_dir, dl.suggested_filename or "report.pdf")
            dl.save_as(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"             → download intercept failed: {e}")
            # fallback: if a new tab opened, grab its URL
            new_pages = ctx.pages
            for p in new_pages:
                if p != page and (".pdf" in p.url or ".ashx" in p.url):
                    cookies = {c["name"]: c["value"] for c in ctx.cookies()}
                    resp = requests.get(p.url, cookies=cookies, timeout=120,
                                        headers={"User-Agent": UA})
                    p.close()
                    if resp.ok:
                        return resp.content
            return None
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    finally:
        page.close()

# ── main ──────────────────────────────────────────────────────────────────────

def run(target_countries, year_from, year_to,
        dry_run: bool, token: str):
    from playwright.sync_api import sync_playwright

    saved, skipped, failed = 0, 0, 0

    with sync_playwright() as pw:
      browser = pw.chromium.launch(headless=True)
      ctx = browser.new_context(user_agent=UA)

      for iso3 in target_countries:
        country_name = COUNTRY_QUERY[iso3]
        print(f"\n{'='*60}")
        print(f"  {iso3}  ({country_name})")

        try:
            results = get_all_results(iso3, token)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                print("  ERROR: Coveo token expired. Re-run debug_search.py to get a new token.")
                sys.exit(1)
            print(f"  ERROR: {e}")
            continue

        print(f"  Coveo returned {len(results)} result(s)")

        # deduplicate by (year) — Coveo returns same article with mixed-case URLs
        seen_years = set()

        for result in results:
            title    = result.get("title", "")
            click_uri = result.get("clickUri", "")
            raw      = result.get("raw", {})
            date_str = str(raw.get("date", ""))

            # year: try title first, then date field
            year = parse_year(title) or parse_year(date_str)
            if year is None or year < year_from or year > year_to:
                continue
            if not title_matches_country(title, iso3):
                continue
            if "article iv" not in title.lower() and "consultation" not in title.lower():
                continue
            if year in seen_years:
                continue
            seen_years.add(year)

            # skip if we already have genuine IMF text
            if is_real(iso3, year):
                print(f"  [{iso3} {year}] already have real text — skip")
                skipped += 1
                continue
            # dummy file or missing → proceed

            out_path = TEXT_DIR / f"{iso3}_{year}.txt"
            print(f"\n  [{iso3} {year}] {title[:75]}")
            print(f"             {click_uri[:80]}")

            if dry_run:
                exists_note = " (will overwrite dummy)" if out_path.exists() else ""
                print(f"             → [dry-run]{exists_note}")
                continue

            # ── find + download PDF via browser ───────────────────────────────
            pdf_bytes = get_pdf_bytes(ctx, click_uri)

            if not pdf_bytes:
                print(f"             → could not retrieve PDF")
                failed += 1
                continue

            text = extract_text_from_bytes(pdf_bytes)
            if len(text) < 200:
                print(f"             → extracted text too short ({len(text)} chars), skip")
                failed += 1
                continue

            out = save_text(iso3, year, text)
            print(f"             → saved {out.name}  ({len(text):,} chars)")
            saved += 1
            time.sleep(1)   # polite delay

      browser.close()

    print(f"\n{'='*60}")
    print(f"Done.  Saved: {saved}  Skipped (exist): {skipped}  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Fetch IMF Article IV text via Coveo API")
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--year-from", type=int, default=min(YEARS))
    parser.add_argument("--year-to",   type=int, default=max(YEARS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token", default=COVEO_TOKEN,
                        help="Coveo Bearer token (re-run debug_search.py if expired)")
    args = parser.parse_args()

    bad = [c for c in args.countries if c not in COUNTRY_QUERY]
    if bad:
        print(f"Unknown ISO3 codes: {bad}"); sys.exit(1)

    print(f"Countries  : {args.countries}")
    print(f"Year range : {args.year_from}–{args.year_to}")
    print(f"Output dir : {TEXT_DIR}  (text only, no PDFs saved)")
    print(f"Dry run    : {args.dry_run}")
    print(f"Token      : {args.token[:12]}…")
    print()
    run(args.countries, args.year_from, args.year_to, args.dry_run, args.token)


if __name__ == "__main__":
    main()
