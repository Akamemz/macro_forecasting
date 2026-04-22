"""
IMF Article IV Staff Reports – PDF downloader.

Navigates https://www.imf.org/en/publications/sprolls/article-iv-staff-reports
with a real browser (Playwright), paginates through all entries, matches
articles against our 22 target countries + year range, and downloads PDFs.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/download_article_iv.py                          # all 22 countries, 2005-2023
    python scripts/download_article_iv.py --countries KAZ UZB     # subset
    python scripts/download_article_iv.py --year-from 2015 --year-to 2023
    python scripts/download_article_iv.py --dry-run               # list matches, no download
"""

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COUNTRIES, PDF_DIR, YEARS

# ── country name variants (as they appear in IMF article titles) ─────────────
COUNTRY_NAMES: dict[str, list[str]] = {
    "KAZ": ["Kazakhstan"],
    "KGZ": ["Kyrgyz Republic", "Kyrgyzstan"],
    "TJK": ["Tajikistan"],
    "TKM": ["Turkmenistan"],
    "UZB": ["Uzbekistan"],
    "ARM": ["Armenia"],
    "AZE": ["Azerbaijan"],
    "GEO": ["Georgia"],
    "MDA": ["Moldova", "Republic of Moldova"],
    "BLR": ["Belarus"],
    "UKR": ["Ukraine"],
    "MNG": ["Mongolia"],
    "ALB": ["Albania"],
    "BIH": ["Bosnia and Herzegovina"],
    "MKD": ["North Macedonia", "FYR Macedonia", "Republic of North Macedonia", "Macedonia"],
    "SRB": ["Serbia"],
    "TUR": ["Turkey", "Türkiye"],
    "ROU": ["Romania"],
    "BGR": ["Bulgaria"],
    "HRV": ["Croatia"],
    "USA": ["United States"],
    "JPN": ["Japan"],
}

BASE_URL = "https://www.imf.org/en/publications/sprolls/article-iv-staff-reports"


def parse_year_from_title(title: str) -> int | None:
    """Extract the consultation year from an article title."""
    m = re.search(r"\b(20\d{2}|19\d{2})\b", title)
    return int(m.group(1)) if m else None


def match_country(title: str, target_iso3s: list[str]) -> str | None:
    """Return ISO3 if the title mentions one of our target countries."""
    title_lower = title.lower()
    for iso3 in target_iso3s:
        for name in COUNTRY_NAMES.get(iso3, []):
            if name.lower() in title_lower:
                return iso3
    return None


def pdf_filename(iso3: str, year: int) -> str:
    """Canonical filename matching IMF convention: 1<iso3_lower>ea<year>001.pdf"""
    return f"1{iso3.lower()}ea{year}001.pdf"


def already_downloaded(iso3: str, year: int, out_dir: Path) -> bool:
    return (out_dir / pdf_filename(iso3, year)).exists()


def run(target_countries: list[str], year_from: int, year_to: int, dry_run: bool) -> None:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    year_set = set(range(year_from, year_to + 1))

    downloaded, skipped, not_found = 0, 0, 0

    # ── intercept the API call the page makes to fetch articles ─────────────
    captured_responses: list[dict] = []

    def handle_response(response):
        url = response.url
        # The IMF sprolls page loads articles via a search/content API
        if any(k in url for k in ["api", "search", "solr", "content", "sproll"]):
            try:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    data = response.json()
                    captured_responses.append({"url": url, "data": data})
                    print(f"  [api] captured JSON from: {url[:100]}")
            except Exception:
                pass

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = ctx.new_page()
        page.on("response", handle_response)

        print(f"Opening {BASE_URL} …")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60_000)
        # give JS time to fire its API requests and render articles
        print("  Waiting 8s for JS to render articles …")
        time.sleep(8)

        page_num = 1
        stop = False  # set True when all remaining articles are before year_from

        while not stop:
            print(f"\n── Page {page_num} ──────────────────────────────────")

            # dump captured API calls on first page so we know the endpoint
            if page_num == 1 and captured_responses:
                print(f"  [api] {len(captured_responses)} JSON API call(s) intercepted:")
                for r in captured_responses[:5]:
                    print(f"        {r['url'][:120]}")

            # save screenshot
            shot = ROOT / "data" / f"debug_page{page_num}.png"
            page.screenshot(path=str(shot), full_page=False)
            print(f"  [debug] screenshot → {shot}")

            # ── try to read article links from the rendered DOM ───────────────
            # wait up to 10s for a link that looks like a country report to appear
            try:
                page.wait_for_selector(
                    "a[href*='/Publications/CR/'], a[href*='/publications/cr/']",
                    timeout=10_000
                )
            except Exception:
                pass

            all_links = page.query_selector_all("a[href]")
            print(f"  [debug] total <a> tags: {len(all_links)}")

            imf_links = []
            for a in all_links:
                href = a.get_attribute("href") or ""
                # get text from the element or its children
                title = a.inner_text().strip()
                if not title:
                    title = a.get_attribute("title") or a.get_attribute("aria-label") or ""
                imf_links.append((href, title))

            # show sample of what we found
            cr_links = [(h, t) for h, t in imf_links if "/CR/" in h or "/cr/" in h]
            print(f"  [debug] country-report links found: {len(cr_links)}")
            for h, t in cr_links[:5]:
                print(f"           {t[:60]!r}  →  {h[:80]}")

            # ── collect article rows ──────────────────────────────────────────
            rows = []
            seen = set()
            for href, title in imf_links:
                if not href or not title:
                    continue
                if href in seen:
                    continue
                if not any(p in href for p in ["/Publications/CR/", "/publications/cr/"]):
                    continue
                if href == BASE_URL or href.rstrip("/") == BASE_URL.rstrip("/"):
                    continue
                seen.add(href)
                rows.append((title, href))

            if not rows:
                print("  No articles found on this page — stopping.")
                break

            earliest_year_on_page = None
            for title, href in rows:
                year = parse_year_from_title(title)
                if year and (earliest_year_on_page is None or year < earliest_year_on_page):
                    earliest_year_on_page = year

            for title, href in rows:
                year = parse_year_from_title(title)
                if year is None:
                    continue
                if year < year_from:
                    continue   # too old
                if year > year_to:
                    continue   # too new (future reports)

                iso3 = match_country(title, target_countries)
                if iso3 is None:
                    continue

                fname = pdf_filename(iso3, year)
                out_path = PDF_DIR / fname

                print(f"  MATCH  {iso3} {year}  '{title[:70]}'")

                if already_downloaded(iso3, year, PDF_DIR):
                    print(f"         → already exists, skipping")
                    skipped += 1
                    continue

                if dry_run:
                    print(f"         → [dry-run] would download {fname}")
                    continue

                # ── navigate to article page and find the PDF link ───────────
                full_url = href if href.startswith("http") else f"https://www.imf.org{href}"
                try:
                    art_page = ctx.new_page()
                    art_page.goto(full_url, wait_until="domcontentloaded", timeout=45_000)
                    try:
                        art_page.wait_for_selector("a[href$='.pdf'], a[href*='.ashx']", timeout=10_000)
                    except Exception:
                        pass
                    time.sleep(1)

                    # Look for a direct PDF link (Staff Report document)
                    pdf_link = None
                    for candidate in art_page.query_selector_all("a[href$='.pdf'], a[href*='.ashx']"):
                        link_text = candidate.inner_text().lower()
                        link_href = candidate.get_attribute("href") or ""
                        # prefer the Staff Report PDF, not the full text
                        if any(kw in link_text for kw in ["staff report", "country report", "pdf"]):
                            pdf_link = link_href
                            break
                    if pdf_link is None:
                        # grab the first .pdf/.ashx link on the page
                        el = art_page.query_selector("a[href$='.pdf'], a[href*='.ashx']")
                        pdf_link = el.get_attribute("href") if el else None

                    if pdf_link is None:
                        print(f"         → PDF link not found on article page")
                        not_found += 1
                        art_page.close()
                        continue

                    pdf_url = pdf_link if pdf_link.startswith("http") else f"https://www.imf.org{pdf_link}"

                    # download via requests (faster than browser download)
                    import requests
                    headers = {"User-Agent": ctx._browser.version or "Mozilla/5.0"}
                    resp = requests.get(pdf_url, timeout=120, stream=True)
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)

                    size_kb = out_path.stat().st_size // 1024
                    print(f"         → saved {fname} ({size_kb} KB)")
                    downloaded += 1
                    art_page.close()
                    time.sleep(1.5)   # polite delay

                except Exception as e:
                    print(f"         → ERROR: {e}")
                    not_found += 1
                    try:
                        art_page.close()
                    except Exception:
                        pass

            # ── stop if all articles on page are older than our window ───────
            if earliest_year_on_page is not None and earliest_year_on_page < year_from:
                print(f"\nEarliest year on page ({earliest_year_on_page}) < {year_from} — done.")
                stop = True
                break

            # ── click Next ───────────────────────────────────────────────────
            next_btn = page.query_selector("a[aria-label='Next'], button[aria-label='Next'], a.next, li.next a, a:has-text('Next')")
            if next_btn is None:
                print("No 'Next' button found — reached last page.")
                break

            try:
                next_btn.click()
                page.wait_for_load_state("domcontentloaded", timeout=30_000)
                time.sleep(2)
                page_num += 1
            except PWTimeout:
                print("Timeout clicking Next — stopping.")
                break

        browser.close()

    print(f"\n{'='*50}")
    print(f"Done.  Downloaded: {downloaded}  Skipped (exist): {skipped}  Not found: {not_found}")


def main():
    parser = argparse.ArgumentParser(description="Download IMF Article IV PDFs")
    parser.add_argument(
        "--countries", nargs="+", default=COUNTRIES,
        help="ISO3 codes to download (default: all 22 project countries)"
    )
    parser.add_argument("--year-from", type=int, default=min(YEARS), help="First year (default: 2005)")
    parser.add_argument("--year-to",   type=int, default=max(YEARS), help="Last year (default: 2023)")
    parser.add_argument("--dry-run", action="store_true", help="List matches without downloading")
    args = parser.parse_args()

    bad = [c for c in args.countries if c not in COUNTRY_NAMES]
    if bad:
        print(f"Unknown country codes: {bad}")
        sys.exit(1)

    print(f"Target countries : {args.countries}")
    print(f"Year range       : {args.year_from}–{args.year_to}")
    print(f"Output dir       : {PDF_DIR}")
    print(f"Dry run          : {args.dry_run}")
    print()
    run(args.countries, args.year_from, args.year_to, args.dry_run)


if __name__ == "__main__":
    main()
