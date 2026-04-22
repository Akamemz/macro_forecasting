"""
Quick diagnostic: open the IMF search page, wait, screenshot, dump API calls and links.
"""
import time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from playwright.sync_api import sync_playwright

URL = "https://www.imf.org/en/publications/search#cf-query=Kazakhstan%20Article%20IV%20consultation&cf-type=CR"

api_calls = []
coveo_requests = []   # capture full request details for Coveo calls

def on_request(req):
    if "coveo.com/rest/search" in req.url:
        coveo_requests.append({
            "url": req.url,
            "method": req.method,
            "headers": dict(req.headers),
            "post_data": req.post_data,
        })

def on_response(resp):
    ct = resp.headers.get("content-type", "")
    if "json" in ct:
        try:
            data = resp.json()
            api_calls.append((resp.url, data))
        except Exception:
            pass

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False)   # visible so you can watch
    ctx = browser.new_context(user_agent=(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ))
    page = ctx.new_page()
    page.on("request", on_request)
    page.on("response", on_response)

    print("Navigating …")
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    print("DOM ready, waiting 10s for JS …")
    time.sleep(10)

    # screenshot
    shot = ROOT / "data" / "debug_search.png"
    page.screenshot(path=str(shot), full_page=True)
    print(f"Screenshot → {shot}")

    # ── dump Coveo request details (token + body) ────────────────────────────
    print(f"\n{len(coveo_requests)} Coveo search request(s) captured:")
    for i, r in enumerate(coveo_requests):
        print(f"\n  [{i}] {r['method']} {r['url'][:100]}")
        auth = r["headers"].get("authorization", r["headers"].get("Authorization", "NOT FOUND"))
        print(f"       Authorization: {auth[:80]}")
        print(f"       POST body: {str(r['post_data'])[:300]}")

    # ── dump Coveo response shape ─────────────────────────────────────────────
    print(f"\nCoveo response shapes:")
    for url, data in api_calls:
        if "coveo.com" in url:
            print(f"  totalCount={data.get('totalCount')}  totalCountFiltered={data.get('totalCountFiltered')}")
            results = data.get("results", [])
            print(f"  results[0] keys: {list(results[0].keys()) if results else 'none'}")
            if results:
                r0 = results[0]
                print(f"  title:  {r0.get('title','')[:80]}")
                print(f"  uri:    {r0.get('uri','')[:80]}")
                print(f"  clickUri: {r0.get('clickUri','')[:80]}")
                raw = r0.get("raw", {})
                print(f"  raw keys: {list(raw.keys())[:12]}")
            break

    # dump all links
    all_links = page.query_selector_all("a[href]")
    cr_links = []
    for a in all_links:
        href = a.get_attribute("href") or ""
        title = a.inner_text().strip()
        if "/CR/" in href or "/cr/" in href:
            cr_links.append((title, href))

    print(f"\nCountry-report links on page: {len(cr_links)}")
    for t, h in cr_links[:15]:
        print(f"  {t[:60]!r}  →  {h[:90]}")

    browser.close()
