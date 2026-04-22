"""Check imfisocode field value and find best Article IV filter for Armenia/Moldova."""
import requests

COVEO_ENDPOINT = "https://imfproduction561s308u.org.coveo.com/rest/search/v2"
COVEO_ORG      = "imfproduction561s308u"
TOKEN          = "xx742a6c66-f427-4f5a-ae1e-770dc7264e8a"
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def search(body):
    r = requests.post(f"{COVEO_ENDPOINT}?organizationId={COVEO_ORG}", headers=H, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

# ── 1. Check imfisocode for Armenia ──────────────────────────────────────────
print("=== 1. imfisocode for Armenia result ===")
d = search({
    "q": "Republic of Armenia Article IV",
    "numberOfResults": 1,
    "fieldsToInclude": ["title", "imfisocode", "imfcountry", "imfformalcountry", "imfcontenttype", "imftype", "imfseries"],
})
for r in d.get("results", []):
    raw = r.get("raw", {})
    print(f"  title          : {r['title'][:70]}")
    print(f"  imfisocode     : {raw.get('imfisocode')}")
    print(f"  imfcountry     : {raw.get('imfcountry')}")
    print(f"  imfformalcountry: {raw.get('imfformalcountry')}")
    print(f"  imfcontenttype : {raw.get('imfcontenttype')}")
    print(f"  imftype        : {raw.get('imftype')}")
    print(f"  imfseries      : {raw.get('imfseries')}")

# ── 2. Filter by imfisocode — try ISO3 ───────────────────────────────────────
print("\n=== 2. Filter @imfisocode==ARM ===")
d = search({
    "q": "",
    "aq": '@imfisocode=="ARM"',
    "numberOfResults": 5,
    "fieldsToInclude": ["title", "imfisocode", "imfcontenttype", "date"],
})
print(f"Total: {d.get('totalCountFiltered')}")
for r in d.get("results", []):
    raw = r.get("raw", {})
    print(f"  [{raw.get('date','')}] {r['title'][:65]}  type={raw.get('imfcontenttype')}")

# ── 3. Filter by imfisocode — try ISO2 ───────────────────────────────────────
print("\n=== 3. Filter @imfisocode==AM ===")
d = search({
    "q": "",
    "aq": '@imfisocode=="AM"',
    "numberOfResults": 5,
    "fieldsToInclude": ["title", "imfisocode", "imfcontenttype", "date"],
})
print(f"Total: {d.get('totalCountFiltered')}")
for r in d.get("results", []):
    raw = r.get("raw", {})
    print(f"  [{raw.get('date','')}] {r['title'][:65]}  type={raw.get('imfcontenttype')}")

# ── 4. imfisocode==ARM without content type filter — what types exist? ────────
print("\n=== 4. @imfisocode==ARM — all content types ===")
d = search({
    "q": "Article IV",
    "aq": '@imfisocode=="ARM"',
    "numberOfResults": 10,
    "fieldsToInclude": ["title", "imfcontenttype", "date"],
})
print(f"Total: {d.get('totalCountFiltered')}")
for r in d.get("results", []):
    raw = r.get("raw", {})
    print(f"  [{raw.get('date','')}] ct={str(raw.get('imfcontenttype',''))[:40]}  {r['title'][:55]}")
