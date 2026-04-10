"""
Text encoding pipeline: Article IV text → 768-dim embeddings.

Modes:
  --mode openai  : text-embedding-3-small via OpenAI API (handles full text, no truncation)
  --mode bert    : BERT-base-uncased (truncates at 512 tokens)
  --mode dummy   : deterministic random vectors (fast, no model needed)

Embeddings saved as:
  data/embeddings/{COUNTRY}_{YEAR}.npy   (float32, shape (768,))

Usage:
  python -m src.pipeline.encode_text --mode openai
  python -m src.pipeline.encode_text --mode bert
  python -m src.pipeline.encode_text --mode dummy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import TEXT_DIR, EMB_DIR, TEXT_EMB_DIM


def _load_env_key() -> str:
    """Load OPENAI_API_KEY from .env file in project root."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    import os
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("[ERROR] OPENAI_API_KEY not found in .env or environment.")
        sys.exit(1)
    return key


def _openai_embeddings(texts: list[str], keys: list[tuple]) -> list[np.ndarray]:
    """
    Embed texts using text-embedding-3-small at 768 dims.
    Handles rate limits with exponential backoff.
    Returns list of (768,) float32 arrays in same order as input.
    """
    from openai import OpenAI

    client = OpenAI(api_key=_load_env_key())
    embeddings = []

    for i, (text, (country, year)) in enumerate(zip(texts, keys)):
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                    dimensions=TEXT_EMB_DIM,
                )
                vec = np.array(resp.data[0].embedding, dtype=np.float32)
                embeddings.append(vec)
                print(f"  [{i+1}/{len(texts)}] {country}_{year}  ({len(text):,} chars)")
                break
            except Exception as e:
                # rate limit → wait longer; other errors → standard backoff
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                wait = 60 if is_rate_limit else 2 ** attempt
                print(f"  [{country}_{year}] attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  [{country}_{year}] all attempts failed, using zero vector")
            embeddings.append(np.zeros(TEXT_EMB_DIM, dtype=np.float32))

        time.sleep(0.5)   # polite delay between requests

    return embeddings


def _dummy_embedding(country: str, year: int) -> np.ndarray:
    """Reproducible pseudo-random embedding (no model needed)."""
    seed = abs(hash(f"{country}_{year}")) % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(TEXT_EMB_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


def _bert_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts with frozen BERT-base-uncased, return (N, 768) float32."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        all_embs.append(out.last_hidden_state.mean(dim=1).cpu().float().numpy())

    return np.vstack(all_embs)


def run(mode: str = "dummy", overwrite: bool = False):
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = TEXT_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"[ERROR] {meta_path} not found. Run scripts/prepare_real_data.py first.")
        sys.exit(1)

    meta = pd.read_csv(meta_path)

    # skip already-encoded files unless overwrite requested
    if not overwrite:
        before = len(meta)
        meta = meta[meta.apply(
            lambda r: not (EMB_DIR / f"{r['country']}_{int(r['ref_year'])}.npy").exists(),
            axis=1
        )]
        if before - len(meta) > 0:
            print(f"Skipping {before - len(meta)} already-encoded files (use --overwrite to redo)")

    if meta.empty:
        print("All embeddings already exist. Done.")
        return

    print(f"Encoding {len(meta)} texts in [{mode}] mode …\n")

    texts, keys = [], []
    for _, row in meta.iterrows():
        txt_file = TEXT_DIR / f"{row['country']}_{int(row['ref_year'])}.txt"
        if txt_file.exists():
            texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
            keys.append((row["country"], int(row["ref_year"])))

    if mode == "openai":
        embeddings = _openai_embeddings(texts, keys)
        for (country, year), emb in zip(keys, embeddings):
            np.save(EMB_DIR / f"{country}_{year}.npy", emb)

    elif mode == "bert":
        emb_matrix = _bert_embeddings(texts)
        for (country, year), emb in zip(keys, emb_matrix):
            np.save(EMB_DIR / f"{country}_{year}.npy", emb)

    else:  # dummy
        for _, row in meta.iterrows():
            country, year = row["country"], int(row["ref_year"])
            np.save(EMB_DIR / f"{country}_{year}.npy", _dummy_embedding(country, year))

    print(f"\nDone. {len(list(EMB_DIR.glob('*.npy')))} embeddings in {EMB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["dummy", "bert", "openai"], default="dummy")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode even if .npy already exists")
    args = parser.parse_args()
    run(args.mode, args.overwrite)
