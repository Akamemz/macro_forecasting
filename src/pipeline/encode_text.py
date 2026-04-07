"""
Text encoding pipeline: Article IV summaries → 768-dim frozen BERT embeddings.

Two modes:
  --mode bert   : use bert-base-uncased (downloads ~440 MB on first run)
  --mode dummy  : deterministic random 768-dim vectors (fast, no download)

Embeddings are saved as:
  data/embeddings/{COUNTRY}_{YEAR}.npy   (float32, shape (768,))

Usage:
  python -m src.pipeline.encode_text --mode dummy
  python -m src.pipeline.encode_text --mode bert
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import TEXT_DIR, EMB_DIR, TEXT_EMB_DIM


def _dummy_embedding(country: str, year: int) -> np.ndarray:
    """Reproducible pseudo-random embedding (no model needed)."""
    seed = abs(hash(f"{country}_{year}")) % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(TEXT_EMB_DIM).astype(np.float32)
    # L2-normalise (BERT [CLS] outputs are typically normalised downstream)
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
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Mean-pool over token dimension
        emb = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
        all_embs.append(emb)

    return np.vstack(all_embs)


def run(mode: str = "dummy"):
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = TEXT_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"[ERROR] {meta_path} not found. Run generate_dummy.py first.")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    print(f"Encoding {len(meta)} summaries in [{mode}] mode …")

    if mode == "bert":
        # Load all texts, batch-encode, then save individually
        texts, keys = [], []
        for _, row in meta.iterrows():
            txt_file = TEXT_DIR.parent / row["summary_path"]
            if txt_file.exists():
                texts.append(txt_file.read_text())
                keys.append((row["country"], int(row["ref_year"])))

        embeddings = _bert_embeddings(texts)
        for (country, year), emb in zip(keys, embeddings):
            out_path = EMB_DIR / f"{country}_{year}.npy"
            np.save(out_path, emb)

    else:  # dummy mode
        for _, row in meta.iterrows():
            country, year = row["country"], int(row["ref_year"])
            emb = _dummy_embedding(country, year)
            out_path = EMB_DIR / f"{country}_{year}.npy"
            np.save(out_path, emb)

    print(f"Saved {len(list(EMB_DIR.glob('*.npy')))} embeddings → {EMB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dummy", "bert"], default="dummy")
    args = parser.parse_args()
    run(args.mode)
