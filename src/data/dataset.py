"""
PyTorch Dataset wrapper for pre-aligned macro samples.

Handles variable-length text sequence padding within each batch
via a custom collate function.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MacroDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "values":     torch.from_numpy(s["values"]),      # (T, D)
            "mask":       torch.from_numpy(s["mask"]),        # (T, D)
            "timestamps": torch.from_numpy(s["timestamps"]),  # (T,)
            "query_ts":   torch.tensor(s["query_ts"]),        # scalar
            "text_embs":  torch.from_numpy(s["text_embs"]),   # (N, 768)
            "text_ts":    torch.from_numpy(s["text_ts"]),     # (N,)
            "target":     torch.from_numpy(s["target"]),      # (2,)
        }


def macro_collate(batch: list[dict]) -> dict:
    """
    Collate with zero-padding for variable-length text sequences.
    Returns tensors with an extra 'text_mask' boolean tensor: True = real entry.
    """
    max_n = max(b["text_embs"].shape[0] for b in batch)
    emb_dim = batch[0]["text_embs"].shape[1]

    padded_embs    = []
    padded_ts      = []
    text_mask_list = []

    for b in batch:
        n = b["text_embs"].shape[0]
        pad = max_n - n
        e = torch.cat([b["text_embs"], torch.zeros(pad, emb_dim)], dim=0)
        t = torch.cat([b["text_ts"],   torch.zeros(pad)], dim=0)
        m = torch.cat([torch.ones(n, dtype=torch.bool),
                       torch.zeros(pad, dtype=torch.bool)], dim=0)
        padded_embs.append(e)
        padded_ts.append(t)
        text_mask_list.append(m)

    return {
        "values":     torch.stack([b["values"]     for b in batch]),   # (B, T, D)
        "mask":       torch.stack([b["mask"]       for b in batch]),   # (B, T, D)
        "timestamps": torch.stack([b["timestamps"] for b in batch]),   # (B, T)
        "query_ts":   torch.stack([b["query_ts"]   for b in batch]),   # (B,)
        "text_embs":  torch.stack(padded_embs),                        # (B, N, 768)
        "text_ts":    torch.stack(padded_ts),                          # (B, N)
        "text_mask":  torch.stack(text_mask_list),                     # (B, N)
        "target":     torch.stack([b["target"]     for b in batch]),   # (B, 2)
    }


def make_loaders(
    train_samples: list,
    val_samples: list,
    test_samples: list,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        MacroDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=macro_collate,
    )
    val_loader = DataLoader(
        MacroDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=macro_collate,
    )
    test_loader = DataLoader(
        MacroDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=macro_collate,
    )
    return train_loader, val_loader, test_loader
