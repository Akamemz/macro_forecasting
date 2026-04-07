"""
MMF Module — GR-Add (GRU-Gated Residual Addition).

Architecture:
  1. Numerical encoder: GRU over the imputed + masked time series
     Input per timestep: cat(values, mask)  →  (T, 2D)
     Output: final hidden state h  →  (B, hidden_dim)
     Base prediction:  num_pred = Linear(hidden_dim, n_targets)

  2. Text projection: RecAvg context (B, 768) → Linear → (B, hidden_dim)

  3. GRU-Gated Residual Addition:
     fused   = cat(h, text_proj)      →  (B, 2*hidden_dim)
     gate    = sigmoid(W_g @ fused)   →  (B, n_targets)  ∈ (0, 1)
     delta   = tanh(W_d @ fused)      →  (B, n_targets)
     output  = num_pred + gate * delta

The gate learns how much text context should correct the base numerical
forecast — if text adds no information the gate collapses to ~0.
"""

import torch
import torch.nn as nn

from src.models.ttf import RecAvgTTF
from src.config import TEXT_EMB_DIM, SIGMA


class GRAddMMF(nn.Module):
    """
    Full multimodal model: numerical GRU encoder + RecAvg TTF + GR-Add fusion.

    Args:
        n_indicators: D (number of WDI indicators, default 7)
        hidden_dim:   GRU hidden size
        n_targets:    number of forecast targets (default 2: GDP growth + inflation)
        sigma:        Gaussian kernel width for RecAvg (normalised-year units)
        dropout:      applied after GRU hidden state
    """

    def __init__(
        self,
        n_indicators: int = 7,
        hidden_dim:   int = 64,
        n_targets:    int = 2,
        sigma:        float = SIGMA,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.n_targets   = n_targets

        # Numerical encoder — input = (values || mask) per step
        self.num_gru = nn.GRU(
            input_size=2 * n_indicators,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Base prediction head (numerical only)
        self.num_head = nn.Linear(hidden_dim, n_targets)

        # TTF module (no learnable params)
        self.ttf = RecAvgTTF(sigma=sigma)

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_EMB_DIM, hidden_dim),
            nn.Tanh(),
        )

        # GR-Add gate and delta
        fused_dim = 2 * hidden_dim
        self.gate  = nn.Linear(fused_dim, n_targets)
        self.delta = nn.Linear(fused_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from macro_collate with keys
                   values, mask, timestamps, query_ts,
                   text_embs, text_ts, text_mask
        Returns:
            pred: (B, n_targets)
        """
        values     = batch["values"]      # (B, T, D)
        mask       = batch["mask"]        # (B, T, D)
        text_embs  = batch["text_embs"]  # (B, N, 768)
        text_ts    = batch["text_ts"]    # (B, N)
        text_mask  = batch["text_mask"]  # (B, N)
        query_ts   = batch["query_ts"]   # (B,)

        # ── numerical path ───────────────────────────────────────────────────
        inp = torch.cat([values, mask], dim=-1)   # (B, T, 2D)
        _, h_n = self.num_gru(inp)                # h_n: (1, B, hidden)
        h = self.dropout(h_n.squeeze(0))          # (B, hidden)
        num_pred = self.num_head(h)               # (B, n_targets)

        # ── text context ─────────────────────────────────────────────────────
        ctx       = self.ttf(text_embs, text_ts, query_ts, text_mask)  # (B, 768)
        text_proj = self.text_proj(ctx)                                  # (B, hidden)

        # ── GR-Add fusion ────────────────────────────────────────────────────
        fused = torch.cat([h, text_proj], dim=-1)   # (B, 2*hidden)
        gate  = torch.sigmoid(self.gate(fused))      # (B, n_targets)
        delta = torch.tanh(self.delta(fused))        # (B, n_targets)
        pred  = num_pred + gate * delta              # (B, n_targets)

        return pred


class NumericalOnlyGRU(nn.Module):
    """
    Unimodal baseline: same GRU encoder as GRAddMMF but ignores text.
    Used for ablation (Text removed → pure numerical).
    """

    def __init__(self, n_indicators: int = 7, hidden_dim: int = 64, n_targets: int = 2):
        super().__init__()
        self.gru  = nn.GRU(2 * n_indicators, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        values = batch["values"]
        mask   = batch["mask"]
        inp    = torch.cat([values, mask], dim=-1)
        _, h_n = self.gru(inp)
        return self.head(h_n.squeeze(0))
