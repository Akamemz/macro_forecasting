"""
MMF Module — GR-Add (GRU-Gated Residual Addition).

Implements the architecture from TIME-IMM Appendix I.2 (eq. 12–16), with
canonical pre-alignment (Appendix H Steps 3-4) applied to the backbone GRU.

Pre-alignment (applied in both GRAddMMF and NumericalOnlyGRU):
  Step 3 — input per timestep: [values | mask | norm_timestamp]  →  (2D+1) features
  Step 4 — query row appended: [0(D) | 0(D) | norm_query_ts]    →  seq_len T+1

GR-Add architecture (eq. 12–16):
  1. Backbone GRU over pre-aligned numerical input → base forecast y_ts
  2. RecAvg TTF over Article IV embeddings → text context e  (shape: 768)
  3. Fusion GRU over z = [y_ts ; e]                        (eq. 12-13)
  4. ΔY = W_Δ · H + b_Δ  (linear correction)              (eq. 14)
  5. G  = σ(W_g · [y_ts ; e] + b_g)  (gate)               (eq. 15)
  6. y_fused = G ⊙ y_ts + (1−G) ⊙ (y_ts + ΔY)            (eq. 16)

When text is absent (e = 0), ΔY and G become deterministic functions of y_ts
alone and the gate is free to suppress the correction entirely.
"""

import torch
import torch.nn as nn

from src.models.ttf import T2VTTF
from src.config import TEXT_EMB_DIM


# ── shared pre-alignment helper ───────────────────────────────────────────────

def _build_numerical_input(batch: dict) -> torch.Tensor:
    """
    Appendix H Steps 3-4: build (B, T+1, 2D+1) input tensor from a batch dict.

      Context rows (T):  [values | mask | norm_timestamp]
      Query row    (1):  [zeros  | zeros | norm_query_ts ]
    """
    values     = batch["values"]      # (B, T, D)
    mask       = batch["mask"]        # (B, T, D)
    timestamps = batch["timestamps"]  # (B, T)
    query_ts   = batch["query_ts"]    # (B,)
    B, T, D    = values.shape

    # Step 3: expand features to 2D+1
    ctx = torch.cat([values, mask, timestamps.unsqueeze(-1)], dim=-1)  # (B, T, 2D+1)

    # Step 4: append query row
    q_row = torch.zeros(B, 1, 2 * D + 1, device=values.device)
    q_row[:, 0, -1] = query_ts                                          # timestamp feature

    return torch.cat([ctx, q_row], dim=1)                               # (B, T+1, 2D+1)


# ── models ────────────────────────────────────────────────────────────────────

class GRAddMMF(nn.Module):
    """
    Full multimodal model: backbone GRU + RecAvg TTF + GR-Add fusion.

    Args:
        n_indicators: D — number of WDI indicators (default 7)
        hidden_dim:   GRU hidden size
        n_targets:    number of forecast targets (default 2: GDP + inflation)
        sigma:        Gaussian kernel width for RecAvg TTF
        dropout:      applied after backbone GRU hidden state
    """

    def __init__(
        self,
        n_indicators: int   = 7,
        hidden_dim:   int   = 64,
        n_targets:    int   = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.n_targets = n_targets

        # ── backbone numerical GRU ────────────────────────────────────────────
        # Input per step: [values | mask | timestamp] = 2D+1 features
        self.num_gru  = nn.GRU(2 * n_indicators + 1, hidden_dim,
                               num_layers=1, batch_first=True)
        self.dropout  = nn.Dropout(dropout)
        self.num_head = nn.Linear(hidden_dim, n_targets)   # → y_ts

        # ── TTF module ────────────────────────────────────────────────────────
        self.ttf = T2VTTF()

        # ── GR-Add fusion GRU (eq. 12-13) ────────────────────────────────────
        # Input: z_k = [y^ts_k ; e_k]  ∈ R^(n_targets + TEXT_EMB_DIM)
        fusion_dim = n_targets + TEXT_EMB_DIM                # 2 + 768 = 770
        self.fusion_gru = nn.GRU(fusion_dim, hidden_dim,
                                 num_layers=1, batch_first=True)

        # ΔY = W_Δ · H + b_Δ  (eq. 14) — linear projection from GRU hidden
        self.w_delta = nn.Linear(hidden_dim, n_targets)

        # G = σ(W_g · [Y^ts ; E] + b_g)  (eq. 15)
        self.w_g = nn.Linear(fusion_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from macro_collate — keys:
                   values, mask, timestamps, query_ts,
                   text_embs, text_ts, text_mask
        Returns:
            y_fused: (B, n_targets)
        """
        text_embs = batch["text_embs"]   # (B, N, 768)
        text_ts   = batch["text_ts"]     # (B, N)
        text_mask = batch["text_mask"]   # (B, N)
        query_ts  = batch["query_ts"]    # (B,)

        # ── backbone numerical path ───────────────────────────────────────────
        x = _build_numerical_input(batch)            # (B, T+1, 2D+1)
        _, h_n = self.num_gru(x)                     # h_n: (1, B, hidden)
        h = self.dropout(h_n.squeeze(0))             # (B, hidden)
        y_ts = self.num_head(h)                      # (B, n_targets)

        # ── text context via TTF ──────────────────────────────────────────────
        e = self.ttf(text_embs, text_ts, query_ts, text_mask)   # (B, 768)

        # ── GR-Add fusion (eq. 12–16) ─────────────────────────────────────────
        # z_k = [y^ts ; e]  for the single query step k=1
        z = torch.cat([y_ts, e], dim=-1)             # (B, n_targets+768)

        # H = GRU({z_k})  — one step → take hidden state
        _, h_fused = self.fusion_gru(z.unsqueeze(1)) # h_fused: (1, B, hidden)
        h_fused = h_fused.squeeze(0)                 # (B, hidden)

        # ΔY = W_Δ H + b_Δ  (linear, not tanh — per eq. 14)
        delta_y = self.w_delta(h_fused)              # (B, n_targets)

        # G = σ(W_g · [Y^ts ; E] + b_g)
        gate = torch.sigmoid(self.w_g(z))            # (B, n_targets)

        # Y_fused = G ⊙ Y^ts + (1−G) ⊙ (Y^ts + ΔY)
        y_fused = gate * y_ts + (1 - gate) * (y_ts + delta_y)

        return y_fused                               # (B, n_targets)


class NumericalOnlyGRU(nn.Module):
    """
    Unimodal ablation: GRU backbone with canonical pre-alignment, no text.

    Uses the same backbone as GRAddMMF (pre-aligned input: 2D+1 features,
    T+1 steps with query row appended) so the two models are directly comparable.
    """

    def __init__(self, n_indicators: int = 7, hidden_dim: int = 64, n_targets: int = 2):
        super().__init__()
        self.gru  = nn.GRU(2 * n_indicators + 1, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)   # (B, T+1, 2D+1)
        _, h_n = self.gru(x)
        return self.head(h_n.squeeze(0))
