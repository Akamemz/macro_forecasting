"""
TTF Module — T2V-XAttn (Time2Vec + Cross-Attention Text-Time Fusion).

Adapted from IMM-TSF fusions/TTF_T2V_XAttn.py for single-step output.

Architecture:
  1. tau_feat  = Time2Vec(text_ts)              [B, N, d_tau]
  2. V_fused   = cat([text_embs, tau_feat])     [B, N, d_txt + d_tau]
  3. KV        = KV_proj(V_fused)               [B, N, d_txt]
  4. Q         = Q_param (learnable)            [1, 1, d_txt]
  5. E_attn    = MultiheadAttention(Q, KV, KV)  [B, 1, d_txt]
  6. E_resid   = LayerNorm(E_attn + Q)
  7. ctx       = proj_out(dropout(E_resid))     [B, d_txt]
  8. zero out  ctx where no text present
"""

import torch
import torch.nn as nn

from src.config import TEXT_EMB_DIM


class Time2Vec(nn.Module):
    """Scalar → d_tau-dim time embedding: [linear, sin, sin, ...]."""

    def __init__(self, d_tau: int):
        super().__init__()
        assert d_tau > 1, "d_tau must be > 1"
        self.linear   = nn.Linear(1, 1)            # ω₀, φ₀  (trend)
        self.periodic = nn.Linear(1, d_tau - 1)    # ω₁..ωₖ₋₁, φ₁..φₖ₋₁

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        lin = self.linear(x)                       # (..., 1)
        per = torch.sin(self.periodic(x))          # (..., d_tau-1)
        return torch.cat([lin, per], dim=-1)       # (..., d_tau)


class T2VTTF(nn.Module):
    """
    Time2Vec cross-attention over text embeddings (matches IMM-TSF TTF_T2V_XAttn).

    T2V is applied to text timestamps and concatenated with text embeddings to
    form time-aware K/V. A learnable query attends over these to produce a
    single context vector.

    Args:
        d_txt     : text embedding dim (default TEXT_EMB_DIM = 768).
        n_heads   : attention heads.
        dropout   : dropout on residual output.
    """

    def __init__(
        self,
        d_txt:   int   = TEXT_EMB_DIM,
        n_heads: int   = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_txt = d_txt
        self.d_tau = d_txt // 2                              # matches paper: d_tau = d_txt // 2

        self.time2vec = Time2Vec(self.d_tau)
        # Project [text_emb ; time_feat] → d_txt for K/V
        self.KV_proj  = nn.Linear(d_txt + self.d_tau, d_txt)
        # Learnable fixed query (like the paper's Q_param)
        self.Q_param  = nn.Parameter(torch.randn(1, 1, d_txt))

        self.attn = nn.MultiheadAttention(
            embed_dim=d_txt, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_txt)
        self.dropout    = nn.Dropout(dropout)
        self.proj_out   = nn.Linear(d_txt, d_txt)

    def forward(
        self,
        text_embs: torch.Tensor,   # (B, N, d_txt)
        text_ts:   torch.Tensor,   # (B, N)   normalised publication dates
        query_ts:  torch.Tensor,   # (B,)     normalised query year (unused in Q)
        text_mask: torch.Tensor,   # (B, N) bool  True = real, False = padding
    ) -> torch.Tensor:             # (B, d_txt)
        B, N, _ = text_embs.shape

        # 1) Time2Vec on text timestamps → time-aware K/V
        tau_feat = self.time2vec(text_ts.unsqueeze(-1))           # (B, N, d_tau)
        V_fused  = torch.cat([text_embs, tau_feat], dim=-1)       # (B, N, d_txt + d_tau)
        KV       = self.KV_proj(V_fused)                          # (B, N, d_txt)

        # 2) Learnable query (broadcast over batch)
        Q = self.Q_param.expand(B, 1, self.d_txt)                 # (B, 1, d_txt)

        # 3) Padding mask for attention (True = ignore)
        key_padding_mask = ~text_mask                             # (B, N)

        # 4) Cross-attention
        E_attn, _ = self.attn(Q, KV, KV, key_padding_mask=key_padding_mask)  # (B, 1, d_txt)

        # 5) Residual + LayerNorm + dropout + projection
        Q2      = self.Q_param.expand(B, 1, self.d_txt)
        E_resid = self.layer_norm(E_attn + Q2)                    # (B, 1, d_txt)
        ctx     = self.proj_out(self.dropout(E_resid)).squeeze(1) # (B, d_txt)

        # 6) Zero out samples with no text
        has_text = text_mask.any(dim=1)                           # (B,)
        ctx = torch.where(has_text.unsqueeze(1), ctx, torch.zeros_like(ctx))

        return ctx
