"""
XAttn-Add MMF on top of any IMM-TSF backbone.

Mirrors TIME-IMM's MMF_XAttn_Add (fusions/MMF_XAttn_Add.py) but adapted
for our single-step output format [B, n_targets] instead of [B, T, C].

Architecture:
  1. backbone(batch)  →  y_ts  [B, n_targets]
  2. T2VTTF           →  e     [B, TEXT_EMB_DIM]
  3. Q = proj_q(y_ts.unsqueeze(1))          [B, 1, d_attn]
     K = proj_k(e.unsqueeze(1))             [B, 1, d_attn]
     V = proj_v(e.unsqueeze(1))             [B, 1, d_attn]
  4. attn_out = MultiheadAttention(Q, K, V) [B, 1, d_attn]
  5. ΔY = residual_head(attn_out).squeeze(1)[B, n_targets]
  6. ΔY zeroed where no text present
  7. Y_fused = (y_ts + κ * ΔY) / (1 + κ)
"""

import torch
import torch.nn as nn

from src.models.ttf import T2VTTF
from src.config import TEXT_EMB_DIM
from src.models.imm_tsf import build_imm_model
from src.models.imm_gr_add import _GRADD_BACKBONES


class IMMXAttnAddMMF(nn.Module):
    """
    XAttn-Add MMF wrapping any IMM-TSF backbone.

    Args:
        backbone   : imm_tsf wrapper returning [B, n_targets]
        n_targets  : forecast output dim (default 2)
        d_attn     : attention projection dim
        n_heads    : attention heads
        kappa      : text residual weight  (Y_fused = (y_ts + κΔY)/(1+κ))
        dropout    : dropout on residual
    """

    def __init__(
        self,
        backbone:  nn.Module,
        n_targets: int   = 2,
        d_attn:    int   = 64,
        n_heads:   int   = 1,
        kappa:     float = 1.0,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.kappa    = kappa

        self.ttf = T2VTTF()

        # Q from y_ts, K/V from text embedding
        self.proj_q = nn.Linear(n_targets,    d_attn, bias=False)
        self.proj_k = nn.Linear(TEXT_EMB_DIM, d_attn, bias=False)
        self.proj_v = nn.Linear(TEXT_EMB_DIM, d_attn, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        self.residual_head = nn.Linear(d_attn, n_targets)
        self.layer_norm    = nn.LayerNorm(n_targets)
        self.dropout       = nn.Dropout(dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        # ── backbone forecast ─────────────────────────────────────────────────
        y_ts = self.backbone(batch)                        # [B, n_targets]

        # ── text context ──────────────────────────────────────────────────────
        e = self.ttf(
            batch["text_embs"],
            batch["text_ts"],
            batch["query_ts"],
            batch["text_mask"],
        )                                                  # [B, 768]

        # text presence mask: True if at least one real text token exists
        has_text = batch["text_mask"].any(dim=1)           # [B]

        # ── cross-attention ───────────────────────────────────────────────────
        Q = self.proj_q(y_ts.unsqueeze(1))                 # [B, 1, d_attn]
        K = self.proj_k(e.unsqueeze(1))                    # [B, 1, d_attn]
        V = self.proj_v(e.unsqueeze(1))                    # [B, 1, d_attn]

        attn_out, _ = self.attn(Q, K, V)                   # [B, 1, d_attn]

        # zero out attention for samples with no text
        mask = has_text.view(-1, 1, 1)                     # [B, 1, 1]
        attn_out = torch.where(mask, attn_out, torch.zeros_like(attn_out))

        # ── residual correction ───────────────────────────────────────────────
        delta_y = self.residual_head(attn_out).squeeze(1)  # [B, n_targets]
        delta_y = self.dropout(self.layer_norm(delta_y))

        # zero out residual for no-text samples
        delta_y = torch.where(has_text.unsqueeze(1), delta_y, torch.zeros_like(delta_y))

        # ── fuse ─────────────────────────────────────────────────────────────
        return (y_ts + self.kappa * delta_y) / (1.0 + self.kappa)


# ── factory ───────────────────────────────────────────────────────────────────

IMM_XATTN_MODELS = [b + "_xattn" for b in _GRADD_BACKBONES]


def build_imm_xattn_model(name: str, device) -> nn.Module:
    if not name.endswith("_xattn"):
        raise ValueError(f"Expected name ending in '_xattn', got: {name}")
    backbone_name = name[: -len("_xattn")]
    backbone = build_imm_model(backbone_name, device)
    return IMMXAttnAddMMF(backbone)
