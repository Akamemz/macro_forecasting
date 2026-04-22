"""
GR-Add MMF on top of any IMM-TSF backbone.

Architecture (TIME-IMM Appendix I.2, eq. 12-16):
  1. backbone(batch)  →  y_ts  [B, n_targets]     (numerical forecast)
  2. T2VTTF           →  e     [B, TEXT_EMB_DIM]  (text context)
  3. z = [y_ts ; e]
  4. H = GRU({z})
  5. ΔY = W_Δ · H
  6. G  = σ(W_g · z)
  7. y_fused = G ⊙ y_ts + (1−G) ⊙ (y_ts + ΔY)

The backbone can be any wrapper whose forward(batch) → [B, n_targets].
All IMM-TSF wrappers in imm_tsf.py already satisfy this contract.
"""

import torch
import torch.nn as nn

from src.models.ttf import T2VTTF
from src.config import TEXT_EMB_DIM, N_INDICATORS, CONTEXT_WINDOW, TARGET_COLS, HIDDEN_DIM
from src.models.imm_tsf import build_imm_model, _N_PATCHES, _PATCH_LEN, _STRIDE


class IMMGRAddMMF(nn.Module):
    """
    GR-Add MMF wrapping any IMM-TSF backbone.

    Args:
        backbone : an imm_tsf wrapper (already returns [B, n_targets])
        n_targets: number of forecast targets (default 2)
        hidden_dim: GRU hidden size for fusion
    """

    def __init__(
        self,
        backbone:   nn.Module,
        n_targets:  int = 2,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.backbone = backbone

        # TTF: Time2Vec cross-attention text fusion (matches IMM-TSF TTF_T2V_XAttn)
        self.ttf = T2VTTF()

        # GR-Add fusion components (eq. 12-16)
        fusion_dim = n_targets + TEXT_EMB_DIM          # 2 + 768 = 770
        self.fusion_gru = nn.GRU(fusion_dim, hidden_dim, num_layers=1, batch_first=True)
        self.w_delta    = nn.Linear(hidden_dim, n_targets)
        self.w_g        = nn.Linear(fusion_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        # ── step 1: backbone numerical forecast ──────────────────────────────
        y_ts = self.backbone(batch)                    # [B, n_targets]

        # ── step 2: text context via T2V TTF ────────────────────────────────
        e = self.ttf(
            batch["text_embs"],   # [B, N, 768]
            batch["text_ts"],     # [B, N]
            batch["query_ts"],    # [B]
            batch["text_mask"],   # [B, N]
        )                                              # [B, 768]

        # ── steps 3-7: GR-Add fusion ─────────────────────────────────────────
        z = torch.cat([y_ts, e], dim=-1)               # [B, 770]
        _, h = self.fusion_gru(z.unsqueeze(1))         # h: [1, B, hidden]
        h = h.squeeze(0)                               # [B, hidden]

        delta_y = self.w_delta(h)                      # [B, n_targets]
        gate    = torch.sigmoid(self.w_g(z))           # [B, n_targets]

        return gate * y_ts + (1 - gate) * (y_ts + delta_y)


# ── factory ───────────────────────────────────────────────────────────────────

# Maps  "imm_{backbone}_gradd"  →  "imm_{backbone}"
_GRADD_BACKBONES = [
    "imm_dlinear",
    "imm_informer",
    "imm_patchtst",
    "imm_timesnet",
    "imm_timemixer",
    "imm_ttm",
    "imm_timellm",
    "imm_cru",
    "imm_latent_ode",
    "imm_neural_flow",
    "imm_tpatchgnn",
]

# Exported list of all GR-Add model names (for train.py / main_all.py)
IMM_GRADD_MODELS = [b + "_gradd" for b in _GRADD_BACKBONES]


def build_imm_gradd_model(name: str, device) -> nn.Module:
    """
    Build an IMMGRAddMMF model by name, e.g. "imm_dlinear_gradd".
    Strips the "_gradd" suffix, builds the backbone, then wraps with GR-Add.
    """
    if not name.endswith("_gradd"):
        raise ValueError(f"Expected name ending in '_gradd', got: {name}")

    backbone_name = name[: -len("_gradd")]              # e.g. "imm_dlinear"
    backbone = build_imm_model(backbone_name, device)   # existing IMM wrapper
    return IMMGRAddMMF(backbone)
