"""
Additional unimodal baseline models (numerical only).

All models use canonical pre-alignment (TIME-IMM Appendix H):
  input shape: (B, T+1, 2D+1)  =  (B, 11, 15) with default config

Models
------
  informer   : Transformer encoder with linear head (Zhou et al. 2021)
  patchtst   : Patch-tokenised Transformer (Nie et al. 2023)
  timesnet   : Multi-scale temporal conv (Wu et al. 2023)
               Note: uses Inception-style conv instead of FFT-based 2D reshape
               because T=11 is too short for reliable FFT period detection.
  timemixer  : Decomposable multi-scale MLP mixing (Wang et al. 2024)

Not implemented (require very heavy external deps / pretrained LLMs):
  TimeLLM, TTM         — need HuggingFace LLM weights (~7B params)
  CRU, Latent-ODE,
  Neural Flow          — need torchdiffeq ODE solver
  t-PatchGNN           — needs PyTorch Geometric + graph construction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mmf import _build_numerical_input


# ── Informer ──────────────────────────────────────────────────────────────────

class Informer(nn.Module):
    """
    Transformer encoder with linear head.
    Represents the Informer family (we use standard attention since T=11
    is far too short to benefit from ProbSparse approximation).
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        d_model:      int   = 64,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim = 2 * n_indicators + 1                          # 15
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(seq_len * d_model, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        x = self.input_proj(x)                     # (B, T+1, d_model)
        x = self.encoder(x)                        # (B, T+1, d_model)
        return self.head(x.reshape(x.size(0), -1)) # (B, n_targets)


# ── PatchTST ──────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Patch-tokenised Transformer.
    Splits the time axis into overlapping patches; each patch is a token.
    With seq_len=11, patch_len=4, stride=2 → 4 patches.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        patch_len:    int   = 4,
        stride:       int   = 2,
        d_model:      int   = 64,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim = 2 * n_indicators + 1
        self.patch_len = patch_len
        self.stride    = stride
        n_patches      = (seq_len - patch_len) // stride + 1  # 4 with defaults
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_len * in_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        B, T, F = x.shape

        patches = [
            x[:, i * self.stride : i * self.stride + self.patch_len, :]
              .reshape(B, -1)
            for i in range(self.n_patches)
        ]                                          # list of (B, patch_len*F)
        x_patch = torch.stack(patches, dim=1)      # (B, n_patches, patch_len*F)
        x_patch = self.patch_proj(x_patch) + self.pos_embed  # (B, n_patches, d_model)
        x_patch = self.encoder(x_patch)            # (B, n_patches, d_model)
        return self.head(x_patch.reshape(B, -1))   # (B, n_targets)


# ── TimesNet ──────────────────────────────────────────────────────────────────

class _InceptionBlock(nn.Module):
    """Multi-kernel temporal conv (3, 5, 7) fused by a linear projection."""

    def __init__(self, channels: int):
        super().__init__()
        self.c3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.c5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.c7 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.proj = nn.Linear(channels * 3, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = torch.cat([F.gelu(self.c3(x)),
                         F.gelu(self.c5(x)),
                         F.gelu(self.c7(x))], dim=1)   # (B, 3C, T)
        out = out.permute(0, 2, 1)                      # (B, T, 3C)
        return self.proj(out).permute(0, 2, 1)          # (B, C, T)


class TimesNet(nn.Module):
    """
    TimesNet adapted for short irregular series.
    Uses stacked Inception-style conv blocks with residual connections.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        d_model:      int   = 64,
        n_layers:     int   = 2,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim = 2 * n_indicators + 1
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([_InceptionBlock(d_model) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(seq_len * d_model, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        x = self.input_proj(x)                     # (B, T+1, d_model)
        x = x.permute(0, 2, 1)                    # (B, d_model, T+1)
        for block in self.blocks:
            x = x + block(x)                       # residual
        x = self.norm(x.permute(0, 2, 1))         # (B, T+1, d_model)
        x = self.dropout(x)
        return self.head(x.reshape(x.size(0), -1)) # (B, n_targets)


# ── TimeMixer ─────────────────────────────────────────────────────────────────

class TimeMixer(nn.Module):
    """
    TimeMixer adapted for short irregular series.
    Projects input to d_model, then applies MLP mixing independently at
    three temporal scales (full, half, quarter length). Each scale produces
    a d_model vector; the three are concatenated and projected to n_targets.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        d_model:      int   = 64,
        n_scales:     int   = 3,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim = 2 * n_indicators + 1
        self.n_scales = n_scales
        self.input_proj = nn.Linear(in_dim, d_model)

        # Build per-scale MLP; track sequence lengths after successive halving
        self.scale_mlps = nn.ModuleList()
        scale_len = seq_len
        for _ in range(n_scales):
            self.scale_mlps.append(nn.Sequential(
                nn.Linear(scale_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            scale_len = max(1, scale_len // 2)

        self.head = nn.Linear(d_model * n_scales, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        B = x.size(0)
        x = self.input_proj(x)                     # (B, T+1, d_model)

        scale_outs = []
        curr = x
        for mlp in self.scale_mlps:
            scale_outs.append(mlp(curr.reshape(B, -1)))   # (B, d_model)
            if curr.size(1) > 1:
                curr = F.avg_pool1d(
                    curr.permute(0, 2, 1),
                    kernel_size=2, stride=2,
                ).permute(0, 2, 1)

        fused = torch.cat(scale_outs, dim=-1)      # (B, d_model * n_scales)
        return self.head(fused)                    # (B, n_targets)
