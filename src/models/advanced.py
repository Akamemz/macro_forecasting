"""
Advanced baseline models.

TimeLLM  : LLM reprogramming for TS forecasting (Zhou et al. 2024)
           Requires: pip install transformers
           Falls back to a small Transformer encoder if not installed.

TTM      : Tiny Time Mixers — stacked TSMixer blocks (Ekambaram et al. 2024)
           Pure PyTorch, no external deps.

LatentODE: Neural ODE latent dynamics (Rubanova et al. 2019)
           Uses manual Euler integration — no torchdiffeq needed.

CRU      : Continuous Recurrent Unit with exponential time-decay gating
           (Schirmer et al. 2022). Pure PyTorch.

tPatchGNN: Patch-based temporal GNN (Sun et al. 2023)
           Manual mean-aggregation GCN — no torch_geometric needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mmf import _build_numerical_input


# ── TimeLLM ───────────────────────────────────────────────────────────────────

class TimeLLM(nn.Module):
    """
    Patch the series → project to LLM hidden space → frozen LLM → forecast.

    Uses a small GPT-2 config (d_llm=256, 2 layers) so it runs on CPU without
    needing pretrained weights. Install `transformers` for the GPT-2 backend;
    otherwise falls back to a standard Transformer encoder of the same size.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        patch_len:    int   = 4,
        stride:       int   = 2,
        d_llm:        int   = 256,
        n_llm_layers: int   = 2,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim     = 2 * n_indicators + 1
        self.patch_len  = patch_len
        self.stride     = stride
        n_patches  = (seq_len - patch_len) // stride + 1
        self.n_patches  = n_patches

        # Reprogramming: flattened patch → LLM hidden dim
        self.reprog = nn.Linear(patch_len * in_dim, d_llm)

        # LLM backbone (frozen after init)
        try:
            from transformers import GPT2Model, GPT2Config
            cfg = GPT2Config(
                n_embd=d_llm, n_layer=n_llm_layers, n_head=4,
                n_positions=64, vocab_size=50257,
                resid_pdrop=0.0, attn_pdrop=0.0,
            )
            self.llm      = GPT2Model(cfg)
            self._gpt2    = True
        except ImportError:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_llm, nhead=4, dim_feedforward=d_llm * 2, batch_first=True,
            )
            self.llm   = nn.TransformerEncoder(enc_layer, num_layers=n_llm_layers)
            self._gpt2 = False

        for p in self.llm.parameters():
            p.requires_grad = False

        self.head = nn.Linear(n_patches * d_llm, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)                      # (B, T+1, 2D+1)
        B, T, F = x.shape

        patches = [
            x[:, i * self.stride : i * self.stride + self.patch_len, :]
              .reshape(B, -1)
            for i in range(self.n_patches)
        ]
        x_p = torch.stack(patches, dim=1)                      # (B, n_patches, patch*F)
        x_p = self.reprog(x_p)                                 # (B, n_patches, d_llm)

        if self._gpt2:
            out = self.llm(inputs_embeds=x_p).last_hidden_state
        else:
            out = self.llm(x_p)                                # (B, n_patches, d_llm)

        return self.head(out.reshape(B, -1))                   # (B, n_targets)


# ── TTM ───────────────────────────────────────────────────────────────────────

class _TSMixerBlock(nn.Module):
    """Time-mix MLP + channel-mix MLP with residuals and LayerNorm."""

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1    = nn.LayerNorm(d_model)
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
        )
        self.norm2    = nn.LayerNorm(d_model)
        self.chan_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        r = self.time_mix(self.norm1(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = x + r
        x = x + self.chan_mix(self.norm2(x))
        return x


class TTM(nn.Module):
    """
    Tiny Time Mixers: stacked TSMixer blocks (IBM Research, 2024).
    Pure PyTorch — no external dependencies.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        d_model:      int   = 64,
        n_blocks:     int   = 2,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim = 2 * n_indicators + 1
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks     = nn.ModuleList([
            _TSMixerBlock(seq_len, d_model, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(seq_len * d_model, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        x = self.input_proj(x)                     # (B, T+1, d_model)
        for block in self.blocks:
            x = block(x)
        return self.head(x.reshape(x.size(0), -1)) # (B, n_targets)


# ── LatentODE ─────────────────────────────────────────────────────────────────

class _ODEFunc(nn.Module):
    """Autonomous ODE function: dz/dt = f(z)."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentODE(nn.Module):
    """
    Latent-ODE: encode observations with GRU → project to latent z0 →
    integrate dz/dt = f(z) via Euler steps → decode to forecast.

    Pure PyTorch — manual Euler integration, no torchdiffeq required.
    """

    def __init__(
        self,
        n_indicators: int   = 7,
        latent_dim:   int   = 32,
        hidden_dim:   int   = 64,
        ode_steps:    int   = 10,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim         = 2 * n_indicators + 1
        self.ode_steps = ode_steps

        self.encoder   = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.ode_func  = _ODEFunc(latent_dim, hidden_dim)
        self.decoder   = nn.Linear(latent_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)

        # Encode context (exclude query row appended at position -1)
        _, h = self.encoder(x[:, :-1, :])          # h: (1, B, hidden)
        z    = self.to_latent(h.squeeze(0))        # (B, latent_dim)

        # Euler ODE integration: 0 → 1  (normalised time horizon)
        dt = 1.0 / self.ode_steps
        for _ in range(self.ode_steps):
            z = z + dt * self.ode_func(z)

        return self.decoder(z)                     # (B, n_targets)


# ── CRU ───────────────────────────────────────────────────────────────────────

class CRU(nn.Module):
    """
    Continuous Recurrent Unit (simplified).

    At each observation the hidden state is first exponentially decayed
    proportionally to the elapsed time Δt, then updated by a GRU cell.
    This lets the model explicitly represent the passage of time between
    irregular observations without an ODE solver.
    """

    def __init__(
        self,
        n_indicators: int   = 7,
        hidden_dim:   int   = 64,
        n_targets:    int   = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()
        in_dim = 2 * n_indicators     # values + mask (timestamp used for Δt gating)
        self.hidden_dim  = hidden_dim
        self.gru_cell    = nn.GRUCell(in_dim, hidden_dim)
        self.decay_gate  = nn.Linear(1, hidden_dim)   # Δt → per-dim decay rate
        self.dropout     = nn.Dropout(dropout)
        self.head        = nn.Linear(hidden_dim, n_targets)

    def forward(self, batch: dict) -> torch.Tensor:
        values     = batch["values"]      # (B, T, D)
        mask       = batch["mask"]        # (B, T, D)
        timestamps = batch["timestamps"]  # (B, T)  normalised years
        B, T, D    = values.shape

        h = torch.zeros(B, self.hidden_dim, device=values.device)

        for t in range(T):
            # Exponential time-decay between steps
            if t > 0:
                dt    = (timestamps[:, t] - timestamps[:, t - 1]).unsqueeze(-1)  # (B,1)
                decay = torch.exp(-F.softplus(self.decay_gate(dt)))               # (B, H)
                h     = h * decay

            inp = torch.cat([values[:, t, :], mask[:, t, :]], dim=-1)            # (B, 2D)
            h   = self.gru_cell(inp, h)

        return self.head(self.dropout(h))            # (B, n_targets)


# ── t-PatchGNN ────────────────────────────────────────────────────────────────

class tPatchGNN(nn.Module):
    """
    t-PatchGNN (simplified): tokenise the series into overlapping patches,
    build a symmetric k-nearest-neighbour temporal graph, apply stacked
    mean-aggregation graph convolution, then pool and forecast.

    Pure PyTorch — manual sparse-free message passing, no torch_geometric.
    """

    def __init__(
        self,
        seq_len:      int   = 11,
        n_indicators: int   = 7,
        patch_len:    int   = 4,
        stride:       int   = 2,
        d_model:      int   = 64,
        n_gnn_layers: int   = 2,
        k_neighbours: int   = 2,
        dropout:      float = 0.1,
        n_targets:    int   = 2,
    ):
        super().__init__()
        in_dim            = 2 * n_indicators + 1
        self.patch_len    = patch_len
        self.stride       = stride
        self.k_neighbours = k_neighbours
        n_patches         = (seq_len - patch_len) // stride + 1
        self.n_patches    = n_patches

        self.patch_proj = nn.Linear(patch_len * in_dim, d_model)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_gnn_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_gnn_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(n_patches * d_model, n_targets)

        # Fixed row-normalised temporal adjacency (built once, moved to device on first call)
        self.register_buffer("adj", self._build_adj(n_patches, k_neighbours))

    @staticmethod
    def _build_adj(n: int, k: int) -> torch.Tensor:
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(max(0, i - k), min(n, i + k + 1)):
                adj[i, j] = 1.0
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg                            # row-normalised

    def forward(self, batch: dict) -> torch.Tensor:
        x = _build_numerical_input(batch)          # (B, T+1, 2D+1)
        B, T, F = x.shape

        patches = [
            x[:, i * self.stride : i * self.stride + self.patch_len, :]
              .reshape(B, -1)
            for i in range(self.n_patches)
        ]
        h = self.patch_proj(torch.stack(patches, dim=1))       # (B, n_patches, d_model)

        for lin, norm in zip(self.gnn_layers, self.norms):
            agg = torch.bmm(self.adj.unsqueeze(0).expand(B, -1, -1), h)  # (B, n, d)
            h   = h + F.gelu(norm(lin(agg)))                              # residual

        return self.head(self.dropout(h).reshape(B, -1))       # (B, n_targets)
