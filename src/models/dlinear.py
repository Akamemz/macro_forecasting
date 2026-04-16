"""
DLinear Baseline — unimodal numerical-only.

Adapted from "Are Transformers Effective for Time Series Forecasting?" (Zeng et al. 2023).

Follows TIME-IMM Appendix H canonical pre-alignment:
  Step 3 — each timestep's feature vector is [values, mask, normalised_timestamp],
            expanding D → 2D+1.
  Step 4 — the query timestamp is appended as one extra row with zero values/mask
            and the normalised query year as the timestamp feature, so the model
            sees WHERE in time it is predicting.

After pre-alignment the input is (B, T+1, 2D+1). We then:
  1. Decompose into trend (moving average) + seasonal residual.
  2. Apply independent linear layers to the flattened trend and seasonal.
  3. Sum to produce the forecast.

Because our series are short (T=10, T+1=11) we use a kernel of 3 for the moving average.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """Centred moving average with reflection padding."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        pad = self.kernel_size // 2
        x_pad = F.pad(x.permute(0, 2, 1), (pad, pad), mode="reflect")  # (B, F, T+2p)
        avg = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1, padding=0)
        return avg.permute(0, 2, 1)  # (B, T, F)


class DLinear(nn.Module):
    """
    DLinear adapted for irregular multivariate series with canonical pre-alignment.

    Args:
        seq_len:      context length T (default 10 years)
        n_indicators: number of WDI features D (default 7)
        n_targets:    forecast outputs (default 2)
    """

    def __init__(
        self,
        seq_len:      int = 10,
        n_indicators: int = 7,
        n_targets:    int = 2,
    ):
        super().__init__()
        self.seq_len      = seq_len
        self.n_indicators = n_indicators

        # After pre-alignment (Steps 3-4): (T+1) timesteps × (2D+1) features
        in_dim = (seq_len + 1) * (2 * n_indicators + 1)

        self.moving_avg = MovingAvg(kernel_size=3)

        # Separate linear layers for trend and seasonal
        self.linear_trend    = nn.Linear(in_dim, n_targets)
        self.linear_seasonal = nn.Linear(in_dim, n_targets)

    # ── pre-alignment helpers ─────────────────────────────────────────────────

    def _impute(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace missing value entries with per-column observed mean (or 0)."""
        obs_sum   = (values * mask).sum(dim=1)          # (B, D)
        obs_count = mask.sum(dim=1).clamp(min=1.0)      # (B, D)
        col_mean  = obs_sum / obs_count                  # (B, D)
        fill = col_mean.unsqueeze(1).expand_as(values)
        return torch.where(mask.bool(), values, fill)

    def _build_input(self, batch: dict) -> torch.Tensor:
        """
        Appendix H Steps 3-4: build (B, T+1, 2D+1) input tensor.

          Context rows (T):  [imputed_values | mask | norm_timestamp]
          Query row    (1):  [zeros(D)       | zeros(D) | norm_query_ts]
        """
        values    = batch["values"]      # (B, T, D)
        mask      = batch["mask"]        # (B, T, D)
        timestamps = batch["timestamps"] # (B, T)
        query_ts  = batch["query_ts"]    # (B,)
        B, T, D   = values.shape

        # Impute missing values; timestamps are always observed — no imputation needed
        imputed = self._impute(values, mask)             # (B, T, D)

        # Context block: [imputed, mask, timestamps] → (B, T, 2D+1)
        ctx = torch.cat([imputed, mask, timestamps.unsqueeze(-1)], dim=-1)

        # Query row: zeros for values/mask, query timestamp in last channel
        q_row = torch.zeros(B, 1, 2 * D + 1, device=values.device)
        q_row[:, 0, -1] = query_ts                       # (B,) broadcast into (B, 1)

        return torch.cat([ctx, q_row], dim=1)            # (B, T+1, 2D+1)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._build_input(batch)                     # (B, T+1, 2D+1)

        # Decompose
        trend    = self.moving_avg(x)                    # (B, T+1, 2D+1)
        seasonal = x - trend                             # (B, T+1, 2D+1)

        # Flatten time × feature
        trend_flat    = trend.reshape(x.size(0), -1)    # (B, (T+1)*(2D+1))
        seasonal_flat = seasonal.reshape(x.size(0), -1)

        pred = self.linear_trend(trend_flat) + self.linear_seasonal(seasonal_flat)
        return pred  # (B, n_targets)
