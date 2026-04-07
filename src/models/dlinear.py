"""
DLinear Baseline — unimodal numerical-only.

Adapted from "Are Transformers Effective for Time Series Forecasting?" (Zeng et al. 2023).
For irregular / gappy series we:
  1. Impute missing values with the column mean over the observed context window
     (falls back to 0 if the column is fully missing).
  2. Decompose into trend (moving average) + seasonal residual.
  3. Apply separate linear layers to each component.
  4. Sum the two linear outputs to produce the forecast.

Because our series are short (T=10) we use a kernel of 3 for the moving average.
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
        # x: (B, T, D)
        pad = self.kernel_size // 2
        # Reflect-pad along the time dimension
        x_pad = F.pad(x.permute(0, 2, 1), (pad, pad), mode="reflect")  # (B, D, T+2p)
        avg = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1, padding=0)
        return avg.permute(0, 2, 1)  # (B, T, D)


class DLinear(nn.Module):
    """
    DLinear adapted for irregular multivariate series.

    Args:
        seq_len:     context length T (default 10 years)
        n_indicators: number of WDI features D (default 7)
        n_targets:   forecast outputs (default 2)
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
        in_dim = seq_len * n_indicators

        self.moving_avg = MovingAvg(kernel_size=3)

        # Separate linear layers for trend and seasonal
        self.linear_trend    = nn.Linear(in_dim, n_targets)
        self.linear_seasonal = nn.Linear(in_dim, n_targets)

    def _impute(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace missing entries with per-column observed mean (or 0)."""
        # values, mask: (B, T, D)
        # compute observed mean per (batch, feature)
        obs_sum   = (values * mask).sum(dim=1)         # (B, D)
        obs_count = mask.sum(dim=1).clamp(min=1.0)     # (B, D)
        col_mean  = obs_sum / obs_count                 # (B, D)

        # Broadcast mean to full (B, T, D) and fill missing
        fill = col_mean.unsqueeze(1).expand_as(values)
        return torch.where(mask.bool(), values, fill)

    def forward(self, batch: dict) -> torch.Tensor:
        values = batch["values"]   # (B, T, D)
        mask   = batch["mask"]     # (B, T, D)

        x = self._impute(values, mask)   # (B, T, D) — no NaNs

        # Decompose
        trend    = self.moving_avg(x)    # (B, T, D)
        seasonal = x - trend             # (B, T, D)

        # Flatten time × feature
        trend_flat    = trend.reshape(trend.size(0), -1)     # (B, T*D)
        seasonal_flat = seasonal.reshape(seasonal.size(0), -1)

        pred = self.linear_trend(trend_flat) + self.linear_seasonal(seasonal_flat)
        return pred  # (B, n_targets)
