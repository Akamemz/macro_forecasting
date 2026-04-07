"""
TTF Module — RecAvg (Timestamp-to-Text Fusion).

For a forecast query time t_q, compute a Gaussian-weighted mean of all
available Article IV embeddings:

    w_i  = exp( -(t_q - t_i)^2 / (2 * sigma^2) )   [un-normalised]
    w_i  = w_i * text_mask_i                         [zero out padding]
    w    = w / (sum(w) + eps)                        [normalise]
    ctx  = w @ text_embs                             [weighted mean, (B, 768)]

Recent reports (smaller |t_q - t_i|) receive higher weight, which naturally
handles the 3-8 month publication lag in Article IV reports.
"""

import torch
import torch.nn as nn


class RecAvgTTF(nn.Module):
    """
    Gaussian-weighted average of text embeddings (no learnable parameters).

    Args:
        sigma: Gaussian kernel width in normalised-year units.
               sigma=1.0 ≈ the full [0,1] range; adjust to taste.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        text_embs: torch.Tensor,  # (B, N, 768)
        text_ts:   torch.Tensor,  # (B, N)      normalised publication dates
        query_ts:  torch.Tensor,  # (B,)        normalised query year
        text_mask: torch.Tensor,  # (B, N) bool  True = real, False = padding
    ) -> torch.Tensor:            # (B, 768)
        # Time differences: (B, N)
        diff = query_ts.unsqueeze(1) - text_ts          # (B, N)
        log_w = -(diff ** 2) / (2 * self.sigma ** 2)   # (B, N)
        w = torch.exp(log_w)                            # (B, N)

        # Zero-out padding positions
        w = w * text_mask.float()                       # (B, N)

        # Normalise
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)    # (B, N)

        # Weighted mean
        ctx = torch.bmm(w.unsqueeze(1), text_embs)     # (B, 1, 768)
        return ctx.squeeze(1)                            # (B, 768)
