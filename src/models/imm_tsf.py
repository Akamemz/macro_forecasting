"""
Real IMM-TSF model implementations wired into the macro forecasting pipeline.

Loads models directly from the IMM-TSF-master repo (no reimplementation).
Each model is wrapped to convert the macro batch format:
    {values [B,T,D], mask [B,T,D], timestamps [B,T], query_ts [B]}
into the unified IMM-TSF interface:
    forecasting(tp_to_predict, observed_data, observed_tp, observed_mask) → [B,Lp,C]
then slices output [B,1,C] → [B,n_targets].

Dependencies
------------
  pip install transformers          # TimeLLM
  pip install torchdiffeq           # LatentODE, NeuralFlow
  GPU required                      # tPatchGNN (hardcoded .cuda() in original)
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# ── add IMM-TSF repo root to path ─────────────────────────────────────────────
# Priority: IMM_TSF_ROOT env var → sibling directory → home directory search
def _find_imm_tsf_root() -> Path:
    if "IMM_TSF_ROOT" in os.environ:
        return Path(os.environ["IMM_TSF_ROOT"])
    _here = Path(__file__).resolve()
    for candidate in [
        _here.parents[3] / "IMM-TSF-master",
        _here.parents[3] / "IMM-TSF",
        Path.home() / "IMM-TSF-master",
        Path.home() / "IMM-TSF",
        Path("/Users/nongming/Downloads/IMM-TSF-master"),
    ]:
        if (candidate / "models").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find IMM-TSF repo. "
        "Set the IMM_TSF_ROOT environment variable, e.g.:\n"
        "  export IMM_TSF_ROOT=/nfs/user_home/nming/IMM-TSF"
    )

IMM_TSF_ROOT = _find_imm_tsf_root()
if str(IMM_TSF_ROOT) not in sys.path:
    sys.path.insert(0, str(IMM_TSF_ROOT))


from src.config import N_INDICATORS, CONTEXT_WINDOW, TARGET_COLS, BATCH_SIZE


# ── shared config builder ─────────────────────────────────────────────────────

def _cfg(device):
    """SimpleNamespace with all attributes any IMM-TSF model might access."""
    return SimpleNamespace(
        # ── core dims ──────────────────────────────────────────────────────
        input_len      = CONTEXT_WINDOW,      # 10
        pred_len       = 1,
        enc_in         = N_INDICATORS,         # 7  (used by most models)
        C              = N_INDICATORS,         # 7  (used by TimeLLM, tPatchGNN)
        batch_size     = 256,                  # large so zeros_pad buffer never runs short
        device         = torch.device(device) if isinstance(device, str) else device,
        # ── transformer / attention ────────────────────────────────────────
        d_model        = 64,
        n_heads        = 4,
        d_ff           = 128,
        e_layers       = 2,
        d_layers       = 1,
        factor         = 3,
        distil         = True,
        dropout        = 0.1,
        embed          = "fixed",
        freq           = "h",
        activation     = "gelu",
        # ── DLinear ────────────────────────────────────────────────────────
        moving_avg     = 3,
        individual     = False,
        # ── TimesNet ───────────────────────────────────────────────────────
        top_k          = 3,
        num_kernels    = 6,
        c_out          = N_INDICATORS,   # output channels (used by TimesNet, Informer)
        # ── TimeMixer ──────────────────────────────────────────────────────
        down_sampling_layers = 2,
        down_sampling_window = 2,
        down_sampling_method = "avg",
        decomp_method        = "moving_avg",
        channel_independence = False,
        # ── PatchTST ───────────────────────────────────────────────────────
        # patch_len / stride passed explicitly to constructor
        # ── TTM ────────────────────────────────────────────────────────────
        patch_size     = 4,
        stride         = 2,
        AP_levels      = 1,
        n_vars         = 2 * N_INDICATORS + 1,  # 15  (2C+1 after irregular encoding)
        mode           = "mix_channel",
        use_decoder    = False,
        use_norm       = True,
        # ── TimeLLM ────────────────────────────────────────────────────────
        llm_model_timellm  = "GPT2",
        llm_layers_timellm = 2,
        input_token_len    = 4,
        ts_vocab_size      = 1000,
        domain_des         = "Macroeconomic indicators for 22 developing economies.",
        # ── CRU ────────────────────────────────────────────────────────────
        cru_lsd                       = 32,
        cru_hidden_units              = 32,
        cru_enc_num_layers            = 1,
        cru_dec_num_layers            = 1,
        cru_num_layers                = 1,
        cru_dropout_type              = "None",
        cru_dropout_rate              = 0.0,
        cru_solver                    = "euler",
        cru_bandwidth                 = 3,
        cru_num_basis                 = 15,
        cru_trans_net_hidden_units    = [],
        cru_trans_net_hidden_activation = "elup1",
        cru_t_sensitive_trans_net     = False,
        cru_trans_var_activation      = "elup1",
        cru_trans_covar               = 0.1,
        cru_enc_var_activation        = "square",
        cru_dec_var_activation        = "exp",
        lr                            = 1e-3,
        # ── NeuralFlow ─────────────────────────────────────────────────────
        nf_latents       = 20,
        nf_rec_dims      = 40,
        nf_gru_units     = 32,
        nf_hidden_layers = 2,
        nf_hidden_dim    = 32,
        nf_flow_model    = "coupling",
        nf_flow_layers   = 2,
        nf_time_net      = "TimeLinear",
        nf_time_hidden_dim = 8,
        nf_solver        = "dopri5",
        nf_obsrv_std     = 0.01,
        nf_extrap        = 0,
        nf_invertible    = 1,
        nf_components    = 8,
        nf_mixing        = 0.0001,
        nf_marks         = 0,
    )


# ── batch format conversion ───────────────────────────────────────────────────

def _to_imm(batch):
    """
    Convert macro batch dict → (tp_to_predict, observed_data, observed_tp, observed_mask).
    NaN values are zeroed out; the mask already marks them as missing.
    """
    observed_data = batch["values"].nan_to_num(0.0)   # [B, T, D]
    observed_mask = batch["mask"]                      # [B, T, D]
    observed_tp   = batch["timestamps"]                # [B, T]
    tp_to_predict = batch["query_ts"].unsqueeze(1)     # [B, 1]
    return tp_to_predict, observed_data, observed_tp, observed_mask


# ── generic wrapper ───────────────────────────────────────────────────────────

class _Wrap(nn.Module):
    """
    Thin wrapper around any IMM-TSF model.
    Converts macro batch, calls model.forecasting(), slices to target channels.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        tp, data, tp_obs, mask = _to_imm(batch)
        out = self.model.forecasting(tp, data, tp_obs, mask)  # [B, 1, C]
        return out[:, 0, :][:, TARGET_COLS]                   # [B, n_targets]


# ── LatentODE wrapper (needs 1-D shared time grid, not [B,T]) ────────────────

class _LatentODEWrap(nn.Module):
    """
    LatentODE's run_odernn iterates over a shared 1-D time vector.
    Our batch carries [B, T] timestamps; we pass observed_tp[0] as the
    shared time axis (safe because all samples share the same normalized grid).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        data  = batch["values"].nan_to_num(0.0)     # [B, T, D]
        mask  = batch["mask"]                        # [B, T, D]
        tp_obs = batch["timestamps"][0]              # [T]  — shared time grid
        tp    = batch["query_ts"][:1]                # [1]  — 1-D, required by odeint
        out = self.model.forecasting(tp, data, tp_obs, mask)  # [B, 1, C]
        return out[:, 0, :][:, TARGET_COLS]


# ── tPatchGNN wrapper (needs [B,M,L,N] patched input) ────────────────────────

_PATCH_LEN = 4
_STRIDE    = 2
_N_PATCHES = (CONTEXT_WINDOW - _PATCH_LEN) // _STRIDE + 1   # 4


def _patchify(x, patch_len=_PATCH_LEN, stride=_STRIDE, n_patches=_N_PATCHES):
    """[B, T, D] → [B, M, patch_len, D]"""
    return torch.stack(
        [x[:, i * stride : i * stride + patch_len, :] for i in range(n_patches)],
        dim=1,
    )


class _tPatchGNNWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        values = batch["values"].nan_to_num(0.0)       # [B, T, D]
        mask   = batch["mask"]                          # [B, T, D]
        ts     = batch["timestamps"]                    # [B, T]
        ts_exp = ts.unsqueeze(-1).expand_as(values)    # [B, T, D]

        X_p  = _patchify(values)   # [B, M, patch_len, D]
        ts_p = _patchify(ts_exp)   # [B, M, patch_len, D]
        m_p  = _patchify(mask)     # [B, M, patch_len, D]

        tp = batch["query_ts"].unsqueeze(1)             # [B, 1]
        out = self.model.forecasting(tp, X_p, ts_p, m_p)  # [B, 1, D]
        return out[:, 0, :][:, TARGET_COLS]


# ── model factory ─────────────────────────────────────────────────────────────

def build_imm_model(name: str, device) -> nn.Module:
    cfg = _cfg(device)

    # ── regular forecasting models ────────────────────────────────────────────
    if name == "imm_dlinear":
        from models.DLinear import DLinear
        return _Wrap(DLinear(cfg))

    if name == "imm_informer":
        from models.Informer import Informer
        cfg_inf = _cfg(device)
        cfg_inf.pred_len = 8   # ProbSparse attention needs len > 1; we still slice [:, 0, :]
        return _Wrap(Informer(cfg_inf))

    if name == "imm_patchtst":
        from models.PatchTST import PatchTST
        return _Wrap(PatchTST(cfg, patch_len=4, stride=2))

    if name == "imm_timesnet":
        from models.TimesNet import TimesNet
        return _Wrap(TimesNet(cfg))

    if name == "imm_timemixer":
        from models.TimeMixer import TimeMixer
        return _Wrap(TimeMixer(cfg))

    # ── foundation models ─────────────────────────────────────────────────────
    if name == "imm_ttm":
        from models.TTM import TTM
        return _Wrap(TTM(cfg))

    if name == "imm_timellm":
        from models.TimeLLM import TimeLLM
        return _Wrap(TimeLLM(cfg))

    # ── irregular-native models ───────────────────────────────────────────────
    if name == "imm_cru":
        from models.CRU import CRU
        return _Wrap(CRU(cfg))

    if name == "imm_latent_ode":
        from models.LatentODE import LatentODE
        args = SimpleNamespace(
            C              = N_INDICATORS,
            device         = device,
            dataset        = "macro",
            hid_dim        = 32,
            pred_len       = 1,
            ode_latents    = 20,
            ode_units      = 32,
            ode_rec_dims   = 32,
            ode_gen_layers = 1,
            ode_rec_layers = 1,
            ode_gru_units  = 32,
            ode_obsrv_std  = 0.01,
            ode_n_traj_samples = 1,
            ode_z0_encoder = "odernn",
            ode_poisson    = False,
            ode_classif    = False,
        )
        return _LatentODEWrap(LatentODE(args))

    if name == "imm_neural_flow":
        from models.NeuralFlow import NeuralFlow
        return _Wrap(NeuralFlow(cfg))

    if name == "imm_tpatchgnn":
        from models.tPatchGNN import tPatchGNN
        args = SimpleNamespace(
            device    = device,
            C         = N_INDICATORS,
            N         = N_INDICATORS,   # number of variables (nodes in graph)
            npatch    = _N_PATCHES,     # 4
            hid_dim   = 64,
            nlayer    = 2,
            te_dim    = 10,
            n_heads   = 4,
            tf_layer  = 1,
            node_dim  = 32,
            hop       = 2,
            outlayer  = "Linear",
        )
        return _tPatchGNNWrap(tPatchGNN(args, supports=None))

    raise ValueError(f"Unknown IMM model: {name}")
