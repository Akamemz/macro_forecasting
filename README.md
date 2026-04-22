# macro_forecasting

Macro-economic forecasting pipeline replicating Time-IMM, with **GR-Add** and **T2V-XAttn-Add** multimodal fusion on top of IMM-TSF backbones using WDI macro panel data and IMF Article IV report text embeddings.

Text-time fusion uses **T2V-XAttn** (Time2Vec + Cross-Attention) — replaces the earlier RecAvg baseline.

## Dependencies

Install Python packages:
```bash
pip install torch transformers torchdiffeq torchvision torchaudio seaborn geotorch reformer-pytorch torch-geometric gluonts
```

> **Note:** `stribor` is **not** required. The `flow.py` and `ode.py` files in
> `IMM-TSF/lib/neural_flow_components/models/` have been patched to use pure
> PyTorch MLP implementations, removing the dependency on `stribor`/`torchtyping`
> which are incompatible with PyTorch ≥ 2.0.

**IMM-TSF** must be cloned separately:
```bash
git clone https://github.com/blacksnail789521/IMM-TSF.git
```
Then either:
- Place it at `~/IMM-TSF` or `~/IMM-TSF-master` (auto-detected), or
- Set the environment variable:
  ```bash
  export IMM_TSF_ROOT=/path/to/IMM-TSF
  ```

### Source patches required

Three IMM-TSF source files must be patched before running:

| File | Issue | Fix |
|------|-------|-----|
| `IMM-TSF/models/tPatchGNN.py` | Hardcoded `.cuda()` calls break CPU-only runs | Replace `.cuda()` with `.to(device)` |
| `IMM-TSF/lib/neural_flow_components/models/flow.py` | `stribor` incompatible with PyTorch ≥ 2.0 | Reimplemented with pure PyTorch |
| `IMM-TSF/lib/neural_flow_components/models/ode.py` | `stribor` incompatible with PyTorch ≥ 2.0 | Reimplemented with pure PyTorch |

Patched versions of all three files are included in `src/patches/`.

## Usage

```bash
# Run all models
python main_all.py

# Run only IMM-TSF backbone models
python main_all.py --imm-only

# Run only GR-Add fusion models
python main_all.py --gradd-only

# Run only XAttn-Add fusion models
python main_all.py --xattn-only

# Skip specific models
python main_all.py --skip imm_timellm imm_latent_ode

# Run on CPU, override epochs
python main_all.py --device cpu --epochs 50
```

## Model groups

| Group | Count | Description |
|-------|-------|-------------|
| Original | 3 | DLinear, NumericalGRU, GR-Add |
| IMM-TSF | 11 | IMM-TSF backbone models |
| GR-Add | 11 | IMM-TSF + GR-Add multimodal fusion |
| T2V-XAttn-Add | 11 | IMM-TSF + T2V-XAttn multimodal fusion |

## Results

### GR-Add fusion (`--gradd-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_dlinear_gradd | 1.1017 | **2.3278** |
| imm_timellm_gradd | 1.0638 | 2.5813 |
| imm_ttm_gradd | 1.1320 | 2.7547 |
| imm_informer_gradd | 1.0420 | 2.7646 |
| imm_patchtst_gradd | 1.2164 | 2.8314 |
| imm_timesnet_gradd | 1.0817 | 2.9355 |
| imm_cru_gradd | 0.9728 | 2.9546 |
| imm_timemixer_gradd | 1.0318 | 2.9587 |
| imm_latent_ode_gradd | 0.9666 | 3.0967 |
| imm_tpatchgnn_gradd | 0.9821 | 3.2002 |
| imm_neural_flow_gradd | 0.9679 | 3.2967 |

### T2V-XAttn-Add fusion (`--xattn-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_dlinear_xattn | 1.1131 | **2.1530** |
| imm_timemixer_xattn | 0.9407 | 2.6751 |
| imm_timesnet_xattn | 1.1807 | 2.6834 |
| imm_tpatchgnn_xattn | 0.9561 | 2.7439 |
| imm_neural_flow_xattn | 0.9396 | 2.7585 |
| imm_ttm_xattn | 1.3729 | 2.7666 |
| imm_latent_ode_xattn | 0.9180 | 2.8329 |
| imm_timellm_xattn | 0.9887 | 2.9176 |
| imm_patchtst_xattn | 1.4466 | 2.9470 |
| imm_cru_xattn | 0.9435 | 3.0127 |
| imm_informer_xattn | 1.0145 | 3.0171 |
