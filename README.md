 # macro_forecasting

Macro-economic forecasting pipeline replicating Time-IMM, with **GR-Add** and **RecAvg-XAttn-Add** multimodal fusion on top of IMM-TSF backbones using WDI macro panel data and IMF Article IV report text embeddings.

Text-time fusion uses **RecAvg** (Gaussian-kernel recency-weighted average) as the TTF baseline.

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
| RecAvg-XAttn-Add | 11 | IMM-TSF + RecAvg Cross-Attention multimodal fusion |

## Results

### GR-Add fusion (`--gradd-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_patchtst_gradd | 1.2138 | **2.1866** |
| imm_dlinear_gradd | 1.0131 | 2.3964 |
| imm_timellm_gradd | 1.0531 | 2.5968 |
| imm_informer_gradd | 0.9882 | 2.6547 |
| imm_ttm_gradd | 1.1467 | 2.7398 |
| imm_timesnet_gradd | 1.1145 | 2.7400 |
| imm_cru_gradd | 0.9548 | 2.7886 |
| imm_timemixer_gradd | 0.9319 | 2.8100 |
| imm_neural_flow_gradd | 0.9292 | 2.8567 |
| imm_latent_ode_gradd | 0.9426 | 2.9292 |
| imm_tpatchgnn_gradd | 0.8922 | 2.9612 |

### RecAvg-XAttn-Add fusion (`--xattn-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_timellm_xattn | 1.1719 | **2.1523** |
| imm_patchtst_xattn | 1.1516 | 2.1813 |
| imm_dlinear_xattn | 1.0746 | 2.2151 |
| imm_informer_xattn | 1.1244 | 2.7355 |
| imm_latent_ode_xattn | 0.9191 | 2.8630 |
| imm_timemixer_xattn | 0.9570 | 2.8679 |
| imm_neural_flow_xattn | 0.9465 | 2.9209 |
| imm_ttm_xattn | 1.4457 | 2.9280 |
| imm_tpatchgnn_xattn | 0.8991 | 2.9538 |
| imm_timesnet_xattn | 1.1489 | 2.9971 |
| imm_cru_xattn | 0.9606 | 3.0512 |

