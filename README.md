# macro_forecasting

Macro-economic forecasting pipeline replicating Time-IMM, with **GR-Add** and **XAttn-Add** multimodal fusion on top of IMM-TSF backbones using WDI macro panel data and IMF Article IV report text embeddings.

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
| XAttn-Add | 11 | IMM-TSF + Cross-Attention multimodal fusion |

## Results

### GR-Add fusion (`--gradd-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_dlinear_gradd | 1.0266 | **2.3640** |
| imm_timesnet_gradd | 1.2048 | 2.4775 |
| imm_timemixer_gradd | 1.0223 | 2.6957 |
| imm_timellm_gradd | 0.9950 | 2.7414 |
| imm_latent_ode_gradd | 0.9663 | 2.8273 |
| imm_cru_gradd | 1.0521 | 2.9046 |
| imm_tpatchgnn_gradd | 0.9356 | 2.9329 |
| imm_informer_gradd | 1.0162 | 3.0030 |
| imm_neural_flow_gradd | 0.9668 | 3.1400 |
| imm_ttm_gradd | 0.9863 | 3.1763 |
| imm_patchtst_gradd | 1.1123 | 3.2597 |

### XAttn-Add fusion (`--xattn-only`)

| Model | Val MSE | Test MSE |
|-------|---------|----------|
| imm_timellm_xattn | 1.1223 | **2.5326** |
| imm_dlinear_xattn | 1.2232 | 2.6083 |
| imm_cru_xattn | 0.9451 | 2.7631 |
| imm_informer_xattn | 1.1359 | 2.7662 |
| imm_neural_flow_xattn | 0.9583 | 2.7843 |
| imm_timemixer_xattn | 0.9621 | 2.8224 |
| imm_patchtst_xattn | 1.4121 | 2.8381 |
| imm_tpatchgnn_xattn | 0.9272 | 2.8483 |
| imm_timesnet_xattn | 1.1867 | 2.9495 |
| imm_latent_ode_xattn | 0.9940 | 3.0770 |
| imm_ttm_xattn | 1.4167 | 3.3890 |
