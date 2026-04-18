# macro_forecasting

Macro-economic forecasting pipeline replicating Time-IMM, with GR-Add and XAttn-Add multimodal fusion on top of IMM-TSF backbones using the wdi data and imf reports text.

## Dependencies

Install Python packages:
```bash
pip install torch transformers torchdiffeq
```

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
| IMM-TSF | 11 | Real IMM-TSF backbone models |
| GR-Add | 11 | IMM-TSF + GR-Add multimodal fusion |
| XAttn-Add | 11 | IMM-TSF + Cross-Attention multimodal fusion |
