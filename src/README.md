# src/

```
macro_forecasting/
├── data/
│   ├── pdf/                                 # real IMF Article IV PDFs
│   ├── macro/                               # per-country WDI CSVs (22 countries)
│   ├── text/                                # 5-sentence summaries + metadata.csv
│   └── embeddings/                          # precomputed 768-dim .npy files
├── src/
│   ├── pipeline/
│   │   ├── extract_pdf.py                   # PyMuPDF extraction of pages 1-6
│   │   ├── generate_dummy.py                # synthetic macro CSVs + text summaries
│   │   └── encode_text.py                   # BERT or dummy 768-dim embeddings
│   ├── data/
│   │   ├── prealign.py                      # TIME-IMM pre-alignment: (T,D) tensors + mask + timestamps
│   │   └── dataset.py                       # PyTorch Dataset + variable-length collate
│   ├── models/
│   │   ├── ttf.py                           # RecAvgTTF: Gaussian-weighted text embedding average
│   │   ├── mmf.py                           # GRAddMMF (full multimodal) + NumericalOnlyGRU (ablation)
│   │   └── dlinear.py                       # DLinear baseline with mean imputation
│   ├── config.py                            # all hyperparams, paths, 22 countries, 7 indicators
│   ├── train.py                             # training loop with early stopping
│   └── evaluate.py                          # MSE/MAE table + missingness-stratified analysis
├── research/
│   ├── papers/                              # reference papers (Time-IMM.pdf)
│   ├── ppt/                                 # presentation slides
│   └── report/                             # progress reports
├── results/                                 # training results JSON + checkpoints
└── run_pipeline.py                          # single entry point for everything
```

## Usage

```bash
python run_pipeline.py               # full run, dummy embeddings
python run_pipeline.py --skip-train  # data generation only
python run_pipeline.py --emb bert    # real BERT encoding
python run_pipeline.py --epochs 20 --models dlinear gr_add

python -m src.train --model gr_add
python -m src.evaluate
```
