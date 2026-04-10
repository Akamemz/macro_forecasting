"""
End-to-end pipeline runner.

Steps:
  1. Generate dummy macro CSVs + text summaries
  2. Encode text to 768-dim embeddings (dummy mode by default)
  3. Test PDF extraction on real IMF reports in data/pdf/
  4. Train all models
  5. Evaluate and print comparison table

Usage:
  python run_pipeline.py                  # dummy embeddings, all models
  python run_pipeline.py --emb bert       # real BERT (downloads ~440 MB)
  python run_pipeline.py --skip-train     # only generate data + evaluate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step_generate_dummy():
    print("\n[1/5] Generating dummy macro data and text summaries …")
    from src.pipeline.generate_dummy import run
    run()


def step_encode_text(mode: str):
    print(f"\n[2/5] Encoding text summaries ({mode} mode) …")
    from src.pipeline.encode_text import run
    run(mode)


def step_test_pdf_extraction():
    print("\n[3/5] Testing PDF extraction on real IMF reports …")
    from src.pipeline.extract_pdf import extract_report_text, parse_country_year_from_filename
    from src.config import PDF_DIR

    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("  No PDFs found in data/pdf/ — skipping")
        return

    for pdf in pdfs:
        country, year = parse_country_year_from_filename(pdf.name)
        text = extract_report_text(pdf)
        print(f"  {pdf.name}  →  {country} {year}  ({len(text)} chars)")
        print(f"  Preview: {text[:200].strip()[:150]} …\n")


def step_train(models: list[str], epochs: int, device: str):
    print(f"\n[4/5] Training models: {models} …")
    from src.train import train
    for m in models:
        train(m, epochs=epochs, batch_size=16, lr=1e-3, device_str=device)


def step_evaluate(device: str):
    print("\n[5/5] Evaluating on test split …")
    from src.evaluate import run
    run(split="test", device_str=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb",        default="dummy", choices=["dummy", "bert"])
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gen",   action="store_true",
                        help="Skip dummy data generation (use when real data is already in place)")
    parser.add_argument("--models",     nargs="+",
                        default=["dlinear", "numerical_gru", "gr_add"])
    args = parser.parse_args()

    if not args.skip_gen:
        step_generate_dummy()
    step_encode_text(args.emb)
    if not args.skip_gen:
        step_test_pdf_extraction()

    if not args.skip_train:
        step_train(args.models, args.epochs, args.device)
        step_evaluate(args.device)
    else:
        print("\n[skip] Training skipped (--skip-train). "
              "Run python -m src.train --model <name> to train individually.")
