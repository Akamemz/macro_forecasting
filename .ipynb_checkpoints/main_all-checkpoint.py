from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
Run all models sequentially and print a summary table at the end.

Usage
-----
  python main_all.py                                        # all models
  python main_all.py --device cpu                           # force CPU
  python main_all.py --epochs 50                            # override epochs
  python main_all.py --skip imm_timellm imm_latent_ode      # skip specific models
  python main_all.py --only dlinear gr_add imm_dlinear      # run only these
  python main_all.py --imm-only                             # only IMM-TSF models
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
torch.set_default_device('cpu') # This forces every new tensor to the CPU

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import EPOCHS, BATCH_SIZE, LR
from src.train import train
from src.models.imm_gr_add import IMM_GRADD_MODELS
from src.models.imm_xattn_add import IMM_XATTN_MODELS

# ── model groups ──────────────────────────────────────────────────────────────

# Original project models (kept as-is)
ORIGINAL_MODELS = [
    "dlinear",
    "numerical_gru",
    "gr_add",
]

IMM_MODELS = [
    "imm_dlinear",
    "imm_informer",
    "imm_patchtst",
    "imm_timesnet",
    "imm_timemixer",
    "imm_ttm",
    "imm_timellm",
    "imm_cru",
    "imm_latent_ode",
    "imm_neural_flow",
    "imm_tpatchgnn",
]

ALL_MODELS = ORIGINAL_MODELS + IMM_MODELS + IMM_GRADD_MODELS + IMM_XATTN_MODELS


# ── summary table ─────────────────────────────────────────────────────────────

def _print_table(results: list[dict]):
    print("\n" + "=" * 65)
    print(f"  {'Model':<22}  {'Val MSE':>10}  {'Test MSE':>10}  {'Status':>8}")
    print("=" * 65)
    for r in sorted(results, key=lambda x: x.get("test_mse", float("inf"))):
        if r["status"] == "ok":
            print(f"  {r['model']:<22}  {r['val_mse']:>10.4f}  {r['test_mse']:>10.4f}  {'ok':>8}")
        else:
            print(f"  {r['model']:<22}  {'—':>10}  {'—':>10}  {r['status']:>8}")
    print("=" * 65)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs",  type=int,   default=EPOCHS)
    parser.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--skip",    nargs="*",  default=[],
                        help="Models to skip")
    parser.add_argument("--only",    nargs="*",  default=[],
                        help="Run only these models (overrides --skip)")
    parser.add_argument("--imm-only", action="store_true",
                        help="Run only the real IMM-TSF models (no GR-Add)")
    parser.add_argument("--gradd-only", action="store_true",
                        help="Run only IMM-TSF + GR-Add models")
    parser.add_argument("--xattn-only", action="store_true",
                        help="Run only IMM-TSF + XAttn-Add models")
    args = parser.parse_args()

    # Build the list of models to run
    if args.only:
        models = args.only
    elif args.imm_only:
        models = IMM_MODELS
    elif args.gradd_only:
        models = IMM_GRADD_MODELS
    elif args.xattn_only:
        models = IMM_XATTN_MODELS
    else:
        models = ALL_MODELS

    models = [m for m in models if m not in args.skip]

    print(f"\nRunning {len(models)} model(s) on {args.device}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"Models: {models}\n")

    results = []
    t_start = time.time()

    for model_name in models:
        print(f"\n{'─'*60}")
        print(f"  Starting: {model_name}")
        print(f"{'─'*60}")
        try:
            r = train(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                device_str=args.device,
            )
            results.append({
                "model":    model_name,
                "val_mse":  r["best_val_mse"],
                "test_mse": r["test_mse"],
                "status":   "ok",
            })
        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {e}")
            traceback.print_exc()
            results.append({
                "model":  model_name,
                "status": "failed",
            })

    total = time.time() - t_start
    print(f"\nFinished all models in {total/60:.1f} min")
    _print_table(results)


if __name__ == "__main__":
    main()
