#!/usr/bin/env bash
# End-to-end Mini-ARC SFPRM baseline (full 12k-step budget).
# Smoke test instead: ./run.sh --smoke
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

echo "[1/3] Downloading Mini-ARC tasks..."
python data/download_miniarc.py

if [[ "$SMOKE" -eq 1 ]]; then
  echo "[2/3] Smoke training (tiny steps, no compile)..."
  python train.py \
    --steps 3 \
    --bptt_k 2 \
    --batch_size 4 \
    --max_tasks 4 \
    --eval_every 3 \
    --ckpt_every 3 \
    --n_color_perms 2 \
    --no-wandb \
    --out_dir ./checkpoints/smoke \
    "$@"

  echo "[3/3] Smoke evaluate..."
  python evaluate.py \
    --checkpoint ./checkpoints/smoke/model_best.pt \
    --max_tasks 4 \
    --max_loops 8 \
    --no-wandb
  echo "Smoke OK."
  exit 0
fi

echo "[2/3] Training SFPRM (Budget: 12,000 steps, ~2 hours on 16GB GPU)..."
python train.py \
  --steps 12000 \
  --bptt_k 6 \
  --batch_size 512 \
  --lr_muon 0.02 \
  --lr_adam 1e-3 \
  --compile \
  --out_dir ./checkpoints \
  "$@"

echo "[3/3] Evaluating Adaptive Chordal Halting on Test Split..."
python evaluate.py \
  --checkpoint ./checkpoints/model_best.pt \
  --epsilon 1e-3 \
  --max_loops 100
