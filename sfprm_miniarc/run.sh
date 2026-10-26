#!/usr/bin/env bash
# End-to-end Mini-ARC SFPRM (paper-faithful, desk 12k-step budget).
# Smoke test instead: ./run.sh --smoke
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

echo "[1/3] Downloading Mini-ARC tasks from Hugging Face (andrewcropper/ilp-datasets mini-arc/)..."
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
    --max_loops 8 \
    --max_iter_train 4 \
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

echo "[2/3] Training SFPRM (12,000 steps; paper solver/arch; 16GB dims)..."
python train.py \
  --steps 12000 \
  --bptt_k 6 \
  --batch_size 512 \
  --lr_muon 5e-3 \
  --lr_adam 1e-3 \
  --weight_decay 0.1 \
  --out_dir ./checkpoints \
  "$@"

echo "[3/3] Evaluating paper chordal halt (τ=0.01, max_iter=200)..."
python evaluate.py \
  --checkpoint ./checkpoints/model_best.pt \
  --tau 0.01 \
  --max_loops 200
