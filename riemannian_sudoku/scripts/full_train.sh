#!/usr/bin/env bash
# Full desk train after tune lock. Locked hparams + stability protocol (cosine / early-stop / clip).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-64}"
STEPS="${STEPS:-50000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
CKPT_DIR="${CKPT_DIR:-checkpoints/full-stable}"
RESUME="${RESUME:-}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EARLY_STOP_DROP="${EARLY_STOP_DROP:-0.15}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
NAME="${WANDB_NAME:-full-stable-lr${LR}-ns${NUM_STEPS}-b${BATCH}-s${STEPS}}"

mkdir -p "$CKPT_DIR" logs
ARGS=(
  train
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
  --steps "$STEPS" --batch-size "$BATCH"
  --lr "$LR" --num-steps "$NUM_STEPS"
  --lr-schedule "$LR_SCHEDULE"
  --grad-clip "$GRAD_CLIP"
  --early-stop-drop "$EARLY_STOP_DROP"
  --early-stop-patience "$EARLY_STOP_PATIENCE"
  --eval-every 500 --log-every 50
  --ckpt-every 1000 --ckpt-dir "$CKPT_DIR"
  --wandb-name "$NAME"
  --seed 42
)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "=== $NAME ==="
# shellcheck disable=SC2068
uv run riemannian-sudoku ${ARGS[@]} 2>&1 | tee "logs/${NAME}.log"
