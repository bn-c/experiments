#!/usr/bin/env bash
# Full desk train after tune lock. Defaults from plan + defaults note.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

LR="${LR:-3e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-64}"
STEPS="${STEPS:-50000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
CKPT_DIR="${CKPT_DIR:-checkpoints/full}"
RESUME="${RESUME:-}"
NAME="${WANDB_NAME:-full-lr${LR}-ns${NUM_STEPS}-b${BATCH}-s${STEPS}}"

mkdir -p "$CKPT_DIR" logs
ARGS=(
  train
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
  --steps "$STEPS" --batch-size "$BATCH"
  --lr "$LR" --num-steps "$NUM_STEPS"
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
