#!/usr/bin/env bash
# Short desk stability A/B: euclid_axial vs cross_coupled (compile off).
# Same seed-42 holdout as m3vi22qo / GPU-fidelity (val 2048 from train head).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-2000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
# First pass: compile off (plan).
COMPILE="${COMPILE:-0}"

common=(
  train
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
  --steps "$STEPS" --batch-size "$BATCH"
  --lr "$LR" --num-steps "$NUM_STEPS"
  --lr-schedule cosine --grad-clip 1.0
  --early-stop-drop 0.15 --early-stop-patience 2
  --kinetic-weight 0.05 --constraint-weight 0 --supervise-all-steps
  --eval-every "$EVAL_EVERY" --log-every 50
  --ckpt-every 1000 --num-workers 4 --pin-memory
  --seed "$SEED" --no-wandb
)

run_one() {
  local mixer="$1"
  local name="cc-ab-${mixer}-b${BATCH}-s${STEPS}"
  local ckpt="checkpoints/cross-coupled-ab/${name}"
  mkdir -p "$ckpt" logs
  echo "=== $name (mixer=$mixer) ==="
  local args=("${common[@]}" --mixer "$mixer" --ckpt-dir "$ckpt" --wandb-name "$name")
  if [[ "$COMPILE" == "1" ]]; then
    args+=(--compile)
  fi
  # shellcheck disable=SC2068
  uv run riemannian-sudoku ${args[@]} 2>&1 | tee "logs/${name}.log"
}

run_one euclid_axial
run_one cross_coupled
