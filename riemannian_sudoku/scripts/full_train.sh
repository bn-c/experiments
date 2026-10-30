#!/usr/bin/env bash
# Full desk train — tip bipartite product + chart + global T^{16} (f7fv0ymk line).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-50000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
MIXER="${MIXER:-bipartite_sn_tm}"
RESUME="${RESUME:-}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EARLY_STOP_DROP="${EARLY_STOP_DROP:-0.05}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
KINETIC_WEIGHT="${KINETIC_WEIGHT:-0}"
GAUGE_KINETIC_WEIGHT="${GAUGE_KINETIC_WEIGHT:-0}"
CONSTRAINT_WEIGHT="${CONSTRAINT_WEIGHT:-0.0}"
NO_AUGMENT="${NO_AUGMENT:-0}"
SUPERVISE_ALL_STEPS="${SUPERVISE_ALL_STEPS:-1}"
USE_GLOBAL_TORUS="${USE_GLOBAL_TORUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPILE="${COMPILE:-1}"
NAME="${WANDB_NAME:-bipartite-sn-tm-product-chart-global-t16-s${STEPS}-b${BATCH}}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/bipartite-sn-tm-product-chart-global-t16-full}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${NAME}}"

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
  --kinetic-weight "$KINETIC_WEIGHT"
  --gauge-kinetic-weight "$GAUGE_KINETIC_WEIGHT"
  --constraint-weight "$CONSTRAINT_WEIGHT"
  --mixer "$MIXER"
  --num-workers "$NUM_WORKERS" --pin-memory
  --eval-every 500 --log-every 50
  --ckpt-every 1000 --ckpt-dir "$CKPT_DIR"
  --wandb-name "$NAME"
  --seed 42
)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi
if [[ "$NO_AUGMENT" == "1" ]]; then
  ARGS+=(--no-augment)
fi
if [[ "$SUPERVISE_ALL_STEPS" == "1" ]]; then
  ARGS+=(--supervise-all-steps)
fi
if [[ "$USE_GLOBAL_TORUS" == "1" ]]; then
  ARGS+=(--use-global-torus)
else
  ARGS+=(--no-use-global-torus)
fi
if [[ "$COMPILE" == "1" ]]; then
  ARGS+=(--compile)
fi

echo "=== $NAME mixer=$MIXER use_global_torus=$USE_GLOBAL_TORUS ==="
# shellcheck disable=SC2068
uv run riemannian-sudoku ${ARGS[@]} 2>&1 | tee "logs/${NAME}.log"
