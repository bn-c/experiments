#!/usr/bin/env bash
# ODE / integration-depth A/B on winning geo_transformer (2× Encoder + SiLU-expand).
# Arms: NUM_STEPS ∈ {4, 8, 16}; 10k cooled cosine; compile ON; else = G recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
DEPTHS="${DEPTHS:-4 8 16}"

mkdir -p logs checkpoints/geo-transformer-ode-depth
SUMMARY="logs/geo-transformer-ode-depth-summary.txt"
{
  echo "G10K_REF=6rp4e5u4 G10K_BLANK=0.512 G_PARAMS=501257"
  echo "RECIPE=geo_transformer d=${D_MODEL} lk=0.5 b=${BATCH} s=${STEPS} cosine T_max=${STEPS} compile=1 seed=${SEED}"
  echo "DEPTHS=${DEPTHS}"
  echo "----"
} | tee "$SUMMARY"

for NS in $DEPTHS; do
  NAME="geo-tf-ode-n${NS}-d${D_MODEL}-b${BATCH}-s${STEPS}"
  CKPT="checkpoints/geo-transformer-ode-depth/${NAME}"
  LOG="logs/${NAME}.log"
  mkdir -p "$CKPT"
  echo "=== $NAME ===" | tee -a "$SUMMARY"
  uv run riemannian-sudoku train \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
    --steps "$STEPS" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NS" \
    --d-model "$D_MODEL" --n-heads "$N_HEADS" \
    --lr-schedule cosine --grad-clip 1.0 \
    --early-stop-drop 0.05 --early-stop-patience 2 \
    --kinetic-weight 0.5 --constraint-weight 0 --supervise-all-steps \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --num-workers 4 --pin-memory \
    --seed "$SEED" \
    --compile \
    --mixer geo_transformer \
    --ckpt-dir "$CKPT" --wandb-name "$NAME" \
    >"$LOG" 2>&1
  {
    echo "NAME=$NAME NUM_STEPS=$NS"
    grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|IT_S|WALL_S|COMPILE)=' "$LOG" || true
    echo "----"
  } | tee -a "$SUMMARY"
done

echo "=== done ode-depth ==="
cat "$SUMMARY"
