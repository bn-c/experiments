#!/usr/bin/env bash
# 10k cooled cosine: geo_transformer (d_model=128) vs SwiGLU no-gauge W1 baseline aychd5jf.
# Recipe: λ_k=0.5, batch 128, lr≈1.414e-4, T_max=10k, seed 42, early-stop 0.05.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"

mkdir -p logs checkpoints/geo-transformer-10k
SUMMARY="logs/geo-transformer-10k-summary.txt"
NAME="geo-tf-d${D_MODEL}-lk0.5-b${BATCH}-s${STEPS}"
CKPT="checkpoints/geo-transformer-10k/${NAME}"
LOG="logs/${NAME}.log"
mkdir -p "$CKPT"

{
  echo "B0_REF=aychd5jf MIXER=cross_coupled ACT=swiglu NO_GAUGE=1 LK=0.5"
  echo "B0_BLANK_5K=0.343 B0_BLANK_10K=0.379 N_PARAMS=1009709"
  echo "ARM=geo_transformer D_MODEL=${D_MODEL} N_HEADS=${N_HEADS} LK=0.5"
  echo "----"
} | tee "$SUMMARY"

echo "=== $NAME ===" | tee -a "$SUMMARY"

uv run riemannian-sudoku train \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
  --steps "$STEPS" --batch-size "$BATCH" \
  --lr "$LR" --num-steps "$NUM_STEPS" \
  --d-model "$D_MODEL" --n-heads "$N_HEADS" \
  --lr-schedule cosine --grad-clip 1.0 \
  --early-stop-drop 0.05 --early-stop-patience 2 \
  --kinetic-weight 0.5 --constraint-weight 0 --supervise-all-steps \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --ckpt-every 1000 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --mixer geo_transformer \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >"$LOG" 2>&1

{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S)=' "$LOG" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
