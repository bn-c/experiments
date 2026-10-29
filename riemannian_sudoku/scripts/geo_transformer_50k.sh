#!/usr/bin/env bash
# Geo transformer G @ 50k: same arch as 6rp4e5u4 (blank 0.512 @10k), cooled cosine over 10k then hold.
# Recipe: λ_k=0.5, batch 128, lr≈1.414e-4, --lr-cosine-steps 10000, seed 42, compile ON.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-50000}"
LR_COSINE_STEPS="${LR_COSINE_STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"

mkdir -p logs checkpoints/geo-transformer-50k
SUMMARY="logs/geo-transformer-50k-summary.txt"
NAME="geo-tf-d${D_MODEL}-lk0.5-b${BATCH}-s${STEPS}-cos${LR_COSINE_STEPS}"
CKPT="checkpoints/geo-transformer-50k/${NAME}"
LOG="logs/${NAME}.log"
mkdir -p "$CKPT"

{
  echo "G10K_REF=6rp4e5u4 G10K_BLANK=0.512 G_PARAMS=501257"
  echo "ARM=geo_transformer D_MODEL=${D_MODEL} N_HEADS=${N_HEADS} NUM_STEPS=${NUM_STEPS} LK=0.5"
  echo "STEPS=${STEPS} LR_COSINE_STEPS=${LR_COSINE_STEPS} COMPILE=1 SEED=${SEED} VAL=${VAL_SIZE}"
  echo "----"
} | tee "$SUMMARY"

echo "=== $NAME ===" | tee -a "$SUMMARY"

# Param gate (known 501257 for d=128 geo_transformer; print live count from train log too).
echo "N_PARAMS_EXPECTED=501257" | tee -a "$SUMMARY"

uv run riemannian-sudoku train \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
  --steps "$STEPS" --batch-size "$BATCH" \
  --lr "$LR" --num-steps "$NUM_STEPS" \
  --d-model "$D_MODEL" --n-heads "$N_HEADS" \
  --lr-schedule cosine --lr-cosine-steps "$LR_COSINE_STEPS" --grad-clip 1.0 \
  --early-stop-drop 0.05 --early-stop-patience 2 \
  --kinetic-weight 0.5 --constraint-weight 0 --supervise-all-steps \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --ckpt-every 2500 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --compile \
  --mixer geo_transformer \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >"$LOG" 2>&1

{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
