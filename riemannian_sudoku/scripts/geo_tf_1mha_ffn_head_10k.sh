#!/usr/bin/env bash
# 10k cooled cosine: geo_transformer 1× residual MHA + L=2 d_ff=4d SiLU head.
# Recipe matches geo-transformer-10k (λ_k=0.5, batch 128, T_max=10k, seed 42).
# MUST use torch.compile (project preference). Baseline blank@10k 0.512 (6rp4e5u4).
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

mkdir -p logs checkpoints/geo-tf-1mha-ffn-head-10k
SUMMARY="logs/geo-tf-1mha-ffn-head-10k-summary.txt"
NAME="geo-tf-1mha-ffn-d${D_MODEL}-lk0.5-b${BATCH}-s${STEPS}"
CKPT="checkpoints/geo-tf-1mha-ffn-head-10k/${NAME}"
LOG="logs/${NAME}.log"
mkdir -p "$CKPT"

{
  echo "BASELINE_REF=6rp4e5u4 MIXER=geo_transformer_2x_enc BLANK_10K=0.512 N_PARAMS=501257"
  echo "ARM=geo_transformer_1mha_ffn_head D_MODEL=${D_MODEL} N_HEADS=${N_HEADS} LK=0.5 COMPILE=1"
  echo "ARCH=encode+e_row+e_col → 1x residual MHA+LN → L=2 SiLU head 128→512→512→9"
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
  --compile \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >"$LOG" 2>&1

{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
