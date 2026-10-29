#!/usr/bin/env bash
# Short W&B: SwiGLU --no-gauge mlp_layers=2 vs Arm W baseline aychd5jf (layers=1).
# Recipe matches activation A/B: λ_k=0.5, batch 128, lr≈1.414e-4 cosine, T_max=STEPS.
# Arm D2 only trains; W1 metrics reused from aychd5jf (blank@10k 0.379).
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

mkdir -p logs checkpoints/swiglu-mlp-depth-ab
SUMMARY="logs/swiglu-mlp-depth-ab-summary.txt"
: >"$SUMMARY"

echo "=== W1 ref aychd5jf (mlp_layers=1, skip train) ===" | tee -a "$SUMMARY"
{
  echo "ARM=W1 ACT=swiglu MLP_LAYERS=1 NAME=cc-act-ab-W-swiglu-nogauge-lk0.5-b128-s10000"
  echo "RUN_ID=aychd5jf"
  echo "N_PARAMS=1009709"
  echo "VAL_BLANK=0.379000"
  echo "BEST_BLANK=0.379000"
  echo "MIXER=cross_coupled"
  echo "USE_GAUGE=0"
  echo "MLP_ACTIVATION=swiglu"
  echo "MLP_LAYERS=1"
  echo "----"
} | tee -a "$SUMMARY"

name="cc-swiglu-depth-D2-nogauge-lk0.5-b${BATCH}-s${STEPS}"
ckpt="checkpoints/swiglu-mlp-depth-ab/${name}"
log="logs/${name}.log"
mkdir -p "$ckpt"
echo "=== $name (mlp_layers=2) ===" | tee -a "$SUMMARY"

uv run riemannian-sudoku train \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
  --steps "$STEPS" --batch-size "$BATCH" \
  --lr "$LR" --num-steps "$NUM_STEPS" \
  --lr-schedule cosine --grad-clip 1.0 \
  --early-stop-drop 0.05 --early-stop-patience 2 \
  --kinetic-weight 0.5 --constraint-weight 0 --supervise-all-steps \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --ckpt-every 1000 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --mixer cross_coupled --no-gauge \
  --mlp-activation swiglu --mlp-layers 2 \
  --ckpt-dir "$ckpt" --wandb-name "$name" \
  >"$log" 2>&1

{
  echo "ARM=D2 ACT=swiglu MLP_LAYERS=2 NAME=$name"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|MLP_ACTIVATION|MLP_LAYERS)=' "$log" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "All arms finished."
cat "$SUMMARY"
