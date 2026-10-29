#!/usr/bin/env bash
# SiLU expanded MLP 10k: 729→1458→SiLU→1458→SiLU→729 on cross_coupled --no-gauge.
# Compare to SiLU cliff 3i4zt8kr (contracted) and SwiGLU W1 aychd5jf.
# Recipe: λ_k=0.5, batch 128, lr≈1.414e-4 cosine T_max=10k, ES drop 0.05, seed 42.
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

mkdir -p logs checkpoints/silu-expand-mlp
name="cc-silu-expand-nogauge-lk0.5-b${BATCH}-s${STEPS}"
ckpt="checkpoints/silu-expand-mlp/${name}"
log="logs/${name}.log"
SUMMARY="logs/silu-expand-mlp-summary.txt"
mkdir -p "$ckpt"
: >"$SUMMARY"

echo "=== $name (silu --mlp-width expand) ===" | tee -a "$SUMMARY"
echo "REF SiLU cliff 3i4zt8kr / SwiGLU W1 aychd5jf" | tee -a "$SUMMARY"

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
  --mlp-activation silu --mlp-width expand \
  --ckpt-dir "$ckpt" --wandb-name "$name" \
  >"$log" 2>&1

{
  echo "NAME=$name"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|MLP_ACTIVATION|MLP_LAYERS|MLP_WIDTH|WALL_S|IT_S)=' "$log" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $name ==="
cat "$SUMMARY"
