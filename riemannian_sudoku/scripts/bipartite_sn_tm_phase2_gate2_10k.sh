#!/usr/bin/env bash
# Phase-2 Gate 2: partial-blind boxes 10k — pin families 0–1 GT; family 2 ~N(0,0.5²).
# Pass: purity >0.99 from ~0.81; family_gram_sim→~0.167.
# ONLY run after Gate 1 passes. Stop if this fails — do not start Gate 3.
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

mkdir -p logs checkpoints/bipartite-sn-tm-phase2-gate2
NAME="bipartite-sn-tm-phase2-gate2-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-phase2-gate2/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "ARM=phase2_sinkhorn Gate2 partial-blind boxes K=${NUM_STEPS}"
  echo "VF=687010 PHI=2187 pin_families=0,1 family2=N(0,0.5^2) PE=t2_grid COMPILE=1"
  echo "PASS: purity>0.99 family_gram_sim~0.167"
  echo "----"
} | tee "$SUMMARY"

echo "=== $NAME ===" | tee -a "$SUMMARY"
echo "START=$(date -Is)" | tee "$LOG"

uv run riemannian-sudoku train \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
  --steps "$STEPS" --batch-size "$BATCH" \
  --lr "$LR" --num-steps "$NUM_STEPS" \
  --lr-schedule cosine --grad-clip 1.0 \
  --early-stop-drop 0.05 --early-stop-patience 2 \
  --kinetic-weight 0 --gauge-kinetic-weight 0 --constraint-weight 0 \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --ckpt-every 1000 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --mixer bipartite_sn_tm \
  --pe-mode t2_grid \
  --routing-mode soft_sinkhorn \
  --routing-init partial_blind \
  --sigma-init 0.5 \
  --gt-boost 2.0 \
  --ortho-weight 0.1 \
  --ent-weight 0.005 \
  --compile \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >>"$LOG" 2>&1

echo "EXIT:$?" | tee -a "$LOG"
{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
