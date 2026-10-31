#!/usr/bin/env bash
# Phase-2 Gate 1: warm-start basin 1k — Φ pinned GT +2.0, τ=0.25, t2_grid, compile on.
# Pass: purity=1.0; entropy<0.05; blank ≈ hard-A Phase-1 noise.
# Stop — do not start Gate 2 if this fails.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-1000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"

mkdir -p logs checkpoints/bipartite-sn-tm-phase2-gate1
NAME="bipartite-sn-tm-phase2-gate1-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-phase2-gate1/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "ARM=phase2_sinkhorn Gate1 warm-start GT+2.0 tau=0.25 t2_grid K=${NUM_STEPS}"
  echo "VF=687010 PHI=2187 ROUTING=soft_sinkhorn PE=t2_grid COMPILE=1"
  echo "PASS: purity=1.0 entropy<0.05 blank≈hard-A Phase-1"
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
  --ckpt-every 500 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --mixer bipartite_sn_tm \
  --pe-mode t2_grid \
  --routing-mode soft_sinkhorn \
  --routing-init gt_pinned \
  --gt-boost 2.0 \
  --tau-hold 0.25 \
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
