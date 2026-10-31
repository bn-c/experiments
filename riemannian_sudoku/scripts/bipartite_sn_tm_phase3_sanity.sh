#!/usr/bin/env bash
# Phase-3 v3.1 MultiBank sanity (desk, compile ON).
# Pass: max(purity_T1,T2) >= 0.98; |h_T1-h_T2| >= 0.8; log 4 alphas + 3 h + pooled_top3.
# Blank not required ≈ hard-A 0.539.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-2000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
ROUTING_INIT="${ROUTING_INIT:-gt_pinned}"
TAU_HOLD="${TAU_HOLD:-0.25}"

mkdir -p logs checkpoints/bipartite-sn-tm-phase3-sanity
NAME="bipartite-sn-tm-phase3-sanity-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-phase3-sanity/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "ARM=phase3_multibank sanity MultiBank Attractor K=${NUM_STEPS}"
  echo "S0=81xd64 T1/T2=F3x9xm16 S1=9xd32 PE=t2_grid COMPILE=1"
  echo "INIT=${ROUTING_INIT} TAU_HOLD=${TAU_HOLD}"
  echo "PASS: max(purity_T1,purity_T2)>=0.98; |h_T1-h_T2|>=0.8; log 4 alphas + 3 h + pooled_top3"
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
  --mixer multibank_attractor \
  --pe-mode t2_grid \
  --routing-mode soft_sinkhorn \
  --routing-init "$ROUTING_INIT" \
  --sigma-init 0.5 \
  --gt-boost 2.0 \
  --tau-hold "$TAU_HOLD" \
  --ortho-weight 0.1 \
  --ent-weight 0.005 \
  --compile \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >>"$LOG" 2>&1

echo "EXIT:$?" | tee -a "$LOG"
{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  grep -E 'alpha_S0_T|h_T1|h_T2|h_S1|sudoku_graph_purity' "$LOG" | tail -40 || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
