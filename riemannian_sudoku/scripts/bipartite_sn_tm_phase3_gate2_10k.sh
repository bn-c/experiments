#!/usr/bin/env bash
# Phase-3 Gate 2: partial_blind 10k — pin S0–T1 families [0,1]; rest ~N(0,σ²).
# Pass: purity_T1≥0.98 + hierarchy (|h_S1-h_T1|≈1 OR α_S0_T2 drops vs α_S0_T1).
# Mid-check ~3k if purity stuck <0.85. Blank not gated.
# ONLY run after Phase-3 sanity (gt_pinned) passes. Stop if this fails — do not start Gate 3.
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
ROUTING_INIT="${ROUTING_INIT:-partial_blind}"

mkdir -p logs checkpoints/bipartite-sn-tm-phase3-gate2
NAME="bipartite-sn-tm-phase3-gate2-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-phase3-gate2/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "ARM=phase3_multibank Gate2 partial_blind MultiBank Attractor K=${NUM_STEPS}"
  echo "S0=81xd64 T1/T2=F3x9xm16 S1=9xd32 PE=t2_grid COMPILE=1"
  echo "INIT=${ROUTING_INIT} (pin S0-T1 fam 0,1; rest ~N; S1 free)"
  echo "PASS: purity_T1>=0.98; |h_S1-h_T1|~1 OR alpha_S0_T2 drops vs alpha_S0_T1"
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
  --ortho-weight 0.1 \
  --ent-weight 0.005 \
  --compile \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >>"$LOG" 2>&1

echo "EXIT:$?" | tee -a "$LOG"
{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  grep -E 'alpha_S0_T|h_T1|h_T2|h_S1|sudoku_graph_purity|routing_clash|family_gram' "$LOG" | tail -60 || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
