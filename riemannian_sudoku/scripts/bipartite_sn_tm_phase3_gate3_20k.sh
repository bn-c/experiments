#!/usr/bin/env bash
# Phase-3 v3.1 Gate 3: full_blind 20k — clear_pins(); all four Φ ~ N(0,σ²).
# Early @4k: |Δh_T|>0.4 and |Δα_S0|>0.25. Final: max purity OR pooled_top3 ≥0.98 with loser α low.
# ONLY run after Gate 2 passes. Stop if this fails — no long run.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-20000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
ROUTING_INIT="${ROUTING_INIT:-full_blind}"

mkdir -p logs checkpoints/bipartite-sn-tm-phase3-gate3
NAME="bipartite-sn-tm-phase3-gate3-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-phase3-gate3/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "ARM=phase3_multibank Gate3 full_blind MultiBank Attractor K=${NUM_STEPS}"
  echo "S0=81xd64 T1/T2=F3x9xm16 S1=9xd32 PE=t2_grid COMPILE=1"
  echo "INIT=${ROUTING_INIT} (all Phi ~N; S1 free)"
  echo "PASS: @4k |dh|>0.4 & |da_S0|>0.25; @20k max(purity) or pooled_top3>=0.98; loser alpha low"
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
  --ortho-weight 0.1 \
  --ent-weight 0.005 \
  --compile \
  --ckpt-dir "$CKPT" --wandb-name "$NAME" \
  >>"$LOG" 2>&1

echo "EXIT:$?" | tee -a "$LOG"
{
  echo "NAME=$NAME"
  grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|IT_S|WALL_S|COMPILE)=' "$LOG" || true
  grep -E 'alpha_S0_T|h_T1|h_T2|h_S1|sudoku_graph_purity|routing_clash|family_gram|entropy' "$LOG" | tail -80 || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
