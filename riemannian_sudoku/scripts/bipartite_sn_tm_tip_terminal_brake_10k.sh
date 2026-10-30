#!/usr/bin/env bash
# Tip f7fv0ymk product-chart + global T^{16}: K=16, terminal-window CE (last 4) + λ_brake=0.01.
# Cooled 10k, compile on. W&B bipartite-sn-tm-tip-terminal-brake-s10000.
# Compare vs d6ffqu57 (K16 final-CE) / f7fv0ymk (tip K8).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-16}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
LAMBDA_BRAKE="${LAMBDA_BRAKE:-0.01}"
TERMINAL_WINDOW="${TERMINAL_WINDOW:-4}"

mkdir -p logs checkpoints/bipartite-sn-tm-tip-terminal-brake-10k
NAME="bipartite-sn-tm-tip-terminal-brake-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-tip-terminal-brake-10k/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "BASELINE_TIP_K8=blank@10k=0.5375 n_params~681877 run=f7fv0ymk K=8 supervise_all_steps=ramp"
  echo "BASELINE_K16_FINALCE=blank@16=0.5861 run=d6ffqu57 K=16 final_CE highK_collapse"
  echo "ARM=bipartite_sn_tm tip product S8xS54 + T16_global + w_x_chart H_Y=304 H_Z=96 H_X=412 terminal_window=${TERMINAL_WINDOW} lambda_brake=${LAMBDA_BRAKE} K=${NUM_STEPS}"
  echo "CE_GAMMA=0.9 BATCH=${BATCH} LR=${LR} SEED=${SEED} VAL=${VAL_SIZE} COMPILE=1 SUPERVISE_ALL_STEPS=0"
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
  --kinetic-weight 0 --gauge-kinetic-weight "$LAMBDA_BRAKE" --constraint-weight 0 \
  --supervise-terminal-window "$TERMINAL_WINDOW" \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --ckpt-every 1000 --num-workers 4 --pin-memory \
  --seed "$SEED" \
  --mixer bipartite_sn_tm \
  --use-global-torus \
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
