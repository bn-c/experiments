#!/usr/bin/env bash
# Tip f7fv0ymk product-chart + global T^{16}: K=32, landing-window CE (k=29..32)
# + λ_kin=0.15 on Σ(‖ω_y‖²+‖ω_z‖²+‖v_c‖²) over landing window (no v_x).
# Cooled 50k, compile on. W&B bipartite-sn-tm-tip-landing-kin-k32-s50000.
# Compare undamped K_eval vs ca48ruu4 (terminal-brake 10k).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-32}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-50000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
LAMBDA_KIN="${LAMBDA_KIN:-0.15}"
TERMINAL_WINDOW="${TERMINAL_WINDOW:-4}"

mkdir -p logs checkpoints/bipartite-sn-tm-tip-landing-kin-k32-50k
NAME="bipartite-sn-tm-tip-landing-kin-k32-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-tip-landing-kin-k32-50k/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "BASELINE_TERMINAL_BRAKE=blank@16=0.5685 run=ca48ruu4 K=16 window=4 lambda_brake=0.01"
  echo "BASELINE_TIP_K8=blank@10k=0.5375 n_params~681877 run=f7fv0ymk K=8 supervise_all_steps=ramp"
  echo "ARM=bipartite_sn_tm tip product S8xS54 + T16_global + w_x_chart H_Y=304 H_Z=96 H_X=412 landing_window=${TERMINAL_WINDOW} lambda_kin=${LAMBDA_KIN} K=${NUM_STEPS} no_vx"
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
  --kinetic-weight 0 --gauge-kinetic-weight "$LAMBDA_KIN" --constraint-weight 0 \
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
