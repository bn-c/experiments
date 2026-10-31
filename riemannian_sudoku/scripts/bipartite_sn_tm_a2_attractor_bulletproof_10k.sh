#!/usr/bin/env bash
# Bulletproof Attractor VF: Kuramoto T^16 soft-cap + signed ETF / gated NP+scratch S^63.
# Final-step CE only (NO --supervise-all-steps). λ_ω=0, cooled 10k, compile-on.
# Override NUM_STEPS=16 for the primary K=16 gate; STEPS=50000 for promote.
# W&B bipartite-sn-tm-a2-attractor-bp-k${NUM_STEPS}-s${STEPS}.
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

mkdir -p logs checkpoints/bipartite-sn-tm-a2-attractor-bp
NAME="bipartite-sn-tm-a2-attractor-bp-k${NUM_STEPS}-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-a2-attractor-bp/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "BASELINE_RELU_MEAN_K8=blank@10k=0.5455 n_params=681441 run=0y20qsx1"
  echo "BASELINE_NAIVE_ATTRACTOR_K8=blank@10k=0.5428 n_params=687443 run=sjjxj1xt"
  echo "BASELINE_FILM_K16=blank@10k=0.5559 n_params=681441"
  echo "BASELINE_NAIVE_ATTRACTOR_K16=blank@10k=0.5696 n_params=687443 run=d6mc81hl"
  echo "ARM=bipartite_sn_tm_A2 BulletproofAttractor KuramotoSoftCap1.25 SignedETF_GatedNPScratch FiLM H_X=432 in_y=178 final_step_CE K=${NUM_STEPS} n_params=687010"
  echo "CE_GAMMA=0.9 BATCH=${BATCH} LR=${LR} SEED=${SEED} VAL=${VAL_SIZE} COMPILE=1 SUPERVISE_ALL_STEPS=0 LAMBDA_OMEGA=0"
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
