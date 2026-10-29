#!/usr/bin/env bash
# A2 ReLU Claims + Mean-Scale (Option A): keep relu q; sum→mean on q / q².
# Final-step CE only (NO --supervise-all-steps). K=8 cooled 10k compile-on.
# W&B bipartite-sn-tm-a2-relu-claim-mean-finalce-s10000.
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

mkdir -p logs checkpoints/bipartite-sn-tm-a2-relu-claim-mean-10k
NAME="bipartite-sn-tm-a2-relu-claim-mean-finalce-s${STEPS}"
CKPT="checkpoints/bipartite-sn-tm-a2-relu-claim-mean-10k/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/${NAME}-summary.txt"
mkdir -p "$CKPT"

{
  echo "BASELINE_UNNORM_LOGITS=blank@10k=0.5450 n_params=681441 run=6g8oc7mq supervise_all_steps"
  echo "BASELINE_Y0_LOGIT=blank@10k=0.5479 n_params=681441 run=yr1h706t y0=pi/4 logit_init=4 unnorm_claims"
  echo "BASELINE_G=geo_transformer blank@10k=0.512 n_params=501257 run=6rp4e5u4"
  echo "ARM=bipartite_sn_tm_A2 FiLM_vf_x H_X=432 y0=pi/4 logit_init=4 unnorm_claims unnorm_logits relu_q mean_scale final_step_CE K=${NUM_STEPS}"
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
