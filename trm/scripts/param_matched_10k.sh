#!/usr/bin/env bash
# Param-matched TRM 10k vs geo_transformer G (501,257 params, blank 0.512, W&B 6rp4e5u4).
# Width: d_model=148, n_layers=2, mlp_expansion=4 → 500,364 params (±5% of G).
# Fair recursion (match geo’s 8 with-grad Exp steps): N_sup=8, T=1, n_latent=1.
# Same HF head + seed-42 / val-2048 split as Riemannian; torch.compile ON; cosine T_max=10k.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_PROJECT="${WANDB_PROJECT:-trm-sudoku}"

D_MODEL="${D_MODEL:-148}"
N_LAYERS="${N_LAYERS:-2}"
MLP_EXPANSION="${MLP_EXPANSION:-4}"
N_LATENT="${N_LATENT:-1}"
T_OUTER="${T_OUTER:-1}"
N_SUP="${N_SUP:-8}"
BATCH="${BATCH:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS="${STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

EFF_BATCH=$((BATCH * GRAD_ACCUM))
NAME="trm-pm-d${D_MODEL}-L${N_LAYERS}-e${MLP_EXPANSION}-n${N_LATENT}-T${T_OUTER}-N${N_SUP}-b${EFF_BATCH}-s${STEPS}"
CKPT="checkpoints/param-matched-10k/${NAME}"
LOG="logs/${NAME}.log"
SUMMARY="logs/param-matched-10k-summary.txt"
mkdir -p logs "$CKPT"

{
  echo "G_REF=6rp4e5u4 G_BLANK_10K=0.512 G_PARAMS=501257"
  echo "ARM=trm_param_matched D_MODEL=${D_MODEL} N_LAYERS=${N_LAYERS} MLP_EXPANSION=${MLP_EXPANSION}"
  echo "FAIR_DEPTH N_SUP=${N_SUP} T=${T_OUTER} N_LATENT=${N_LATENT}"
  echo "EFFECTIVE_BATCH=${EFF_BATCH} BATCH=${BATCH} ACCUM=${GRAD_ACCUM}"
  echo "COMPILE=1 LR_SCHEDULE=cosine T_MAX=${STEPS} SEED=${SEED} VAL=${VAL_SIZE}"
  echo "----"
} | tee "$SUMMARY"

echo "=== $NAME ===" | tee -a "$SUMMARY"

uv run python - <<PY | tee -a "$SUMMARY"
from trm.config import ModelConfig
from trm.model.trm_mlp import TRMMLP
cfg = ModelConfig(
    d_model=${D_MODEL},
    n_layers=${N_LAYERS},
    mlp_expansion=${MLP_EXPANSION},
    n_latent=${N_LATENT},
    T=${T_OUTER},
    max_supervision_steps=${N_SUP},
)
n = sum(p.numel() for p in TRMMLP(cfg).parameters() if p.requires_grad)
print(f'N_PARAMS={n}')
lo, hi = int(501257 * 0.95), int(501257 * 1.05)
print(f'PARAM_GATE={"OK" if lo <= n <= hi else "FAIL"} window=[{lo},{hi}]')
PY

uv run python -m trm.main train \
  --d-model "$D_MODEL" --n-layers "$N_LAYERS" --mlp-expansion "$MLP_EXPANSION" \
  --n "$N_LATENT" --T "$T_OUTER" --max-steps "$N_SUP" \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
  --steps "$STEPS" --batch-size "$BATCH" --grad-accum-steps "$GRAD_ACCUM" \
  --lr "$LR" --lr-schedule cosine \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --seed "$SEED" --split-seed "$SEED" \
  --compile \
  --num-workers "$NUM_WORKERS" --pin-memory \
  --checkpoint-dir "$CKPT" --wandb-name "$NAME" \
  >"$LOG" 2>&1

{
  echo "NAME=$NAME"
  grep -E '^(N_PARAMS|VAL_BLANK|VAL_PUZZLE|STOP_STEP)=' "$LOG" || true
  rg -N 'wandb:.*(View run|Run data|run[: ])' "$LOG" | tail -8 || true
  echo "----"
} | tee -a "$SUMMARY"

echo "=== done $NAME ==="
cat "$SUMMARY"
