#!/usr/bin/env bash
# Short ≤5k W&B A/B: pin whether missing gauge kinetic drives blank collapse.
# Same recipe as cross_coupled_ab.sh except W&B ON, early-stop-drop 0.05, mixer=cross_coupled.
# Arms: (λ_k, λ_g) = A0(0.05,0) A1(0.5,0) A2(0.05,0.5) A3(0.5,0.5).
# Kill switch: if train/kinetic > 0.3 after step 2k AND blank falling → stop arm as fail.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-5000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"

ARMS=(
  "A0 0.05 0"
  "A1 0.5 0"
  "A2 0.05 0.5"
  "A3 0.5 0.5"
)

mkdir -p logs checkpoints/cross-coupled-kinetic-ab
SUMMARY="logs/cc-kin-ab-summary.txt"
: >"$SUMMARY"

run_arm() {
  local arm="$1"
  local lam_k="$2"
  local lam_g="$3"
  local name="cc-kin-ab-${arm}-lk${lam_k}-lg${lam_g}-b${BATCH}-s${STEPS}"
  local ckpt="checkpoints/cross-coupled-kinetic-ab/${name}"
  local log="logs/${name}.log"
  mkdir -p "$ckpt"
  echo "=== $name (λ_k=$lam_k λ_g=$lam_g) ===" | tee -a "$SUMMARY"

  uv run riemannian-sudoku train \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
    --steps "$STEPS" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NUM_STEPS" \
    --lr-schedule cosine --grad-clip 1.0 \
    --early-stop-drop 0.05 --early-stop-patience 2 \
    --kinetic-weight "$lam_k" --gauge-kinetic-weight "$lam_g" \
    --constraint-weight 0 --supervise-all-steps \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --num-workers 4 --pin-memory \
    --seed "$SEED" \
    --mixer cross_coupled \
    --ckpt-dir "$ckpt" --wandb-name "$name" \
    >"$log" 2>&1 &
  local pid=$!
  echo "pid=$pid log=$log"

  local killed=0
  local last_blank=""
  local prev_blank=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    # Latest train kinetic line + val blank lines.
    local kin_line blank_line
    kin_line=$(tr '\r' '\n' <"$log" 2>/dev/null | grep -E 'step [0-9]+ train loss=.*kinetic=' | tail -1 || true)
    blank_line=$(tr '\r' '\n' <"$log" 2>/dev/null | grep -E 'step [0-9]+ val blank=' | tail -1 || true)
    if [[ -n "$blank_line" ]]; then
      local step blank
      step=$(echo "$blank_line" | sed -n 's/.*step \([0-9]*\) val blank=\([0-9.]*\).*/\1/p')
      blank=$(echo "$blank_line" | sed -n 's/.*step \([0-9]*\) val blank=\([0-9.]*\).*/\2/p')
      if [[ -n "$blank" && "$blank" != "$last_blank" ]]; then
        prev_blank="$last_blank"
        last_blank="$blank"
      fi
      if [[ -n "$step" && "$step" -ge 2000 && -n "$kin_line" ]]; then
        local kin
        # Match " kinetic=" (leading space) so gauge_kinetic= is not grabbed.
        kin=$(echo "$kin_line" | grep -oE ' kinetic=[0-9.eE+-]+' | head -1 | cut -d= -f2)
        local falling=0
        if [[ -n "$prev_blank" && -n "$last_blank" ]]; then
          awk -v a="$last_blank" -v b="$prev_blank" 'BEGIN{exit !(a+0 < b+0)}' && falling=1 || true
        fi
        echo "[kill-check] arm=$arm step=$step kin=$kin blank=$last_blank prev=$prev_blank falling=$falling"
        if [[ -n "$kin" ]] && awk -v k="$kin" 'BEGIN{exit !(k+0 > 0.3)}' && [[ "$falling" -eq 1 ]]; then
          echo "FAIL kill-switch: kinetic=$kin > 0.3 after step $step blank falling ($prev_blank -> $last_blank)"
          echo "KILL_SWITCH=1 KIN=$kin STEP=$step BLANK=$last_blank PREV_BLANK=$prev_blank" >>"$log"
          kill "$pid" 2>/dev/null || true
          wait "$pid" 2>/dev/null || true
          killed=1
          break
        fi
      fi
    fi
  done

  if [[ "$killed" -eq 0 ]]; then
    wait "$pid" || true
  fi

  {
    echo "ARM=$arm NAME=$name LAM_K=$lam_k LAM_G=$lam_g KILLED=$killed"
    grep -E '^(RUN_ID|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER)=' "$log" || true
    grep -E 'KILL_SWITCH=' "$log" || true
    echo "----"
  } | tee -a "$SUMMARY"
  echo "=== done $name (killed=$killed) ==="
}

for row in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  run_arm "$1" "$2" "$3"
done

echo "All arms finished."
cat "$SUMMARY"
