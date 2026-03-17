#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Aging-SJF-specific label (set before sourcing lib) ──────────────────
export CANDIDATE_LABEL="${CANDIDATE_LABEL:-aging-sjf}"

# ── Source shared library ────────────────────────────────────────────────
source "${SCRIPT_DIR}/bench_lib.sh"

# ── Profile: 'high' (aging-sjf specific) ────────────────────────────────
if [[ "$PROFILE" == "high" ]]; then
  if [[ "$PRESSURE_SPLIT_SET" -eq 0 ]]; then
    PRESSURE_SPLIT=0
  fi
  if [[ "$REQUEST_RATES_SET" -eq 0 ]]; then
    REQUEST_RATES="128 256"
  fi
  if [[ "$BURSTINESS_SET" -eq 0 ]]; then
    BURSTINESS="2"
  fi
elif [[ -n "$PROFILE" ]] && \
     [[ "$PROFILE" != "pressure" && "$PROFILE" != "low" && \
        "$PROFILE" != "split" && "$PROFILE" != "default" ]]; then
  echo "Unknown PROFILE: $PROFILE (supported: default, low, high, pressure, split)."
  exit 1
fi

# ── Aging-SJF specific config ───────────────────────────────────────────
MLFQ_AGING_SECONDS="${MLFQ_AGING_SECONDS:-5.0}"
MLFQ_AGING_DYNAMIC_WAITING_LOW="${MLFQ_AGING_DYNAMIC_WAITING_LOW:-4}"
MLFQ_AGING_DYNAMIC_WAITING_HIGH="${MLFQ_AGING_DYNAMIC_WAITING_HIGH:-32}"
MLFQ_AGING_DYNAMIC_MIN_FACTOR="${MLFQ_AGING_DYNAMIC_MIN_FACTOR:-0.0}"
MLFQ_AGING_DYNAMIC_MAX_FACTOR="${MLFQ_AGING_DYNAMIC_MAX_FACTOR:-1.0}"
MLFQ_PRESSURE_WAITING_THRESHOLD="${MLFQ_PRESSURE_WAITING_THRESHOLD:-${MLFQ_LOW_PRESSURE_WAITING_THRESHOLD:-8}}"
MLFQ_SJF_TOKEN_CHUNK_SIZE="${MLFQ_SJF_TOKEN_CHUNK_SIZE:-256}"
MLFQ_SJF_WEIGHT="${MLFQ_SJF_WEIGHT:-1.0}"
MLFQ_SJF_PREFILL_WEIGHT="${MLFQ_SJF_PREFILL_WEIGHT:-$MLFQ_SJF_WEIGHT}"
MLFQ_SJF_DECODE_WEIGHT="${MLFQ_SJF_DECODE_WEIGHT:-0.0}"
MLFQ_SJF_DYNAMIC_WAITING_LOW="${MLFQ_SJF_DYNAMIC_WAITING_LOW:-4}"
MLFQ_SJF_DYNAMIC_WAITING_HIGH="${MLFQ_SJF_DYNAMIC_WAITING_HIGH:-32}"
MLFQ_SJF_DYNAMIC_MIN_FACTOR="${MLFQ_SJF_DYNAMIC_MIN_FACTOR:-0.2}"
MLFQ_SJF_DYNAMIC_MAX_FACTOR="${MLFQ_SJF_DYNAMIC_MAX_FACTOR:-1.0}"
MLFQ_LOCALITY_WEIGHT="${MLFQ_LOCALITY_WEIGHT:-0.0}"

MLFQ_SJF_LOW_TOKEN_CHUNK_SIZE="${MLFQ_SJF_LOW_TOKEN_CHUNK_SIZE:-0}"
MLFQ_SJF_LOW_WEIGHT="${MLFQ_SJF_LOW_WEIGHT:-0.0}"
MLFQ_SJF_LOW_PREFILL_WEIGHT="${MLFQ_SJF_LOW_PREFILL_WEIGHT:-0.0}"
MLFQ_SJF_LOW_DECODE_WEIGHT="${MLFQ_SJF_LOW_DECODE_WEIGHT:-0.0}"
MLFQ_SJF_LOW_DYNAMIC_WAITING_LOW="${MLFQ_SJF_LOW_DYNAMIC_WAITING_LOW:-0}"
MLFQ_SJF_LOW_DYNAMIC_WAITING_HIGH="${MLFQ_SJF_LOW_DYNAMIC_WAITING_HIGH:-0}"
MLFQ_SJF_LOW_DYNAMIC_MIN_FACTOR="${MLFQ_SJF_LOW_DYNAMIC_MIN_FACTOR:-0.0}"
MLFQ_SJF_LOW_DYNAMIC_MAX_FACTOR="${MLFQ_SJF_LOW_DYNAMIC_MAX_FACTOR:-0.0}"

MLFQ_LOW_TOKEN_CHUNK_SIZE="${MLFQ_LOW_TOKEN_CHUNK_SIZE:-$MLFQ_TOKEN_CHUNK_SIZE}"
MLFQ_LOW_WAITING_BUDGET_FRACTION="${MLFQ_LOW_WAITING_BUDGET_FRACTION:-$MLFQ_WAITING_BUDGET_FRACTION}"
MLFQ_LOW_AGING_SECONDS="${MLFQ_LOW_AGING_SECONDS:-0.0}"
MLFQ_LOW_AGING_DYNAMIC_WAITING_LOW="${MLFQ_LOW_AGING_DYNAMIC_WAITING_LOW:-0}"
MLFQ_LOW_AGING_DYNAMIC_WAITING_HIGH="${MLFQ_LOW_AGING_DYNAMIC_WAITING_HIGH:-0}"
MLFQ_LOW_AGING_DYNAMIC_MIN_FACTOR="${MLFQ_LOW_AGING_DYNAMIC_MIN_FACTOR:-0.0}"
MLFQ_LOW_AGING_DYNAMIC_MAX_FACTOR="${MLFQ_LOW_AGING_DYNAMIC_MAX_FACTOR:-0.0}"
MLFQ_LOW_LOCALITY_WEIGHT="${MLFQ_LOW_LOCALITY_WEIGHT:-0.0}"
MLFQ_LOW_PROFILE_PRESSURE_WAITING_THRESHOLD="${MLFQ_LOW_PROFILE_PRESSURE_WAITING_THRESHOLD:-0}"

# ── Candidate arg arrays ────────────────────────────────────────────────
CANDIDATE_SCHED_ARGS=(
  "--mlfq-enable-experimental"
  "--mlfq-aging-seconds" "$MLFQ_AGING_SECONDS"
  "--mlfq-aging-dynamic-waiting-low" "$MLFQ_AGING_DYNAMIC_WAITING_LOW"
  "--mlfq-aging-dynamic-waiting-high" "$MLFQ_AGING_DYNAMIC_WAITING_HIGH"
  "--mlfq-aging-dynamic-min-factor" "$MLFQ_AGING_DYNAMIC_MIN_FACTOR"
  "--mlfq-aging-dynamic-max-factor" "$MLFQ_AGING_DYNAMIC_MAX_FACTOR"
  "--mlfq-low-pressure-waiting-threshold" "$MLFQ_PRESSURE_WAITING_THRESHOLD"
  "--mlfq-low-pressure-pending-tokens-threshold" "$MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD"
  "--mlfq-low-pressure-ratio-threshold" "$MLFQ_LOW_PRESSURE_RATIO_THRESHOLD"
  "--mlfq-phase-rps-window-s" "$MLFQ_PHASE_RPS_WINDOW_S"
  "--mlfq-phase-rps-ema-alpha" "$MLFQ_PHASE_RPS_EMA_ALPHA"
  "--mlfq-phase-rps-hysteresis" "$MLFQ_PHASE_RPS_HYSTERESIS"
  "--mlfq-phase-min-seconds" "$MLFQ_PHASE_MIN_SECONDS"
  "--mlfq-phase-pressure-baseline-max" "$MLFQ_PHASE_PRESSURE_BASELINE_MAX"
  "--mlfq-phase-pressure-las-max" "$MLFQ_PHASE_PRESSURE_LAS_MAX"
  "--mlfq-phase-pressure-pending-tokens" "$MLFQ_PHASE_PRESSURE_PENDING_TOKENS"
  "--mlfq-sjf-token-chunk-size" "$MLFQ_SJF_TOKEN_CHUNK_SIZE"
  "--mlfq-sjf-weight" "$MLFQ_SJF_WEIGHT"
  "--mlfq-sjf-prefill-weight" "$MLFQ_SJF_PREFILL_WEIGHT"
  "--mlfq-sjf-decode-weight" "$MLFQ_SJF_DECODE_WEIGHT"
  "--mlfq-sjf-dynamic-waiting-low" "$MLFQ_SJF_DYNAMIC_WAITING_LOW"
  "--mlfq-sjf-dynamic-waiting-high" "$MLFQ_SJF_DYNAMIC_WAITING_HIGH"
  "--mlfq-sjf-dynamic-min-factor" "$MLFQ_SJF_DYNAMIC_MIN_FACTOR"
  "--mlfq-sjf-dynamic-max-factor" "$MLFQ_SJF_DYNAMIC_MAX_FACTOR"
  "--mlfq-locality-weight" "$MLFQ_LOCALITY_WEIGHT"
)

CANDIDATE_SCHED_ARGS_LOW=(
  "--mlfq-enable-experimental"
  "--mlfq-token-chunk-size" "$MLFQ_LOW_TOKEN_CHUNK_SIZE"
  "--mlfq-waiting-budget-fraction" "$MLFQ_LOW_WAITING_BUDGET_FRACTION"
  "--mlfq-aging-seconds" "$MLFQ_LOW_AGING_SECONDS"
  "--mlfq-aging-dynamic-waiting-low" "$MLFQ_LOW_AGING_DYNAMIC_WAITING_LOW"
  "--mlfq-aging-dynamic-waiting-high" "$MLFQ_LOW_AGING_DYNAMIC_WAITING_HIGH"
  "--mlfq-aging-dynamic-min-factor" "$MLFQ_LOW_AGING_DYNAMIC_MIN_FACTOR"
  "--mlfq-aging-dynamic-max-factor" "$MLFQ_LOW_AGING_DYNAMIC_MAX_FACTOR"
  "--mlfq-low-pressure-waiting-threshold" "$MLFQ_LOW_PROFILE_PRESSURE_WAITING_THRESHOLD"
  "--mlfq-low-pressure-pending-tokens-threshold" "$MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD"
  "--mlfq-low-pressure-ratio-threshold" "$MLFQ_LOW_PRESSURE_RATIO_THRESHOLD"
  "--mlfq-phase-rps-window-s" "$MLFQ_PHASE_RPS_WINDOW_S"
  "--mlfq-phase-rps-ema-alpha" "$MLFQ_PHASE_RPS_EMA_ALPHA"
  "--mlfq-phase-rps-hysteresis" "$MLFQ_PHASE_RPS_HYSTERESIS"
  "--mlfq-phase-min-seconds" "$MLFQ_PHASE_MIN_SECONDS"
  "--mlfq-phase-pressure-baseline-max" "$MLFQ_PHASE_PRESSURE_BASELINE_MAX"
  "--mlfq-phase-pressure-las-max" "$MLFQ_PHASE_PRESSURE_LAS_MAX"
  "--mlfq-phase-pressure-pending-tokens" "$MLFQ_PHASE_PRESSURE_PENDING_TOKENS"
  "--mlfq-sjf-token-chunk-size" "$MLFQ_SJF_LOW_TOKEN_CHUNK_SIZE"
  "--mlfq-sjf-weight" "$MLFQ_SJF_LOW_WEIGHT"
  "--mlfq-sjf-prefill-weight" "$MLFQ_SJF_LOW_PREFILL_WEIGHT"
  "--mlfq-sjf-decode-weight" "$MLFQ_SJF_LOW_DECODE_WEIGHT"
  "--mlfq-sjf-dynamic-waiting-low" "$MLFQ_SJF_LOW_DYNAMIC_WAITING_LOW"
  "--mlfq-sjf-dynamic-waiting-high" "$MLFQ_SJF_LOW_DYNAMIC_WAITING_HIGH"
  "--mlfq-sjf-dynamic-min-factor" "$MLFQ_SJF_LOW_DYNAMIC_MIN_FACTOR"
  "--mlfq-sjf-dynamic-max-factor" "$MLFQ_SJF_LOW_DYNAMIC_MAX_FACTOR"
  "--mlfq-locality-weight" "$MLFQ_LOW_LOCALITY_WEIGHT"
)

# ── Dispatch ─────────────────────────────────────────────────────────────
if [[ "$PRESSURE_SPLIT" == "1" ]]; then
  if [[ "$PROFILE" == "split" ]]; then
    run_group "low" "$LOW_REQUEST_RATES" "CANDIDATE_SCHED_ARGS_LOW"
    run_group "high" "$HIGH_REQUEST_RATES" "CANDIDATE_SCHED_ARGS"
  else
    run_group "low" "$LOW_REQUEST_RATES"
    run_group "high" "$HIGH_REQUEST_RATES"
  fi
else
  run_case "baseline" "$BASE_OUTDIR" "$REQUEST_RATES" "${BASELINE_SCHED_ARGS[@]}"
  run_case "$CANDIDATE_LABEL" "$CAND_OUTDIR" "$REQUEST_RATES" "${CANDIDATE_SCHED_ARGS[@]}"

  python3 tools/bench_compare_runs.py \
    "$BASE_OUTDIR" \
    "$CAND_OUTDIR" \
    --baseline-name "baseline" \
    --candidate-name "$CANDIDATE_LABEL"
fi

echo "Results saved under: $ROOT_OUTDIR"
