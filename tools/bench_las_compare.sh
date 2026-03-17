#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── LAS-specific label (set before sourcing lib) ────────────────────────
export CANDIDATE_LABEL="${CANDIDATE_LABEL:-las}"

# ── Source shared library ────────────────────────────────────────────────
source "${SCRIPT_DIR}/bench_lib.sh"

# ── Profile: 'tri' (LAS specific) ───────────────────────────────────────
if [[ "$PROFILE" == "tri" ]]; then
  TRI_SPLIT=1
  if [[ "$LOW_REQUEST_RATES_SET" -eq 0 ]]; then
    LOW_REQUEST_RATES="2 4 8 16"
  fi
  if [[ "$MID_REQUEST_RATES_SET" -eq 0 ]]; then
    MID_REQUEST_RATES="32 64"
  fi
  if [[ "$HIGH_REQUEST_RATES_SET" -eq 0 ]]; then
    HIGH_REQUEST_RATES="128 256"
  fi
  if [[ "$BURSTINESS_SET" -eq 0 ]]; then
    BURSTINESS="2"
  fi
  if [[ "$GOODPUT_SET" -eq 0 ]]; then
    GOODPUT="ttft:2000 tpot:200 e2el:30000"
  fi
elif [[ -n "$PROFILE" ]] && \
     [[ "$PROFILE" != "pressure" && "$PROFILE" != "low" && \
        "$PROFILE" != "split" && "$PROFILE" != "default" ]]; then
  echo "Unknown PROFILE: $PROFILE (supported: default, low, pressure, split, tri)."
  exit 1
fi

# ── LAS specific config ─────────────────────────────────────────────────
MLFQ_LAS_TOKEN_CHUNK_SIZE="${MLFQ_LAS_TOKEN_CHUNK_SIZE:-256}"
MLFQ_LAS_WEIGHT="${MLFQ_LAS_WEIGHT:-1.0}"
MLFQ_LAS_LOW_TOKEN_CHUNK_SIZE="${MLFQ_LAS_LOW_TOKEN_CHUNK_SIZE:-0}"
MLFQ_LAS_LOW_WEIGHT="${MLFQ_LAS_LOW_WEIGHT:-0.0}"
MLFQ_LAS_MID_TOKEN_CHUNK_SIZE="${MLFQ_LAS_MID_TOKEN_CHUNK_SIZE:-1024}"
MLFQ_LAS_MID_WEIGHT="${MLFQ_LAS_MID_WEIGHT:-0.2}"

# ── Candidate arg arrays ────────────────────────────────────────────────
CANDIDATE_SCHED_ARGS=(
  "--mlfq-enable-experimental"
  "--mlfq-low-pressure-waiting-threshold" "$MLFQ_LOW_PRESSURE_WAITING_THRESHOLD"
  "--mlfq-aging-seconds" "0"
  "--mlfq-aging-dynamic-waiting-low" "0"
  "--mlfq-aging-dynamic-waiting-high" "0"
  "--mlfq-aging-dynamic-min-factor" "0"
  "--mlfq-aging-dynamic-max-factor" "0"
  "--mlfq-sjf-token-chunk-size" "0"
  "--mlfq-sjf-weight" "0"
  "--mlfq-sjf-prefill-weight" "0"
  "--mlfq-sjf-decode-weight" "0"
  "--mlfq-sjf-dynamic-waiting-low" "0"
  "--mlfq-sjf-dynamic-waiting-high" "0"
  "--mlfq-sjf-dynamic-min-factor" "0"
  "--mlfq-sjf-dynamic-max-factor" "0"
  "--mlfq-locality-weight" "0"
  "--mlfq-locality-max-boost" "0"
  "--mlfq-las-token-chunk-size" "$MLFQ_LAS_TOKEN_CHUNK_SIZE"
  "--mlfq-las-weight" "$MLFQ_LAS_WEIGHT"
)

CANDIDATE_SCHED_ARGS_LOW=(
  "--mlfq-enable-experimental"
  "--mlfq-low-pressure-waiting-threshold" "$MLFQ_LOW_PRESSURE_WAITING_THRESHOLD"
  "--mlfq-aging-seconds" "0"
  "--mlfq-aging-dynamic-waiting-low" "0"
  "--mlfq-aging-dynamic-waiting-high" "0"
  "--mlfq-aging-dynamic-min-factor" "0"
  "--mlfq-aging-dynamic-max-factor" "0"
  "--mlfq-sjf-token-chunk-size" "0"
  "--mlfq-sjf-weight" "0"
  "--mlfq-sjf-prefill-weight" "0"
  "--mlfq-sjf-decode-weight" "0"
  "--mlfq-sjf-dynamic-waiting-low" "0"
  "--mlfq-sjf-dynamic-waiting-high" "0"
  "--mlfq-sjf-dynamic-min-factor" "0"
  "--mlfq-sjf-dynamic-max-factor" "0"
  "--mlfq-locality-weight" "0"
  "--mlfq-locality-max-boost" "0"
  "--mlfq-las-token-chunk-size" "$MLFQ_LAS_LOW_TOKEN_CHUNK_SIZE"
  "--mlfq-las-weight" "$MLFQ_LAS_LOW_WEIGHT"
)

CANDIDATE_SCHED_ARGS_MID=(
  "--mlfq-enable-experimental"
  "--mlfq-low-pressure-waiting-threshold" "$MLFQ_LOW_PRESSURE_WAITING_THRESHOLD"
  "--mlfq-aging-seconds" "0"
  "--mlfq-aging-dynamic-waiting-low" "0"
  "--mlfq-aging-dynamic-waiting-high" "0"
  "--mlfq-aging-dynamic-min-factor" "0"
  "--mlfq-aging-dynamic-max-factor" "0"
  "--mlfq-sjf-token-chunk-size" "0"
  "--mlfq-sjf-weight" "0"
  "--mlfq-sjf-prefill-weight" "0"
  "--mlfq-sjf-decode-weight" "0"
  "--mlfq-sjf-dynamic-waiting-low" "0"
  "--mlfq-sjf-dynamic-waiting-high" "0"
  "--mlfq-sjf-dynamic-min-factor" "0"
  "--mlfq-sjf-dynamic-max-factor" "0"
  "--mlfq-locality-weight" "0"
  "--mlfq-locality-max-boost" "0"
  "--mlfq-las-token-chunk-size" "$MLFQ_LAS_MID_TOKEN_CHUNK_SIZE"
  "--mlfq-las-weight" "$MLFQ_LAS_MID_WEIGHT"
)

# ── Dispatch ─────────────────────────────────────────────────────────────
if [[ "$TRI_SPLIT" == "1" ]]; then
  run_group "low" "$LOW_REQUEST_RATES" "CANDIDATE_SCHED_ARGS_LOW"
  run_group "mid" "$MID_REQUEST_RATES" "CANDIDATE_SCHED_ARGS_MID"
  run_group "high" "$HIGH_REQUEST_RATES" "CANDIDATE_SCHED_ARGS"
elif [[ "$PRESSURE_SPLIT" == "1" ]]; then
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
