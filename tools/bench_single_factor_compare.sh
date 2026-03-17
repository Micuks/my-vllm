#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${BENCH_ENV_FILE:-tools/bench_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set MODEL to the model path or name (e.g., MODEL=Qwen/Qwen3-4B)."
  exit 1
fi

FACTOR="${FACTOR:-}"
FACTORS="${FACTORS:-aging}"
if [[ -n "$FACTOR" ]]; then
  FACTORS="$FACTOR"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROOT_OUTDIR="${OUTDIR:-bench_results/$(date +%Y%m%d-%H%M%S)}"
REQUEST_RATES="${REQUEST_RATES:-2 4 128 256}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,e2el}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-50,95,99}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-600}"

DATASET_NAME="${DATASET_NAME:-random-mix}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.2}"
RANDOM_MIX_INPUT_LENS="${RANDOM_MIX_INPUT_LENS:-256,1024,2048}"
RANDOM_MIX_OUTPUT_LENS="${RANDOM_MIX_OUTPUT_LENS:-32,128,512}"
RANDOM_MIX_WEIGHTS="${RANDOM_MIX_WEIGHTS:-0.5,0.3,0.2}"
BURSTINESS="${BURSTINESS:-1.0}"
GOODPUT="${GOODPUT:-}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"
MLFQ_TOKEN_CHUNK_SIZE="${MLFQ_TOKEN_CHUNK_SIZE:-256}"
MLFQ_WAITING_BUDGET_FRACTION="${MLFQ_WAITING_BUDGET_FRACTION:-0.2}"

AGING_SECONDS="${AGING_SECONDS:-5.0}"
AGING_DYNAMIC_WAITING_LOW="${AGING_DYNAMIC_WAITING_LOW:-0}"
AGING_DYNAMIC_WAITING_HIGH="${AGING_DYNAMIC_WAITING_HIGH:-0}"
AGING_DYNAMIC_MIN_FACTOR="${AGING_DYNAMIC_MIN_FACTOR:-0.0}"
AGING_DYNAMIC_MAX_FACTOR="${AGING_DYNAMIC_MAX_FACTOR:-0.0}"

SJF_TOKEN_CHUNK_SIZE="${SJF_TOKEN_CHUNK_SIZE:-256}"
SJF_WEIGHT="${SJF_WEIGHT:-1.0}"
SJF_DYNAMIC_WAITING_LOW="${SJF_DYNAMIC_WAITING_LOW:-0}"
SJF_DYNAMIC_WAITING_HIGH="${SJF_DYNAMIC_WAITING_HIGH:-0}"
SJF_DYNAMIC_MIN_FACTOR="${SJF_DYNAMIC_MIN_FACTOR:-0.0}"
SJF_DYNAMIC_MAX_FACTOR="${SJF_DYNAMIC_MAX_FACTOR:-0.0}"

LOCALITY_WEIGHT="${LOCALITY_WEIGHT:-1.0}"
LOCALITY_MAX_BOOST="${LOCALITY_MAX_BOOST:-0}"

LAS_TOKEN_CHUNK_SIZE="${LAS_TOKEN_CHUNK_SIZE:-256}"
LAS_WEIGHT="${LAS_WEIGHT:-1.0}"

BP_PENDING_TOKENS="${BP_PENDING_TOKENS:-64}"
BP_PENALTY="${BP_PENALTY:-1}"
BP_MAX_TOKENS="${BP_MAX_TOKENS:-8}"
BP_MIN_TOKENS="${BP_MIN_TOKENS:-1}"
BP_LAG_SECONDS="${BP_LAG_SECONDS:-0.0}"
BP_UPDATE_INTERVAL_S="${BP_UPDATE_INTERVAL_S:-0.1}"

TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
PREFETCH_MODEL="${PREFETCH_MODEL:-0}"

if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
else
  PY_RUN=(python3)
fi

SERVER_ARGS=(
  "--host" "$HOST"
  "--port" "$PORT"
)

COMMON_SCHED_ARGS=(
  "--max-model-len" "$MAX_MODEL_LEN"
  "--max-num-seqs" "$MAX_NUM_SEQS"
  "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
  "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
  "--scheduling-policy" "$SCHEDULING_POLICY"
  "--mlfq-token-chunk-size" "$MLFQ_TOKEN_CHUNK_SIZE"
  "--mlfq-waiting-budget-fraction" "$MLFQ_WAITING_BUDGET_FRACTION"
)

BASELINE_SCHED_ARGS=(
  "--no-mlfq-enable-experimental"
  "--mlfq-aging-seconds" "0"
  "--mlfq-aging-dynamic-waiting-low" "0"
  "--mlfq-aging-dynamic-waiting-high" "0"
  "--mlfq-aging-dynamic-min-factor" "0"
  "--mlfq-aging-dynamic-max-factor" "0"
  "--mlfq-sjf-token-chunk-size" "0"
  "--mlfq-sjf-weight" "0"
  "--mlfq-sjf-dynamic-waiting-low" "0"
  "--mlfq-sjf-dynamic-waiting-high" "0"
  "--mlfq-sjf-dynamic-min-factor" "0"
  "--mlfq-sjf-dynamic-max-factor" "0"
  "--mlfq-sjf-prefill-weight" "0"
  "--mlfq-sjf-decode-weight" "0"
  "--mlfq-locality-weight" "0"
  "--mlfq-locality-max-boost" "0"
  "--mlfq-las-token-chunk-size" "0"
  "--mlfq-las-weight" "0"
  "--mlfq-low-pressure-waiting-threshold" "0"
  "--mlfq-low-pressure-pending-tokens-threshold" "0"
  "--mlfq-low-pressure-ratio-threshold" "0"
  "--mlfq-phase-rps-baseline-max" "0"
  "--mlfq-phase-rps-las-max" "0"
  "--mlfq-phase-rps-window-s" "1"
  "--mlfq-phase-rps-ema-alpha" "0.2"
  "--mlfq-phase-rps-hysteresis" "0"
  "--mlfq-phase-min-seconds" "0"
  "--mlfq-phase-pressure-baseline-max" "0"
  "--mlfq-phase-pressure-las-max" "0"
  "--mlfq-phase-pressure-pending-tokens" "0"
  "--output-backpressure-pending-tokens" "0"
  "--output-backpressure-penalty" "0"
  "--output-backpressure-max-tokens" "0"
  "--output-backpressure-min-tokens" "0"
  "--output-backpressure-lag-seconds" "0"
  "--output-backpressure-update-interval-s" "0"
)

BENCH_ARGS=(
  "--backend" "openai"
  "--dataset-name" "$DATASET_NAME"
  "--num-prompts" "$NUM_PROMPTS"
  "--max-concurrency" "$MAX_CONCURRENCY"
  "--burstiness" "$BURSTINESS"
  "--ready-check-timeout-sec" "$READY_TIMEOUT_SEC"
  "--model" "$MODEL"
  "--percentile-metrics" "$PERCENTILE_METRICS"
  "--metric-percentiles" "$METRIC_PERCENTILES"
  "--save-result"
)

if [[ "$DATASET_NAME" == "random-mix" ]]; then
  BENCH_ARGS+=(
    "--random-mix-input-lens" "$RANDOM_MIX_INPUT_LENS"
    "--random-mix-output-lens" "$RANDOM_MIX_OUTPUT_LENS"
    "--random-mix-weights" "$RANDOM_MIX_WEIGHTS"
    "--random-range-ratio" "$RANDOM_RANGE_RATIO"
  )
fi

if [[ -n "$GOODPUT" ]]; then
  IFS=" " read -r -a goodput_items <<< "$GOODPUT"
  BENCH_ARGS+=("--goodput" "${goodput_items[@]}")
fi

PY_ENV=(
  env
  MODEL="$MODEL"
  TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  HF_ENDPOINT="$HF_ENDPOINT"
  HF_HUB_DISABLE_TELEMETRY="$HF_HUB_DISABLE_TELEMETRY"
  HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-600}"
)

cleanup_server() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    wait "$pid" || true
  fi
}

run_case() {
  local label="$1"
  local outdir="$2"
  local request_rates="$3"
  shift 3
  local extra_args=("$@")

  if [[ "$PREFETCH_MODEL" == "1" && ! -d "$MODEL" ]]; then
    "${PY_ENV[@]}" "${PY_RUN[@]}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "")
if not model:
    raise SystemExit("MODEL is empty")
snapshot_download(repo_id=model, resume_download=True, local_files_only=False)
PY
  fi

  echo "Starting serve benchmark: ${label}"

  "${PY_ENV[@]}" "${PY_RUN[@]}" -m vllm.entrypoints.cli.main serve "$MODEL" \
    "${SERVER_ARGS[@]}" \
    "${COMMON_SCHED_ARGS[@]}" \
    "${extra_args[@]}" \
    >"${outdir}/${label}_server.log" 2>&1 &

  local server_pid="$!"
  trap "cleanup_server $server_pid" EXIT

  for rate in $request_rates; do
    "${PY_ENV[@]}" "${PY_RUN[@]}" -m vllm.entrypoints.cli.main bench serve \
      "${BENCH_ARGS[@]}" \
      "--label" "$label" \
      "--request-rate" "$rate" \
      "--result-dir" "$outdir" \
      "--result-filename" "${label}-rps${rate}.json"
  done

  cleanup_server "$server_pid"
  trap - EXIT
}

append_if_set() {
  local name="$1"
  local var="$2"
  if [[ -n ${!var+x} ]]; then
    CAND_ARGS+=("$name" "${!var}")
  fi
}

run_factor() {
  local factor="$1"
  local base_dir="${ROOT_OUTDIR}/${factor}/baseline"
  local cand_dir="${ROOT_OUTDIR}/${factor}/${factor}"
  local cand_label="$factor"

  mkdir -p "$base_dir" "$cand_dir"

  CAND_ARGS=("${BASELINE_SCHED_ARGS[@]}")
  case "$factor" in
    aging)
      CAND_ARGS+=("--mlfq-enable-experimental")
      CAND_ARGS+=(
        "--mlfq-aging-seconds" "$AGING_SECONDS"
        "--mlfq-aging-dynamic-waiting-low" "$AGING_DYNAMIC_WAITING_LOW"
        "--mlfq-aging-dynamic-waiting-high" "$AGING_DYNAMIC_WAITING_HIGH"
        "--mlfq-aging-dynamic-min-factor" "$AGING_DYNAMIC_MIN_FACTOR"
        "--mlfq-aging-dynamic-max-factor" "$AGING_DYNAMIC_MAX_FACTOR"
      )
      ;;
    sjf)
      CAND_ARGS+=("--mlfq-enable-experimental")
      CAND_ARGS+=(
        "--mlfq-sjf-token-chunk-size" "$SJF_TOKEN_CHUNK_SIZE"
        "--mlfq-sjf-weight" "$SJF_WEIGHT"
        "--mlfq-sjf-dynamic-waiting-low" "$SJF_DYNAMIC_WAITING_LOW"
        "--mlfq-sjf-dynamic-waiting-high" "$SJF_DYNAMIC_WAITING_HIGH"
        "--mlfq-sjf-dynamic-min-factor" "$SJF_DYNAMIC_MIN_FACTOR"
        "--mlfq-sjf-dynamic-max-factor" "$SJF_DYNAMIC_MAX_FACTOR"
      )
      append_if_set "--mlfq-sjf-prefill-weight" SJF_PREFILL_WEIGHT
      append_if_set "--mlfq-sjf-decode-weight" SJF_DECODE_WEIGHT
      ;;
    locality)
      CAND_ARGS+=("--mlfq-enable-experimental")
      CAND_ARGS+=(
        "--mlfq-locality-weight" "$LOCALITY_WEIGHT"
        "--mlfq-locality-max-boost" "$LOCALITY_MAX_BOOST"
      )
      ;;
    las)
      CAND_ARGS+=("--mlfq-enable-experimental")
      CAND_ARGS+=(
        "--mlfq-las-token-chunk-size" "$LAS_TOKEN_CHUNK_SIZE"
        "--mlfq-las-weight" "$LAS_WEIGHT"
      )
      ;;
    backpressure)
      CAND_ARGS+=("--mlfq-enable-experimental")
      CAND_ARGS+=(
        "--output-backpressure-pending-tokens" "$BP_PENDING_TOKENS"
        "--output-backpressure-penalty" "$BP_PENALTY"
        "--output-backpressure-max-tokens" "$BP_MAX_TOKENS"
        "--output-backpressure-min-tokens" "$BP_MIN_TOKENS"
        "--output-backpressure-lag-seconds" "$BP_LAG_SECONDS"
        "--output-backpressure-update-interval-s" "$BP_UPDATE_INTERVAL_S"
      )
      ;;
    *)
      echo "Unknown FACTOR: $factor (supported: aging, sjf, locality, las, backpressure)."
      exit 1
      ;;
  esac

  run_case "baseline" "$base_dir" "$REQUEST_RATES" "${BASELINE_SCHED_ARGS[@]}"
  run_case "$cand_label" "$cand_dir" "$REQUEST_RATES" "${CAND_ARGS[@]}"

  python3 tools/bench_compare_runs.py \
    "$base_dir" \
    "$cand_dir" \
    --baseline-name "baseline" \
    --candidate-name "$cand_label"
}

for factor in $FACTORS; do
  run_factor "$factor"
done
