# bench_lib.sh - shared infrastructure for benchmark comparison scripts.
# Source this file; do NOT execute it directly.

# ── env file ────────────────────────────────────────────────────────────
ENV_FILE="${BENCH_ENV_FILE:-tools/bench_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# ── MODEL validation ────────────────────────────────────────────────────
MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set MODEL to the model path or name (e.g., MODEL=Qwen/Qwen3-4B)."
  exit 1
fi

# ── SET flags (detect whether the caller pre-set env vars) ──────────────
PROFILE="${PROFILE:-}"
REQUEST_RATES_SET=0
RPS_LIST_SET=0
PRESSURE_SPLIT_SET=0
LOW_REQUEST_RATES_SET=0
HIGH_REQUEST_RATES_SET=0
MID_REQUEST_RATES_SET=0
BURSTINESS_SET=0
GOODPUT_SET=0
if [[ -n ${REQUEST_RATES+x} ]]; then REQUEST_RATES_SET=1; fi
if [[ -n ${RPS_LIST+x} ]]; then RPS_LIST_SET=1; fi
if [[ -n ${PRESSURE_SPLIT+x} ]]; then PRESSURE_SPLIT_SET=1; fi
if [[ -n ${LOW_REQUEST_RATES+x} ]]; then LOW_REQUEST_RATES_SET=1; fi
if [[ -n ${HIGH_REQUEST_RATES+x} ]]; then HIGH_REQUEST_RATES_SET=1; fi
if [[ -n ${MID_REQUEST_RATES+x} ]]; then MID_REQUEST_RATES_SET=1; fi
if [[ -n ${BURSTINESS+x} ]]; then BURSTINESS_SET=1; fi
if [[ -n ${GOODPUT+x} ]]; then GOODPUT_SET=1; fi
TRI_SPLIT="${TRI_SPLIT:-0}"

# ── Common env vars ─────────────────────────────────────────────────────
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROOT_OUTDIR="${OUTDIR:-bench_results/$(date +%Y%m%d-%H%M%S)}"
BASE_OUTDIR="${ROOT_OUTDIR}/baseline"
CAND_OUTDIR="${ROOT_OUTDIR}/${CANDIDATE_LABEL}"
REQUEST_RATES="${REQUEST_RATES:-2 4 8 16 32 64 128 256}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
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
PRESSURE_SPLIT="${PRESSURE_SPLIT:-0}"
LOW_REQUEST_RATES="${LOW_REQUEST_RATES:-2 4 8 16 32 64}"
MID_REQUEST_RATES="${MID_REQUEST_RATES:-32 64}"
HIGH_REQUEST_RATES="${HIGH_REQUEST_RATES:-128 256}"

# ── Server config vars ──────────────────────────────────────────────────
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"
MLFQ_TOKEN_CHUNK_SIZE="${MLFQ_TOKEN_CHUNK_SIZE:-256}"
MLFQ_WAITING_BUDGET_FRACTION="${MLFQ_WAITING_BUDGET_FRACTION:-0.2}"

# ── Common scheduling config vars ───────────────────────────────────────
MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD="${MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD:-0}"
MLFQ_LOW_PRESSURE_RATIO_THRESHOLD="${MLFQ_LOW_PRESSURE_RATIO_THRESHOLD:-0.0}"
MLFQ_LOW_PRESSURE_WAITING_THRESHOLD="${MLFQ_LOW_PRESSURE_WAITING_THRESHOLD:-0}"
MLFQ_PHASE_RPS_WINDOW_S="${MLFQ_PHASE_RPS_WINDOW_S:-1.0}"
MLFQ_PHASE_RPS_EMA_ALPHA="${MLFQ_PHASE_RPS_EMA_ALPHA:-0.2}"
MLFQ_PHASE_RPS_HYSTERESIS="${MLFQ_PHASE_RPS_HYSTERESIS:-0.0}"
MLFQ_PHASE_MIN_SECONDS="${MLFQ_PHASE_MIN_SECONDS:-1.0}"
MLFQ_PHASE_PRESSURE_BASELINE_MAX="${MLFQ_PHASE_PRESSURE_BASELINE_MAX:-0.0}"
MLFQ_PHASE_PRESSURE_LAS_MAX="${MLFQ_PHASE_PRESSURE_LAS_MAX:-0.0}"
MLFQ_PHASE_PRESSURE_PENDING_TOKENS="${MLFQ_PHASE_PRESSURE_PENDING_TOKENS:-0}"

# ── HF / Transformers env vars ──────────────────────────────────────────
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
PREFETCH_MODEL="${PREFETCH_MODEL:-0}"

# ── Common profile cases ────────────────────────────────────────────────
if [[ -n "$PROFILE" ]]; then
  case "$PROFILE" in
    pressure)
      if [[ "$PRESSURE_SPLIT_SET" -eq 0 ]]; then
        PRESSURE_SPLIT=1
      fi
      if [[ "$LOW_REQUEST_RATES_SET" -eq 0 ]]; then
        LOW_REQUEST_RATES="2 4 8 16 32 64"
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
      ;;
    low)
      if [[ "$PRESSURE_SPLIT_SET" -eq 0 ]]; then
        PRESSURE_SPLIT=0
      fi
      if [[ "$REQUEST_RATES_SET" -eq 0 ]]; then
        REQUEST_RATES="2 4 8 16 32 64"
      fi
      if [[ "$BURSTINESS_SET" -eq 0 ]]; then
        BURSTINESS="1"
      fi
      ;;
    split)
      if [[ "$PRESSURE_SPLIT_SET" -eq 0 ]]; then
        PRESSURE_SPLIT=1
      fi
      if [[ "$LOW_REQUEST_RATES_SET" -eq 0 ]]; then
        LOW_REQUEST_RATES="2 4 8 16 32 64"
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
      ;;
    default)
      ;;
    # Wrapper scripts may define additional profiles (e.g., "high", "tri").
    # Unknown profiles are handled after the lib is sourced.
  esac
fi

if [[ "$RPS_LIST_SET" -eq 1 && "$REQUEST_RATES_SET" -eq 0 ]]; then
  REQUEST_RATES="$RPS_LIST"
  REQUEST_RATES_SET=1
fi

# ── Create output directories ───────────────────────────────────────────
mkdir -p "$BASE_OUTDIR" "$CAND_OUTDIR"

# ── Python runner ────────────────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
else
  PY_RUN=(python3)
fi

# ── Arg arrays ───────────────────────────────────────────────────────────
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

# Comprehensive baseline: zero-out ALL experimental flags from both scripts.
BASELINE_SCHED_ARGS=(
  "--no-mlfq-enable-experimental"
  "--mlfq-aging-seconds" "0"
  "--mlfq-aging-dynamic-waiting-low" "0"
  "--mlfq-aging-dynamic-waiting-high" "0"
  "--mlfq-aging-dynamic-min-factor" "0"
  "--mlfq-aging-dynamic-max-factor" "0"
  "--mlfq-low-pressure-waiting-threshold" "0"
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
  "--mlfq-las-token-chunk-size" "0"
  "--mlfq-las-weight" "0"
)

# ── BENCH_ARGS ───────────────────────────────────────────────────────────
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
else
  BENCH_ARGS+=(
    "--input-len" "$INPUT_LEN"
    "--output-len" "$OUTPUT_LEN"
    "--random-range-ratio" "$RANDOM_RANGE_RATIO"
  )
fi

if [[ -n "$GOODPUT" ]]; then
  IFS=" " read -r -a goodput_items <<< "$GOODPUT"
  BENCH_ARGS+=("--goodput" "${goodput_items[@]}")
fi

# ── PY_ENV ───────────────────────────────────────────────────────────────
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

# ── Functions ────────────────────────────────────────────────────────────
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

run_group() {
  local group_label="$1"
  local request_rates="$2"
  local candidate_args_name="${3:-CANDIDATE_SCHED_ARGS}"
  local cand_label="${4:-${CANDIDATE_LABEL}}"
  local base_dir="${ROOT_OUTDIR}/${group_label}/baseline"
  local cand_dir="${ROOT_OUTDIR}/${group_label}/${cand_label}"

  mkdir -p "$base_dir" "$cand_dir"

  local -n candidate_args_ref="$candidate_args_name"

  run_case "baseline" "$base_dir" "$request_rates" "${BASELINE_SCHED_ARGS[@]}"
  run_case "$cand_label" "$cand_dir" "$request_rates" "${candidate_args_ref[@]}"

  python3 tools/bench_compare_runs.py \
    "$base_dir" \
    "$cand_dir" \
    --baseline-name "baseline" \
    --candidate-name "$cand_label"
}
