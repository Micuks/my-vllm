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

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROOT_OUTDIR="${OUTDIR:-bench_results/$(date +%Y%m%d-%H%M%S)}"
CONFIG_ROOT="${ROOT_OUTDIR}/configs"

REQUEST_RATES="${REQUEST_RATES:-2 4 8 16 32 64 128 256}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,e2el}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-50,95,99}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"

DATASET_NAME="${DATASET_NAME:-random-mix}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.2}"
RANDOM_MIX_INPUT_LENS="${RANDOM_MIX_INPUT_LENS:-256,1024,2048}"
RANDOM_MIX_OUTPUT_LENS="${RANDOM_MIX_OUTPUT_LENS:-32,128,512}"
RANDOM_MIX_WEIGHTS="${RANDOM_MIX_WEIGHTS:-0.5,0.3,0.2}"
BURSTINESS="${BURSTINESS:-1.0}"
GOODPUT="${GOODPUT:-}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-48 64 80}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-6144 8192 10240}"
GPU_MEMORY_UTILIZATION_LIST="${GPU_MEMORY_UTILIZATION_LIST:-0.88 0.90}"

TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
PREFETCH_MODEL="${PREFETCH_MODEL:-0}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-1}"
BENCH_RETRIES="${BENCH_RETRIES:-4}"
BENCH_RETRY_SLEEP_SEC="${BENCH_RETRY_SLEEP_SEC:-2}"
BENCH_RETRY_BACKOFF="${BENCH_RETRY_BACKOFF:-2}"
BENCH_RETRY_FORCE_OFFLINE_AFTER_FAILURE="${BENCH_RETRY_FORCE_OFFLINE_AFTER_FAILURE:-1}"
BENCH_TOKENIZER="${BENCH_TOKENIZER:-}"
BENCH_FORCE_LOCAL_TOKENIZER="${BENCH_FORCE_LOCAL_TOKENIZER:-1}"
BENCH_TOKENIZER_ALLOW_ONLINE="${BENCH_TOKENIZER_ALLOW_ONLINE:-1}"

if [[ "$TRANSFORMERS_OFFLINE" == "1" || "$HF_HUB_OFFLINE" == "1" ]]; then
  BENCH_TOKENIZER_ALLOW_ONLINE=0
fi

mkdir -p "$CONFIG_ROOT"

if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
else
  PY_RUN=(python3)
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

PY_ENV_OFFLINE=(
  env
  MODEL="$MODEL"
  TRANSFORMERS_OFFLINE="1"
  HF_HUB_OFFLINE="1"
  HF_ENDPOINT="$HF_ENDPOINT"
  HF_HUB_DISABLE_TELEMETRY="$HF_HUB_DISABLE_TELEMETRY"
  HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-600}"
)

SERVER_ARGS=(
  "--host" "$HOST"
  "--port" "$PORT"
  "--max-model-len" "$MAX_MODEL_LEN"
  "--scheduling-policy" "$SCHEDULING_POLICY"
  "--no-mlfq-enable-experimental"
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

cleanup_server() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    wait "$pid" || true
  fi
}

prefetch_model_once() {
  if [[ "$PREFETCH_MODEL" != "1" || -d "$MODEL" ]]; then
    return
  fi
  "${PY_ENV[@]}" "${PY_RUN[@]}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "")
if not model:
    raise SystemExit("MODEL is empty")
snapshot_download(repo_id=model, resume_download=True, local_files_only=False)
PY
}

resolve_local_tokenizer_path() {
  if [[ -n "$BENCH_TOKENIZER" ]]; then
    echo "$BENCH_TOKENIZER"
    return 0
  fi
  if [[ "$BENCH_FORCE_LOCAL_TOKENIZER" != "1" || -d "$MODEL" ]]; then
    echo ""
    return 0
  fi

  "${PY_ENV[@]}" "${PY_RUN[@]}" - <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "")
allow_online = os.environ.get("BENCH_TOKENIZER_ALLOW_ONLINE", "1") == "1"
if not model:
    raise SystemExit("MODEL is empty")

try:
    # First try cache-only to avoid network flakiness.
    local_path = snapshot_download(
        repo_id=model,
        local_files_only=True,
        resume_download=True,
    )
    print(local_path)
    raise SystemExit(0)
except Exception:
    if not allow_online:
        raise

local_path = snapshot_download(
    repo_id=model,
    local_files_only=False,
    resume_download=True,
)
print(local_path)
PY
}

run_bench_with_retry() {
  local config_label="$1"
  local outdir="$2"
  local rate="$3"
  shift 3
  local tokenizer_args=("$@")

  local attempt=1
  local sleep_s="$BENCH_RETRY_SLEEP_SEC"
  while (( attempt <= BENCH_RETRIES )); do
    local -a env_arr=("${PY_ENV[@]}")
    if (( attempt > 1 )) && [[ "$BENCH_RETRY_FORCE_OFFLINE_AFTER_FAILURE" == "1" ]]; then
      env_arr=("${PY_ENV_OFFLINE[@]}")
    fi

    if "${env_arr[@]}" "${PY_RUN[@]}" -m vllm.entrypoints.cli.main bench serve \
      "${BENCH_ARGS[@]}" \
      "${tokenizer_args[@]}" \
      "--label" "$config_label" \
      "--request-rate" "$rate" \
      "--result-dir" "$outdir" \
      "--result-filename" "rps${rate}.json"; then
      return 0
    fi

    if (( attempt == BENCH_RETRIES )); then
      echo "bench serve failed after ${BENCH_RETRIES} attempts: config=${config_label}, rps=${rate}"
      return 1
    fi

    echo "bench serve failed (attempt ${attempt}/${BENCH_RETRIES}) config=${config_label}, rps=${rate}, retry in ${sleep_s}s..."
    sleep "$sleep_s"
    sleep_s=$(( sleep_s * BENCH_RETRY_BACKOFF ))
    attempt=$(( attempt + 1 ))
  done
}

run_config() {
  local config_label="$1"
  local max_num_seqs="$2"
  local max_num_batched_tokens="$3"
  local gpu_mem_util="$4"
  local outdir="${CONFIG_ROOT}/${config_label}"
  local tokenizer_path="${BENCH_TOKENIZER_RESOLVED:-}"
  local -a tokenizer_args=()
  if [[ -n "$tokenizer_path" ]]; then
    tokenizer_args=( "--tokenizer" "$tokenizer_path" )
  fi

  mkdir -p "$outdir"
  cat >"${outdir}/config.env" <<EOF
MODEL=${MODEL}
MAX_NUM_SEQS=${max_num_seqs}
MAX_NUM_BATCHED_TOKENS=${max_num_batched_tokens}
GPU_MEMORY_UTILIZATION=${gpu_mem_util}
REQUEST_RATES=${REQUEST_RATES}
DATASET_NAME=${DATASET_NAME}
BURSTINESS=${BURSTINESS}
GOODPUT=${GOODPUT}
EOF

  echo "Starting baseline sweep config: ${config_label}"

  "${PY_ENV[@]}" "${PY_RUN[@]}" -m vllm.entrypoints.cli.main serve "$MODEL" \
    "${SERVER_ARGS[@]}" \
    "--max-num-seqs" "$max_num_seqs" \
    "--max-num-batched-tokens" "$max_num_batched_tokens" \
    "--gpu-memory-utilization" "$gpu_mem_util" \
    >"${outdir}/server.log" 2>&1 &

  local server_pid="$!"
  trap "cleanup_server $server_pid" EXIT

  for rate in $REQUEST_RATES; do
    local result_file="${outdir}/rps${rate}.json"
    if [[ "$SKIP_EXISTING_RESULTS" == "1" && -f "$result_file" ]]; then
      echo "Skip existing result: ${result_file}"
      continue
    fi
    run_bench_with_retry "$config_label" "$outdir" "$rate" "${tokenizer_args[@]}"
  done

  cleanup_server "$server_pid"
  trap - EXIT
}

prefetch_model_once
BENCH_TOKENIZER_RESOLVED="$(resolve_local_tokenizer_path || true)"
if [[ -n "$BENCH_TOKENIZER_RESOLVED" ]]; then
  echo "Using local tokenizer path: $BENCH_TOKENIZER_RESOLVED"
else
  echo "Using tokenizer from model id: $MODEL"
fi

for max_num_seqs in $MAX_NUM_SEQS_LIST; do
  for max_num_batched_tokens in $MAX_NUM_BATCHED_TOKENS_LIST; do
    for gpu_mem_util in $GPU_MEMORY_UTILIZATION_LIST; do
      gpu_tag="${gpu_mem_util//./p}"
      label="seq${max_num_seqs}_tok${max_num_batched_tokens}_gpu${gpu_tag}"
      run_config "$label" "$max_num_seqs" "$max_num_batched_tokens" "$gpu_mem_util"
    done
  done
done

SWEEP_SUMMARY_JSON="${ROOT_OUTDIR}/sweep_summary.json"
"${PY_ENV[@]}" "${PY_RUN[@]}" - <<'PY' "$CONFIG_ROOT" "$SWEEP_SUMMARY_JSON"
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

config_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

rows = []
for config_dir in sorted(config_root.iterdir()):
    if not config_dir.is_dir():
        continue
    for result in sorted(config_dir.glob("rps*.json")):
        data = json.loads(result.read_text(encoding="utf-8"))
        data["config"] = config_dir.name
        rows.append(data)

if not rows:
    raise SystemExit("No benchmark JSON found for sweep summary.")

per_rps = defaultdict(list)
for row in rows:
    rps = row.get("request_rate")
    if rps is None:
        continue
    per_rps[int(rps)].append(row)

high_rps_env = os.environ.get("HIGH_RPS_LIST", "128 256")
high_rps = [int(x) for x in high_rps_env.split() if x.strip()]
if not high_rps:
    high_rps = sorted(per_rps.keys())[-2:]
if not any(rps in per_rps for rps in high_rps):
    high_rps = sorted(per_rps.keys())

def success_rate(row):
    completed = row.get("completed")
    failed = row.get("failed")
    if completed is None or failed is None:
        return None
    total = completed + failed
    if total <= 0:
        return None
    return completed / total

def obj_value(row):
    val = row.get("request_goodput")
    if val is None:
        val = row.get("request_throughput")
    return float(val) if val is not None else float("-inf")

def top_by_rps():
    result = {}
    for rps, items in sorted(per_rps.items()):
        best = max(items, key=obj_value)
        result[str(rps)] = {
            "best_config": best["config"],
            "objective": obj_value(best),
            "request_throughput": best.get("request_throughput"),
            "request_goodput": best.get("request_goodput"),
            "total_token_throughput": best.get("total_token_throughput"),
            "mean_ttft_ms": best.get("mean_ttft_ms"),
            "mean_e2el_ms": best.get("mean_e2el_ms"),
            "success_rate": success_rate(best),
            "completed": best.get("completed"),
            "failed": best.get("failed"),
        }
    return result

def rank_configs():
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)

    ranked = []
    for config, items in by_config.items():
        item_by_rps = {int(i["request_rate"]): i for i in items if i.get("request_rate") is not None}
        scores = []
        ttft = []
        e2el = []
        succ = []
        for rps in high_rps:
            row = item_by_rps.get(rps)
            if not row:
                continue
            scores.append(obj_value(row))
            if row.get("mean_ttft_ms") is not None:
                ttft.append(float(row["mean_ttft_ms"]))
            if row.get("mean_e2el_ms") is not None:
                e2el.append(float(row["mean_e2el_ms"]))
            s = success_rate(row)
            if s is not None:
                succ.append(s)
        if not scores:
            continue
        ranked.append(
            {
                "config": config,
                "high_rps": high_rps,
                "avg_objective": sum(scores) / len(scores),
                "avg_ttft_ms": (sum(ttft) / len(ttft)) if ttft else None,
                "avg_e2el_ms": (sum(e2el) / len(e2el)) if e2el else None,
                "avg_success_rate": (sum(succ) / len(succ)) if succ else None,
            }
        )
    ranked.sort(key=lambda x: x["avg_objective"], reverse=True)
    return ranked

payload = {
    "high_rps_list": high_rps,
    "top_by_rps": top_by_rps(),
    "ranked_configs": rank_configs(),
}
summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSweep summary (top configs by high-RPS objective):")
for idx, row in enumerate(payload["ranked_configs"][:5], start=1):
    succ = row["avg_success_rate"]
    succ_str = f"{succ * 100:.2f}%" if succ is not None else "n/a"
    ttft_str = f"{row['avg_ttft_ms']:.2f}" if row["avg_ttft_ms"] is not None else "n/a"
    e2el_str = f"{row['avg_e2el_ms']:.2f}" if row["avg_e2el_ms"] is not None else "n/a"
    print(
        f"{idx}. {row['config']}: "
        f"avg_obj={row['avg_objective']:.3f}, "
        f"avg_success={succ_str}, "
        f"avg_ttft_ms={ttft_str}, "
        f"avg_e2el_ms={e2el_str}"
    )

print(f"\nSummary JSON: {summary_path}")
PY

echo "Results saved under: $ROOT_OUTDIR"
