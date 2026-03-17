# MLFQ Backpressure Benchmark

This guide records how to run the baseline vs mlfq-backpressure benchmarks
using the provided scripts and vLLM bench tooling.

## What This Measures

- Online serving: `vllm bench serve` via OpenAI-compatible API server.
- Offline throughput: `vllm bench throughput` (no API server).

Backpressure only affects the API server path, so online results are the
primary signal. Offline throughput is included for completeness.

## Prereqs

- Model must be available locally or cached in HuggingFace cache.
- `uv` is optional; the script falls back to `python` if not present.
- CUDA build uses a precompiled wheel if available.

## Scripts

- Runner: `tools/bench_mlfq_backpressure.sh`
- Summary: `tools/bench_mlfq_backpressure_summary.py`

## Default Parameters

- `REQUEST_RATES="2 8"`
- `NUM_PROMPTS=100`
- `INPUT_LEN=256`
- `OUTPUT_LEN=128`
- `MAX_CONCURRENCY=32`
- `PERCENTILE_METRICS="ttft,e2el"`
- `METRIC_PERCENTILES="95,99"`

## Run Baseline vs MLFQ Backpressure

```bash
MODEL=facebook/opt-125m tools/bench_mlfq_backpressure.sh
```

Results are saved under `bench_results/<timestamp>/`.

## Higher Load Example

```bash
MODEL=facebook/opt-125m \
REQUEST_RATES="64 96" \
NUM_PROMPTS=300 \
MAX_CONCURRENCY=128 \
INPUT_LEN=512 \
OUTPUT_LEN=256 \
tools/bench_mlfq_backpressure.sh
```

## Summarize Results

```bash
tools/bench_mlfq_backpressure_summary.py bench_results/<timestamp>
```

## Environment Overrides

You can override any of the following:

- `REQUEST_RATES`, `NUM_PROMPTS`, `INPUT_LEN`, `OUTPUT_LEN`, `MAX_CONCURRENCY`
- `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`
- `PERCENTILE_METRICS`, `METRIC_PERCENTILES`
- `BACKPRESSURE_YELLOW_FREE_RATIO`, `BACKPRESSURE_RED_FREE_RATIO`
- `BACKPRESSURE_SHORT_MAX_TOKENS`, `BACKPRESSURE_PRIORITY_CUTOFF`
- `BACKPRESSURE_REFRESH_S`

Example:

```bash
MODEL=facebook/opt-125m \
REQUEST_RATES="128 160" \
NUM_PROMPTS=400 \
MAX_CONCURRENCY=256 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_NUM_BATCHED_TOKENS=4096 \
tools/bench_mlfq_backpressure.sh
```
