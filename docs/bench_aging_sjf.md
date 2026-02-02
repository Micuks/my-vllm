# Aging-SJF Benchmarks

This note captures the best observed split-profile results so far for the
aging-SJF scheduler changes.

## Best Observed Split-Profile Result (Qwen3-4B)

Run:

```bash
PROFILE=split MODEL=Qwen/Qwen3-4B tools/bench_aging_sjf_compare.sh
```

Context:

- Dataset: `random-mix` (inputs 256/1024/2048, outputs 32/128/512, weights 0.5/0.3/0.2).
- Burstiness: 2.0 (pressure profile defaults).
- Goodput thresholds: `ttft:2000 tpot:200 e2el:30000`.
- Results under: `bench_results/20260202-121115/high`.

RPS 128 (baseline -> aging-sjf):

- request_throughput: 3.534 -> 4.275 (+21.0%)
- request_goodput: 2.302 -> 3.119 (+35.5%)
- total_token_throughput: 2976.538 -> 3418.793 (+14.9%)
- mean_ttft_ms: 1683.397 -> 1166.045 (-30.7%)
- p95_ttft_ms: 6619.136 -> 3652.495 (-44.8%)
- mean_tpot_ms: 128.868 -> 118.186 (-8.3%)
- mean_e2el_ms: 14699.704 -> 12410.723 (-15.6%)
- completed/failed: 218/82 -> 244/56

RPS 256 (baseline -> aging-sjf):

- request_throughput: 4.222 -> 4.271 (+1.2%)
- request_goodput: 3.361 -> 3.571 (+6.3%)
- total_token_throughput: 3426.542 -> 3415.935 (-0.3%)
- mean_ttft_ms: 933.016 -> 922.175 (-1.2%)
- p95_ttft_ms: 3302.518 -> 2919.526 (-11.6%)
- mean_tpot_ms: 113.142 -> 113.358 (+0.2%)
- mean_e2el_ms: 12540.580 -> 12336.948 (-1.6%)
- completed/failed: 245/55 -> 244/56
