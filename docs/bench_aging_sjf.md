# Aging-SJF Benchmarks

本文记录 aging‑SJF 调度改动的探索过程与当前最好结果。

## 当前最好结果（Qwen3‑4B，split profile）

运行命令：

```bash
PROFILE=split MODEL=Qwen/Qwen3-4B tools/bench_aging_sjf_compare.sh
```

实验上下文：

- 数据集：`random-mix`（输入 256/1024/2048，输出 32/128/512，权重 0.5/0.3/0.2）
- 突发度：2.0（pressure profile 默认）
- Goodput 阈值：`ttft:2000 tpot:200 e2el:30000`
- 结果路径：`bench_results/20260202-121115/high`

RPS 128（baseline -> aging‑sjf）：

- request_throughput: 3.534 -> 4.275 (+21.0%)
- request_goodput: 2.302 -> 3.119 (+35.5%)
- total_token_throughput: 2976.538 -> 3418.793 (+14.9%)
- mean_ttft_ms: 1683.397 -> 1166.045 (-30.7%)
- p95_ttft_ms: 6619.136 -> 3652.495 (-44.8%)
- mean_tpot_ms: 128.868 -> 118.186 (-8.3%)
- mean_e2el_ms: 14699.704 -> 12410.723 (-15.6%)
- completed/failed: 218/82 -> 244/56

RPS 256（baseline -> aging‑sjf）：

- request_throughput: 4.222 -> 4.271 (+1.2%)
- request_goodput: 3.361 -> 3.571 (+6.3%)
- total_token_throughput: 3426.542 -> 3415.935 (-0.3%)
- mean_ttft_ms: 933.016 -> 922.175 (-1.2%)
- p95_ttft_ms: 3302.518 -> 2919.526 (-11.6%)
- mean_tpot_ms: 113.142 -> 113.358 (+0.2%)
- mean_e2el_ms: 12540.580 -> 12336.948 (-1.6%)
- completed/failed: 245/55 -> 244/56

## 探索与改进过程

- 初始问题：MLFQ + backpressure 在中高 RPS 会拒绝大量请求，导致“只看成功请求”
  的延迟改善失去意义。目标转向**不拒绝请求**且最大化吞吐。
- 第一阶段：在 priority 调度里加入 Aging + SJF。离线吞吐基本不变，但低/中 RPS
  时延退化，整体吞吐提升不明显。
- 加入低压旁路：等待队列小的时候跳过 SJF/aging/locality，尽量贴近 baseline 行为。
- 引入 split profile：低压（RPS 2–64）目标“低延迟/高成功率”，高压（RPS
  128–256）目标“goodput/吞吐最大化”。
- 完全分离 low/high 参数：低压接近 baseline，高压启用 SJF/aging。
- 最佳结果记录：`bench_results/20260202-121115/high`（见上文）。
- 后续 split 跑法（如 `bench_results/20260203-005810`）显示：低压对 SJF/aging
  权重非常敏感，RPS 2–32 常退化；RPS 64/128 有时能明显提升 goodput/成功率。

## 关键结论

- 低压敏感性真实存在：低压区间只要 SJF/aging 不为 0，TTFT/TPOT/E2EL 容易退化。
- 高压收益不稳定但可见：RPS 128 常提升吞吐/成功率，同时 TTFT 改善；RPS 256
  收益较小，p99 可能回退。
- 成功率/Goodput 比均值延迟更能反映高压效果。

## 下一步方向（候选）

- LAS（Least Attained Service）：按“已服务 token 数”惩罚，更接近公平且降低
  队头阻塞，不依赖作业大小预测。
- Prefill/Decode 分治：长 prompt prefill 分块，让短请求 decode 插空以改善 TTFT。
- 背压改为“配额”而非“拒绝”：输出缓冲满时降低该请求每轮 token 配额。

## LAS Bench 结果（Qwen3-4B，split profile）

运行命令（脚本：`tools/bench_las_compare.sh`）：

```bash
PROFILE=split MODEL=Qwen/Qwen3-4B tools/bench_las_compare.sh
```

结果路径：`bench_results/20260203-171716`

低压段（RPS 2/4/8/16/32/64）总结：

- RPS 2/4：吞吐与 goodput 回退，TTFT/TPOT/E2EL 明显变差，失败数略增。
- RPS 8/16/32：吞吐与 goodput 改善，TTFT/TPOT/E2EL 明显变好。
- RPS 64：成功率上升，但吞吐和时延严重退化（说明 LAS 权重在该负载过重）。

高压段（RPS 128/256）总结：

- RPS 128：吞吐/goodput/成功率提升，均值延迟改善，p95/p99 TTFT 小幅变差。
- RPS 256：吞吐/goodput/TTFT/TPOT/E2EL 全面改善，尾部延迟显著改善。

## LAS 参数建议（基于 20260203-171716）

目标：低压不退化，中高压保留收益。

- 低压旁路：`MLFQ_LOW_PRESSURE_WAITING_THRESHOLD=8` 或更大（如 16）。
- 低压权重：`MLFQ_LAS_LOW_TOKEN_CHUNK_SIZE=0`、`MLFQ_LAS_LOW_WEIGHT=0.0`
- 高压权重：建议从温和开始（例如 `MLFQ_LAS_TOKEN_CHUNK_SIZE=512`、
  `MLFQ_LAS_WEIGHT=0.5`），观察 RPS 64 的退化是否消失，再逐步加大。
- 压力段划分：如果 RPS 64 仍退化，可把 split 调整为低压 `2 4 8 16 32`、
  高压 `64 128 256`，单独对 RPS 64 选更温和配置。
