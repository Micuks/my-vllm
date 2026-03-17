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
- 2026-02-04：针对低压退化与相位误触发，新增“低压旁路 + 压力相位判定”：
  - 低压旁路判定升级为 waiting + pending_tokens + in/out RPS ratio。
  - 相位切换优先使用压力比值（可选 pending gate），回退到纯 RPS。
  - 加入日志：低压旁路启停 + 相位切换明细（含 rps_in/rps_out/ratio/pending）。

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

## 低压旁路 + 压力相位判定（验证用）

新增参数（CLI）：

- `--mlfq-low-pressure-pending-tokens-threshold`
- `--mlfq-low-pressure-ratio-threshold`
- `--mlfq-phase-pressure-baseline-max`
- `--mlfq-phase-pressure-las-max`
- `--mlfq-phase-pressure-pending-tokens`

建议初始值（便于验证，不保证最优）：

- 低压旁路：`--mlfq-low-pressure-waiting-threshold 8`
- 低压 pending：`--mlfq-low-pressure-pending-tokens-threshold 256`
- 低压 ratio：`--mlfq-low-pressure-ratio-threshold 1.10`
- 压力相位：`--mlfq-phase-pressure-baseline-max 1.10`
- 压力相位：`--mlfq-phase-pressure-las-max 1.35`
- 相位 pending gate：`--mlfq-phase-pressure-pending-tokens 256`
- 相位滞回与停留：`--mlfq-phase-rps-hysteresis 0.05 --mlfq-phase-min-seconds 5`
- 说明：当 ratio 还不可用（例如起跑阶段）时保持 baseline，不会回退到纯 RPS 相位。

验证命令（只跑低压 2/4 与高压 128/256）：

```bash
# 低压验证：确认 low-pressure bypass 生效
PROFILE=low \
MODEL=Qwen/Qwen3-4B \
MLFQ_LOW_PRESSURE_WAITING_THRESHOLD=8 \
MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD=256 \
MLFQ_LOW_PRESSURE_RATIO_THRESHOLD=1.10 \
MLFQ_PHASE_PRESSURE_BASELINE_MAX=1.10 \
MLFQ_PHASE_PRESSURE_LAS_MAX=1.35 \
MLFQ_PHASE_PRESSURE_PENDING_TOKENS=256 \
MLFQ_PHASE_RPS_HYSTERESIS=0.05 \
MLFQ_PHASE_MIN_SECONDS=5 \
REQUEST_RATES="2 4" \
tools/bench_aging_sjf_compare.sh

# 高压验证：确认相位切换和高压收益
PROFILE=high \
MODEL=Qwen/Qwen3-4B \
MLFQ_LOW_PRESSURE_WAITING_THRESHOLD=8 \
MLFQ_LOW_PRESSURE_PENDING_TOKENS_THRESHOLD=256 \
MLFQ_LOW_PRESSURE_RATIO_THRESHOLD=1.10 \
MLFQ_PHASE_PRESSURE_BASELINE_MAX=1.10 \
MLFQ_PHASE_PRESSURE_LAS_MAX=1.35 \
MLFQ_PHASE_PRESSURE_PENDING_TOKENS=256 \
MLFQ_PHASE_RPS_HYSTERESIS=0.05 \
MLFQ_PHASE_MIN_SECONDS=5 \
REQUEST_RATES="128 256" \
tools/bench_aging_sjf_compare.sh
```

日志检查点（APIServer 输出）：

- `MLFQ low-pressure bypass enabled/disabled`
- `MLFQ phase switch: baseline -> las` / `las -> aging-sjf`

## 单因子 A/B 验证脚本

脚本：`tools/bench_single_factor_compare.sh`

注意：所有实验性调度特性默认关闭，需显式传
`--mlfq-enable-experimental` 才会生效；脚本已为 candidate 自动开启。

默认参数（可通过环境变量覆盖）：

- RPS：`2 4 128 256`
- Dataset：`random-mix`
- Aging：`AGING_SECONDS=5.0`
- SJF：`SJF_TOKEN_CHUNK_SIZE=256`、`SJF_WEIGHT=1.0`
- Locality：`LOCALITY_WEIGHT=1.0`
- LAS：`LAS_TOKEN_CHUNK_SIZE=256`、`LAS_WEIGHT=1.0`
- Backpressure：`BP_PENDING_TOKENS=64`、`BP_PENALTY=1`、`BP_MAX_TOKENS=8`

用法示例（单个因素）：

```bash
FACTOR=aging MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
FACTOR=sjf MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
FACTOR=locality MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
FACTOR=las MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
FACTOR=backpressure MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
```

批量（多因素顺序跑）：

```bash
FACTORS="aging sjf locality" MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
```

## 单因子结果结论（20260205-113542）

结论概览（RPS 2/4/128/256）：

- aging：低压明显退化，高压（RPS 128）明显改善，RPS 256 基本持平。
- sjf：低压轻微退化/混合，高压改善最稳（吞吐、TTFT/E2EL 皆改善）。
- locality：几乎全线负收益，建议禁用。
- las：整体偏负，RPS 256 成功率明显下降，建议禁用。
- backpressure：低压略有改善但中高压退化，效果不稳定，建议禁用。

建议策略：

- 低压完全 baseline（禁用 SJF/aging/locality/LAS/backpressure）。
- 高压只启 SJF（优先），aging 作为备选；locality/LAS/backpressure 默认关闭。
- SJF 温和参数起步：`SJF_WEIGHT=0.5`、`SJF_TOKEN_CHUNK_SIZE=512`，
  `SJF_PREFILL_WEIGHT=0`、`SJF_DECODE_WEIGHT=0.5`。

下一步验证命令（高压 128/256）：

```bash
PROFILE=high \
MODEL=Qwen/Qwen3-4B \
MLFQ_SJF_TOKEN_CHUNK_SIZE=512 \
MLFQ_SJF_WEIGHT=0.5 \
MLFQ_SJF_PREFILL_WEIGHT=0 \
MLFQ_SJF_DECODE_WEIGHT=0.5 \
MLFQ_AGING_SECONDS=0 \
MLFQ_LOCALITY_WEIGHT=0 \
MLFQ_LOW_PRESSURE_WAITING_THRESHOLD=0 \
REQUEST_RATES="128 256" \
tools/bench_aging_sjf_compare.sh
```

## 单因子结果结论（20260205-155128）

结论概览（RPS 2/4/128/256）：

- aging：低压明显退化；RPS 128 有延迟改善但吞吐略降；RPS 256 基本持平偏负。
- sjf：RPS 256 明显改善（吞吐与 TTFT/E2EL 变好），RPS 128 与低压退化；适合“极高压专用”。
- locality：多数点吞吐/延迟不占优，建议关闭。
- las：低压略有改善但 RPS 128 明显退化，整体不稳，建议关闭。
- backpressure：RPS 128 明显改善；RPS 256 吞吐小涨但 TTFT 变差，偏混合。

建议策略：

- 默认关闭实验性调度；仅在“明确高压场景”手动开启。
- 若要试用：RPS 128 优先 backpressure；RPS 256 优先 sjf。

下一步复测命令（验证稳定性）：

```bash
FACTOR=backpressure REQUEST_RATES="128" MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
FACTOR=sjf REQUEST_RATES="256" MODEL=Qwen/Qwen3-4B tools/bench_single_factor_compare.sh
```

## 复测结论（20260205-174624 / 20260205-194017）

复测结果显示策略收益不稳定，且多为负收益：

- backpressure @ RPS 128（20260205-174624）：吞吐下降 ~16%，TTFT/TPOT/E2EL 全面变差，成功率无变化。
- sjf @ RPS 256（20260205-194017）：吞吐下降 ~19%，TTFT/TPOT/E2EL 全面变差，成功率无变化。

最终建议：

- 默认关闭所有实验性调度（aging/SJF/LAS/locality/backpressure/phase）。
- 仅在明确“特定负载 + 预先验证”时手动开启，并优先短跑验证。

## 最终推荐配置（默认关闭实验性调度）

建议在生产/默认 benchmark 中显式关闭实验性调度：

```bash
--scheduling-policy priority \
--no-mlfq-enable-experimental
```

如需短跑验证某个策略，再显式开启并设置对应参数：

```bash
--mlfq-enable-experimental \
--mlfq-sjf-token-chunk-size 512 \
--mlfq-sjf-weight 0.5
```

## 基线容量扫描（新增）

先做 baseline 容量扫描，再做调度策略对比。这样可以避免把“配置不佳”误判为“调度策略退化”。

脚本：`tools/bench_baseline_sweep.sh`

默认会扫描：

- `MAX_NUM_SEQS_LIST=48 64 80`
- `MAX_NUM_BATCHED_TOKENS_LIST=6144 8192 10240`
- `GPU_MEMORY_UTILIZATION_LIST=0.88 0.90`
- `REQUEST_RATES=2 4 8 16 32 64 128 256`

推荐命令（Qwen3-4B）：

```bash
MODEL=Qwen/Qwen3-4B \
GOODPUT="ttft:2000 tpot:200 e2el:30000" \
tools/bench_baseline_sweep.sh
```

输出内容：

- 每个配置的 benchmark JSON：`bench_results/<ts>/configs/<config>/rps*.json`
- 自动汇总：`bench_results/<ts>/sweep_summary.json`
- 终端会打印高压目标（默认 `HIGH_RPS_LIST=128 256`）的 Top 配置。

抗抖动开关（`tools/bench_baseline_sweep.sh`）：

- `BENCH_RETRIES`：每个 RPS 的 `bench serve` 重试次数（默认 `4`）
- `BENCH_RETRY_SLEEP_SEC`：首次重试等待秒数（默认 `2`）
- `BENCH_RETRY_BACKOFF`：重试退避倍数（默认 `2`）
- `BENCH_RETRY_FORCE_OFFLINE_AFTER_FAILURE`：首个失败后用离线模式重试（默认 `1`）
- `BENCH_FORCE_LOCAL_TOKENIZER`：优先使用本地 tokenizer snapshot（默认 `1`）
- `SKIP_EXISTING_RESULTS`：断点续跑时跳过已存在 `rps*.json`（默认 `1`）

网络不稳定时推荐：

```bash
MODEL=Qwen/Qwen3-4B \
BENCH_RETRIES=6 \
BENCH_RETRY_SLEEP_SEC=2 \
BENCH_RETRY_BACKOFF=2 \
BENCH_RETRY_FORCE_OFFLINE_AFTER_FAILURE=1 \
tools/bench_baseline_sweep.sh
```

建议流程：

1. 先运行 baseline sweep，选出高压 `request_goodput` 最优配置。
2. 固定该配置，再运行 `tools/bench_aging_sjf_compare.sh` 或 `tools/bench_single_factor_compare.sh`。
3. 最后再对比策略收益，减少实验噪声。
