#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare vLLM serving benchmark JSONs between two result directories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percentile_from_list(
    percentiles: Iterable[Tuple[float, float]], target: float
) -> float | None:
    for p, value in percentiles:
        if abs(p - target) < 1e-6:
            return float(value)
    return None


def get_percentile(obj: Dict[str, Any], metric: str, p: int) -> float | None:
    key = f"p{p}_{metric}_ms"
    if key in obj:
        return obj.get(key)
    return percentile_from_list(obj.get(f"percentiles_{metric}_ms", []), p)


def load_dir(result_dir: str) -> Dict[float, Dict[str, Any]]:
    data: Dict[float, Dict[str, Any]] = {}
    for path in Path(result_dir).glob("*.json"):
        obj = load_json(path)
        rps = obj.get("request_rate")
        if rps is None:
            continue
        if rps in data:
            raise SystemExit(
                f"Duplicate request_rate {rps} in {result_dir}. "
                f"Files must be unique per RPS."
            )
        data[rps] = obj
    return data


def pct_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%"


def print_metric(
    label: str, base: float | None, cand: float | None, higher_is_better: bool
) -> None:
    if base is None or cand is None:
        delta = None
    else:
        delta = pct_change(cand, base)
        if delta is not None and not higher_is_better:
            delta = -delta
    base_str = "n/a" if base is None else f"{base:.3f}"
    cand_str = "n/a" if cand is None else f"{cand:.3f}"
    print(f"  {label}: {base_str} -> {cand_str} ({format_pct(delta)})")


def success_rate(obj: Dict[str, Any]) -> float | None:
    completed = obj.get("completed")
    failed = obj.get("failed")
    if completed is None or failed is None:
        return None
    total = completed + failed
    if total == 0:
        return None
    return completed / total * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", help="Baseline result directory.")
    parser.add_argument("candidate_dir", help="Candidate result directory.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    if not os.path.isdir(args.baseline_dir):
        raise SystemExit(f"Baseline dir not found: {args.baseline_dir}")
    if not os.path.isdir(args.candidate_dir):
        raise SystemExit(f"Candidate dir not found: {args.candidate_dir}")

    base = load_dir(args.baseline_dir)
    cand = load_dir(args.candidate_dir)
    rps_list = sorted(set(base) & set(cand))
    if not rps_list:
        raise SystemExit("No overlapping request_rate values to compare.")

    for rps in rps_list:
        b = base[rps]
        c = cand[rps]
        print(f"\nRPS {int(rps)} ({args.baseline_name} -> {args.candidate_name})")
        print_metric(
            "request_throughput (req/s)",
            b.get("request_throughput"),
            c.get("request_throughput"),
            True,
        )
        print_metric(
            "total_token_throughput (tok/s)",
            b.get("total_token_throughput"),
            c.get("total_token_throughput"),
            True,
        )
        print_metric(
            "request_goodput (req/s)",
            b.get("request_goodput"),
            c.get("request_goodput"),
            True,
        )
        print_metric(
            "success_rate (%)",
            success_rate(b),
            success_rate(c),
            True,
        )
        print_metric(
            "mean_ttft_ms",
            b.get("mean_ttft_ms"),
            c.get("mean_ttft_ms"),
            False,
        )
        print_metric(
            "p95_ttft_ms",
            get_percentile(b, "ttft", 95),
            get_percentile(c, "ttft", 95),
            False,
        )
        print_metric(
            "p99_ttft_ms",
            get_percentile(b, "ttft", 99),
            get_percentile(c, "ttft", 99),
            False,
        )
        print_metric(
            "mean_tpot_ms",
            b.get("mean_tpot_ms"),
            c.get("mean_tpot_ms"),
            False,
        )
        print_metric(
            "p99_tpot_ms",
            get_percentile(b, "tpot", 99),
            get_percentile(c, "tpot", 99),
            False,
        )
        print_metric(
            "mean_e2el_ms",
            b.get("mean_e2el_ms"),
            c.get("mean_e2el_ms"),
            False,
        )
        print_metric(
            "p95_e2el_ms",
            get_percentile(b, "e2el", 95),
            get_percentile(c, "e2el", 95),
            False,
        )
        print_metric(
            "p99_e2el_ms",
            get_percentile(b, "e2el", 99),
            get_percentile(c, "e2el", 99),
            False,
        )
        print(
            "  completed/failed: "
            f"{b.get('completed')}/{b.get('failed')} -> "
            f"{c.get('completed')}/{c.get('failed')}"
        )
        print(
            "  max_concurrency: "
            f"{b.get('max_concurrency')} -> {c.get('max_concurrency')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
