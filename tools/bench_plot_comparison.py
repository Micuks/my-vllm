#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate latency-vs-throughput comparison charts from benchmark JSON dirs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ── data helpers ────────────────────────────────────────────────────────────

def _load_dir(result_dir: str) -> Dict[float, Dict[str, Any]]:
    data: Dict[float, Dict[str, Any]] = {}
    for path in Path(result_dir).glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        rps = obj.get("request_rate")
        if rps is not None:
            data[float(rps)] = obj
    return data


def _percentile(obj: Dict[str, Any], metric: str, p: int) -> Optional[float]:
    key = f"p{p}_{metric}_ms"
    if key in obj:
        return obj[key]
    for pv, val in obj.get(f"percentiles_{metric}_ms", []):
        if abs(pv - p) < 1e-6:
            return float(val)
    return None


def _success_rate(obj: Dict[str, Any]) -> Optional[float]:
    c, f = obj.get("completed"), obj.get("failed")
    if c is None or f is None:
        return None
    total = c + f
    return c / total * 100.0 if total else None


# ── metric definitions ──────────────────────────────────────────────────────

Metric = Tuple[str, str, bool]  # (title, unit, higher_is_better)

METRICS: List[Tuple[Metric, Any]] = [
    (("Throughput", "req/s", True),      lambda o: o.get("request_throughput")),
    (("Goodput", "req/s", True),         lambda o: o.get("request_goodput")),
    (("Mean TTFT", "ms", False),         lambda o: o.get("mean_ttft_ms")),
    (("P95 TTFT", "ms", False),          lambda o: _percentile(o, "ttft", 95)),
    (("Mean E2EL", "ms", False),         lambda o: o.get("mean_e2el_ms")),
    (("Success Rate", "%", True),        _success_rate),
]


# ── plotting ────────────────────────────────────────────────────────────────

def _extract(data: Dict[float, Dict[str, Any]], rates: List[float], fn):
    """Return (x, y) lists for rates where fn produces a value."""
    xs, ys = [], []
    for r in rates:
        v = fn(data[r])
        if v is not None:
            xs.append(r)
            ys.append(v)
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_dir")
    ap.add_argument("candidate_dir")
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--output", "-o", default="comparison.png")
    args = ap.parse_args()

    for d in (args.baseline_dir, args.candidate_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"Directory not found: {d}")

    base = _load_dir(args.baseline_dir)
    cand = _load_dir(args.candidate_dir)
    rates = sorted(set(base) & set(cand))
    if not rates:
        raise SystemExit("No overlapping request_rate values to compare.")

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass  # fall back to default style

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{args.baseline_name}  vs  {args.candidate_name}", fontsize=14)

    for ax, ((title, unit, higher), fn) in zip(axes.flat, METRICS):
        bx, by = _extract(base, rates, fn)
        cx, cy = _extract(cand, rates, fn)
        if not bx and not cx:
            ax.set_visible(False)
            continue
        ax.plot(bx, by, "o-", label=args.baseline_name, markersize=5)
        ax.plot(cx, cy, "s-", label=args.candidate_name, markersize=5)
        direction = "higher is better" if higher else "lower is better"
        ax.set_title(f"{title}  ({direction})", fontsize=10)
        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel(f"{title} ({unit})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
