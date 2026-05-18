# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-level integration tests for the goodput closed-loop wiring.

These tests construct a real `Scheduler` with `policy="goodput"` and
exercise the path `update_from_output → record_iteration → gamma_tick
→ effective_max_running → admission gate`. Unit tests in
`test_goodput_scorer.py` cover the scorer alone; this file confirms that
the scheduler wires it correctly and respects the dynamic concurrency
cap without violating its own invariants.

Heavy fixtures (full ModelConfig, KV cache config) are reused from
`tests.v1.core.utils.create_scheduler` by passing the goodput knobs
through a thin wrapper.
"""
from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler

from ..utils import create_requests, create_scheduler


def _goodput_scheduler(
    max_num_seqs: int,
    gamma_max: float = 200.0,
    concurrency_floor: int = 1,
    tpot_target_s: float = 0.08,
    gamma_kp: float = 2000.0,
    gamma_ki: float = 0.0,
) -> Scheduler:
    """Build a real `Scheduler` with `policy='goodput'` through the
    standard constructor path. Going through the constructor (not
    mutating private internals after the fact) ensures these tests
    cover the end-to-end wiring: SchedulerConfig → Scheduler init →
    GoodputScorer build → kv_cache_manager hook → waiting-queue
    construction → admission gate.
    """
    return create_scheduler(
        max_num_seqs=max_num_seqs,
        scheduling_policy="goodput",
        goodput_overrides={
            "goodput_tpot_target_s": tpot_target_s,
            "goodput_gamma_kp": gamma_kp,
            "goodput_gamma_ki": gamma_ki,
            "goodput_gamma_max": gamma_max,
            "goodput_concurrency_floor": concurrency_floor,
        },
    )


# ── Tests ────────────────────────────────────────────────────────────────


def test_admission_gate_respects_effective_cap_when_gamma_high():
    """When the controller forces `effective_max_running` below the
    static cap, `schedule()` admits at most that many concurrent decoders
    even with many requests waiting.
    """
    scheduler = _goodput_scheduler(max_num_seqs=8, concurrency_floor=2)
    # Force aggressive reduction: gamma = gamma_max ⇒ effective_cap = floor.
    scheduler._goodput_scorer.gamma = scheduler._goodput_scorer.config.gamma_max
    assert (
        scheduler._goodput_scorer.effective_max_running(static_cap=8)
        == 2
    )

    # Add 8 short requests; only the first 2 should be admitted.
    requests = create_requests(num_requests=8, num_tokens=16)
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()
    assert len(scheduler.running) == 2, (
        f"effective_cap=2 ⇒ at most 2 running; got {len(scheduler.running)}"
    )
    # And the scheduler invariant still holds.
    assert len(scheduler.running) <= scheduler.max_num_running_reqs


def test_no_admission_when_running_already_above_shrunk_cap():
    """If the controller shrinks `effective_max_running` below
    `len(self.running)`, the scheduler must not preempt — but it also
    must not admit new requests. Running drains naturally.
    """
    scheduler = _goodput_scheduler(max_num_seqs=8, concurrency_floor=2)

    # First, prime the running set with 4 requests under no controller pressure.
    primer = create_requests(num_requests=4, num_tokens=16)
    for req in primer:
        scheduler.add_request(req)
    _ = scheduler.schedule()
    assert len(scheduler.running) == 4

    # Now activate the controller fully so effective_cap = 2 < 4.
    scheduler._goodput_scorer.gamma = scheduler._goodput_scorer.config.gamma_max
    assert scheduler._goodput_scorer.effective_max_running(static_cap=8) == 2

    # Add 4 more waiters. Schedule should not admit any (no preemption either).
    waiters = create_requests(
        num_requests=4,
        num_tokens=16,
        req_ids=[f"w{i}" for i in range(4)],
    )
    for req in waiters:
        scheduler.add_request(req)
    running_before = list(scheduler.running)
    _ = scheduler.schedule()
    # Running set unchanged; no admissions allowed while above cap.
    assert scheduler.running == running_before
    assert len(scheduler.waiting) == 4
    # Invariant: running never exceeds static cap.
    assert len(scheduler.running) <= scheduler.max_num_running_reqs


def test_disabled_controller_is_byte_equal_to_open_loop():
    """`gamma_max == 0` skips the per-iteration hook AND keeps
    `effective_max_running` equal to the static cap, so the goodput
    scoring path is byte-equal to the pre-Phase-2 open-loop behaviour.
    """
    scheduler = _goodput_scheduler(max_num_seqs=4, gamma_max=0.0)
    scorer = scheduler._goodput_scorer
    assert scorer.config.gamma_max == 0.0

    # Inflate gamma manually; effective_max_running should still ignore it.
    scorer.gamma = 999.0
    assert scorer.effective_max_running(static_cap=4) == 4

    # And gamma_tick should be a no-op (gamma stays as we set it).
    scorer.gamma_tick(tpot_obs_s=1.0, now=0.0)
    scorer.gamma_tick(tpot_obs_s=1.0, now=1.0)
    assert scorer.gamma == 999.0  # untouched

    # Schedule should admit up to static cap, never more.
    requests = create_requests(num_requests=8, num_tokens=16)
    for req in requests:
        scheduler.add_request(req)
    _ = scheduler.schedule()
    assert len(scheduler.running) == 4
    assert len(scheduler.running) <= scheduler.max_num_running_reqs


def test_update_from_output_feeds_controller_when_enabled():
    """End-to-end: a controller-enabled scheduler that runs `schedule()`
    then `update_from_output()` with non-empty sampled tokens must have
    its TPOT window populated and `_goodput_prev_iter_ts` advanced.

    With `gamma_max == 0`, the same sequence must NOT touch either, so
    the open-loop path remains byte-equal.
    """
    from vllm.v1.outputs import ModelRunnerOutput

    def _run_one_step(scheduler: Scheduler) -> None:
        for req in create_requests(num_requests=2, num_tokens=16):
            scheduler.add_request(req)
        sched_out = scheduler.schedule()
        # Build a minimal ModelRunnerOutput that mimics one sampled token
        # per running request.
        req_ids = list(sched_out.num_scheduled_tokens.keys())
        runner_out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[1] for _ in req_ids],
        )
        # Two calls — first seeds `_goodput_prev_iter_ts`, second produces
        # a real iter_dur so record_iteration appends to the window.
        scheduler.update_from_output(sched_out, runner_out)
        scheduler.update_from_output(sched_out, runner_out)

    # Controller enabled: window should fill, prev_ts should advance.
    on = _goodput_scheduler(max_num_seqs=4, gamma_max=200.0)
    _run_one_step(on)
    assert on._goodput_prev_iter_ts is not None
    assert len(on._goodput_scorer._tpot_window) > 0

    # Controller disabled: window stays empty, prev_ts stays None.
    off = _goodput_scheduler(max_num_seqs=4, gamma_max=0.0)
    _run_one_step(off)
    assert off._goodput_prev_iter_ts is None
    assert len(off._goodput_scorer._tpot_window) == 0


def test_floor_above_static_cap_does_not_violate_scheduler_invariant():
    """Regression for the BLOCKER bug Codex caught: a user can
    mis-configure `goodput_concurrency_floor > max_num_seqs`. The
    scheduler must still respect `len(running) <= max_num_running_reqs`.
    """
    scheduler = _goodput_scheduler(
        max_num_seqs=4, concurrency_floor=64  # floor > static cap
    )
    # Activate the controller fully.
    scheduler._goodput_scorer.gamma = scheduler._goodput_scorer.config.gamma_max
    eff = scheduler._goodput_scorer.effective_max_running(
        static_cap=scheduler.max_num_running_reqs
    )
    assert eff <= scheduler.max_num_running_reqs

    # Schedule 8 requests; must not admit more than 4 (the static cap).
    requests = create_requests(num_requests=8, num_tokens=16)
    for req in requests:
        scheduler.add_request(req)
    _ = scheduler.schedule()
    assert len(scheduler.running) <= scheduler.max_num_running_reqs
