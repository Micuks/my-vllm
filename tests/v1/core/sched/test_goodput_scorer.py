# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the goodput scheduler scoring + queue.

Uses SimpleNamespace request mocks so the tests run without vLLM's compiled
C extensions, which are required by `from vllm.v1.request import Request`.
"""

# Stub the optional CUDA-only extension before any vllm.* import. When vLLM
# is properly built (production setup) the real module wins; when it isn't,
# the stub lets the goodput tests still run because they don't actually
# execute any CUDA code paths.
import sys as _sys
import types as _types

if "vllm._C_stable_libtorch" not in _sys.modules:
    _sys.modules["vllm._C_stable_libtorch"] = _types.ModuleType(
        "vllm._C_stable_libtorch"
    )

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.sched.goodput_scorer import GoodputConfig, GoodputScorer
from vllm.v1.core.sched.request_queue import (
    GoodputRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


def make_request(
    request_id: str = "r0",
    *,
    arrival_time: float = 0.0,
    num_prompt_tokens: int = 100,
    num_computed_tokens: int = 0,
    num_output_tokens: int = 0,
    num_output_placeholders: int = 0,
    max_tokens: int = 100,
) -> Any:
    """Duck-typed Request mock with the fields the scorer reads."""
    return SimpleNamespace(
        request_id=request_id,
        arrival_time=arrival_time,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=num_output_tokens,
        num_output_placeholders=num_output_placeholders,
        max_tokens=max_tokens,
    )


# ── GoodputScorer ──────────────────────────────────────────────────────


class TestPiecewise:
    def test_below_tau_is_zero(self):
        assert GoodputScorer._piecewise(0.0, 0.5) == 0.0
        assert GoodputScorer._piecewise(0.4, 0.5) == 0.0
        assert GoodputScorer._piecewise(0.499, 0.5) == 0.0

    def test_at_tau_is_zero(self):
        assert GoodputScorer._piecewise(0.5, 0.5) == 0.0

    def test_at_deadline_is_one(self):
        assert GoodputScorer._piecewise(1.0, 0.5) == pytest.approx(1.0)

    def test_halfway_between_tau_and_deadline(self):
        # tau=0.5, deadline=1.0; midpoint 0.75 → ratio 0.5 in normalized space
        assert GoodputScorer._piecewise(0.75, 0.5) == pytest.approx(0.5)

    def test_past_deadline_keeps_growing(self):
        assert GoodputScorer._piecewise(1.5, 0.5) == pytest.approx(2.0)

    def test_different_tau(self):
        # tau=0.3, ratio=0.65 → (0.65-0.3)/(1-0.3) = 0.5
        assert GoodputScorer._piecewise(0.65, 0.3) == pytest.approx(0.5)

    def test_tau_at_or_above_one_disables_pressure(self):
        # Regression: tau=1.0 used to hit a divide-by-zero. The function
        # now treats tau>=1 as "SLO_pressure disabled" — always 0,
        # regardless of ratio.
        for tau in (1.0, 1.5, 10.0, 1000.0):
            for ratio in (0.0, 0.5, 1.0, 1.5, 100.0):
                assert GoodputScorer._piecewise(ratio, tau) == 0.0


class TestSloPressure:
    def test_waiting_below_tau_returns_zero(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(arrival_time=0.0)
        # ttft_slo=2.0, tau=0.5 → activation at t=1.0
        assert scorer._slo_pressure(req, now=0.5) == 0.0
        assert scorer._slo_pressure(req, now=0.99) == 0.0

    def test_waiting_at_deadline_returns_M(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(arrival_time=0.0)
        # at ratio=1.0 (now=2.0s), pressure = M
        assert scorer._slo_pressure(req, now=2.0) == pytest.approx(scorer.M)

    def test_waiting_past_deadline_exceeds_M(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(arrival_time=0.0)
        assert scorer._slo_pressure(req, now=4.0) > scorer.M

    def test_decoding_uses_e2el_not_ttft(self):
        # E2EL_SLO=30s; if request has emitted tokens AND prefill is
        # complete (i.e., currently in self.running and decoding), the
        # ratio should use projected E2EL not (now - arrived) / TTFT_SLO.
        # Note: num_computed_tokens must equal num_prompt_tokens to
        # signal "decoding"; with num_computed_tokens==0 the scorer
        # treats it as preempted-mid-decode and uses TTFT pressure.
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(
            arrival_time=0.0,
            num_prompt_tokens=200,
            num_computed_tokens=200,  # prefill complete
            num_output_tokens=10,
            max_tokens=100,
        )
        # Even at t=2.5s (past TTFT_SLO=2s for waiting), this decoding
        # request should not be at deadline because E2EL_SLO=30s
        pressure = scorer._slo_pressure(req, now=2.5)
        assert pressure < scorer.M

    def test_first_token_emitted_switches_branch(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(arrival_time=0.0, num_output_tokens=0)
        p_waiting = scorer._slo_pressure(req, now=1.5)  # past tau on TTFT
        # Move into the "in running, decoding" regime: produced 1 token,
        # has done some computation. Should switch to E2EL pressure.
        req.num_output_tokens = 1
        req.num_computed_tokens = req.num_prompt_tokens  # prefill complete
        p_decoding = scorer._slo_pressure(req, now=1.5)  # below tau on E2EL
        # Switching from waiting to decoding should drop pressure to 0
        assert p_waiting > 0
        assert p_decoding == 0.0

    def test_cache_aware_length_signal_via_peek(self):
        """Regression test: scorer must peek the KV cache so length_signal
        reflects POST-cache prefill work, not raw prompt size. Without
        this, a long mid-session prompt with massive cache hit ranks
        WORSE than a short fresh prompt, even though admitting the
        cached one is cheap. Agentic workloads degenerate to FCFS
        without cache awareness.
        """
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)

        class FakeCoord:
            def __init__(self, hits_for):
                self.hits_for = hits_for  # request_id -> int

            def find_longest_cache_hit(self, block_hashes, max_hit):
                # Return matched-blocks placeholder + hit count
                return ([], self.hits_for.get(id(block_hashes), 0))

        class FakeKVM:
            enable_caching = True
            def __init__(self, hits_by_req):
                self.coordinator = FakeCoord(hits_by_req)

        # Mid-session: long prompt (5000) but 4500 already cached
        mid_bh = ["m1", "m2"]  # non-empty placeholder; id() is the lookup key
        mid = make_request(
            "mid", num_prompt_tokens=5000, num_computed_tokens=0
        )
        mid.block_hashes = mid_bh
        mid.num_tokens = 5000
        # Fresh: small prompt (1000) but 0 cached
        fresh_bh = ["f1"]
        fresh = make_request(
            "fresh", num_prompt_tokens=1000, num_computed_tokens=0
        )
        fresh.block_hashes = fresh_bh
        fresh.num_tokens = 1000

        kvm = FakeKVM({id(mid_bh): 4500, id(fresh_bh): 0})
        scorer.kv_cache_manager = kvm  # type: ignore[assignment]

        # Without cache awareness: mid would score 5000, fresh 1000.
        # With cache awareness: mid_effective = 5000 - 4500 = 500;
        # fresh_effective = 1000. Mid should now WIN (lower score).
        s_mid = scorer.score(mid, now=0.5)
        s_fresh = scorer.score(fresh, now=0.5)
        assert s_mid < s_fresh, (
            f"cache-aware mid (effective 500) should beat fresh (effective "
            f"1000): mid={s_mid} fresh={s_fresh}"
        )

    def test_peek_cache_hit_safe_without_kv_cache_manager(self):
        """Without the scheduler wiring kv_cache_manager, the scorer
        must still work and just treat all requests as zero-cached."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        # kv_cache_manager unset (None) by default
        req = make_request("r", num_prompt_tokens=1000)
        # length_signal must equal full prompt (no cache awareness)
        assert scorer._length_signal(req) == 1000
        assert scorer._cache_signal(req) == 0

    def test_pressure_clamp_caps_unbounded_ramp(self):
        """Regression test: with pressure_clamp=1.0, SLO_pressure stops
        growing past deadline. Many requests can be past deadline at
        once (saturated serving) and they all tie at M, letting length
        and cache signals decide ordering. Without the clamp, the
        earliest-arrived request's pressure dominates and ordering
        reverts to FCFS."""
        scorer_clamped = GoodputScorer(
            GoodputConfig(pressure_clamp=1.0), max_prompt_tokens=8192
        )
        scorer_unclamped = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        early = make_request("early", arrival_time=0.0)
        late = make_request("late", arrival_time=18.0)
        # early at now=20: ratio = 20/2 = 10
        # late at now=20:  ratio = 2/2 = 1
        # Unclamped: SLO_pressure_early >> SLO_pressure_late (10× difference)
        # Clamped at 1: both reach the cap, so they tie on pressure.
        p_early_unclamped = scorer_unclamped._slo_pressure(early, now=20.0)
        p_late_unclamped = scorer_unclamped._slo_pressure(late, now=20.0)
        assert p_early_unclamped > p_late_unclamped * 5, "unclamped should heavily favor older"

        p_early_clamped = scorer_clamped._slo_pressure(early, now=20.0)
        p_late_clamped = scorer_clamped._slo_pressure(late, now=20.0)
        assert p_early_clamped == p_late_clamped == scorer_clamped.M, \
            f"clamped should tie at M: early={p_early_clamped} late={p_late_clamped} M={scorer_clamped.M}"

    def test_preempted_mid_decode_uses_ttft_branch(self):
        """Regression test for the preempted-mid-decode bug.

        When a request is preempted while decoding, the scheduler resets
        num_computed_tokens=0 but num_output_tokens stays at the count it
        had. Such a request is back in the waiting queue and the user
        observes a token gap. The scorer must rank it by TTFT pressure
        (small denominator 2s), NOT by projected E2EL (large denominator
        30s), or it gets crowded out by fresh arrivals and triggers
        preemption cascades.
        """
        cfg = GoodputConfig()
        scorer = GoodputScorer(cfg, max_prompt_tokens=8192)
        # A request preempted after 50 output tokens, has been waiting 30s
        preempted = make_request(
            "preempted",
            arrival_time=0.0,
            num_prompt_tokens=1024,
            num_computed_tokens=0,
            num_output_tokens=50,
            max_tokens=128,
        )
        # A fresh request that just arrived 1s ago — still in TTFT window
        fresh = make_request(
            "fresh",
            arrival_time=29.0,
            num_prompt_tokens=1024,
            num_computed_tokens=0,
            num_output_tokens=0,
        )
        # Preempted: elapsed=30s, must use ttft denom → ratio=15 → very urgent
        # Fresh:     elapsed=1s,  must use ttft denom → ratio=0.5 → at tau
        p_preempted = scorer._slo_pressure(preempted, now=30.0)
        p_fresh = scorer._slo_pressure(fresh, now=30.0)
        assert p_preempted > p_fresh, (
            "Preempted-mid-decode req must outrank fresh on SLO_pressure; "
            "otherwise admission keeps choosing fresh and re-preempting decodes"
        )


class TestLengthSignal:
    def test_prefill_only_default(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(num_prompt_tokens=200, num_computed_tokens=50)
        assert scorer._length_signal(req) == 150

    def test_prefill_only_zero_when_decoding(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(
            num_prompt_tokens=200,
            num_computed_tokens=200,  # prefill done
            num_output_tokens=10,
        )
        assert scorer._length_signal(req) == 0

    def test_remaining_mode_with_decode(self):
        cfg = GoodputConfig(length_mode="remaining", kappa=0.5)
        scorer = GoodputScorer(cfg, max_prompt_tokens=8192)
        req = make_request(
            num_prompt_tokens=200,
            num_computed_tokens=200,
            num_output_tokens=10,
            max_tokens=100,
        )
        # prefill_left=0, decode_left=90, kappa=0.5 → 0 + 45 = 45
        assert scorer._length_signal(req) == 45


class TestCacheSignal:
    def test_unobserved_returns_zero(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(request_id="r1")
        assert scorer._cache_signal(req) == 0

    def test_observation_persists(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(request_id="r1")
        scorer.observe_cached_tokens(req, 128)
        assert scorer._cache_signal(req) == 128

    def test_observation_overwrite(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(request_id="r1")
        scorer.observe_cached_tokens(req, 128)
        scorer.observe_cached_tokens(req, 256)
        assert scorer._cache_signal(req) == 256

    def test_negative_clamped_to_zero(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(request_id="r1")
        scorer.observe_cached_tokens(req, -10)
        assert scorer._cache_signal(req) == 0


class TestTpotWindow:
    def test_initial_value_used_when_empty(self):
        cfg = GoodputConfig(initial_tpot_s=0.07)
        scorer = GoodputScorer(cfg, max_prompt_tokens=8192)
        assert scorer.tpot_avg_s == pytest.approx(0.07)

    def test_rolling_mean(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        # Each iteration produces 10 tokens in 0.5s → 50 ms/token
        scorer.record_iteration(0.5, 10)
        scorer.record_iteration(0.6, 10)  # 60 ms/token
        # Mean should be 55 ms/token
        assert scorer.tpot_avg_s == pytest.approx(0.055)

    def test_window_size_caps_history(self):
        cfg = GoodputConfig(tpot_window_n=3)
        scorer = GoodputScorer(cfg, max_prompt_tokens=8192)
        # Push 4 samples; only last 3 retained
        for tpot in [0.1, 0.1, 0.1, 0.05]:
            scorer.record_iteration(tpot * 10, 10)
        assert scorer.tpot_avg_s == pytest.approx((0.1 + 0.1 + 0.05) / 3)

    def test_zero_or_negative_input_ignored(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        scorer.record_iteration(0, 10)
        scorer.record_iteration(0.5, 0)
        scorer.record_iteration(-1, 10)
        assert scorer.tpot_avg_s == pytest.approx(0.05)  # initial


class TestGammaController:
    """Phase-2 closed-loop TPOT controller.

    The controller is a PI law that adapts a gain `gamma` based on
    `tpot_obs - tpot_target_s`. `gamma` then shrinks the effective
    concurrent-decoder cap via `effective_max_running()`.
    """

    def _cfg(self, **overrides) -> GoodputConfig:
        base = dict(
            tpot_target_s=0.08,
            gamma_kp=100.0,  # 1 s of error → gamma = 100
            gamma_ki=10.0,
            gamma_max=200.0,
            gamma_integral_clamp_s=4.0,
            concurrency_floor=2,
        )
        base.update(overrides)
        return GoodputConfig(**base)

    def test_disabled_by_default(self):
        # Default config has gamma_max == 0, controller disabled.
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        scorer.gamma_tick(tpot_obs_s=1.0, now=0.0)
        scorer.gamma_tick(tpot_obs_s=1.0, now=1.0)
        assert scorer.gamma == 0.0
        assert scorer.effective_max_running(static_cap=32) == 32

    def test_gamma_rises_when_tpot_above_target(self):
        scorer = GoodputScorer(self._cfg(), max_prompt_tokens=8192)
        # First call seeds the timestamp; subsequent calls integrate.
        scorer.gamma_tick(tpot_obs_s=0.08, now=0.0)
        scorer.gamma_tick(tpot_obs_s=0.18, now=0.1)  # error +0.10s
        assert scorer.gamma > 0.0

    def test_gamma_returns_to_zero_when_tpot_below_target(self):
        scorer = GoodputScorer(self._cfg(), max_prompt_tokens=8192)
        scorer.gamma_tick(tpot_obs_s=0.08, now=0.0)
        scorer.gamma_tick(tpot_obs_s=0.20, now=0.1)
        elevated = scorer.gamma
        assert elevated > 0.0
        # Many recovery steps with negative error should drive gamma to 0.
        for k in range(1, 200):
            scorer.gamma_tick(tpot_obs_s=0.02, now=0.1 + 0.1 * k)
        assert scorer.gamma == 0.0

    def test_gamma_clamped_to_max(self):
        scorer = GoodputScorer(self._cfg(gamma_max=50.0), max_prompt_tokens=8192)
        scorer.gamma_tick(tpot_obs_s=0.08, now=0.0)
        # Massive error → P-term alone exceeds gamma_max.
        scorer.gamma_tick(tpot_obs_s=10.0, now=0.1)
        assert scorer.gamma == pytest.approx(50.0)

    def test_integral_anti_windup(self):
        clamp = 2.0
        scorer = GoodputScorer(
            self._cfg(gamma_integral_clamp_s=clamp), max_prompt_tokens=8192
        )
        scorer.gamma_tick(tpot_obs_s=0.08, now=0.0)
        # Sustained big positive error for *much longer* than the clamp
        # window. Without anti-windup, integral would grow to ~100;
        # with it, integral must be ≤ clamp.
        for k in range(1, 1000):
            scorer.gamma_tick(tpot_obs_s=1.08, now=0.1 * k)
        assert scorer._pi_integral <= clamp + 1e-6
        # And symmetric on the negative side.
        for k in range(1, 1000):
            scorer.gamma_tick(tpot_obs_s=-100.0, now=200.0 + 0.1 * k)
        assert scorer._pi_integral >= -clamp - 1e-6

    def test_dt_clamped_so_idle_gap_does_not_explode_integral(self):
        scorer = GoodputScorer(self._cfg(), max_prompt_tokens=8192)
        scorer.gamma_tick(tpot_obs_s=0.08, now=0.0)
        # Simulate a 1-hour idle gap. dt must be clamped to ≤ 1 s so this
        # does not bombard the integral.
        scorer.gamma_tick(tpot_obs_s=0.18, now=3600.0)
        # Just one tick: integral term contributes at most
        # gamma_ki * (1.0 s * 0.10 s error) = 1.0; combined with the P
        # term gamma_kp * 0.10 = 10 → gamma ≤ 11.
        assert scorer.gamma <= 11.0 + 1e-6

    def test_effective_max_running_floor(self):
        scorer = GoodputScorer(
            self._cfg(concurrency_floor=4), max_prompt_tokens=8192
        )
        # Force gamma = gamma_max.
        scorer.gamma = scorer.config.gamma_max
        assert scorer.effective_max_running(static_cap=32) == 4
        # And floor obeys static_cap when static_cap < floor:
        scorer.gamma = 0.0
        assert scorer.effective_max_running(static_cap=2) == 2

    def test_effective_max_running_linear_between_zero_and_max(self):
        scorer = GoodputScorer(self._cfg(), max_prompt_tokens=8192)
        scorer.gamma = scorer.config.gamma_max * 0.5
        # Halve cap (rounded) but never below floor.
        assert scorer.effective_max_running(static_cap=32) == 16

    def test_effective_cap_never_exceeds_static_cap_even_when_floor_too_high(self):
        """Regression: even if a user mis-configures
        `concurrency_floor > max_num_seqs`, `effective_max_running` must
        never return a value above `static_cap` or it would violate the
        scheduler invariant `len(running) <= max_num_running_reqs`.
        """
        scorer = GoodputScorer(
            self._cfg(concurrency_floor=64), max_prompt_tokens=8192
        )
        # gamma at max ⇒ aggressive reduction. With static_cap < floor,
        # the static_cap wins and we never exceed it.
        scorer.gamma = scorer.config.gamma_max
        assert scorer.effective_max_running(static_cap=32) == 32
        # Same expectation at every gamma between 0 and gamma_max.
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            scorer.gamma = scorer.config.gamma_max * frac
            cap = scorer.effective_max_running(static_cap=32)
            assert 1 <= cap <= 32, (
                f"effective_max_running={cap} out of [1,32] at frac={frac}"
            )


class TestScoreCombination:
    def test_below_tau_proxies_decide_order(self):
        """Below tau, SLO_pressure=0; length and cache decide order."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        # Two waiting requests, both below tau
        short_req = make_request("short", num_prompt_tokens=100)
        long_req = make_request("long", num_prompt_tokens=500)
        s_short = scorer.score(short_req, now=0.5)
        s_long = scorer.score(long_req, now=0.5)
        # Shorter request scores lower (= higher priority)
        assert s_short < s_long
        assert s_short == 100  # alpha=1, length=100, no SLO_pressure, no cache
        assert s_long == 500

    def test_at_deadline_pressure_dominates_proxies(self):
        """At deadline, pressure ≥ M, must beat any proxy combination."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        # A late but huge request: should still beat a fresh small one
        late = make_request("late", arrival_time=0.0, num_prompt_tokens=8000)
        fresh = make_request("fresh", arrival_time=2.0, num_prompt_tokens=10)
        # at now=2.0, late has elapsed 2s = at TTFT_SLO; fresh just arrived
        s_late = scorer.score(late, now=2.0)
        s_fresh = scorer.score(fresh, now=2.0)
        # Late should win despite being huge, because it's at deadline
        assert s_late < s_fresh

    def test_cache_subtracts_from_score(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(num_prompt_tokens=200)
        without_cache = scorer.score(req, now=0.5)
        scorer.observe_cached_tokens(req, 100)
        with_cache = scorer.score(req, now=0.5)
        # cache hit should reduce score (= raise priority)
        assert with_cache < without_cache
        assert without_cache - with_cache == pytest.approx(100)

    def test_discard_removes_observations(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request("r1")
        scorer.observe_cached_tokens(req, 50)
        assert scorer.num_observations() == 1
        scorer.discard("r1")
        assert scorer.num_observations() == 0


# ── GoodputRequestQueue ────────────────────────────────────────────────


class TestGoodputRequestQueue:
    def test_empty_queue(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        assert len(q) == 0
        assert not q
        with pytest.raises(IndexError):
            q.pop_request()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_add_and_pop_ordering(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        # Add three with different prompt lengths; all below tau.
        # With prefill_only length signal, shorter prompt → lower score → pops first.
        long_req = make_request("long", num_prompt_tokens=500)
        mid_req = make_request("mid", num_prompt_tokens=200)
        short_req = make_request("short", num_prompt_tokens=100)
        for r in [long_req, mid_req, short_req]:
            q.add_request(r)
        assert len(q) == 3
        assert q.pop_request().request_id == "short"
        assert q.pop_request().request_id == "mid"
        assert q.pop_request().request_id == "long"

    def test_refresh_reorders_after_time_passes(self):
        """A waiting long request should overtake a fresh short one when
        the long one crosses the SLO_pressure activation threshold."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        long_old = make_request(
            "long_old", arrival_time=0.0, num_prompt_tokens=500
        )
        short_new = make_request(
            "short_new", arrival_time=1.5, num_prompt_tokens=10
        )
        q.add_request(long_old)
        q.add_request(short_new)
        # At t=0 (queue creation), short_new ranks first (lower length)
        # but really both were added at _last_refresh=0 with stale scores.
        # After refresh at t=2.0 (long_old at deadline), order should flip.
        q.refresh(now=2.0)
        first = q.peek_request()
        assert first.request_id == "long_old"

    def test_remove_request(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        a = make_request("a", num_prompt_tokens=100)
        b = make_request("b", num_prompt_tokens=200)
        q.add_request(a)
        q.add_request(b)
        q.remove_request(a)
        assert len(q) == 1
        assert q.peek_request().request_id == "b"

    def test_remove_requests_batch(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        reqs = [make_request(f"r{i}", num_prompt_tokens=100 + i) for i in range(5)]
        for r in reqs:
            q.add_request(r)
        q.remove_requests([reqs[0], reqs[2], reqs[4]])
        assert len(q) == 2
        ids = sorted(r.request_id for r in list(q))
        assert ids == ["r1", "r3"]

    def test_iter_yields_in_score_order(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        reqs = [make_request(f"r{i}", num_prompt_tokens=500 - 100 * i) for i in range(5)]
        for r in reqs:
            q.add_request(r)
        # iter should yield shortest first; lengths are 500, 400, 300, 200, 100
        ordered = [r.request_id for r in q]
        assert ordered == ["r4", "r3", "r2", "r1", "r0"]

    def test_iter_does_not_consume_queue(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        for i in range(3):
            q.add_request(make_request(f"r{i}"))
        list(iter(q))
        assert len(q) == 3


class TestFactory:
    def test_create_goodput_requires_scorer(self):
        with pytest.raises(ValueError, match="GoodputScorer"):
            create_request_queue(SchedulingPolicy.GOODPUT)

    def test_create_goodput_returns_goodput_queue(self):
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = create_request_queue(SchedulingPolicy.GOODPUT, scorer=scorer)
        assert isinstance(q, GoodputRequestQueue)

    def test_create_fcfs_ignores_scorer(self):
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = create_request_queue(SchedulingPolicy.FCFS, scorer=scorer)
        assert isinstance(q, FCFSRequestQueue)


# ── Determinism / regression smoke ─────────────────────────────────────


class TestDeterminism:
    def test_score_is_pure_given_state(self):
        """Calling score() twice with same args returns same value."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        req = make_request(num_prompt_tokens=200)
        s1 = scorer.score(req, now=0.5)
        s2 = scorer.score(req, now=0.5)
        assert s1 == s2

    def test_M_constant_at_init(self):
        cfg = GoodputConfig(alpha=1.0, beta=2.0)
        scorer = GoodputScorer(cfg, max_prompt_tokens=1000)
        # M = 1.0 * 1000 + 2.0 * 1000 + 1.0 = 3001
        assert scorer.M == pytest.approx(3001.0)

    def test_tie_breaker_is_seq_not_request_identity(self):
        """Two requests with same score: earlier add wins (lower seq)."""
        scorer = GoodputScorer(GoodputConfig(), max_prompt_tokens=8192)
        q = GoodputRequestQueue(scorer)
        a = make_request("a", num_prompt_tokens=100)
        b = make_request("b", num_prompt_tokens=100)
        q.add_request(a)
        q.add_request(b)
        assert q.pop_request().request_id == "a"
