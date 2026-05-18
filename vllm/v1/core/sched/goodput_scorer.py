# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SLO-aware goodput scoring for the goodput scheduling policy.

Score backbone: piecewise SLO_pressure on TTFT (waiting/prefill) and
projected E2EL (decoding). Proxy signals: prefill-only length, observed
prefix-cache hit. Lower score = higher priority.

Phase 2 adds an optional closed-loop PI controller that adapts a hidden
gain `gamma` from observed TPOT (fed via `record_iteration`) and shrinks
the effective concurrent-decoder cap returned by `effective_max_running`.
The controller is opt-in (`gamma_max=0` keeps the original open-loop
behaviour and skips the per-iteration hook entirely).
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass
class GoodputConfig:
    """Hyperparameters for the goodput scorer.

    The four primary tunables are tau, alpha, beta (and the SLO triple).
    All others have defaults that should not need tuning in practice.
    """

    ttft_slo_s: float = 2.0
    """TTFT deadline (seconds). Default matches vLLM bench Standard preset."""

    tpot_slo_s: float = 0.1
    """TPOT deadline (seconds). Used by the goodput post-hoc evaluator;
    Phase-1 scoring does not consume it directly."""

    e2el_slo_s: float = 30.0
    """End-to-end latency deadline (seconds)."""

    tau: float = 0.5
    """SLO_pressure activation threshold. Below tau · SLO the pressure
    is zero and proxy signals decide ordering; above tau, linear ramp."""

    alpha: float = 1.0
    """Length signal weight (prefill remaining tokens favor short)."""

    beta: float = 1.0
    """Cache signal weight (prefix cache hits boost priority)."""

    length_mode: Literal["prefill_only", "remaining"] = "prefill_only"
    """`prefill_only`: decode contributes 0. `remaining`: decode counted
    with weight kappa."""

    kappa: float = 0.0
    """Decode-vs-prefill weight when length_mode = remaining."""

    tpot_window_n: int = 20
    """Rolling-mean window for TPOT estimate."""

    initial_tpot_s: float = 0.05
    """Conservative starting TPOT estimate (50 ms) before window fills."""

    pressure_clamp: float = float("inf")
    """Cap on the raw SLO_pressure ratio (before M scaling). Default
    `inf` reproduces the original unbounded ramp (one badly-late request
    drowns out everything else). Setting this to e.g. 1.0 makes every
    deadline-violating request tie at pressure = M, letting length_signal
    and cache_signal break the ties — usually the right default for
    workloads where many requests miss their deadline simultaneously."""

    # ── Phase 2: closed-loop TPOT controller ────────────────────────────
    # The PI controller below adapts a hidden gain `gamma` based on
    # observed TPOT vs `tpot_target_s`. The current admission gate uses
    # gamma to dynamically shrink the concurrent-decoder cap when the
    # measured TPOT exceeds the target, and grows it back to the static
    # cap when TPOT recovers. Set `gamma_max=0` to disable the controller
    # (cleanly recovers Phase-1 open-loop behaviour).

    tpot_target_s: float = 0.08
    """Target steady-state TPOT (seconds). The PI error is
    `tpot_obs - tpot_target_s`. Default 80 ms = 0.8 × default TPOT SLO
    of 100 ms, giving a 20 ms safety margin before SLO miss."""

    gamma_kp: float = 50.0
    """Proportional gain. Units: gamma per second of TPOT error."""

    gamma_ki: float = 5.0
    """Integral gain. Units: gamma per (second of error × second of dt)."""

    gamma_max: float = 0.0
    """Upper bound on gamma. When gamma == gamma_max the effective
    concurrent-decoder cap is reduced to `concurrency_floor`. Default 0
    disables the controller — opt in by setting > 0 (typical: 200)."""

    gamma_integral_clamp_s: float = 4.0
    """Anti-windup: integral term is clamped to ±this many seconds of
    accumulated error. Prevents persistent SLO miss from inflating gamma
    so far that recovery overshoots."""

    concurrency_floor: int = 4
    """Lower bound on the effective running-set cap. Never starve the
    GPU below this even under maximum gamma."""


@dataclass
class _RequestObservation:
    """Per-request mutable state owned by the scorer (not on Request)."""

    cached_tokens: int = 0


class GoodputScorer:
    """Stateful scorer that computes a per-request goodput score.

    The scorer owns:
      * rolling mean TPOT (updated by record_iteration)
      * per-request observed prefix-cache token count
      * derived constant M that bounds proxy contributions

    `score(req, now)` is pure given current state; the queue calls it
    every refresh.
    """

    def __init__(self, config: GoodputConfig, max_prompt_tokens: int) -> None:
        self.config = config
        self.max_prompt_tokens = max(1, max_prompt_tokens)

        # M is a constant chosen so that any deadline-violating request
        # (SLO_pressure >= 1) has a score that exceeds any plausible
        # proxy-only score. length_signal and cache_signal are bounded
        # above by max_prompt_tokens.
        self._M = (
            abs(config.alpha) * self.max_prompt_tokens
            + abs(config.beta) * self.max_prompt_tokens
            + 1.0
        )

        self._tpot_window: deque[float] = deque(maxlen=config.tpot_window_n)
        self._observations: dict[str, _RequestObservation] = {}
        # Set by the scheduler after KVCacheManager is constructed. When
        # present, the scorer can peek at the cache to make the length /
        # cache signals reflect the *actual* prefill work after prefix
        # cache hits — without this, both signals are blind to caching
        # potential at admission time, which causes goodput to degenerate
        # to FCFS on cache-heavy agentic workloads.
        self.kv_cache_manager: object | None = None

        # Phase 2 closed-loop TPOT controller state. Read by
        # `effective_max_running()`; mutated by `gamma_tick()`. The
        # controller is disabled when `gamma_max <= 0`.
        self.gamma: float = 0.0
        self._pi_integral: float = 0.0
        self._last_gamma_tick_ts: float | None = None

    # ── Score components ────────────────────────────────────────────────

    @staticmethod
    def _piecewise(ratio: float, tau: float, clamp: float = float("inf")) -> float:
        """0 below tau, linearly ramps to 1 at deadline (ratio = 1),
        keeps growing past deadline unless `clamp` is set (clamp=1 caps
        at the deadline value).

        `tau >= 1` is treated as "SLO_pressure disabled" — useful as an
        ablation knob to make scoring depend only on the proxy signals
        without ever activating the deadline pressure term.
        """
        if tau >= 1.0 or ratio < tau:
            return 0.0
        raw = (ratio - tau) / (1.0 - tau)
        return min(clamp, raw) if clamp != float("inf") else raw

    def _slo_pressure(self, req: "Request", now: float) -> float:
        cfg = self.config
        elapsed = max(0.0, now - req.arrival_time)
        # A preempted-mid-decode request (num_computed_tokens==0 AND
        # num_output_tokens>0) belongs in the same TTFT-pressure regime as a
        # fresh request: the user is experiencing a token gap equivalent to
        # TTFT. Using e2el_slo as denominator here deprioritises preempted
        # reqs relative to fresh arrivals and causes preemption cascades.
        in_waiting_pre_first_token = (
            req.num_output_tokens == 0 or req.num_computed_tokens == 0
        )
        if in_waiting_pre_first_token:
            ratio = elapsed / cfg.ttft_slo_s
        else:
            remaining_decode = max(
                0,
                req.max_tokens - req.num_output_tokens - req.num_output_placeholders,
            )
            projected_e2el = elapsed + remaining_decode * self.tpot_avg_s
            ratio = projected_e2el / cfg.e2el_slo_s
        return self._M * self._piecewise(ratio, cfg.tau, cfg.pressure_clamp)

    def _peek_cache_hit(self, req: "Request") -> int:
        """Side-effect-free lookup of how many prefix tokens are already
        in the KV cache for this request. Returns 0 when the scheduler
        hasn't wired `kv_cache_manager` yet, when prefix caching is
        off, or when the request hasn't computed `block_hashes` yet.
        """
        kvm = self.kv_cache_manager
        if kvm is None or not getattr(kvm, "enable_caching", False):
            return 0
        if getattr(req, "skip_reading_prefix_cache", False):
            return 0
        bh = getattr(req, "block_hashes", None)
        if not bh:
            return 0
        max_hit = max(0, req.num_tokens - 1)
        try:
            _, hits = kvm.coordinator.find_longest_cache_hit(bh, max_hit)
        except Exception:
            return 0
        return int(hits)

    def _length_signal(self, req: "Request") -> int:
        cfg = self.config
        # Effective prefill work = prompt tokens not yet computed and not
        # served by the prefix cache. This is the actual GPU work that
        # admission will incur, so it is the right thing to compare across
        # waiting requests.
        cache_hits = self._peek_cache_hit(req)
        prefill_left = max(
            0,
            req.num_prompt_tokens - req.num_computed_tokens - cache_hits,
        )
        if cfg.length_mode == "prefill_only":
            return prefill_left
        decode_left = max(
            0,
            req.max_tokens - req.num_output_tokens - req.num_output_placeholders,
        )
        return prefill_left + int(cfg.kappa * decode_left)

    def _cache_signal(self, req: "Request") -> int:
        # Prefer the live peek over the previously-recorded observation.
        # The observation path is kept for callers that explicitly
        # pre-record a hit before scoring (e.g. a future PD-disaggregated
        # backend), but for the normal admission path the live peek
        # captures fresh requests that observe_cached_tokens has never
        # been called on.
        peeked = self._peek_cache_hit(req)
        if peeked > 0:
            return peeked
        obs = self._observations.get(req.request_id)
        return obs.cached_tokens if obs is not None else 0

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def tpot_avg_s(self) -> float:
        if not self._tpot_window:
            return self.config.initial_tpot_s
        return sum(self._tpot_window) / len(self._tpot_window)

    def score(self, req: "Request", now: float) -> float:
        """Compute a per-request goodput score. Lower = higher priority.

        SLO_pressure grows as a request approaches its deadline, so we
        SUBTRACT it: high pressure → very negative score → top of heap.
        Length is added (longer prompts deprioritised); cache_signal is
        subtracted (cache-hit-rich requests get a boost).
        """
        cfg = self.config
        return (
            -self._slo_pressure(req, now)
            + cfg.alpha * self._length_signal(req)
            - cfg.beta * self._cache_signal(req)
        )

    def observe_cached_tokens(self, req: "Request", cached_tokens: int) -> None:
        """Record the most recent prefix-cache hit length for a request.

        Called by the scheduler when prefix lookup runs; subsequent
        scoring uses this value via cache_signal.
        """
        obs = self._observations.get(req.request_id)
        if obs is None:
            self._observations[req.request_id] = _RequestObservation(
                cached_tokens=max(0, cached_tokens)
            )
        else:
            obs.cached_tokens = max(0, cached_tokens)

    def record_iteration(
        self, iter_duration_s: float, num_output_tokens: int
    ) -> None:
        """Feed a TPOT sample. `iter_duration_s` is the wallclock duration
        of the most recent forward pass; `num_output_tokens` is the total
        tokens produced by that pass across all running requests. Per-token
        TPOT is appended to the rolling window used by `tpot_avg_s`.
        Wired by the scheduler at end of `update_from_output` when the
        closed-loop controller is enabled (see `gamma_max`); otherwise the
        scheduler skips this call to keep open-loop behaviour byte-equal
        to the pre-Phase-2 path.
        """
        if iter_duration_s <= 0 or num_output_tokens <= 0:
            return
        per_token = iter_duration_s / num_output_tokens
        self._tpot_window.append(per_token)

    def gamma_tick(self, tpot_obs_s: float, now: float) -> None:
        """PI step that adapts `gamma` based on observed TPOT.

        Call once per engine forward pass with the most recent TPOT
        sample. `now` should be a monotonic wallclock (e.g.
        `time.perf_counter()`); only differences matter.

        No-op when `gamma_max <= 0` (controller disabled).
        """
        cfg = self.config
        if cfg.gamma_max <= 0.0:
            return
        if self._last_gamma_tick_ts is None:
            self._last_gamma_tick_ts = now
            return
        dt = now - self._last_gamma_tick_ts
        self._last_gamma_tick_ts = now
        # Clamp dt so a long pause (e.g. server idle) does not blow up I.
        dt = max(0.0, min(1.0, dt))
        error = tpot_obs_s - cfg.tpot_target_s
        bound = cfg.gamma_integral_clamp_s
        self._pi_integral = max(
            -bound, min(bound, self._pi_integral + error * dt)
        )
        raw = cfg.gamma_kp * error + cfg.gamma_ki * self._pi_integral
        self.gamma = max(0.0, min(cfg.gamma_max, raw))

    def effective_max_running(self, static_cap: int) -> int:
        """Return the dynamically reduced concurrent-decoder cap.

        Reduces the static cap proportionally to `gamma / gamma_max`. The
        return value is always in `[1, static_cap]`: it is never above
        the scheduler's static cap (which would violate the invariant
        `len(running) <= max_num_running_reqs`), and never below 1.
        `concurrency_floor` only acts as a lower bound *within* the
        `[1, static_cap]` range — when the floor exceeds the static cap
        the static cap wins, so this method is safe to call regardless of
        the user-supplied floor value.

        When the controller is disabled (`gamma_max == 0`) or gamma is
        zero, returns the unchanged static cap.
        """
        cfg = self.config
        if cfg.gamma_max <= 0.0 or self.gamma <= 0.0:
            return static_cap
        reduction = self.gamma / cfg.gamma_max
        effective = int(round(static_cap * (1.0 - reduction)))
        # Floor below static_cap, clamp to [1, static_cap]. The outer
        # `min(static_cap, ...)` guarantees we never exceed the scheduler's
        # invariant even if `concurrency_floor > static_cap`.
        floor = max(1, cfg.concurrency_floor)
        return max(1, min(static_cap, max(floor, effective)))

    def discard(self, request_id: str) -> None:
        """Drop the per-request observation when a request finishes."""
        self._observations.pop(request_id, None)

    # ── Introspection (for tests / logs) ────────────────────────────────

    @property
    def M(self) -> float:
        return self._M

    def num_observations(self) -> int:
        return len(self._observations)
