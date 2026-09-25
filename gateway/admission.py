"""
Admission control, per-key fairness, and gateway-side metrics — v2.

WHY THIS EXISTS
---------------
v1 accepted every request and forwarded it. Under overload that produces the
worst possible behaviour: Stage 9 measured TTFT climbing past 13 s and still
rising, with every user waiting and nobody served well. An unbounded queue
converts "too much load" into "everyone gets a bad experience" instead of
"some people get told to come back".

The measured basis for the numbers here (Finalization Phases 2-3, 1.5B-AWQ):

    concurrency 4   p95 TTFT  696-928 ms    PASS, +38..54% margin
    concurrency 6   p95 TTFT 1005-1135 ms   PASS, +24..33% margin
    concurrency 8   p95 TTFT 1401-1605 ms   marginal, -7..+7%
    concurrency 12  p95 TTFT     2043 ms    FAIL, -36%

So six concurrent in-flight requests is the largest level that holds
p95 TTFT < 1.5 s with margin outside the ~25% thermal noise band. That is
MAX_INFLIGHT's default, and it is a measurement rather than a guess.

THE QUEUE IS DELIBERATELY SHALLOW, AND THE ARITHMETIC SAYS WHY
--------------------------------------------------------------
A queued request's TTFT includes its queue wait. At concurrency 6 the service
completes roughly 1.4 requests/second, so every extra queue slot adds ~0.7 s to
the TTFT of whoever sits in it. Two slots of queue already consumes the entire
1.5 s SLO budget.

The honest consequence: **you cannot hold a tight TTFT SLO and also queue
deeply.** The queue here exists to absorb sub-second arrival jitter, not to
store backlog. Anything beyond it is refused immediately, which is the whole
point - a fast, honest 503 is a better product than a 30-second success.

FAIRNESS — AND THE FIXED CAP THAT DID NOT WORK
----------------------------------------------
A global limit alone lets one caller take every slot, so there is also a
per-key limit. The first version was a FIXED cap of 3 against 6 total slots,
and the adversarial scenario showed it failing badly: one abusive key at 50
concurrent, alongside ten normal users, pushed **10/10 normal users past the
SLO** with p95 TTFT of 4,825 ms.

The cap was working exactly as written, and the arithmetic was the problem. A
fixed cap of 3 out of 6 hands one tenant **half the service no matter how many
other tenants exist**. That is a quota, not fairness - with eleven active keys,
a fair share is 6/11, not 3.

The fix is a DYNAMIC share: divide the pool by the number of keys currently
contending, and use the static value only as an upper bound for the case where
nobody else wants the capacity.

    limit = clamp(ceil(MAX_INFLIGHT / active_keys), 1, MAX_INFLIGHT_PER_KEY)

With 1 active key it stays 3; with 7 it becomes 1. The abusive key's share
falls from 50% of the service to 14% the moment other tenants show up, and
rises back when they leave. This is the mechanism worst-case #2 tests.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Concurrent requests forwarded to the engine. Measured ceiling, see above.
MAX_INFLIGHT = _int_env("GATEWAY_MAX_INFLIGHT", 6)

# Requests allowed to WAIT for a slot. Shallow on purpose - see the docstring.
MAX_QUEUE = _int_env("GATEWAY_MAX_QUEUE", 12)

# How long a request may wait before we give up and shed it. This is the term
# that BOUNDS p95 TTFT under overload: worst-case TTFT is roughly
# QUEUE_TIMEOUT + engine TTFT, and it does not grow with offered load.
QUEUE_TIMEOUT_S = _float_env("GATEWAY_QUEUE_TIMEOUT_S", 2.0)

# Per-key share of the pool. Half, so one key can never starve the rest.
MAX_INFLIGHT_PER_KEY = _int_env("GATEWAY_MAX_INFLIGHT_PER_KEY", max(1, MAX_INFLIGHT // 2))

# A stream that outlives this is abandoned and its slot released. Without it a
# client reading one byte per second holds a slot indefinitely (worst case #8).
STREAM_TIMEOUT_S = _float_env("GATEWAY_STREAM_TIMEOUT_S", 300.0)

# Advertised in Retry-After on a shed request.
RETRY_AFTER_S = _int_env("GATEWAY_RETRY_AFTER_S", 2)

# Weighted admission. A prompt longer than this counts as more than one slot.
# 2048 is not arbitrary - it is `--max-num-batched-tokens`, the engine's prefill
# chunk size, so cost is measured in "how many scheduler steps of prefill will
# this monopolise".
LONG_PROMPT_TOKENS = _int_env("GATEWAY_LONG_PROMPT_TOKENS", 2048)

# Ceiling on one request's cost, so a giant prompt can never take the whole
# pool. At 4 of 6 slots, two short requests can always still be served.
MAX_REQUEST_COST = _int_env("GATEWAY_MAX_REQUEST_COST", max(1, MAX_INFLIGHT - 2))


class Rejected(Exception):
    """Raised when a request cannot be admitted. Carries the reason for metrics."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Admission:
    """Bounded concurrency with a shallow queue and a per-key share.

    Deliberately not a token bucket or a rate limiter. Those cap *arrivals*,
    which is the wrong variable: what actually degrades TTFT is the number of
    sequences the engine is decoding at once. Limiting in-flight work maps
    directly onto the thing that was measured.
    """

    def __init__(self) -> None:
        # A counter guarded by a Condition rather than a Semaphore, because
        # admission is WEIGHTED: a request may need several slots at once, and
        # a Semaphore cannot be acquired N-at-a-time atomically.
        self._used = 0
        self._cond = asyncio.Condition()
        self._inflight = 0
        self._waiting = 0
        self._per_key: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

        # Counters for /metrics. Plain ints; the GIL makes += safe enough here
        # and a lock on the metrics path would be a self-inflicted bottleneck.
        self.admitted_total = 0
        self.rejected_total: dict[str, int] = defaultdict(int)
        self.queue_wait_total_s = 0.0
        self.queue_waits = 0
        self.peak_inflight = 0
        self.peak_waiting = 0
        self.weighted_admissions = 0

    # -- introspection, for /metrics and the dashboard ---------------------

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def slots_used(self) -> int:
        """Weighted slots currently held.

        Distinct from `inflight`, which counts REQUESTS. A single long prompt
        can hold four slots while being one request, so these two gauges
        diverge exactly when head-of-line blocking is the thing you are
        looking for - which is why both are exported.
        """
        return self._used

    def snapshot(self) -> dict:
        return {
            "inflight": self._inflight,
            "waiting": self._waiting,
            "max_inflight": MAX_INFLIGHT,
            "max_queue": MAX_QUEUE,
            "max_inflight_per_key": MAX_INFLIGHT_PER_KEY,
            "slots_used": self.slots_used,
            "long_prompt_tokens": LONG_PROMPT_TOKENS,
            "max_request_cost": MAX_REQUEST_COST,
            "weighted_admissions": self.weighted_admissions,
            "active_keys": len(self._per_key),
            "current_share_per_key": max(1, min(
                MAX_INFLIGHT_PER_KEY, -(-MAX_INFLIGHT // max(1, len(self._per_key))))),
            "admitted_total": self.admitted_total,
            "rejected_total": dict(self.rejected_total),
            "peak_inflight": self.peak_inflight,
            "peak_waiting": self.peak_waiting,
            "mean_queue_wait_ms": round(
                1000 * self.queue_wait_total_s / self.queue_waits, 2
            ) if self.queue_waits else 0.0,
        }

    # -- the admission decision --------------------------------------------

    def _key_limit(self, api_key: str) -> int:
        """This key's current fair share of the pool.

        Counted over keys currently CONTENDING (in flight or waiting), not over
        every key that exists - a tenant that is idle is not owed a slot, and
        dividing by the total key count would throttle everyone to nothing on a
        service with many registered but quiet users.

        The requesting key is included in the count whether or not it already
        holds a slot, so a newly arriving tenant immediately tightens everyone
        else's share rather than having to wait for the next arrival.
        """
        active = len(self._per_key) + (0 if api_key in self._per_key else 1)
        share = -(-MAX_INFLIGHT // max(1, active))  # ceiling division
        return max(1, min(MAX_INFLIGHT_PER_KEY, share))

    @staticmethod
    def cost_of(prompt_tokens: int) -> int:
        """How many admission slots this request consumes.

        ADMITTING BY REQUEST COUNT SYSTEMATICALLY MIS-ADMITS, because requests
        are not the same size. Worst case #3 measured a ~20k-token prompt
        pushing concurrent short requests from 82 ms to 12,201 ms of TTFT: one
        request occupied the engine's prefill capacity for twelve seconds while
        counting as a single unit of load.

        vLLM has the right controls for this engine-side
        (`--max-num-partial-prefills`), but enabling them on this build forces
        the V0 engine and the 0.11.0 API server then fails to start - so the
        bound has to be applied here instead.

        Cost is prompt length divided by the prefill chunk size, capped so that
        no single request can consume the entire pool: a giant prompt takes
        several slots, which limits how many can prefill concurrently AND
        leaves capacity for short requests. It is a coarse proxy for GPU
        seconds, and coarse is enough - the thing being prevented is one
        request quietly counting as a sixth of the service when it is really
        most of it.
        """
        if prompt_tokens <= LONG_PROMPT_TOKENS:
            return 1
        return min(MAX_REQUEST_COST, 1 + prompt_tokens // LONG_PROMPT_TOKENS)

    async def acquire(self, api_key: str, cost: int = 1) -> float:
        """Take `cost` slots, or raise Rejected. Returns seconds spent queued.

        Three refusal paths, each with its own metric label so an operator can
        tell them apart on a dashboard - "we are full" and "you personally are
        using too much" are different problems with different fixes.
        """
        cost = max(1, min(MAX_REQUEST_COST, cost))
        async with self._lock:
            limit = self._key_limit(api_key)
            if self._per_key.get(api_key, 0) >= limit:
                self.rejected_total["per_key_limit"] += 1
                raise Rejected(
                    "per_key_limit",
                    f"Too many concurrent requests for this API key "
                    f"({limit} in flight; the limit is your share of "
                    f"{MAX_INFLIGHT} slots across active clients). Retry shortly.",
                )
            if self._waiting >= MAX_QUEUE:
                self.rejected_total["queue_full"] += 1
                raise Rejected(
                    "queue_full",
                    f"Server is at capacity ({MAX_INFLIGHT} concurrent, "
                    f"{MAX_QUEUE} queued). Retry shortly.",
                )
            self._waiting += 1
            self._per_key[api_key] += 1
            self.peak_waiting = max(self.peak_waiting, self._waiting)

        queued_at = time.perf_counter()
        try:
            async def take():
                async with self._cond:
                    await self._cond.wait_for(
                        lambda: self._used + cost <= MAX_INFLIGHT)
                    self._used += cost
            await asyncio.wait_for(take(), timeout=QUEUE_TIMEOUT_S)
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting -= 1
                self._per_key[api_key] -= 1
                if self._per_key[api_key] <= 0:
                    self._per_key.pop(api_key, None)
                self.rejected_total["queue_timeout"] += 1
            raise Rejected(
                "queue_timeout",
                f"Timed out waiting {QUEUE_TIMEOUT_S:g}s for a free slot. Retry shortly.",
            )

        waited = time.perf_counter() - queued_at
        async with self._lock:
            self._waiting -= 1
            self._inflight += 1
            self.peak_inflight = max(self.peak_inflight, self._inflight)
            self.admitted_total += 1
            self.queue_wait_total_s += waited
            self.queue_waits += 1
            if cost > 1:
                self.weighted_admissions += 1
        return waited

    async def release(self, api_key: str, cost: int = 1) -> None:
        async with self._lock:
            self._inflight -= 1
            self._per_key[api_key] -= 1
            if self._per_key[api_key] <= 0:
                self._per_key.pop(api_key, None)
        async with self._cond:
            self._used -= cost
            self._cond.notify_all()


# --------------------------------------------------------------------------
# Metrics exposition
# --------------------------------------------------------------------------
#
# Hand-rolled Prometheus text format rather than pulling in prometheus_client.
# The exposition format is a dozen lines of text and adding a dependency to
# emit it would be a poor trade - especially on Boundary 2 grounds, where every
# pinned dependency is one more thing to keep pinned.


class Stats:
    """Request-outcome counters and a TTFT histogram, for /metrics."""

    # Buckets chosen around the 1.5 s SLO so the panel can show the SLO line.
    TTFT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)

    def __init__(self) -> None:
        self.requests_total: dict[int, int] = defaultdict(int)
        self.ttft_bucket_counts = [0] * (len(self.TTFT_BUCKETS) + 1)
        self.ttft_sum_s = 0.0
        self.ttft_count = 0
        self.context_trimmed_total = 0
        self.context_trimmed_messages = 0
        self.upstream_errors_total = 0
        # Set by main.record() when the ledger write fails. Non-zero means the
        # service is serving work it cannot account for - an alertable state.
        self.billing_errors = 0

    def record_status(self, code: int) -> None:
        self.requests_total[code] += 1

    def record_ttft(self, seconds: float) -> None:
        self.ttft_sum_s += seconds
        self.ttft_count += 1
        for i, edge in enumerate(self.TTFT_BUCKETS):
            if seconds <= edge:
                self.ttft_bucket_counts[i] += 1
                return
        self.ttft_bucket_counts[-1] += 1

    def render(self, adm: Admission) -> str:
        out: list[str] = []
        a = out.append

        a("# HELP gateway_requests_total Requests completed, by HTTP status.")
        a("# TYPE gateway_requests_total counter")
        for code, n in sorted(self.requests_total.items()):
            a(f'gateway_requests_total{{status="{code}"}} {n}')

        a("# HELP gateway_inflight Requests currently forwarded to the engine.")
        a("# TYPE gateway_inflight gauge")
        a(f"gateway_inflight {adm.inflight}")

        a("# HELP gateway_queue_depth Requests currently waiting for a slot.")
        a("# TYPE gateway_queue_depth gauge")
        a(f"gateway_queue_depth {adm.waiting}")

        a("# HELP gateway_max_inflight Configured concurrency limit.")
        a("# TYPE gateway_max_inflight gauge")
        a(f"gateway_max_inflight {MAX_INFLIGHT}")

        a("# HELP gateway_admitted_total Requests that acquired a slot.")
        a("# TYPE gateway_admitted_total counter")
        a(f"gateway_admitted_total {adm.admitted_total}")

        a("# HELP gateway_rejected_total Requests shed by admission control.")
        a("# TYPE gateway_rejected_total counter")
        for reason in ("queue_full", "queue_timeout", "per_key_limit"):
            a(f'gateway_rejected_total{{reason="{reason}"}} {adm.rejected_total.get(reason, 0)}')

        a("# HELP gateway_queue_wait_seconds_total Cumulative time spent queued.")
        a("# TYPE gateway_queue_wait_seconds_total counter")
        a(f"gateway_queue_wait_seconds_total {adm.queue_wait_total_s:.6f}")

        a("# HELP gateway_ttft_seconds Time to first token, as the CLIENT sees it.")
        a("# TYPE gateway_ttft_seconds histogram")
        cumulative = 0
        for i, edge in enumerate(self.TTFT_BUCKETS):
            cumulative += self.ttft_bucket_counts[i]
            a(f'gateway_ttft_seconds_bucket{{le="{edge}"}} {cumulative}')
        cumulative += self.ttft_bucket_counts[-1]
        a(f'gateway_ttft_seconds_bucket{{le="+Inf"}} {cumulative}')
        a(f"gateway_ttft_seconds_sum {self.ttft_sum_s:.6f}")
        a(f"gateway_ttft_seconds_count {self.ttft_count}")

        a("# HELP gateway_context_trimmed_total Requests whose history was trimmed to fit.")
        a("# TYPE gateway_context_trimmed_total counter")
        a(f"gateway_context_trimmed_total {self.context_trimmed_total}")

        a("# HELP gateway_context_trimmed_messages_total Messages dropped by trimming.")
        a("# TYPE gateway_context_trimmed_messages_total counter")
        a(f"gateway_context_trimmed_messages_total {self.context_trimmed_messages}")

        a("# HELP gateway_upstream_errors_total Failures reaching the engine.")
        a("# TYPE gateway_upstream_errors_total counter")
        a(f"gateway_upstream_errors_total {self.upstream_errors_total}")

        a("# HELP gateway_weighted_admissions_total Requests that cost more than one slot.")
        a("# TYPE gateway_weighted_admissions_total counter")
        a(f"gateway_weighted_admissions_total {adm.weighted_admissions}")

        a("# HELP gateway_slots_used Admission slots currently held (weighted).")
        a("# TYPE gateway_slots_used gauge")
        a(f"gateway_slots_used {adm.slots_used}")

        a("# HELP gateway_billing_write_errors_total Ledger writes that failed.")
        a("# TYPE gateway_billing_write_errors_total counter")
        a(f"gateway_billing_write_errors_total {self.billing_errors}")

        return "\n".join(out) + "\n"
