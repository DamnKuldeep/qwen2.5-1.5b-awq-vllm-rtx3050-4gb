"""
Concurrency ramp - Stage 9.

Finds the concurrency level at which the system stops meeting a stated SLO, and
provides the evidence needed to say WHICH resource ran out first.

WHY A CUSTOM SCRIPT RATHER THAN LOCUST OR k6 (DECISIONS.md)
-----------------------------------------------------------
No new framework to install or learn, and every line of what is being measured
is visible here. The timing logic below is the same manual approach used
throughout this project: record a timestamp before the request, another at the
first streamed token, another at the last.

THREE DESIGN DECISIONS THAT COME FROM EARLIER STAGES
-----------------------------------------------------
1. UNIQUE PROMPTS PER REQUEST, BY DEFAULT.
   Stage 8 measured two identical benchmark runs differing by 16% throughput
   and 45% on the ITL tail, purely because the second run's prompts were
   already in the prefix cache. A ramp with repeated prompts would measure the
   cache, not the hardware. Each request here gets a distinct prompt.

   The --shared-prefix flag inverts this: every request shares one long
   preamble, which is the workload Stage 11 needs to prove prefix caching
   actually does something. One script, two experiments.

2. IT TALKS TO THE GATEWAY (8080), NOT vLLM (8000).
   This is a product load test. Auth, budget checks, SQLite writes and the
   streaming proxy are all in the path here because they are all in the path in
   production. It also exercises the Stage 4 prediction that budget overshoot
   is bounded by (prompt_tokens + max_tokens) x requests_in_flight - a much
   larger number at concurrency 32 than at 1.

3. A DISCARDED WARM-UP LEVEL, AND THE FIRST REQUEST OF EACH LEVEL IS KEPT.
   Stage 7 measured the first request after startup at 3,292 ms against 310 ms
   later - one-time kernel autotuning and tokenizer warm-up. A warm-up level
   absorbs that. Within a level nothing is discarded, because at that point the
   slow requests are real queueing, which is what we are trying to measure.

Requires: the stack running (docker compose up -d) and the database seeded.

Usage:
    python load_testing/ramp.py
    python load_testing/ramp.py --levels 1,4,8,16,32 --slo-ttft 1.5
    python load_testing/ramp.py --shared-prefix        # Stage 11 variant
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field

import httpx

# A word pool for synthetic prompts. Real words rather than random characters
# because the tokenizer handles them predictably - random character soup
# fragments into far more tokens per character and makes prompt length hard to
# control.
WORDS = (
    "system memory kernel latency throughput cache tensor batch scheduler token "
    "gradient inference pipeline compute bandwidth register kernel occupancy "
    "quantization attention embedding transformer decoder prefill sequence "
    "allocation fragmentation eviction preemption utilisation saturation queue"
).split()

SHARED_PREAMBLE = (
    "You are a technical assistant. Follow these standing instructions on every "
    "request. Be precise. Prefer concrete numbers over adjectives. When you are "
    "uncertain, say so explicitly rather than hedging. Never invent citations. "
    "Structure longer answers with short paragraphs. Define any acronym on first "
    "use. Assume the reader is an engineer who wants the mechanism, not an "
    "analogy. If a question has a false premise, say so before answering. "
) * 6


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    prompt_tokens: int = 0
    itls_ms: list[float] = field(default_factory=list)
    status: int = 0
    error: str = ""


@dataclass
class LevelResult:
    concurrency: int
    duration_s: float
    results: list[RequestResult]

    @property
    def ok(self) -> list[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> int:
        return len(self.results) - len(self.ok)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.ok)


def pct(values: list[float], p: float) -> float:
    """Percentile by nearest-rank. Explicit rather than via numpy so the
    definition being used is visible - percentile implementations differ, and a
    p95 that means something slightly different run-to-run is worse than none.

    IMPORTANT LIMITATION, found the hard way in Stage 10(c): with nearest-rank,
    p95 of 16 samples is s[15] - the MAXIMUM. A single outlier then defines it
    entirely. That is exactly what happened when a cold-start request after a
    vLLM restart made p95 TTFT read 4,700 ms against 1,419 ms on an otherwise
    identical run.

    Minimum sample size per level was raised to 24 as a result (p95 of 24 is the
    23rd value, not the largest), and the runner warns below 40 samples, where
    p95 is still dominated by one or two requests.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def make_prompt(index: int, target_words: int, shared_prefix: bool, echo_task: bool = False) -> str:
    """Build a prompt that is either unique per request or shares a long preamble.

    Unique mode seeds the RNG from the request index, so prompts differ between
    requests (defeating the prefix cache) while the whole run stays reproducible
    across invocations.
    """
    if echo_task:
        # A task whose OUTPUT overlaps its INPUT, for testing n-gram
        # speculative decoding.
        #
        # N-gram speculation drafts tokens by looking for the current suffix
        # earlier in the context and proposing what followed it. It therefore
        # helps only when the model reproduces text it has already seen -
        # summarisation of a quoted passage, code editing, structured
        # extraction. On this project's default random-word prompts the output
        # shares nothing with the input, so speculation would draft garbage,
        # fail verification, and cost time rather than save it.
        #
        # Measuring speculative decoding on a workload it cannot help would
        # produce a true number that answers the wrong question.
        rng = random.Random(index)
        body = " ".join(rng.choice(WORDS) for _ in range(max(20, target_words // 2)))
        return (
            "Repeat the following notes back verbatim, exactly as written, "
            f"with no commentary or preamble.\n\n{body}\n\nRepeat them now:"
        )

    if shared_prefix:
        # Identical preamble, tiny unique suffix. The preamble is a prefix-cache
        # hit after the first request; only the suffix needs prefilling.
        return f"{SHARED_PREAMBLE}\n\nQuestion {index}: summarise the above in one sentence."

    rng = random.Random(index)
    body = " ".join(rng.choice(WORDS) for _ in range(target_words))
    # Leading unique marker guarantees divergence from token zero, so not even
    # the first block can be shared with another request.
    return f"Request {index}. Summarise the following notes in one sentence.\n\n{body}"


async def one_request(
    client: httpx.AsyncClient, index: int, args, sem: asyncio.Semaphore
) -> RequestResult:
    """Issue one streamed request and time it token by token."""
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": make_prompt(
                    index, args.prompt_words, args.shared_prefix, args.echo_task
                ),
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }

    # ignore_eos forces exactly max_tokens outputs instead of stopping at EOS.
    # The first ramp run showed replies averaging ~42 tokens against a
    # max_tokens of 128 - the same defect that invalidated the first Stage 2
    # benchmark, reappearing here. Harmless when measuring queueing, but fatal
    # for Stage 11's before/after comparison, where output volume must be
    # identical between runs.
    #
    # Worth noting WHY this needs no gateway change: `ignore_eos` is a vLLM
    # extension, not part of the OpenAI schema. It reaches the engine only
    # because the gateway forwards unmodelled fields untouched - the Boundary 1
    # decision from Stage 3, paying for itself in a place we did not anticipate.
    if args.ignore_eos:
        body["ignore_eos"] = True

    async with sem:
        t0 = time.perf_counter()
        t_first: float | None = None
        t_prev: float | None = None
        itls: list[float] = []
        usage: dict = {}

        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {args.api_key}"},
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    return RequestResult(
                        ok=False, status=resp.status_code, error=resp.text[:200]
                    )

                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    # SSE events do not align with network reads, so buffer and
                    # split on the blank-line delimiter (same reasoning as the
                    # browser parser in ui/index.html).
                    while "\n\n" in buf:
                        event, buf = buf.split("\n\n", 1)
                        for line in event.split("\n"):
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                obj = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            delta = ""
                            choices = obj.get("choices") or []
                            if choices:
                                delta = (choices[0].get("delta") or {}).get("content") or ""

                            if delta:
                                now = time.perf_counter()
                                if t_first is None:
                                    t_first = now
                                else:
                                    itls.append((now - t_prev) * 1000)
                                t_prev = now

                            if obj.get("usage"):
                                usage = obj["usage"]

            t_end = time.perf_counter()
            if t_first is None:
                return RequestResult(ok=False, status=200, error="no tokens received")

            return RequestResult(
                ok=True,
                ttft_ms=(t_first - t0) * 1000,
                e2e_ms=(t_end - t0) * 1000,
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                itls_ms=itls,
                status=200,
            )

        except Exception as exc:  # noqa: BLE001 - a load test wants every failure
            return RequestResult(ok=False, error=f"{type(exc).__name__}: {exc}")


async def run_level(client: httpx.AsyncClient, concurrency: int, n: int, args) -> LevelResult:
    """Run n requests with at most `concurrency` in flight."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    tasks = [one_request(client, args.next_index + i, args, sem) for i in range(n)]
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - t0
    # Advance the index so the next level's prompts differ from this one's too,
    # otherwise level N+1 would replay level N's prompts straight out of cache.
    args.next_index += n
    return LevelResult(concurrency=concurrency, duration_s=duration, results=list(results))


def print_row(lr: LevelResult, slo_ttft_s: float) -> bool:
    ttfts = [r.ttft_ms for r in lr.ok if r.ttft_ms is not None]
    itls = [x for r in lr.ok for x in r.itls_ms]
    e2es = [r.e2e_ms for r in lr.ok if r.e2e_ms is not None]

    p95_ttft = pct(ttfts, 95) / 1000 if ttfts else float("nan")
    meets = p95_ttft < slo_ttft_s
    tput = lr.output_tokens / lr.duration_s if lr.duration_s else 0.0

    # Mean prompt tokens actually sent. Reported because Experiment 2 compares
    # shared-prefix against unique prompts, and that comparison is only valid
    # if the prompts are of comparable length - the shared preamble is a fixed
    # string while unique prompts are generated to a word count, so the two
    # cannot be assumed equal. Print it rather than trust it.
    prompts = [r.prompt_tokens for r in lr.ok if r.prompt_tokens]
    mean_prompt = sum(prompts) / len(prompts) if prompts else 0

    print(
        f"| {lr.concurrency:>4} "
        f"| {len(lr.ok):>3}/{len(lr.results):<3} "
        f"| {mean_prompt:>6.0f} "
        f"| {tput:>8.1f} "
        f"| {pct(ttfts, 50):>9.0f} "
        f"| {p95_ttft * 1000:>9.0f} "
        f"| {pct(itls, 50):>8.1f} "
        f"| {pct(itls, 99):>8.1f} "
        f"| {pct(e2es, 50) / 1000:>8.2f} "
        f"| {'PASS' if meets else 'FAIL':>4} "
        # Margin matters as much as the verdict. The 3B passed this SLO once by
        # 24 ms and failed it later by 30 ms - the same measurement either side
        # of the line, with thermal drift (~24%) far larger than the gap. A bare
        # PASS/FAIL turns noise into a fact; the margin shows when a verdict is
        # meaningless. Positive = headroom.
        f"| {(slo_ttft_s - p95_ttft) / slo_ttft_s * 100:>+6.0f}% |"
    )
    return meets


async def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrency ramp against the gateway.")
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--api-key", default="dev-key-alpha")
    ap.add_argument("--model", default="",
                    help="Model to request. Empty (default) asks the gateway via "
                         "/v1/models, so a model swap needs no flag change.")
    ap.add_argument("--levels", default="1,2,4,8,16,24,32",
                    help="comma-separated concurrency levels")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-words", type=int, default=380,
                    help="~380 words is roughly 512 tokens, matching the Stage 2 baseline")
    ap.add_argument("--slo-ttft", type=float, default=1.5,
                    help="SLO: p95 TTFT in seconds. Stated before measuring, on purpose.")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="All requests share a long preamble (Stage 11 prefix-caching test)")
    ap.add_argument("--echo-task", action="store_true",
                    help="Ask the model to repeat its input verbatim. Required for a "
                         "meaningful n-gram speculative-decoding measurement: speculation "
                         "only helps when output overlaps input, and the default random "
                         "prompts share nothing with their answers.")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Force exactly --max-tokens outputs. Required for Stage 11 "
                         "before/after comparisons; without it replies stop at EOS and "
                         "output volume varies between runs.")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--min-requests", type=int, default=24,
                    help="Minimum requests per level. Stage 11's protocol wants 40+: "
                         "with nearest-rank percentiles, p95 below ~40 samples is "
                         "still set by one or two requests.")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="Starting prompt index. Prompts are seeded from the request "
                         "index, and this counter resets to 0 on every invocation - so "
                         "a warm-up run and the measured run that follows it generate "
                         "THE SAME PROMPTS, and the measured run's early levels are "
                         "served entirely from the prefix cache the warm-up populated. "
                         "Found in Finalization Phase 2: a warm-up of 80 requests made "
                         "the measured ramp report TTFT p50 of 77 ms at concurrency 2 "
                         "against 229 ms at concurrency 1 - latency improving as "
                         "concurrency rises, which is impossible without cache hits. "
                         "Give the measured run an offset past the end of the warm-up "
                         "(the standard form uses 1000) so its prompts are genuinely "
                         "cold. Levels within one run are already unique.")
    args = ap.parse_args()

    args.next_index = args.index_offset
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout, limits=limits) as client:
        try:
            health = (await client.get("/health")).json()
        except Exception as exc:  # noqa: BLE001
            print(f"Cannot reach the gateway at {args.base_url}: {exc}", file=sys.stderr)
            sys.exit(1)
        if health.get("vllm") != "ok":
            print(f"Gateway is up but vLLM is not: {health}", file=sys.stderr)
            sys.exit(1)

        # Verify the engine is actually serving the model being benchmarked.
        #
        # Added after a full ramp - 320 requests over several minutes - returned
        # nothing but 404s because a `docker compose up` without its -f override
        # flags had silently reverted the engine to a different model. Compose
        # reconciles EVERY service against the base file, not just the one
        # named, so omitting -f is not a no-op: it reverts.
        #
        # One request here turns a multi-minute silent failure into an
        # immediate, specific error.
        try:
            listed = await client.get(
                "/v1/models", headers={"Authorization": f"Bearer {args.api_key}"}
            )
            served = [m["id"] for m in listed.json().get("data", [])]
        except Exception:  # noqa: BLE001 - a missing model list is not fatal
            served = []

        # Empty --model means "use whatever the engine is serving". This makes a
        # model swap require no flag change, and removes the class of error where
        # a whole ramp 404s because the engine was quietly changed underneath it.
        if not args.model:
            if not served:
                print("Cannot determine the served model and --model was not given.",
                      file=sys.stderr)
                sys.exit(1)
            args.model = served[0]
            print(f"Auto-detected model: {args.model}")

        if served and args.model not in served:
            print(f"STOP: the server is serving {served}, not '{args.model}'.", file=sys.stderr)
            print("      Every request would return 404.", file=sys.stderr)
            print("      Either pass --model with one of the names above, or check that your", file=sys.stderr)
            print("      `docker compose` command carried the right -f override flags.", file=sys.stderr)
            sys.exit(1)

        mode = "SHARED PREFIX" if args.shared_prefix else "unique prompts"
        print(f"\nRamp against {args.base_url}  |  {mode}  |  "
              f"max_tokens={args.max_tokens}  |  SLO: p95 TTFT < {args.slo_ttft}s\n")

        if not args.no_warmup:
            print("Warm-up level (discarded) ...", flush=True)
            await run_level(client, 4, 8, args)

        print("| conc |  ok/n   | prompt |    tok/s |  TTFT p50 |  TTFT p95 |  ITL p50 |  ITL p99 |  E2E p50 |  SLO | margin |")
        print("| ---: | :-----: | -----: | -------: | --------: | --------: | -------: | -------: | -------: | :--: | -----: |")

        broke_at: int | None = None
        for c in levels:
            # 24 minimum, not 16: with nearest-rank percentiles, p95 of 16
            # samples IS the maximum, so one outlier defines it. See pct().
            n = max(args.min_requests, 2 * c)
            lr = await run_level(client, c, n, args)
            meets = print_row(lr, args.slo_ttft)
            if n < 40:
                print(f"       note: n={n}, so p95 is dominated by the slowest "
                      f"1-2 requests. Treat it as indicative, not stable.")
            if not meets and broke_at is None:
                broke_at = c
            if lr.failed:
                # Report the first FAILED result, not results[0] - which is
                # usually a SUCCESS, and printed an empty error string during
                # the Stage 10(c) kill test, hiding the actual failure reason.
                first_err = next((r for r in lr.results if not r.ok), None)
                detail = ""
                if first_err:
                    detail = f"HTTP {first_err.status} " if first_err.status else ""
                    detail += first_err.error
                print(f"       {lr.failed}/{len(lr.results)} FAILED at concurrency {c}: "
                      f"{detail[:160]}")

        print()
        if broke_at is None:
            print(f"SLO held at every level tested (up to {levels[-1]}). "
                  f"The ceiling is above this range - extend --levels.")
        else:
            print(f"SLO (p95 TTFT < {args.slo_ttft}s) FIRST BREAKS AT CONCURRENCY {broke_at}.")
        print(
            "\nNow read the Grafana dashboard for the same window to say WHICH resource ran out:\n"
            "  - KV cache usage near 100% and Preemptions non-zero -> MEMORY ceiling\n"
            "  - KV cache low, Queue time rising, running pinned at --max-num-seqs\n"
            "    -> the engine's own concurrency cap, not the hardware\n"
            "  - KV cache low, queue time low, ITL rising -> COMPUTE/thermal ceiling\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
