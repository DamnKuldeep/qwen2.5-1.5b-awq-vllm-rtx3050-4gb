"""
Multi-user chat simulator - the instrument v2's capacity model is built on.

WHY ramp.py IS NOT ENOUGH
--------------------------
`ramp.py` holds N requests permanently in flight with unique prompts. That
measures the ENGINE, and it measured it well - the SLO ceiling of ~6 concurrent
requests comes from it. But it is not what a chat service experiences:

  * Real users think between turns. A request is in flight maybe 12% of the
    time, so "100 users" and "100 concurrent requests" differ by ~8x.
  * Real conversations ACCUMULATE. Turn 8 resends turns 1-7, so the prompt
    grows every turn and the cost per turn grows with it.
  * Real conversations SHARE A PREFIX with themselves across turns, which is
    the entire reason prefix caching matters - and `ramp.py`'s unique prompts
    deliberately defeat it.
  * Real arrivals are bursty, not uniform.

THE MEASUREMENT THIS EXISTS FOR
--------------------------------
Prefix cache hit rate as a function of user count, alongside p95 TTFT.

The KV cache is ONE SHARED POOL of ~69,760 tokens, not a per-user allocation.
With 4,000-token conversations only ~17 fit. Past that, LRU eviction means a
returning user's history has been pushed out and their next turn re-prefills
everything - roughly 9x the cost of a cache hit. The prediction on the record
is that capacity shows a KNEE rather than a slope, because eviction is a
threshold effect. If it is a gentle slope instead, the model is wrong somewhere
and that is the more interesting result.

PER-USER, NOT PER-REQUEST
--------------------------
Aggregating across requests hides the user who was unlucky every single turn.
A user cares about the p95 of THEIR turns. This reports both, and the gap
between them is itself a fairness signal.

Usage:
    python load_testing/chat_sim.py --users 20 --duration 120 --out results.json
    python load_testing/chat_sim.py --users 40 --pattern burst --out burst.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

# --------------------------------------------------------------------------
# Prompt material
# --------------------------------------------------------------------------

# A shared system prompt, because every real chat product has one and it is the
# single most valuable thing for prefix caching: identical leading tokens for
# every user of every conversation, so those blocks are stored once and hit by
# everyone. Stage 11 Experiment 2 measured a shared prefix doubling SLO-
# compliant concurrency, so leaving it out would model a worse system than
# anyone would actually build.
SYSTEM_PROMPT = (
    "You are a helpful, concise technical assistant for a software engineering "
    "team. Answer accurately. Prefer short paragraphs and bulleted lists. When "
    "you show code, use fenced code blocks with a language tag. If you are "
    "unsure about something, say so plainly rather than guessing. Keep answers "
    "focused on what was asked."
)

TOPICS = [
    "database indexing", "TCP congestion control", "Python asyncio", "Docker layer caching",
    "CUDA memory coalescing", "Kubernetes readiness probes", "SQL query planning",
    "HTTP caching headers", "Rust borrow checker", "distributed consensus",
    "garbage collection", "vector databases", "load balancer algorithms",
    "TLS handshakes", "message queue delivery guarantees", "B-tree structure",
    "CPU branch prediction", "filesystem journaling", "OAuth token flows",
    "gRPC streaming", "columnar storage", "consistent hashing",
]

FOLLOW_UPS = [
    "Can you expand on that with a concrete example?",
    "What are the main trade-offs there?",
    "How would that behave under heavy load?",
    "What is the most common mistake people make with this?",
    "How does that compare to the usual alternative?",
    "Can you show a short code sample?",
    "What would you monitor to know it is working?",
    "Where does that approach break down?",
]

# Filler used to inflate a turn to a target size, for the long-conversation and
# long-tail scenarios. Real words rather than random characters because the
# tokenizer handles them predictably - character soup tokenizes at a wildly
# different ratio and would make token-count targets meaningless.
FILLER_WORDS = (
    "system request buffer latency throughput pipeline scheduler allocator "
    "kernel pointer segment offset checksum protocol handshake replica shard "
    "partition consumer producer transaction isolation durability snapshot "
).split()


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class TurnRecord:
    user: int
    turn: int
    status: int
    ttft_ms: float | None
    e2e_ms: float
    itl_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    queue_wait_ms: int
    trimmed: int
    started_at: float
    conversation_tokens: int
    shed_reason: str | None = None


@dataclass
class UserSummary:
    user: int
    turns: int
    ok: int
    shed: int
    errors: int
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_max: float | None
    mean_conversation_tokens: float


def pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated because Stage 10(c) established that
    with small n the interpolated value invents precision the sample does not
    have. With n < 20, p95 IS effectively the maximum and should look like it.
    """
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(p / 100 * len(s)))
    return round(s[k - 1], 1)


# --------------------------------------------------------------------------
# Traffic shapes
# --------------------------------------------------------------------------


def start_delay(pattern: str, user_index: int, n_users: int, duration: float,
                think_mean: float, rng: random.Random) -> float:
    """When a given simulated user first sends a message.

    This is the only place traffic shape is expressed. Everything downstream -
    think time, conversation growth, request issuing - is identical across
    shapes, so a difference between two runs is attributable to arrival timing
    alone rather than to two different simulators.
    """
    if pattern == "herd":
        # Every client reconnecting at once, e.g. right after an outage.
        return 0.0
    if pattern == "ramp":
        # Linearly increasing population across the run: finds the knee in one
        # pass instead of one run per user count.
        return duration * user_index / max(1, n_users)
    if pattern == "burst":
        # A steady baseline, then a spike partway in. The spike is what
        # admission control has to survive, and the graph it produces is the
        # clearest evidence that degradation is bounded.
        baseline = max(1, n_users // 4)
        if user_index < baseline:
            return rng.uniform(0, think_mean)
        return duration * 0.35 + rng.uniform(0, 2.0)
    if pattern == "diurnal":
        # Slow sine over the run, so we can see whether the service RECOVERS
        # between peaks or stays degraded once pushed.
        u = user_index / max(1, n_users)
        return duration * (0.5 - 0.5 * math.cos(math.pi * u)) * 0.9
    # steady (and adversarial's normal users): spread arrivals over one think
    # interval so the population is in a stationary state almost immediately
    # rather than starting as a thundering herd by accident.
    return rng.uniform(0, think_mean)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


class ChatSim:
    def __init__(self, args):
        self.args = args
        self.records: list[TurnRecord] = []
        self.t0 = 0.0
        self.stop_at = 0.0
        self.model = ""

    # -- one user's whole session ------------------------------------------

    async def user_session(self, client: httpx.AsyncClient, user: int,
                           api_key: str, target_tokens: int, seed: int) -> None:
        rng = random.Random(seed)
        delay = start_delay(self.args.pattern, user, self.args.users,
                            self.args.duration, self.args.think_mean, rng)
        await asyncio.sleep(delay)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        topic = rng.choice(TOPICS)

        for turn in range(self.args.turns):
            if time.perf_counter() >= self.stop_at:
                return

            if turn == 0:
                text = f"I'm working on {topic}. Give me a practical overview of how it works."
            else:
                text = rng.choice(FOLLOW_UPS)

            # Pad the opening turn when this user is simulating a long
            # conversation, so the working set reaches its target size quickly
            # instead of needing 40 turns of real accumulation to get there.
            if turn == 0 and target_tokens > 600:
                pad_words = int((target_tokens - 200) * 0.75)
                text += "\n\nFor context, here are my notes:\n" + " ".join(
                    rng.choice(FILLER_WORDS) for _ in range(pad_words)
                )

            messages.append({"role": "user", "content": text})
            record = await self.one_turn(client, user, turn, api_key, messages)
            self.records.append(record)

            if record.status == 200 and record.completion_tokens:
                # The reply is appended so the NEXT turn resends it. This is
                # what makes the conversation accumulate, and it is the whole
                # reason this simulator exists.
                messages.append({"role": "assistant", "content": "x " * record.completion_tokens})
            else:
                messages.pop()  # a shed turn is not part of the conversation

            # Log-normal think time. Real users are not uniform: most reply
            # quickly, a few take a very long time, and that right tail is what
            # decides how many conversations sit idle in the cache being
            # evicted. A uniform distribution would understate eviction.
            think = rng.lognormvariate(
                math.log(max(0.5, self.args.think_mean)), self.args.think_sigma
            )
            think = min(think, self.args.think_max)
            if time.perf_counter() + think >= self.stop_at:
                return
            await asyncio.sleep(think)

    # -- one request --------------------------------------------------------

    async def one_turn(self, client: httpx.AsyncClient, user: int, turn: int,
                       api_key: str, messages: list) -> TurnRecord:
        started = time.perf_counter()
        conversation_tokens = sum(len(m["content"]) for m in messages) // 4

        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.args.max_tokens,
            "temperature": 0,
            "ignore_eos": True,  # fixed output volume, per the measurement protocol
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        ttft = None
        completion = 0
        prompt_tokens = 0
        queue_wait = 0
        trimmed = 0
        status = 0
        shed_reason = None
        deltas = 0
        last_token_at = None
        itl_samples: list[float] = []

        try:
            async with client.stream("POST", "/v1/chat/completions",
                                     json=body, headers=headers) as resp:
                status = resp.status_code
                queue_wait = int(resp.headers.get("X-Queue-Wait-Ms", 0) or 0)
                trimmed = int(resp.headers.get("X-Context-Trimmed-Messages", 0) or 0)
                shed_reason = resp.headers.get("X-Shed-Reason")

                if status != 200:
                    await resp.aread()
                    return TurnRecord(user, turn, status, None,
                                      (time.perf_counter() - started) * 1000, None,
                                      0, 0, queue_wait, trimmed, started,
                                      conversation_tokens, shed_reason)

                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n\n" in buf:
                        event, buf = buf.split("\n\n", 1)
                        for line in event.split("\n"):
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw or raw == "[DONE]":
                                continue
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                            if delta:
                                now = time.perf_counter()
                                if ttft is None:
                                    ttft = (now - started) * 1000
                                else:
                                    itl_samples.append((now - last_token_at) * 1000)
                                last_token_at = now
                                deltas += 1
                            if obj.get("usage"):
                                prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                                completion = obj["usage"].get("completion_tokens", 0)
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            status = status or 599
            shed_reason = shed_reason or type(exc).__name__

        return TurnRecord(
            user, turn, status, round(ttft, 1) if ttft else None,
            round((time.perf_counter() - started) * 1000, 1),
            round(statistics.median(itl_samples), 2) if itl_samples else None,
            prompt_tokens, completion or deltas, queue_wait, trimmed, started,
            conversation_tokens, shed_reason,
        )

    # -- orchestration ------------------------------------------------------

    async def run(self) -> dict:
        args = self.args
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

        async with httpx.AsyncClient(base_url=args.base_url, limits=limits,
                                     timeout=timeout) as client:
            self.model = await self.resolve_model(client)
            engine = await self.engine_config()
            before = await self.prefix_cache_counters()

            # Discard-the-first-request rule: one throwaway request absorbs the
            # cold-start cost, measured at 3,292 ms against 310 ms in v1.
            await self.one_turn(client, -1, 0, args.api_key,
                                [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": "Say OK."}])

            rng = random.Random(args.seed)
            # Conversation-length mix: most conversations are ordinary, a few
            # are very long. The long ones dominate cache pressure, so the mix
            # matters far more than the mean.
            mix = [int(x) for x in args.conversation_mix.split(",")]

            self.t0 = time.perf_counter()
            self.stop_at = self.t0 + args.duration

            tasks = []
            # --abusive-only turns this process into just the attacker, so the
            # victim's latency can be measured in a process the attacker cannot
            # starve. See the note below.
            if args.abusive_only:
                self.t0 = time.perf_counter()
                self.stop_at = self.t0 + args.duration
                await asyncio.gather(*[
                    self.abusive_session(client, 10_000 + a, args.abusive_key)
                    for a in range(args.abusive_concurrency)
                ], return_exceptions=True)
                after = await self.prefix_cache_counters()
                admission = await self.admission_snapshot(client)
                return self.summarise(engine, before, after, admission)

            for u in range(args.users):
                target = rng.choice(mix)
                key = (f"{args.user_key_prefix}{u % args.user_keys:02d}"
                       if args.user_keys > 0 else args.api_key)
                tasks.append(asyncio.create_task(
                    self.user_session(client, u, key, target, args.seed * 1000 + u)
                ))

            # The adversarial shape adds one key issuing continuous concurrent
            # requests alongside the normal population - the fairness case.
            #
            # RUN THE ABUSER IN A SEPARATE PROCESS (--abusive-only) WHENEVER THE
            # VICTIM'S LATENCY IS THE MEASUREMENT.
            #
            # Doing both in one event loop measured 10.8 s p95 for the normal
            # users while the gateway's own histogram showed every one of those
            # requests served in under 1.0 s. Fifty abusive coroutines issuing
            # 2,643 requests starved the ten normal coroutines of scheduler
            # time, and the simulator reported its own scheduling delay as
            # server latency - a ~10x error, in the direction that would have
            # made a working fairness mechanism look broken.
            if args.pattern == "adversarial" and not args.abusive_external:
                for a in range(args.abusive_concurrency):
                    tasks.append(asyncio.create_task(
                        self.abusive_session(client, 10_000 + a, args.abusive_key)
                    ))

            await asyncio.gather(*tasks, return_exceptions=True)
            after = await self.prefix_cache_counters()
            admission = await self.admission_snapshot(client)

        return self.summarise(engine, before, after, admission)

    async def abusive_session(self, client: httpx.AsyncClient, user: int, api_key: str) -> None:
        """One key hammering with no think time at all. Worst case #2."""
        turn = 0
        while time.perf_counter() < self.stop_at:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Explain {random.choice(TOPICS)} in detail. Request {turn}."},
            ]
            self.records.append(await self.one_turn(client, user, turn, api_key, messages))
            turn += 1

    async def resolve_model(self, client: httpx.AsyncClient) -> str:
        """Never hardcode what the server is serving - v1 lost a whole ramp to that."""
        if self.args.model:
            return self.args.model
        resp = await client.get("/v1/models",
                                headers={"Authorization": f"Bearer {self.args.api_key}"})
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

    async def engine_config(self) -> dict:
        """Embed the engine's own config in the results file.

        Protocol rule: a results file six weeks from now is uninterpretable
        without the flags, model and revision that produced it.
        """
        out = {}
        try:
            async with httpx.AsyncClient(base_url=self.args.engine_url, timeout=5.0) as c:
                r = await c.get("/v1/models")
                data = r.json()["data"][0]
                out = {"model": data.get("id"), "max_model_len": data.get("max_model_len")}
        except Exception as exc:
            out = {"error": str(exc)}
        return out

    async def prefix_cache_counters(self) -> dict:
        """Read vLLM's prefix cache counters straight from the engine.

        THE headline metric. This build does not populate per-request
        `cached_tokens` (verified in Finalization Phase 2.5), so the aggregate
        counters are the only source of hit rate - and taking a delta across
        the run is exactly what is needed anyway.
        """
        out = {"queries": 0.0, "hits": 0.0, "preemptions": 0.0}
        try:
            async with httpx.AsyncClient(base_url=self.args.engine_url, timeout=5.0) as c:
                text = (await c.get("/metrics")).text
            for line in text.splitlines():
                if line.startswith("vllm:prefix_cache_queries_total"):
                    out["queries"] = float(line.rsplit(" ", 1)[1])
                elif line.startswith("vllm:prefix_cache_hits_total"):
                    out["hits"] = float(line.rsplit(" ", 1)[1])
                elif line.startswith("vllm:num_preemptions_total"):
                    out["preemptions"] = float(line.rsplit(" ", 1)[1])
        except Exception:
            pass
        return out

    async def admission_snapshot(self, client: httpx.AsyncClient) -> dict:
        try:
            return (await client.get("/admission")).json()
        except Exception:
            return {}

    # -- reporting ----------------------------------------------------------

    def summarise(self, engine: dict, before: dict, after: dict, admission: dict) -> dict:
        args = self.args
        real = [r for r in self.records if r.user >= 0 and r.user < 10_000]
        abusive = [r for r in self.records if r.user >= 10_000]

        ok = [r for r in real if r.status == 200]
        shed = [r for r in real if r.status == 503]
        errs = [r for r in real if r.status not in (200, 503)]

        per_user: list[UserSummary] = []
        by_user: dict[int, list[TurnRecord]] = {}
        for r in real:
            by_user.setdefault(r.user, []).append(r)
        for uid, rs in sorted(by_user.items()):
            t = [r.ttft_ms for r in rs if r.ttft_ms is not None]
            per_user.append(UserSummary(
                user=uid, turns=len(rs),
                ok=sum(1 for r in rs if r.status == 200),
                shed=sum(1 for r in rs if r.status == 503),
                errors=sum(1 for r in rs if r.status not in (200, 503)),
                ttft_p50=pct(t, 50), ttft_p95=pct(t, 95),
                ttft_max=round(max(t), 1) if t else None,
                mean_conversation_tokens=round(
                    statistics.mean([r.prompt_tokens for r in rs if r.prompt_tokens]) , 1)
                if any(r.prompt_tokens for r in rs) else 0.0,
            ))

        all_ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        # The per-user view: take each user's own p95, then look at the
        # distribution of those. The worst user's p95 is the number that
        # decides whether anyone had a bad time, and an aggregate p95 hides it.
        user_p95s = [u.ttft_p95 for u in per_user if u.ttft_p95 is not None]

        dq = after.get("queries", 0) - before.get("queries", 0)
        dh = after.get("hits", 0) - before.get("hits", 0)
        hit_rate = round(100 * dh / dq, 2) if dq > 0 else None

        wall = max((r.started_at for r in real), default=0) - min(
            (r.started_at for r in real), default=0)
        total_out = sum(r.completion_tokens for r in ok)

        return {
            "config": {
                "users": args.users, "pattern": args.pattern, "turns": args.turns,
                "think_mean": args.think_mean, "think_sigma": args.think_sigma,
                "conversation_mix": args.conversation_mix, "duration": args.duration,
                "max_tokens": args.max_tokens, "seed": args.seed,
                "slo_ttft_ms": args.slo_ttft * 1000,
                "abusive_concurrency": args.abusive_concurrency if args.pattern == "adversarial" else 0,
            },
            "engine": engine,
            "admission": admission,
            "totals": {
                "turns_attempted": len(real),
                "turns_ok": len(ok),
                "turns_shed_503": len(shed),
                "turns_error": len(errs),
                "shed_rate_pct": round(100 * len(shed) / len(real), 2) if real else 0,
                "output_tokens": total_out,
                "output_tok_per_s": round(total_out / wall, 1) if wall > 0 else 0,
                "mean_prompt_tokens": round(
                    statistics.mean([r.prompt_tokens for r in ok]), 1) if ok else 0,
            },
            "ttft_ms_all_requests": {
                "p50": pct(all_ttft, 50), "p95": pct(all_ttft, 95),
                "p99": pct(all_ttft, 99), "max": round(max(all_ttft), 1) if all_ttft else None,
            },
            "ttft_ms_per_user_p95": {
                "median_user": pct(user_p95s, 50),
                "worst_user": round(max(user_p95s), 1) if user_p95s else None,
                "users_breaching_slo": sum(1 for v in user_p95s if v > args.slo_ttft * 1000),
                "users_total": len(user_p95s),
            },
            "prefix_cache": {
                "queries_delta": dq, "hits_delta": dh, "hit_rate_pct": hit_rate,
                "preemptions_delta": after.get("preemptions", 0) - before.get("preemptions", 0),
            },
            "abusive": {
                "requests": len(abusive),
                "ok": sum(1 for r in abusive if r.status == 200),
                "shed_503": sum(1 for r in abusive if r.status == 503),
            } if abusive else None,
            "per_user": [asdict(u) for u in per_user],
            "turns": [asdict(r) for r in self.records] if args.record_turns else [],
        }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--engine-url", default="http://localhost:8000",
                    help="vLLM directly, for prefix-cache counters the gateway cannot know.")
    ap.add_argument("--api-key", default="dev-key-alpha",
                    help="Used for model resolution and the warm-up request.")
    ap.add_argument("--user-keys", type=int, default=48,
                    help="Size of the per-user key pool. Each simulated user "
                         "presents its own key, because per-key admission limits "
                         "are keyed on the API key - a population sharing one key "
                         "is one tenant, correctly throttled to its share, and "
                         "measuring that instead of service capacity is a mistake "
                         "this simulator made on its first run. 0 = share --api-key.")
    ap.add_argument("--user-key-prefix", default="dev-user-")
    ap.add_argument("--abusive-key", default="dev-key-gamma",
                    help="Separate key for the adversarial pattern, so per-key limits apply.")
    ap.add_argument("--model", default="", help="Blank = resolve from /v1/models.")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--pattern", default="steady",
                    choices=["steady", "ramp", "burst", "herd", "diurnal", "adversarial"])
    ap.add_argument("--turns", type=int, default=8, help="Turns per user before departing.")
    ap.add_argument("--think-mean", type=float, default=20.0,
                    help="Median think time in seconds (log-normal).")
    ap.add_argument("--think-sigma", type=float, default=0.6)
    ap.add_argument("--think-max", type=float, default=90.0)
    ap.add_argument("--conversation-mix", default="300,300,300,800,800,2000",
                    help="Target opening-context sizes, sampled per user. Skewed on "
                         "purpose: a few large conversations dominate cache pressure.")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--slo-ttft", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--abusive-concurrency", type=int, default=50)
    ap.add_argument("--abusive-only", action="store_true",
                    help="Run ONLY the abusive load. Use in a second process "
                         "alongside a normal run, so the attacker cannot starve "
                         "the victim's coroutines and be mistaken for server "
                         "latency - that error was worth ~10x.")
    ap.add_argument("--abusive-external", action="store_true",
                    help="Adversarial pattern, but the abuser is another process.")
    ap.add_argument("--record-turns", action="store_true",
                    help="Include every per-turn record in the JSON (large).")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    sim = ChatSim(args)
    result = asyncio.run(sim.run())
    if args.label:
        result["config"]["label"] = args.label

    t = result["totals"]
    pc = result["prefix_cache"]
    pu = result["ttft_ms_per_user_p95"]
    print(f"\n{args.pattern} | {args.users} users | {args.duration:g}s | "
          f"think ~{args.think_mean:g}s | seed {args.seed}")
    print(f"  turns        {t['turns_ok']} ok, {t['turns_shed_503']} shed (503), "
          f"{t['turns_error']} error   [{t['shed_rate_pct']}% shed]")
    print(f"  throughput   {t['output_tok_per_s']} tok/s out, "
          f"mean prompt {t['mean_prompt_tokens']:.0f} tok")
    print(f"  TTFT all     p50 {result['ttft_ms_all_requests']['p50']}  "
          f"p95 {result['ttft_ms_all_requests']['p95']}  "
          f"p99 {result['ttft_ms_all_requests']['p99']} ms")
    print(f"  TTFT p95/user median {pu['median_user']} ms, worst {pu['worst_user']} ms, "
          f"{pu['users_breaching_slo']}/{pu['users_total']} users over SLO")
    print(f"  prefix cache {pc['hit_rate_pct']}% hit  "
          f"({pc['hits_delta']:.0f}/{pc['queries_delta']:.0f} tokens), "
          f"preemptions +{pc['preemptions_delta']:.0f}")
    if result.get("abusive"):
        a = result["abusive"]
        print(f"  abusive key  {a['ok']} ok, {a['shed_503']} shed of {a['requests']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
