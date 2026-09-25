"""
Failure injection — the worst cases, each with a stated expectation.

    python load_testing/failure_matrix.py --case all
    python load_testing/failure_matrix.py --case slow_client

Every case declares EXPECTED behaviour before it runs, and reports measured
behaviour next to it. A case that behaves badly is a finding to record, not a
test to make pass - the point is to know what happens, not to assert that
something convenient happens.

Cases needing container manipulation (engine crash, gateway restart) are driven
from the shell and documented in docs/FAILURE_MATRIX.md; this file covers
everything reachable over HTTP, plus the accounting checks that make the
container cases meaningful.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8080"
ENGINE = "http://localhost:8000"
KEY = "dev-key-alpha"
SMALL_KEY = "dev-key-beta"

RESULTS: list[dict] = []


def report(case: str, expected: str, measured: str, verdict: str, detail: str = "") -> None:
    RESULTS.append({"case": case, "expected": expected, "measured": measured,
                    "verdict": verdict, "detail": detail})
    mark = {"PASS": "ok  ", "FINDING": "note", "FAIL": "FAIL"}.get(verdict, "?   ")
    print(f"\n[{mark}] {case}")
    print(f"       expected: {expected}")
    print(f"       measured: {measured}")
    if detail:
        print(f"       {detail}")


async def model_id(client: httpx.AsyncClient) -> str:
    r = await client.get(f"{GATEWAY}/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    return r.json()["data"][0]["id"]


async def max_len(client: httpx.AsyncClient) -> int:
    r = await client.get(f"{GATEWAY}/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    return int(r.json()["data"][0]["max_model_len"])


async def stream_turn(client: httpx.AsyncClient, model: str, messages: list,
                      key: str = KEY, max_tokens: int = 64) -> dict:
    """One streamed turn, returning status, timing and usage."""
    t0 = time.perf_counter()
    ttft = None
    usage = None
    body = {"model": model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    async with client.stream("POST", f"{GATEWAY}/v1/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {key}"}) as r:
        if r.status_code != 200:
            await r.aread()
            return {"status": r.status_code, "ttft_ms": None, "usage": None,
                    "headers": dict(r.headers)}
        buf = ""
        async for chunk in r.aiter_text():
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            buf += chunk
            while "\n\n" in buf:
                event, buf = buf.split("\n\n", 1)
                for line in event.split("\n"):
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw and raw != "[DONE]":
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if obj.get("usage"):
                                usage = obj["usage"]
        return {"status": 200, "ttft_ms": round(ttft or 0, 1), "usage": usage,
                "headers": dict(r.headers)}


# --------------------------------------------------------------------------
# Case 3 — a single very large prompt must not monopolise the service
# --------------------------------------------------------------------------


async def case_long_prompt_isolation(client: httpx.AsyncClient) -> None:
    """EXPECTED: chunked prefill splits the giant prompt into 2048-token pieces
    that interleave with other sequences' decode, so concurrent short requests
    keep being served. Their p95 TTFT should rise but stay in the low seconds,
    not stall for the ~20 s the big prefill takes."""
    model = await model_id(client)

    # ~26k tokens: large, and safely inside the 32,768 window with room to reply.
    #
    # THE NONCE IS LOAD-BEARING, and its absence produced a false PASS.
    # The prompt used to be a fixed string, so the second run of this case hit
    # the prefix cache completely: TTFT fell from 13,159 ms to 370.9 ms and the
    # case reported PASS while testing nothing. vLLM's prefix hash is a chain
    # from block 0, so the nonce has to be at the FRONT to make the whole prompt
    # cold - putting it at the end would leave every preceding block cached.
    nonce = f"{time.time_ns()}"
    huge = nonce + " " + " ".join(
        ["system request buffer latency throughput pipeline"] * 3400)

    baseline = []
    for i in range(6):
        r = await stream_turn(client, model,
                              [{"role": "user", "content": f"Say hello. {nonce}{i}"}])
        if r["ttft_ms"]:
            baseline.append(r["ttft_ms"])

    async def big():
        return await stream_turn(client, model,
                                 [{"role": "user", "content": huge}], max_tokens=32)

    async def smalls():
        out = []
        await asyncio.sleep(0.3)  # let the big one start prefilling first
        for i in range(8):
            r = await stream_turn(client, model,
                                  [{"role": "user", "content": f"Say hello. {nonce}s{i}"}])
            out.append(r)
        return out

    big_task = asyncio.create_task(big())
    small_results = await smalls()
    big_result = await big_task

    during = [r["ttft_ms"] for r in small_results if r["status"] == 200 and r["ttft_ms"]]
    shed = sum(1 for r in small_results if r["status"] == 503)
    p95_base = round(statistics.quantiles(baseline, n=20)[-1], 1) if len(baseline) > 1 else baseline[0]
    p95_during = round(max(during), 1) if during else None

    verdict = "PASS" if during and max(during) < 10_000 else "FINDING"
    report(
        "3. Single ~26k-token prompt alongside normal traffic",
        "short requests keep flowing; their TTFT rises but stays in low seconds",
        f"big prompt {big_result['usage']['prompt_tokens'] if big_result.get('usage') else '?'} tok, "
        f"TTFT {big_result['ttft_ms']} ms; small requests p95 TTFT "
        f"{p95_base} ms alone -> {p95_during} ms during; {shed} shed",
        verdict,
        f"{len(during)}/8 concurrent short requests completed",
    )


# --------------------------------------------------------------------------
# Case 4 — a conversation that outgrows the context window
# --------------------------------------------------------------------------


async def case_context_overflow(client: httpx.AsyncClient) -> None:
    """EXPECTED: the gateway trims the oldest turns and reports how many in
    X-Context-Trimmed-Messages. A 200 with a trim header, never a 400."""
    model = await model_id(client)
    window = await max_len(client)

    # Build a conversation that clearly exceeds the window: ~40k tokens.
    chunk = " ".join(["buffer latency throughput scheduler allocator kernel"] * 400)
    messages = [{"role": "system", "content": "You are a concise assistant."}]
    for i in range(18):
        messages.append({"role": "user", "content": f"Note {i}: {chunk}"})
        messages.append({"role": "assistant", "content": f"Noted {i}."})
    messages.append({"role": "user", "content": "In one sentence, what was note 17 about?"})

    approx = sum(len(m["content"]) for m in messages) // 4
    r = await stream_turn(client, model, messages, max_tokens=48)
    trimmed = r["headers"].get("x-context-trimmed-messages")

    verdict = "PASS" if r["status"] == 200 and trimmed else (
        "FAIL" if r["status"] != 200 else "FINDING")
    report(
        "4. Conversation outgrows the context window",
        f"200 with oldest turns dropped and reported; never a hard 400 (window {window:,})",
        f"HTTP {r['status']}, X-Context-Trimmed-Messages={trimmed}, "
        f"prompt sent ~{approx:,} tok -> engine saw "
        f"{r['usage']['prompt_tokens'] if r.get('usage') else '?'} tok",
        verdict,
        "system message and the current question are preserved by policy",
    )


# --------------------------------------------------------------------------
# Case 7 — client disconnects mid-stream
# --------------------------------------------------------------------------


async def case_client_disconnect(client: httpx.AsyncClient) -> None:
    """EXPECTED: the slot is released promptly and the request is still
    recorded in the ledger. Boundary 4 says accounting must survive the client
    vanishing, and a leaked slot would degrade the service permanently."""
    model = await model_id(client)
    before = await client.get(f"{GATEWAY}/admission")
    before_j = before.json()

    body = {"model": model, "messages": [{"role": "user", "content": "Count slowly to fifty."}],
            "stream": True, "max_tokens": 400, "temperature": 0, "ignore_eos": True}

    got = 0
    async with client.stream("POST", f"{GATEWAY}/v1/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {KEY}"}) as r:
        async for _chunk in r.aiter_bytes():
            got += 1
            if got >= 3:
                break  # abandon the stream mid-flight

    await asyncio.sleep(2.0)
    after_j = (await client.get(f"{GATEWAY}/admission")).json()

    verdict = "PASS" if after_j["inflight"] == 0 else "FAIL"
    report(
        "7. Client disconnects mid-stream",
        "slot released within seconds, in-flight returns to 0, usage still recorded",
        f"in-flight {before_j['inflight']} -> {after_j['inflight']} after abandoning "
        f"the stream at chunk {got}; admitted_total {after_j['admitted_total']}",
        verdict,
        "accounting is written from the generator's finally block",
    )


# --------------------------------------------------------------------------
# Case 8 — a slow client must not hold a slot forever
# --------------------------------------------------------------------------


async def case_slow_client(client: httpx.AsyncClient) -> None:
    """EXPECTED: a client reading very slowly applies backpressure and holds
    its slot, but is bounded by GATEWAY_STREAM_TIMEOUT_S. The important part is
    that it holds ONE slot, not all of them - other users keep being served."""
    model = await model_id(client)
    body = {"model": model, "messages": [{"role": "user", "content": "Write a long essay."}],
            "stream": True, "max_tokens": 512, "temperature": 0, "ignore_eos": True}

    slow_done = asyncio.Event()

    async def slow_reader():
        try:
            async with client.stream("POST", f"{GATEWAY}/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {KEY}"}) as r:
                async for _ in r.aiter_bytes():
                    await asyncio.sleep(1.0)  # ~1 chunk/second
                    if slow_done.is_set():
                        break
        except Exception:
            pass

    task = asyncio.create_task(slow_reader())
    await asyncio.sleep(3.0)

    ok, ttfts = 0, []
    for _ in range(5):
        r = await stream_turn(client, model, [{"role": "user", "content": "Say hello."}])
        if r["status"] == 200:
            ok += 1
            if r["ttft_ms"]:
                ttfts.append(r["ttft_ms"])

    snap = (await client.get(f"{GATEWAY}/admission")).json()
    slow_done.set()
    task.cancel()
    await asyncio.sleep(0.5)

    verdict = "PASS" if ok >= 4 else "FINDING"
    report(
        "8. Slow client reading ~1 chunk/second",
        "holds at most one slot; other users unaffected; bounded by stream timeout",
        f"{ok}/5 concurrent normal requests succeeded while the slow reader ran; "
        f"their TTFT p50 {round(statistics.median(ttfts), 1) if ttfts else '-'} ms; "
        f"in-flight during test {snap['inflight']}",
        verdict,
        f"stream timeout caps a stuck stream; per-key limit caps how many one client can hold",
    )


# --------------------------------------------------------------------------
# Case 11 — budget exhausted mid-conversation
# --------------------------------------------------------------------------


async def case_budget_exhaustion(client: httpx.AsyncClient) -> None:
    """EXPECTED: a clean 429 in the OpenAI error envelope with budget headers,
    and the conversation is resumable - the failure is per-request, and no
    server-side state is corrupted by it."""
    model = await model_id(client)
    messages = [{"role": "user", "content": "Hello, tell me about databases."}]

    statuses = []
    for _ in range(12):
        r = await stream_turn(client, model, messages, key=SMALL_KEY, max_tokens=128)
        statuses.append(r["status"])
        if r["status"] == 429:
            hdrs = r["headers"]
            # Resumability: the same conversation on a healthy key must work.
            resumed = await stream_turn(client, model, messages, key=KEY, max_tokens=32)
            report(
                "11. Budget exhausted mid-conversation",
                "clean 429 with budget headers; conversation resumable on a funded key",
                f"429 after {len(statuses) - 1} served requests; "
                f"X-Budget-Remaining={hdrs.get('x-budget-remaining')}; "
                f"same conversation on a funded key -> HTTP {resumed['status']}",
                "PASS" if resumed["status"] == 200 else "FAIL",
                "429 costs ~8 ms against ~5,000 ms to serve - refusing is ~600x cheaper",
            )
            return
    report("11. Budget exhausted mid-conversation",
           "clean 429 with budget headers",
           f"no 429 in 12 requests (statuses {set(statuses)}) - budget not exhausted",
           "FINDING", "run gateway/seed_db.py --reset then retry")


# --------------------------------------------------------------------------
# Case 12 — the billing database is unavailable
# --------------------------------------------------------------------------


async def case_db_unavailable(client: httpx.AsyncClient) -> None:
    """EXPECTED: FAIL CLOSED. If the ledger cannot be read, the gateway must
    refuse to serve GPU work it cannot bill, with a 503 and Retry-After -
    never serve it for free."""
    model = await model_id(client)
    import subprocess

    def _compose(*args: str) -> None:
        subprocess.run(
            ["docker", "compose",
             "-f", "deployment/docker/docker-compose.yml",
             "-f", "deployment/docker/docker-compose.1_5b-awq.yml", *args],
            capture_output=True, text=True, check=False,
        )

    # HOW THIS BREAKS WRITES, and why the obvious approach does not.
    #
    # The first version held an EXCLUSIVE sqlite lock and expected the gateway
    # to fail. It did not: the database runs in WAL mode, where readers are
    # never blocked by a writer, so every request returned 200 and the test
    # reported a false pass on a service that was in fact serving unbilled
    # work. Making the FILE unwritable is the honest simulation of the real
    # conditions this guards against - a full disk or a permissions change.
    _compose("exec", "-T", "-u", "0", "gateway", "sh", "-c",
             "chmod 444 /data/usage.db; chmod 444 /data/usage.db-wal; "
             "chown root:root /data/usage.db /data/usage.db-wal")
    await asyncio.sleep(1.0)
    # Several requests, not one. In WAL mode the budget READ keeps working
    # while the WRITE fails, so the breaker can only trip after a run of failed
    # writes - and the first few requests are necessarily served unbilled,
    # because you learn the write failed only after doing the work.
    statuses = []
    try:
        for _ in range(6):
            r = await stream_turn(client, model,
                                  [{"role": "user", "content": "Hi."}], max_tokens=16)
            statuses.append(r["status"])
    except Exception as exc:
        statuses.append(f"exception {type(exc).__name__}")
    finally:
        # ALWAYS restore, on every path. A harness that can leave the product's
        # database read-only after a crash is a worse problem than the one it
        # is testing.
        _compose("exec", "-T", "-u", "0", "gateway", "sh", "-c",
                 "chown 10001:10001 /data/usage.db /data/usage.db-wal; "
                 "chmod 644 /data/usage.db; chmod 644 /data/usage.db-wal")

    during_cooldown = await stream_turn(client, model,
                                        [{"role": "user", "content": "Hi."}], max_tokens=16)
    # The breaker is half-open, not latched: it waits out a cooldown, then lets
    # one probe through. Version one reset only on a successful write and so
    # could never recover - restoring the database left the service returning
    # 503 forever. This sleep is testing that fix.
    await asyncio.sleep(11.0)
    recovered = await stream_turn(client, model,
                                  [{"role": "user", "content": "Hi."}], max_tokens=16)

    tripped = 503 in statuses
    unbilled = sum(1 for s in statuses if s == 200)
    verdict = "PASS" if tripped and recovered["status"] == 200 else "FINDING"
    report(
        "12. Billing ledger unwritable (disk full / permissions)",
        "bounded unbilled requests, then 503 (fail CLOSED), then automatic recovery",
        f"{statuses} while unwritable -> {unbilled} served unbilled before the breaker "
        f"tripped; HTTP {during_cooldown['status']} during cooldown; "
        f"HTTP {recovered['status']} after cooldown",
        verdict,
        "WAL mode means readers are never blocked by a writer, so the budget CHECK "
        "keeps succeeding - only a run of failed WRITES reveals the outage",
    )


# --------------------------------------------------------------------------
# Case 10 — KV cache pressure and exact accounting
# --------------------------------------------------------------------------


async def case_kv_pressure(client: httpx.AsyncClient) -> None:
    """EXPECTED: with many long concurrent sequences the engine may preempt.
    Whatever it does, the gateway's token accounting must be exact - Boundary 4."""
    model = await model_id(client)

    def db_totals() -> tuple[int, int]:
        import subprocess
        out = subprocess.run(
            ["docker", "compose",
             "-f", "deployment/docker/docker-compose.yml",
             "-f", "deployment/docker/docker-compose.1_5b-awq.yml",
             "exec", "-T", "gateway", "python", "-c",
             "import sqlite3;c=sqlite3.connect('/data/usage.db');"
             "print(*c.execute('select count(*),coalesce(sum(completion_tokens),0) "
             "from requests').fetchone())"],
            capture_output=True, text=True,
        )
        a, b = out.stdout.strip().split()
        return int(a), int(b)

    before_n, before_tok = db_totals()
    engine_before = (await client.get(f"{ENGINE}/metrics")).text
    pre_before = next((float(l.rsplit(" ", 1)[1]) for l in engine_before.splitlines()
                       if l.startswith("vllm:num_preemptions_total")), 0.0)

    n, out_tokens = 12, 700
    long_prompt = " ".join(["pipeline scheduler allocator kernel buffer"] * 300)
    tasks = [stream_turn(client, model, [{"role": "user", "content": f"{long_prompt} #{i}"}],
                         max_tokens=out_tokens) for i in range(n)]
    results = await asyncio.gather(*tasks)
    await asyncio.sleep(2.0)

    after_n, after_tok = db_totals()
    engine_after = (await client.get(f"{ENGINE}/metrics")).text
    pre_after = next((float(l.rsplit(" ", 1)[1]) for l in engine_after.splitlines()
                      if l.startswith("vllm:num_preemptions_total")), 0.0)

    ok = [r for r in results if r["status"] == 200]
    shed = [r for r in results if r["status"] == 503]
    expected_tokens = len(ok) * out_tokens
    delta_tokens = after_tok - before_tok

    verdict = "PASS" if delta_tokens == expected_tokens else "FINDING"
    report(
        "10. KV pressure / preemption with exact accounting",
        f"every served request bills exactly {out_tokens} completion tokens; zero drift",
        f"{len(ok)} served, {len(shed)} shed; ledger +{after_n - before_n} rows, "
        f"+{delta_tokens} completion tokens (expected {expected_tokens}); "
        f"engine preemptions +{pre_after - pre_before:.0f}",
        verdict,
        "ignore_eos makes output length deterministic, so any drift is real drift",
    )


# --------------------------------------------------------------------------
# Case 1/2 helpers — overload and fairness are covered by chat_sim patterns
# --------------------------------------------------------------------------


async def case_overload_shedding(client: httpx.AsyncClient) -> None:
    """EXPECTED: at ~10x the admission limit, p95 TTFT stays BOUNDED and the
    excess is refused with 503 + Retry-After rather than queued."""
    model = await model_id(client)
    n = 60
    tasks = [stream_turn(client, model,
                         [{"role": "user", "content": f"Explain indexing. #{i}"}],
                         key=f"dev-user-{i % 48:02d}", max_tokens=64)
             for i in range(n)]
    results = await asyncio.gather(*tasks)
    ok = [r for r in results if r["status"] == 200]
    shed = [r for r in results if r["status"] == 503]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
    retry_after = next((r["headers"].get("retry-after") for r in shed), None)
    p95 = round(sorted(ttfts)[max(0, int(0.95 * len(ttfts)) - 1)], 1) if ttfts else None

    verdict = "PASS" if shed and p95 and p95 < 5000 else "FINDING"
    report(
        "1. Burst 10x over admission limit (60 simultaneous)",
        "p95 TTFT bounded; excess refused with 503 + Retry-After, not queued",
        f"{len(ok)} served, {len(shed)} shed (Retry-After: {retry_after}); "
        f"served p95 TTFT {p95} ms",
        verdict,
        f"admission limit is 6 in flight, so this is 10x offered load",
    )


CASES = {
    "overload": case_overload_shedding,
    "long_prompt": case_long_prompt_isolation,
    "context_overflow": case_context_overflow,
    "disconnect": case_client_disconnect,
    "slow_client": case_slow_client,
    "budget": case_budget_exhaustion,
    "db_locked": case_db_unavailable,
    "kv_pressure": case_kv_pressure,
}


async def main_async(which: list[str], out: str) -> int:
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for name in which:
            try:
                await CASES[name](client)
            except Exception as exc:
                report(name, "case runs to completion",
                       f"harness error: {type(exc).__name__}: {exc}", "FAIL")
            await asyncio.sleep(3.0)  # let the engine settle between cases
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
        print(f"\n-> {out}")
    bad = sum(1 for r in RESULTS if r["verdict"] == "FAIL")
    print(f"\n{len(RESULTS)} cases: "
          f"{sum(1 for r in RESULTS if r['verdict'] == 'PASS')} pass, "
          f"{sum(1 for r in RESULTS if r['verdict'] == 'FINDING')} findings, {bad} fail")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--out", default="load_testing/results/failure_matrix.json")
    args = ap.parse_args()
    which = list(CASES) if args.case == "all" else args.case.split(",")
    return asyncio.run(main_async(which, args.out))


if __name__ == "__main__":
    sys.exit(main())
