"""
Worst cases 5 and 6: killing the engine, and restarting the gateway, under load.

    python load_testing/kill_test.py --case engine
    python load_testing/kill_test.py --case gateway

Both are in the definition of done and both need container manipulation, so
they live here rather than in failure_matrix.py.

WHAT CASE 5 IS ACTUALLY CHECKING, and it is not "does it come back"
--------------------------------------------------------------------
Three separate things, and the third is the one v1 got wrong first:

  1. In-flight requests fail with 502, not a hang or a silent truncation.
  2. The engine is restarted automatically and recovery time is measured.
  3. The GATEWAY stays alive - Running, /health 200 - while reporting itself
     NOT READY on /ready.

That third point is the whole liveness/readiness distinction. A failing
liveness probe means "restart this container", and restarting a healthy gateway
because its upstream died helps nothing and drops every request that was about
to succeed. Stage 10(c) had /health return 503 when vLLM was killed, which made
Docker mark the gateway unhealthy - the gateway was fine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time

import httpx

GATEWAY = "http://localhost:8080"
KEY = "dev-key-alpha"
COMPOSE = ["docker", "compose",
           "-f", "deployment/docker/docker-compose.yml",
           "-f", "deployment/docker/docker-compose.1_5b-awq.yml"]


def compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*COMPOSE, *args], capture_output=True, text=True)


def ledger_totals() -> tuple[int, int]:
    out = compose("exec", "-T", "gateway", "python", "-c",
                  "import sqlite3;c=sqlite3.connect('/data/usage.db');"
                  "print(*c.execute('select count(*),coalesce(sum(total_tokens),0) "
                  "from requests').fetchone())")
    a, b = out.stdout.strip().split()
    return int(a), int(b)


async def probe(client: httpx.AsyncClient, path: str) -> int:
    try:
        return (await client.get(f"{GATEWAY}{path}", timeout=5.0)).status_code
    except httpx.HTTPError:
        return 0


async def background_load(client: httpx.AsyncClient, model: str, stop: asyncio.Event,
                          results: list) -> None:
    """Continuous light traffic, so the kill lands on real in-flight work."""
    i = 0
    while not stop.is_set():
        i += 1
        body = {"model": model, "messages": [{"role": "user", "content": f"Count to ten. #{i}"}],
                "stream": True, "max_tokens": 96, "temperature": 0, "ignore_eos": True}
        try:
            async with client.stream("POST", f"{GATEWAY}/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {KEY}"}) as r:
                await r.aread()
                results.append(r.status_code)
        except httpx.HTTPError as exc:
            results.append(type(exc).__name__)
        await asyncio.sleep(0.4)


async def case_engine(args_kill_mode: str = "enginecore") -> None:
    timeout = httpx.Timeout(connect=5.0, read=None, write=15.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        model = (await client.get(f"{GATEWAY}/v1/models",
                                  headers={"Authorization": f"Bearer {KEY}"})).json()["data"][0]["id"]
        before_rows, before_tokens = ledger_totals()

        stop = asyncio.Event()
        results: list = []
        task = asyncio.create_task(background_load(client, model, stop, results))
        await asyncio.sleep(4.0)

        # HOW THE CRASH IS SIMULATED, and two ways that do not work.
        #
        #   docker compose kill vllm  - the container exits 137, but
        #     `restart: unless-stopped` does NOT restart it: Docker treats an
        #     operator kill as intentional. Measured: restarts=0. Worth knowing,
        #     because it is easy to assume that policy covers this.
        #   docker exec vllm kill -9 1  - ignored. The kernel does not deliver
        #     signals to a PID namespace's init from inside that namespace.
        #
        # Killing the EngineCore worker is the realistic crash: the process
        # doing the GPU work dies, the API server notices, and the container
        # exits on its own - which the restart policy does cover.
        print(f"crashing engine ({args_kill_mode})...")
        t_kill = time.perf_counter()
        if args_kill_mode == "enginecore":
            compose("exec", "-T", "vllm", "bash", "-lc",
                    "pkill -9 -f EngineCore || pkill -9 -f VLLM::EngineCore || true")
        else:
            compose("kill", "vllm")

        # WAIT FOR THE CRASH TO LAND BEFORE TIMING RECOVERY.
        # The first version measured "recovery" at 0.8 s, because it probed
        # /ready before the API server had noticed its EngineCore was gone -
        # so it timed the gap between issuing a kill and the kill taking
        # effect, and reported it as a recovery time. Confirm the outage
        # exists before starting the clock for it ending.
        crash_confirmed = None
        while time.perf_counter() - t_kill < 60:
            if (await probe(client, "/ready")) != 200:
                crash_confirmed = time.perf_counter() - t_kill
                break
            await asyncio.sleep(0.5)

        # Watch the gateway's own state while the engine is gone. This is the
        # measurement that matters: liveness must stay green.
        health_codes, ready_codes = set(), set()
        recovered_at = None
        while time.perf_counter() - t_kill < 300:
            health_codes.add(await probe(client, "/health"))
            ready_codes.add(await probe(client, "/ready"))
            if (await probe(client, "/ready")) == 200:
                recovered_at = time.perf_counter() - t_kill
                break
            await asyncio.sleep(2.0)

        stop.set()
        await task
        await asyncio.sleep(3.0)
        after_rows, after_tokens = ledger_totals()

        gw = compose("ps", "--format", "{{.Service}} {{.Status}}").stdout.strip()
        codes = [c for c in results]
        print(json.dumps({
            "crash_detected_after_seconds": round(crash_confirmed, 1) if crash_confirmed else None,
            "recovery_seconds": round(recovered_at, 1) if recovered_at else None,
            "container_restarts": compose(
                "ps", "-q", "vllm").stdout.strip() and subprocess.run(
                ["docker", "inspect", "--format", "{{.RestartCount}}",
                 compose("ps", "-q", "vllm").stdout.strip()],
                capture_output=True, text=True).stdout.strip(),
            "gateway_health_codes_during_outage": sorted(health_codes),
            "gateway_ready_codes_during_outage": sorted(ready_codes),
            "request_outcomes": {str(k): codes.count(k) for k in set(codes)},
            "ledger_rows_added": after_rows - before_rows,
            "ledger_tokens_added": after_tokens - before_tokens,
            "requests_issued": len(codes),
            "accounting_drift": len(codes) - (after_rows - before_rows),
            "compose_ps": gw,
        }, indent=2))


async def case_gateway() -> None:
    timeout = httpx.Timeout(connect=5.0, read=None, write=15.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        model = (await client.get(f"{GATEWAY}/v1/models",
                                  headers={"Authorization": f"Bearer {KEY}"})).json()["data"][0]["id"]
        before_rows, before_tokens = ledger_totals()

        stop = asyncio.Event()
        results: list = []
        task = asyncio.create_task(background_load(client, model, stop, results))
        await asyncio.sleep(4.0)

        print("restarting gateway...")
        t0 = time.perf_counter()
        compose("restart", "gateway")
        back = None
        while time.perf_counter() - t0 < 120:
            if (await probe(client, "/ready")) == 200:
                back = time.perf_counter() - t0
                break
            await asyncio.sleep(1.0)

        stop.set()
        await task
        await asyncio.sleep(2.0)
        after_rows, after_tokens = ledger_totals()
        codes = [c for c in results]
        print(json.dumps({
            "gateway_back_after_seconds": round(back, 1) if back else None,
            "request_outcomes": {str(k): codes.count(k) for k in set(codes)},
            "ledger_rows_before": before_rows,
            "ledger_rows_after": after_rows,
            "ledger_tokens_before": before_tokens,
            "ledger_tokens_after": after_tokens,
            "budgets_survived_restart": after_tokens >= before_tokens,
        }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["engine", "gateway"], required=True)
    ap.add_argument("--kill-mode", choices=["enginecore", "container"], default="enginecore",
                    help="enginecore = realistic crash (restart policy applies); "
                         "container = operator kill (it does not)")
    args = ap.parse_args()
    asyncio.run(case_engine(args.kill_mode) if args.case == "engine" else case_gateway())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
