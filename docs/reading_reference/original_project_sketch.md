# Capstone Project — A Self-Hosted Inference Product on Your RTX 3050

*Single GPU, start to finish: best-in-class inference, wrapped as a real product, fully observed, deliberately broken and fixed.*

---

## 0. Why this project, and what it actually proves

Every other project idea in this space is "run a model and hit it with curl." That proves you can follow a README. This project proves something a hiring manager actually cares about: that you understand the difference between *a model running* and *a system serving*, because you built both halves and instrumented the seam between them.

Concretely, by the end you will have: a properly quantized, properly configured vLLM server; a real API gateway in front of it with auth, rate limiting, and contract testing; a Prometheus/Grafana observability stack reading the engine's *actual* metrics, not a lossy subset; a load-testing suite that finds your GPU's real breaking point; three deliberately-induced failures that you diagnosed and fixed the way the self-hosting article described; and a written benchmark report with your own numbers, not someone else's. That's a portfolio piece and an interview story at the same time.

**Scope discipline, stated up front:** this project stays on one GPU the entire time. No tensor parallelism, no multi-node anything — those solve problems you don't have yet, and manufacturing them on hardware that can't represent them honestly would teach you the wrong lesson. "Scaling" here means finding and extending the ceiling of *one* card through configuration, quantization, and caching — which is the actual skill, and the one that transfers.

---

## 1. Architecture — the whole system, named

```
                    ┌─────────────────────┐
   Load test /      │   Gateway (FastAPI)  │   Auth, rate limit,
   Chat UI  ───────►│   :8080              │   request logging,
                    └──────────┬───────────┘   contract testing
                               │ forwards
                    ┌──────────▼───────────┐
                    │   vLLM (OpenAI API)   │   PagedAttention,
                    │   :8000               │   continuous batching,
                    └──────────┬───────────┘   prefix caching
                               │ /metrics
              ┌────────────────┼────────────────┐
              ▼                                  ▼
     ┌─────────────────┐               ┌──────────────────┐
     │   Prometheus     │◄──────────────│  GPU exporter     │
     │   :9090          │   scrapes     │  (nvidia-smi)     │
     └────────┬─────────┘               └──────────────────┘
              │
              ▼
     ┌─────────────────┐
     │   Grafana         │   dashboards: KV cache %, TTFT/ITL,
     │   :3000           │   throughput, GPU util/temp/mem
     └─────────────────┘
```

Every box above is something you already know how to run: Docker (Module 1/3), a container on your GPU (Module 3), a serving engine (Module 7), and — if you want the Kubernetes version — a Deployment + Service on your GPU-enabled Minikube cluster (Module 4/5).

---

## Stage 1 — Pick the model and quantization, matched to your card

**Why:** Section 1 of the deployment pipeline: memory capacity first, always. Get this wrong and every later stage inherits the mistake.

**What to do:** Your RTX 3050 has 4096MiB. Use Qwen2.5-3B-Instruct-AWQ as your primary model — Qwen ships official AWQ checkpoints day-one (a "safe to bet on" signal from the quantization article), and AWQ is specifically the format vLLM's Marlin kernel is fastest with. At AWQ 4-bit, weights are roughly 2GB, leaving real room for KV cache. Keep `Qwen2.5-1.5B-Instruct-AWQ` on hand as a fallback if you want more KV cache headroom for concurrency testing later.

```bash
# Quick capacity sanity check before committing:
# 3B params × 0.5GB/B (4-bit rule of thumb) ≈ 1.6-2GB weights
# leaves ~2GB+ for KV cache, CUDA context, and framework overhead on a 4GB card
```

**Metric that matters here:** none yet — this is a planning stage. The check is arithmetic, not measurement.

**You'll know this stage is done when:** you can state, in one sentence, why you picked this model+format+kernel combination and not another one. If you can't, go back to the cheat sheet's Section 4–5.

---

## Stage 2 — Stand up the inference engine, configured deliberately

**Why:** this is Section 4 of the pipeline — the knobs that determine whether your 4GB card serves gracefully or falls over under the first real load.

**What to do:**
```bash
docker run --gpus all \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct-AWQ \
  --quantization awq_marlin \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --enable-prefix-caching
```

- `--quantization awq_marlin` explicitly requests the Marlin kernel path — don't leave this to auto-detection; verify it in the startup logs.
- `--max-model-len 4096` is a deliberate KV cache budget, not a default left untouched — this is your worst-case-per-request memory commitment.
- `--max-num-seqs 16` caps how many requests batch together; you'll tune this empirically in Stage 7, this is just a sane starting point.
- `--enable-prefix-caching` turns on the block-reuse mechanism your gateway's shared system prompt will benefit from in Stage 9.

**Metric that matters here:** confirm in the startup logs that Marlin was actually selected, and that the reported number of KV cache blocks is nonzero and reasonable for your VRAM. If the log shows a fallback kernel, something above is misconfigured — fix it before moving on, since every later benchmark number is meaningless against the wrong kernel.

---

## Stage 3 — Establish your honest baseline (this is "doing inference best")

**Why:** you cannot improve what you haven't measured, and a single-request "it feels fast" is exactly the benchmarking mistake called out repeatedly in the reading.

**What to do:** use vLLM's actual bench tools, not a hand-rolled script this time:
```bash
# Latency: small batch, realistic single-turn shape
vllm bench latency --model Qwen/Qwen2.5-3B-Instruct-AWQ \
  --input-len 128 --output-len 128 --batch-size 8

# Throughput: saturate the engine, find its ceiling
vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct-AWQ \
  --input-len 128 --output-len 128 --num-prompts 200
```
Now find your own roofline crossover (`B_sat`) directly: repeat the throughput run at batch sizes 1, 2, 4, 8, 16, 32, plotting tokens/sec and per-token latency at each. There's a point where latency starts climbing noticeably faster than throughput climbs — that's your card's real bandwidth-to-compute crossover, not a number from an article about an H100.

**Metric that matters here:** TTFT, ITL, throughput (tok/s), and — critically — **your own measured B_sat**. Write these four numbers down. This is your baseline; every later stage gets compared against it.

**You'll know this stage is done when:** you have a small table or chart, in your own numbers, showing throughput and ITL across at least 5 batch sizes.

---

## Stage 4 — Build the product layer (the gateway)

**Why:** a raw inference server isn't a product. A product has identity, limits, and a contract it actually keeps — which is exactly the gap Boundary 1 from the self-hosting article warns about.

**What to build**, a small FastAPI service in front of vLLM:
```python
from fastapi import FastAPI, Header, HTTPException
import httpx, time, json

app = FastAPI()
VLLM_URL = "http://localhost:8000/v1/chat/completions"

# --- toy in-memory key/budget store; swap for SQLite/Redis for anything real ---
API_KEYS = {"key-abc123": {"budget": 100_000, "used": 0}}

@app.post("/v1/chat/completions")
async def chat(payload: dict, authorization: str = Header(...)):
    key = authorization.replace("Bearer ", "")
    if key not in API_KEYS:
        raise HTTPException(401, "invalid key")
    if API_KEYS[key]["used"] >= API_KEYS[key]["budget"]:
        raise HTTPException(429, "budget exhausted")

    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(VLLM_URL, json=payload)
    latency = time.time() - t0

    body = r.json()
    tokens_used = body.get("usage", {}).get("total_tokens", 0)
    API_KEYS[key]["used"] += tokens_used

    # log every request — this is your gateway-level observability
    print(json.dumps({"key": key, "latency_s": latency, "tokens": tokens_used}))
    return body
```
This is a deliberately small version of the JD's "Token Factory" pattern from your very first course document — tenant identity, budget enforcement, and request logging, at toy scale.

**The contract test you must write (Boundary 1, made concrete):** send a request with `response_format={"type": "json_object"}` and a prompt that gives the model no natural reason to produce JSON. **Parse the response body and assert it's valid JSON.** Do not just check for HTTP 200 — that's precisely the check the self-hosting article shows failing silently.

```python
import httpx, json

def test_structured_output_contract():
    r = httpx.post("http://localhost:8080/v1/chat/completions",
        headers={"Authorization": "Bearer key-abc123"},
        json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
              "messages": [{"role": "user", "content": "Tell me about your day."}],
              "response_format": {"type": "json_object"}})
    content = r.json()["choices"][0]["message"]["content"]
    json.loads(content)  # raises if it's prose instead of JSON — this is the actual test
```

**Metric that matters here:** gateway-added latency overhead (compare end-to-end latency through the gateway vs. hitting vLLM directly — it should be small, single-digit milliseconds) and the contract test's pass/fail, run on every change you make from here on.

---

## Stage 5 — Containerize and deploy properly

**Why:** this is Modules 1/3/4/5 paying off directly — a product should be reproducible, not a script someone remembers how to run.

**What to do:** Dockerfile the gateway (reuse the Module 1 pattern — dependencies before code, layer caching intact), then either:
- **Docker Compose** for the simplest path: gateway + vLLM + Prometheus + Grafana as one `docker compose up`, or
- **Kubernetes (your GPU-enabled Minikube)** for the version that matches production patterns: a Deployment for the gateway, a Deployment for vLLM requesting `nvidia.com/gpu: 1`, and Services connecting them.

If you go the Kubernetes route, **repeat your Module 4 Lab C exercise for real**: `kubectl delete pod` on the gateway mid-load-test and confirm the Deployment replaces it while vLLM keeps serving — a genuine small-scale self-healing demonstration, not a toy example anymore.

**Metric that matters here:** time-to-recovery after a killed pod (how long until a replacement is `Ready` and serving again) — write this number down, it's a real reliability metric.

---

## Stage 6 — Observability: read the engine's real metrics, not a subset

**Why:** Boundary 3, directly. The whole point is not repeating the mistake of a monitoring bridge that drops KV-cache-% and prefix-hit-rate — the two numbers that would have warned you before something broke.

**What to do:** scrape vLLM's `/metrics` endpoint directly with Prometheus — don't route metrics through the gateway, go straight to the source:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["localhost:8000"]
  - job_name: gpu
    static_configs:
      - targets: ["localhost:9835"]   # nvidia-smi exporter port
```
Build (or import) a Grafana dashboard with these panels, non-negotiably including the two the reading flagged as commonly missing:

| Panel | vLLM metric | Why it's on the dashboard |
|---|---|---|
| **KV cache usage %** | `vllm:gpu_cache_usage_perc` | The single most important number — alert above ~90% sustained |
| Prefix cache hit rate | `vllm:gpu_prefix_cache_hit_rate` | Tells you if Stage 9's prefix caching test is actually working |
| Requests running/waiting | `vllm:num_requests_running` / `_waiting` | Queue depth — your real-time load signal |
| TTFT (p50/p95/p99) | `vllm:time_to_first_token_seconds` | User-facing latency, the metric from Module 6 finally on a real dashboard |
| ITL (p50/p95/p99) | `vllm:time_per_output_token_seconds` | Same |
| Throughput | `vllm:generation_tokens_total` (rate) | Tokens/sec, live |
| GPU utilization / temp / VRAM used | from the GPU exporter | Hardware health, ties straight back to Module 2/3 |

**Metric that matters here:** the dashboard itself, functioning, with real numbers moving as you send traffic. Screenshot it — this becomes documentation later.

---

## Stage 7 — Load testing and finding your GPU's real ceiling ("scaling," honestly defined)

**Why:** "scaling" on one GPU means finding how far configuration and concurrency can go before quality of service breaks — not adding hardware.

**What to do:** write (or use Locust for) a script that ramps concurrent users — 1, 5, 10, 20, 40 — against your gateway, holding a fixed SLO in mind (e.g., "p95 TTFT under 1.5s"). Watch the Grafana dashboard live during each ramp step.

```python
import asyncio, httpx, time

async def one_request(client):
    t0 = time.time()
    r = await client.post("http://localhost:8080/v1/chat/completions",
        headers={"Authorization": "Bearer key-abc123"},
        json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
              "messages": [{"role": "user", "content": "Write two sentences about rivers."}]})
    return time.time() - t0

async def ramp(concurrency):
    async with httpx.AsyncClient(timeout=60) as client:
        t0 = time.time()
        durations = await asyncio.gather(*[one_request(client) for _ in range(concurrency)])
        total = time.time() - t0
    print(f"concurrency={concurrency} total={total:.2f}s p95={sorted(durations)[int(len(durations)*0.95)]:.2f}s")

for c in [1, 5, 10, 20, 40]:
    asyncio.run(ramp(c))
```

**Metric that matters here:** find the concurrency level where p95 latency crosses your SLO, and correlate it against the KV-cache-% panel — you're looking for whether the SLO breaks *because* the cache filled up (a memory ceiling) or because compute saturated (a throughput ceiling). That distinction is the entire "goodput vs. throughput" lesson, now empirically observed on your own card.

**You'll know this stage is done when:** you can state a real answer to "how many concurrent users does my RTX 3050 serve before p95 breaks 1.5 seconds," backed by a chart.

---

## Stage 8 — Deliberately break it, then fix it (the four boundaries, hands-on)

**Why:** reading about failure modes and diagnosing one under pressure are different skills. This stage builds the second one, safely, on purpose.

**Exercise A — force KV cache exhaustion (Boundary 4):** temporarily set `--max-model-len` high and `--max-num-seqs` high, then hold many long-context requests open simultaneously until VRAM actually runs low. Watch the KV-cache-% panel climb past 90%, then watch vLLM preempt (evict) a request. Confirm in your gateway logs whether your own request-tracking code handled the eviction gracefully or got confused by a shrinking token count — this is the exact rewind scenario from Boundary 4, reproduced on purpose.

**Exercise B — the silent field drop (Boundary 1):** temporarily comment out the `response_format` forwarding in your gateway code, rerun the contract test from Stage 4, and confirm it now **fails** — proving the test actually catches the regression it was written for, not just passing by coincidence.

**Exercise C — pod death under load (Boundary 2-adjacent, Kubernetes version):** during an active load test, `kubectl delete pod` on the vLLM deployment itself (not just the gateway) and observe: does the gateway return clean errors during the gap, or does it hang? This is where a production system needs retry/circuit-breaker logic — note it as a real limitation of the current build rather than silently ignoring it.

**Metric that matters here:** for each exercise, one sentence: what broke, what the dashboard showed *before* it visibly broke (the leading indicator), and what you changed to fix or mitigate it. This is your incident-report practice, and it's exactly the kind of story an interview question is fishing for.

---

## Stage 9 — Optimization pass, measured against your Stage 3 baseline

**Why:** now that the system is real, instrumented, and load-tested, you can make deliberate changes and *prove* their effect instead of assuming it.

Run each of these, and re-run your Stage 3 benchmark after each one:

1. **Quantization comparison:** benchmark the unquantized Qwen2.5-1.5B-Instruct against the AWQ 3B version. You're checking whether AWQ's kernel speedup actually beats unquantized-but-smaller on your card — don't assume the cheat sheet's numbers transfer directly to your hardware.
2. **Prefix caching, proven, not assumed:** send 20 requests sharing a long identical system prompt, then 20 requests with no shared prefix. Compare TTFT between the two sets and watch the prefix-cache-hit-rate panel — you should see a real, visible difference.
3. **Speculative decoding (n-gram, free to try):** enable `speculative_config` with the n-gram method from the vLLM internals reading, and re-measure ITL. Small models sometimes see less benefit than large ones — report what you actually find, not what you expected.

**Metric that matters here:** a before/after table for each change — throughput, ITL, TTFT — against your Stage 3 baseline. This table is the centerpiece of your written report.

---

## Stage 10 — Product polish

**Why:** "a product" needs a face, not just an API.

**What to do:** a minimal static HTML+JS chat page (no framework needed) that calls your gateway's `/v1/chat/completions` and streams the response. This doesn't need to be elaborate — it needs to exist, because "I can point you at a working thing" is a different sentence than "I have some scripts."

---

## Stage 11 — Documentation and packaging

Write these up as a single README or short report:
1. The architecture diagram from Section 1, with your real component choices labeled.
2. Your Stage 3 baseline numbers and your Stage 9 before/after table.
3. Your Stage 7 concurrency ceiling and what limited it.
4. Your three Stage 8 incident write-ups.
5. A "what I'd do differently at real scale" closing section — this is where you demonstrate you understand the boundary between what one GPU can do and what the disaggregated-P/D, multi-GPU material in the cheat sheet describes, without ever having pretended to build that part.

---

## Managing the project as you go

- **Keep an experiment log from day one** — one line per change: what you changed, what you measured before and after, date. This single habit is what turns "I tried some stuff" into a defensible engineering narrative.
- **Pin your versions deliberately** (the vLLM image tag, the model revision, Python deps) the moment Stage 2 works — Boundary 2 exists precisely because nobody does this until after something breaks.
- **Don't chase every optimization.** Stop tuning when a change moves a number by less than your measurement noise (run any "before/after" comparison at least twice to know your own noise floor first).
- **Suggested order of effort if time is short:** Stages 1–4 and 6 are the non-negotiable core (a working, observed product). Stage 7–8 are what make it *impressive* rather than merely functional. Stage 9–10 are the polish that turns it into something you'd actually show someone.

---

## What's next

This project sits at the exact edge of what one consumer GPU can honestly demonstrate. When you're ready to see what's on the other side of that edge — multi-GPU tensor parallelism, disaggregated prefill/decode, the fleet-scale GPU Operator material from Module 5 — that's Kaggle T4×2 or a rented multi-GPU box, and it's a natural sequel project, not a redo of this one.
