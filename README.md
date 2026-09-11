# The 4GB Chatbot

[![CI](https://github.com/DamnKuldeep/the-4gb-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/DamnKuldeep/the-4gb-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![vLLM 0.11.0](https://img.shields.io/badge/vLLM-0.11.0-blue)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/model-Qwen2.5--1.5B--Instruct--AWQ-blueviolet)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ)

**Serving an LLM to many people at once from a single 4 GiB laptop GPU —
finding the hardware's real ceiling, building a gateway that refuses to exceed
it, and recording every prediction that turned out wrong.**

vLLM serving `Qwen2.5-1.5B-Instruct-AWQ` behind a FastAPI gateway that does
admission control, per-tenant fairness, output bounding, context management,
auth and token budgets — with Prometheus, Grafana, ten predictive alert rules,
a realistic multi-user load simulator, and a 14-case failure matrix.

The hardware is an **NVIDIA RTX 3050 Laptop GPU, 4096 MiB VRAM, 35 W**, which
thermally throttles from ~1,500 MHz to ~950–1,200 MHz sustained. The
constraints are severe enough that nothing could be assumed, so everything was
measured — and most of the interesting numbers contradicted a prediction that
had been written down first.

---

## The headline

> **~22 simultaneous chat users** within a p95 time-to-first-token of 1.5 s.
> Beyond that, **latency stays flat and excess load is refused** rather than
> everyone being served badly.

That is a derivation from one measured number, not a benchmark result:

```text
measured engine ceiling      6 concurrent requests    p95 TTFT 1,005-1,135 ms (+24..33% margin)
                            12 concurrent requests    p95 TTFT 2,043 ms       (-36%, fails)

measured service time       ~3 s per turn (192 output tokens at ~14 ms/token)
duty cycle                   3 / (3 + 12 s think)     = 20%
users at 100% utilisation    6 / 0.20                 = 30
users at ~75% (usable)                                ≈ 22
```

## The envelope — what "22 users" actually bounds

Every capacity number has a shape it was measured in. This is the shape, and
what happens outside it.

| Dimension | Where 22 holds | Outside it |
| --- | --- | --- |
| **Concurrency** | 6 requests in flight at the engine — the gateway's limit | The engine's own `--max-num-seqs` is 32, deliberately above this, so it never binds first |
| **Think time** | 12 s median (log-normal). Real users often think longer | 30 s → ~49 users; 45 s → ~72. It scales with `(think + service) / service` |
| **Output per turn** | 192 tokens in the simulator (~3 s of decode) | 512 tokens → ~12 users; 1,024 → ~8; 2,048 → ~6. **Output length is the strongest lever after concurrency** |
| **Conversation length** | ~1,000 tokens mean. Holds to ~1,300 at 20 users (0/20 over SLO) | At ~1,700 tokens, 6 of 20 users breach; at ~2,500, worst-user p95 hits 9.6 s. Prefill is quadratic and the first turn of a long conversation is a full miss |
| **Prompt size per request** | Chat-length, ~500–1,000 tokens | A single 20k-token prompt holds every short request for ~20 s — the one documented failure (case 3 below) |
| **Context window** | 32,768 tokens, hard (`rope_scaling: null`). Costs zero KV cache vs 4k — measured | Conversations that exceed it have their oldest turns dropped, reported in `X-Context-Trimmed-Messages`. Never a 400 |
| **Max output per call** | Gateway default **1,024** if the client sends none; **ceiling 2,048** (clamped, reported in a header) | Without this, vLLM defaults an absent `max_tokens` to *the rest of the window* — ~30k tokens, holding a slot for minutes |
| **No EOS emitted** | Generation runs to `max_tokens`, `finish_reason: "length"`. The chat UI auto-continues up to 3 times (~6k tokens per turn) | There is no priority scheme. Continuous batching gives every running sequence one token per step; a slot is held for the *whole* stream. That is why the output bound exists |
| **Stream duration** | Bounded at 300 s; slot released after | A client reading one byte per second cannot hold capacity |
| **Thermal state** | All numbers taken heat-soaked at 86–87 °C die temperature | A cold card runs up to ~40% faster for the first two minutes. Compare only runs from the same idle temperature |
| **SLO definition** | p95 TTFT < 1.5 s, **client-observed, including queue wait** | Not end-to-end latency. Once streaming starts, ITL is ~14 ms/token at 6 concurrent |

Full derivation, including which constraint binds first and when that changes,
in **[docs/CAPACITY_MODEL.md](docs/CAPACITY_MODEL.md)**.

---

## What each decision bought

The inference-engineering story, as a table of before/after numbers. Every
row is a measurement on this hardware; the order is roughly the order they
happened.

| Decision | Before | After | Bought |
| --- | ---: | ---: | --- |
| **Host GPU power mode** → *Prefer maximum performance* | 4.15 tok/s, card stuck in P8 | 47.99 tok/s, P0 | **11.6x**, zero code. Silent — no engine metric showed it |
| **Model: 3B-AWQ → 1.5B-AWQ** | SLO holds to conc 2 @ 64.7 tok/s; 28k-token KV | conc 6 @ 295–332 tok/s; 69,760-token KV | **4.3x SLO-compliant throughput**, 2.4x KV cache. Decode ∝ weight bytes (−38%), prefill ∝ params (−51%), both predicted within 3% |
| **Shared system prompt** (prefix cache) | SLO fails at conc 4 | holds at conc 8 | **2x concurrency, TTFT −82%**, KV usage halved |
| **`--max-model-len` 4,096 → 32,768** | 4k window | 32k window | **Zero KV cost** — 69,616 tokens at both. A ceiling, not an allocation |
| **`--max-num-seqs` 8 → 32** | peak 367 tok/s | peak 686 tok/s | +87% peak, **0% SLO gain** — but the engine can no longer be a hidden limiter |
| **Admission control** (6 in flight) | TTFT 13 s and rising under overload | bounded | Unbounded → bounded |
| **Queue timeout 2.0 → 0.6 s** | flat at 2,070 ms, **49/58 users over SLO** | flat at 778 ms, **0/58 over** | Zero breaches for +10 pts shedding |
| **Per-key share: fixed → dynamic** | one abuser takes 50%, 10/10 users over SLO | 14%, 0/10 over, 98.3% of abuse shed | Fairness that survives a real attacker |
| **Admission weighted by prompt size** | three 20k prompts all admitted | one at a time | Bounds head-of-line damage to one prefill |
| **Context trim policy** | hard 400 past 32k | 200 + trim header | Long conversations never crash |
| **Billing circuit breaker** | serves unbilled work indefinitely if the ledger fails | 3 unbilled, then 503, auto-recovers | Fails closed |
| **Output bound** (default 1,024 / cap 2,048) | absent `max_tokens` → ~30k-token hold | bounded ~30 s worst case | Slot hold time is finite |
| **Measurement protocol** (heat-soak, nonce, control run) | 42% swings between identical runs | reproduces within ~2% | The numbers above mean something |

### Degradation is bounded, and this is the graph that shows it

Measured with `chat_sim.py` — Poisson arrivals, log-normal think time,
multi-turn conversations that accumulate context. Shipped configuration,
one `#` per 50 ms:

```text
users | p95 TTFT                       SLO      | shed % | users over SLO
------+--------------------------------|--------+--------+---------------
   10 | ####                 191 ms    |        |   0.0  |   0 / 10
   20 | #########            434 ms    |        |   8.0  |   0 / 20
   30 | ##############       706 ms    |        |  25.1  |   0 / 30
   40 | ##############       717 ms    |        |  38.9  |   0 / 40
   60 | ################     778 ms    |        |  56.9  |   0 / 58
      +--------------------------------|
                                    1500 ms
```

**Offered load rose 6x; p95 TTFT rose from 191 ms to 778 ms and stayed there.
Not one user out of 58 breached the SLO at any level.** The excess was refused
with `503` + `Retry-After` instead of being absorbed into a queue.

**Bounded is not automatically acceptable.** The first configuration used a
2.0 s queue timeout and produced an equally flat curve — but flat at 2,070 ms,
above the objective, with 49 of 58 users breaching. Worst-case TTFT ≈ queue
timeout + engine TTFT, so a 1.5 s target minus ~0.8 s of engine time leaves
~0.6 s of queue budget. That is arithmetic, not tuning.

> Refusing one request in ten more, so every request you *do* accept is served
> within its target. That trade is the whole thesis.

---

## What watches it

Failures on a GPU service are mostly silent: the 11.6x power-state loss had no
error in any log, and a prefix-cache miss costs 9x and prints nothing. So the
observability is built around **what predicts failure, not what reports it**.

| Signal | Why it is predictive | Source |
| --- | --- | --- |
| **Prefix cache hit rate** | Falls *before* latency rises — spare capacity absorbs the extra prefill first | vLLM `/metrics`, scraped directly |
| **Admission queue depth** | A request has to wait before it can be slow | gateway `/metrics` |
| **KV cache %** and **preemptions** | 8–17% is normal here; 85% means something changed qualitatively | vLLM |
| **Client-observed TTFT histogram** | Includes queue time, which the engine cannot see | gateway |
| **Billing write failures** | Non-zero means GPU work is being served unbilled | gateway |
| **Shed rate by reason** | `queue_full` vs `per_key_limit` are different problems with different fixes | gateway |
| **Sustained throughput below floor with a busy queue** | The only software-visible symptom of thermal throttling — stated as a proxy, because WSL2 gives Prometheus no GPU telemetry | vLLM + gateway |

Ten alert rules in [`observability/prometheus/alerts.yml`](observability/prometheus/alerts.yml),
each with the measurement its threshold came from. Two Grafana dashboards: the
engine's own metrics, and the gateway's admission view.

**And when something does break** — the 14-case
[failure matrix](docs/FAILURE_MATRIX.md), each case with the behaviour
*expected before testing* next to the behaviour measured:

| Case | Measured |
| --- | --- |
| Engine crashes mid-stream | Detected in 1.4 s, auto-recovered in 83 s. Gateway `/health` stayed 200 throughout (never restarted), `/ready` went 503→200. **162 of 162 requests in the ledger, drift 0** |
| Gateway restarts under load | Back in 10.4 s; in-flight requests fail with a dropped connection, budgets intact |
| Client disconnects mid-stream | Slot released, usage recorded — from the generator's `finally` |
| Billing database unwritable | `200, 200, 200, 503…` — three unbilled, then closed, then auto-recovered |
| One key at 50 concurrent | Normal users' p95: 191 → 536 ms. **Still 64% inside the SLO.** 4,669 of 4,749 abusive requests shed |
| Burst of 60 simultaneous (10x limit) | 6 served at p95 263 ms, 54 shed with `Retry-After` |
| Conversation exceeds 32k | 95k tokens sent → 27 oldest messages dropped → engine saw 9.7k. HTTP 200 |
| **A single 20k-token prompt** | **Short requests blocked for ~20 s.** The engine-side fix exists and crash-loops this vLLM build (V0-only). The gateway bounds how many such prompts run; it cannot remove the blocking. Documented, not passed |

12 pass, 2 documented limitations, 0 unexplained.

---

## Architecture

```text
                    ┌──────────────────────────────────────────┐
  browser ────────► │  gateway  (FastAPI, :8080)               │
  OpenAI client ──► │                                          │
                    │   auth ─► budget ─► context policy ─►    │
                    │   output bound ─►   admission control ─┐ │
                    │                                        │ │
                    │   /chat  /dashboard  /metrics  /ready   │ │
                    └────────────────────────────────────────┼─┘
                              │                              │
                    ┌─────────▼─────────┐        ┌───────────▼──────────────┐
                    │ SQLite            │        │ vLLM (:8000)             │
                    │ keys, budgets,    │        │ Qwen2.5-1.5B-AWQ         │
                    │ request ledger    │        │ awq_marlin · prefix cache│
                    └───────────────────┘        │ 69,760-token KV pool     │
                                                 └───────────┬──────────────┘
                    ┌───────────────────┐                    │
                    │ Prometheus :9090  │◄───────────────────┘
                    │ Grafana    :3000  │◄─── gateway /metrics
                    └───────────────────┘
```

**The gateway is the only limiter.** `--max-num-seqs` is set to 32 — far above
the gateway's 6 — specifically so the engine's own cap can never silently shape
traffic the gateway believes it is controlling. One limiter, where the policy
and the metrics live.

---

## Quickstart

**You need:** an NVIDIA GPU with ≥ 4 GiB VRAM and a driver providing **CUDA ≥ 12.8**
(the vLLM image refuses to start below that — checked before any of your code
runs), and Docker with GPU passthrough. On Windows that is Docker Desktop with
the WSL2 backend; on Linux, the NVIDIA Container Toolkit.

```powershell
git clone https://github.com/DamnKuldeep/the-4gb-chatbot.git
cd the-4gb-chatbot/deployment/docker

# The model cache is an external volume so `compose down` can never delete it.
docker volume create hf-cache

# Both files, always. The base file alone runs the older 3B model.
$env:COMPOSE_FILE="docker-compose.yml;docker-compose.1_5b-awq.yml"     # PowerShell
# export COMPOSE_FILE="docker-compose.yml:docker-compose.1_5b-awq.yml"  # bash

docker compose --profile observability up -d
docker compose ps          # wait until vllm reads "healthy"
```

**First launch is slow and that is normal:** it pulls the vLLM image (~10 GB),
downloads the model (~1.2 GB), and compiles kernels (~60 s). Subsequent starts
take 60–90 s. `docker compose up -d` returns in about six seconds — that is not
the engine being ready. Wait for `healthy`.

| URL | What it is |
| --- | --- |
| <http://localhost:8080/chat> | Chat UI. API key `dev-key-alpha` |
| <http://localhost:8080/dashboard?key=dev-key-alpha> | Per-key usage, budgets, request history |
| <http://localhost:8080/metrics> | Gateway metrics: queue depth, shedding, client TTFT |
| <http://localhost:8080/admission> | Live admission state, human-readable |
| <http://localhost:3000> | Grafana — engine dashboard + capacity dashboard |
| <http://localhost:9090/alerts> | 10 alert rules |
| <http://localhost:8000/metrics> | vLLM's own metrics, scraped directly |

Any OpenAI client works by changing the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dev-key-alpha")
```

**If throughput is terrible on a laptop**, check `nvidia-smi` for `P8` under
load. Windows' default GPU power policy held this card at idle clocks during
LLM decode and cost **11.6x** — silently. Fix: NVIDIA Control Panel → Power
management mode → *Prefer maximum performance*.

Shut down with `docker compose --profile observability down`. The model cache
and the usage ledger survive.

**Everything is pinned**: image `vllm/vllm-openai:v0.11.0`, model revision
`3ecffa0ceb27851800f45519bab9c457a04405e1`, driver 616.56 / CUDA 13.4.

---

## Reproducing the measurements

```powershell
.\load_testing\run_matrix.ps1                    # capacity sweep, cache ablation, traffic shapes
python load_testing/failure_matrix.py --case all # the worst cases reachable over HTTP
python load_testing/kill_test.py --case engine   # crash the engine under load
python load_testing/analyze.py                   # tables from results/*.json
```

Every run embeds the engine config that produced it, uses a fixed seed, runs a
discarded warm-up, and records the GPU's idle temperature — because
**two runs of an identical command on this machine differed by 42% purely from
thermal state**, and comparing across that would be measuring the weather.

The ten rules that protocol is built from, each traceable to the measurement
that produced it, are in
[benchmarks/optimization_results.md](benchmarks/optimization_results.md).

### Tests

```powershell
pip install -r gateway/requirements-dev.txt
pytest                           # 25 contract tests, no GPU needed (this is what CI runs)
pytest -m integration            # 1 test against the live stack
node ui/test_markdown.mjs        # 25 renderer tests incl. XSS
.\gateway\test_gateway.ps1       # auth, streaming, 401s
.\gateway\test_budget.ps1        # budget enforcement to a 429
```

CI runs the contract tests and validates both compose files parse. It never
starts vLLM — GitHub's runners have no GPU, and the numbers above cannot be
validated there.

---

## What this turned out to be about

Four results that were not the goal when the project started:

**1. Decode is core-clock-gated, not memory-bandwidth-bound.** Memory clock was
pinned at 5,501 MHz across every run while ITL swung 56%, tracking SM clock to
within 6%. The card never saturates its 192 GB/s bus. This was filed in the
plan as *"a testable prediction this project cannot test"* — and it got tested
by accident.

**2. Prefill is quadratic and everyone here assumed it was linear.** Fitting
three prompt sizes gives `TTFT ≈ 0.13 + n/3070 + n²·1.11e-8`, matching measured
values within 5%. The attention term is 1.6% of prefill at 471 tokens and
**43% at 22,586**. A full 32,768-token prompt costs ~23 s.

**3. The binding constraint changes identity with think time.** At 12 s think
time admission control binds at ~22 users while the prefix cache would not bind
until ~70. At 45 s think time with longer conversations the cache binds at ~17
while admission would not bind until ~72. A capacity model that reports one
number is hiding which regime it was measured in.

**4. Load shedding protects the prefix cache as a side effect.** The predicted
cache-collapse knee never appeared, because shedding keeps the number of
*progressing* conversations low enough that the working set never outgrows the
pool. Two mechanisms designed independently reinforce each other.

The full list — **twenty-four wrong predictions and what each one taught** — is
in [docs/FINAL_REPORT.md §11](docs/FINAL_REPORT.md#11-what-this-project-got-wrong-and-what-that-taught).

---

## Where to read next

| If you want… | Read |
| --- | --- |
| The capacity number and exactly where it stops being true | [docs/CAPACITY_MODEL.md](docs/CAPACITY_MODEL.md) |
| What happens under each of 14 failures, expected vs measured | [docs/FAILURE_MATRIX.md](docs/FAILURE_MATRIX.md) |
| The whole account, v1 through v2, with the wrong-predictions table | [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) |
| Why each architectural choice was made | [DECISIONS.md](DECISIONS.md) |
| The ten measurement rules and the failure behind each | [benchmarks/optimization_results.md](benchmarks/optimization_results.md) |
| The raw chronological log — every stage, every bug, every correction | [PROGRESS_LOG.md](PROGRESS_LOG.md) |
| Every new tool or flag, explained the first time it appeared | [docs/CONCEPTS_EXPLAINED.md](docs/CONCEPTS_EXPLAINED.md) |

---

## Honest limitations

* **Answer quality is barely measured.** On this project's control question the
  1.5B describes the KV cache as a generic key-value store — the same error the
  3B made. So there is no observable regression, but that is one prompt.
* **A single very long prompt still blocks everyone** for ~20 s. vLLM has the
  scheduler flags to fix this; on 0.11.0 they silently disable the V1 engine
  and the server crash-loops. The gateway bounds how many such prompts run at
  once; it cannot remove the blocking.
  [Case 3](docs/FAILURE_MATRIX.md#case-3--the-one-that-does-not-pass-and-why-the-fix-is-not-available).
* **Single replica.** SQLite has one writer; budgets across two gateway
  processes would need Postgres or Redis with atomic decrements.
* **Thermal throttling is not alertable.** It is a hardware fact invisible to
  vLLM's metrics, and `dcgm-exporter` needs driver access WSL2 under WDDM does
  not reliably provide. `benchmarks/watch_gpu.ps1` covers it; a proxy alert
  rule is labelled as a substitute.
* **`--gpu-memory-utilization` is 0.78, not the 0.9 vLLM recommends.** Windows
  WDDM reserves VRAM the engine cannot use. Headroom exists but taking it makes
  startup depend on what else is running on the desktop.
* **The numbers do not transfer.** A 35 W thermally-limited laptop part is not a
  datacenter GPU, WSL2 forces `pin_memory=False`, and the sustained clock varies
  with how long the machine has been working. This is a characterisation of one
  stack on one machine, with the derivation of why it behaves as it does.
* **Not measured at all:** speculative decoding, Kubernetes self-healing,
  multi-replica accounting, real user traffic.

---

## Repository map

| Path | What's in it |
| --- | --- |
| `gateway/` | FastAPI service: auth, budgets, admission, context policy, output bound, dashboard |
| `gateway/admission.py` | Bounded concurrency weighted by prompt size, dynamic per-key share, metrics |
| `gateway/context.py` | Sliding-window context policy and why it is that shape |
| `ui/` | Chat UI (no framework, no build step) + renderer tests |
| `load_testing/chat_sim.py` | Multi-user simulator: Poisson arrivals, think time, growing context |
| `load_testing/failure_matrix.py` | The worst cases, each with a stated expectation |
| `load_testing/run_matrix.ps1` | The whole matrix, one command, seeded |
| `load_testing/results/` | Every measurement behind every table, engine config embedded |
| `deployment/docker/` | Compose stack; every flag carries its derivation in a comment |
| `observability/` | Prometheus config, 10 alert rules, 2 Grafana dashboards |
| `docs/` | Capacity model, failure matrix, final report, concepts glossary |
| `benchmarks/` | Baseline, optimization results, the measurement protocol, GPU sampler |

Built stage by stage as a learning project, with an AI assistant explaining each
piece and the human running every command. `CLAUDE.md`, `PRODUCT_SPEC.md` and
`PROJECT_PLAN.md` are the working agreement that produced it, kept because they
explain the shape of everything else.
