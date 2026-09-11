# The 4GB Chatbot

[![CI](https://github.com/DamnKuldeep/the-4gb-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/DamnKuldeep/the-4gb-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![vLLM 0.11.0](https://img.shields.io/badge/vLLM-0.11.0-blue)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/model-Qwen2.5--1.5B--Instruct--AWQ-blueviolet)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ)

**A multi-user chat service running entirely on one 4 GiB laptop GPU — where
every capacity claim is a real measurement, and every wrong prediction is
written down instead of quietly fixed.**

vLLM serving `Qwen2.5-1.5B-Instruct-AWQ` behind a FastAPI gateway that does
auth, per-key token budgets, admission control, per-tenant fairness and context
management — with Prometheus, Grafana, a chat UI, a realistic load simulator,
and a 14-case failure matrix.

It runs on an **NVIDIA RTX 3050 Laptop GPU with 4096 MiB of VRAM**, a 35 W part
that thermally throttles from ~1,500 MHz to ~950–1,200 MHz sustained. That is
the point: the constraints are severe enough that nothing could be assumed.

---

## The headline

> **~22 simultaneous chat users** within a p95 time-to-first-token of 1.5 s, at
> 12-second think time and ~1,000-token conversations.
> Beyond that, **latency stays flat and excess load is refused**, rather than
> everyone being served badly.

That number is a derivation, not a benchmark result:

```text
measured engine ceiling      6 concurrent requests    p95 TTFT 1,005-1,135 ms (+24..33% margin)
                            12 concurrent requests    p95 TTFT 2,043 ms       (-36%, fails)

measured service time       ~3 s per turn
duty cycle                   3 / (3 + 12)             = 20%
users at 100% utilisation    6 / 0.20                 = 30
users at ~75% (usable)                                ≈ 22
```

**It moves with think time, because think time is a property of the product,
not the hardware:**

| think time | duty cycle | usable users |
| ---: | ---: | ---: |
| 12 s | 20% | **~22** |
| 30 s | 9.1% | ~49 |
| 45 s | 6.3% | ~72 |

Full derivation, including where it stops being true, in
**[docs/CAPACITY_MODEL.md](docs/CAPACITY_MODEL.md)**.

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
with `503` + `Retry-After` instead of being absorbed into a queue. Without
admission control the same load produced TTFT climbing past 13 s and rising.

**Bounded is not automatically acceptable, and getting that wrong is instructive.**
The first configuration used a 2.0 s queue timeout and produced an equally flat
curve — p95 rose just **2.2% while offered load doubled** — but the flat line sat
at **2,070 ms**, above the objective, and **49 of 58 users breached the SLO**.

The fix was arithmetic, not tuning: worst-case TTFT ≈ queue timeout + engine
TTFT, so a 1.5 s objective minus ~0.8 s of engine time leaves ~0.6 s of queue
budget. Dropping the timeout to 0.6 s bought **zero SLO breaches for ten
percentage points more shedding** (47% → 57%).

> Refusing one request in ten more, so every request you *do* accept is served
> within its target. That trade is the whole thesis.

The 10-user level was **re-run last as a control** (260 ms → 266 ms, 83.4% →
83.9% cache hit), which is what makes the curve trustworthy: on this hardware,
thermal drift between runs can otherwise reach 42%.

---

## Architecture

```text
                    ┌──────────────────────────────────────────┐
  browser ────────► │  gateway  (FastAPI, :8080)               │
  OpenAI client ──► │                                          │
                    │   auth ─► budget ─► context policy ─►    │
                    │                     admission control ─┐ │
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
(the vLLM image refuses to start below that — it is checked before any of your
code runs), and Docker with GPU passthrough. On Windows that is Docker Desktop
with the WSL2 backend; on Linux, the NVIDIA Container Toolkit.

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
LLM decode and cost **11.6x** — silently, with no error anywhere. Fix:
NVIDIA Control Panel → Power management mode → *Prefer maximum performance*.

Shut down with `docker compose --profile observability down`. The model cache
and the usage ledger survive.

**Everything is pinned**: image `vllm/vllm-openai:v0.11.0`, model revision
`3ecffa0ceb27851800f45519bab9c457a04405e1`, and the driver version the numbers
were taken on (616.56 / CUDA 13.4).

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
pytest                           # 22 contract tests, no GPU needed (this is what CI runs)
pytest -m integration            # 1 test against the live stack
node ui/test_markdown.mjs        # 25 renderer tests incl. XSS
.\gateway\test_gateway.ps1       # auth, streaming, 401s
.\gateway\test_budget.ps1        # budget enforcement to a 429
```

CI runs the contract tests and validates both compose files parse. It never
starts vLLM — GitHub's runners have no GPU, and the numbers above cannot be
validated there. The workflow says so in its own comments.

---

## What this turned out to be about

Four results that were not the goal when the project started:

**1. Decode is core-clock-gated, not memory-bandwidth-bound.** Memory clock was
pinned at 5,501 MHz across every run while ITL swung 56%, tracking SM clock to
within 6%. The card never saturates its 192 GB/s bus. This was a standing
prediction in the plan, filed as *"a testable prediction this project cannot
test"* — and it got tested by accident.

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
cache-collapse knee never appeared in the user sweep, because shedding keeps
the number of *progressing* conversations low enough that the working set never
outgrows the pool. Two mechanisms that were designed independently reinforce
each other.

The full list — **twenty-four wrong predictions and what each one taught** — is
in [docs/FINAL_REPORT.md §11](docs/FINAL_REPORT.md#11-what-this-project-got-wrong-and-what-that-taught).
That table is the most useful thing in this repository for anyone doing
similar work.

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
* **A single very long prompt still blocks everyone.** A 20k-token prompt holds
  short requests for ~20 s. vLLM has the scheduler flags to fix this; on 0.11.0
  they silently disable the V1 engine and the server crash-loops. The gateway
  bounds how many such prompts run at once; it cannot remove the blocking.
  [Case 3 in the failure matrix](docs/FAILURE_MATRIX.md#case-3--the-one-that-does-not-pass-and-why-the-fix-is-not-available).
* **Single replica.** SQLite has one writer; budgets across two gateway
  processes would need Postgres or Redis with atomic decrements. Documented,
  not solved.
* **Thermal throttling is not alertable.** It is a hardware fact invisible to
  vLLM's metrics, and `dcgm-exporter` needs driver access WSL2 under WDDM does
  not reliably provide. Prometheus covers engine and gateway state;
  `benchmarks/watch_gpu.ps1` covers hardware; a proxy rule infers throttling
  from software symptoms and is labelled as a substitute.
* **`--gpu-memory-utilization` is 0.78, not the 0.9 vLLM recommends.** Windows
  WDDM reserves VRAM the engine cannot use. Headroom exists — desktop apps were
  holding 290 MiB — but taking it makes engine startup depend on what else is
  running on the desktop, which breaks "one command brings the stack up".
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
| `gateway/` | FastAPI service: auth, budgets, admission, context policy, dashboard |
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
