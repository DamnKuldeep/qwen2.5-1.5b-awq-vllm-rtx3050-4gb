# Self-Hosted LLM Inference Product — Final Report

An authenticated, budget-limited API in front of a tuned vLLM server, with separate product and infrastructure dashboards, deployed by one command, benchmarked on real hardware, and hardened — provably — against four named self-hosting failure boundaries.

Built 2026-09-07/08 on a single RTX 3050 laptop GPU with 4 GiB of VRAM.

**Every number in this report was measured on that machine.** Where a figure is an estimate, it says so. Where a measurement is unreliable, it says why.

> ### ⚠ Sections 1–11 describe v1, which shipped `Qwen2.5-3B-Instruct-AWQ`
>
> They are left as written, because the reasoning in them is what produced the
> v2 decisions and rewriting history would hide that. **The finalized stack is
> different**, and sections 12 onward describe it:
>
> | | v1 (sections 1–11) | **Finalized (section 12+)** |
> | --- | --- | --- |
> | Model | Qwen2.5-**3B**-Instruct-AWQ | **Qwen2.5-1.5B-Instruct-AWQ** @ `3ecffa0c…` |
> | Weights / KV cache | 1.9542 GiB / 28,480 tok | **1.1018 GiB / 69,760 tok** |
> | Context window | 4,096 | **32,768** |
> | `--max-num-seqs` | 8 | **32** |
> | Limiter | none | **gateway: 6 in flight, 12 queued, 0.6 s timeout** |
>
> For the current picture start at [README.md](../README.md) and
> [CAPACITY_MODEL.md](CAPACITY_MODEL.md). For what changed and why, section 12.

---

## 1. Architecture

```
                        ┌──────────────────────────────────┐
   browser ───────────► │  GATEWAY  (FastAPI, port 8080)   │
   OpenAI SDK ────────► │                                  │
                        │  • Bearer-token auth             │
                        │  • Per-key token budgets → 429   │
                        │  • Streaming proxy (no buffering)│
                        │  • Request log → SQLite          │
                        │  • /chat        chat UI          │
                        │  • /dashboard   product usage    │
                        │  • /health      liveness         │
                        │  • /ready       readiness        │
                        └───────────────┬──────────────────┘
                                        │ HTTP, streamed
                                        ▼
                        ┌──────────────────────────────────┐
                        │  vLLM  (port 8000)               │
                        │  Qwen2.5-3B-Instruct-AWQ         │
                        │  awq_marlin · FlashAttention     │
                        │  PagedAttention · prefix caching │
                        └───────────────┬──────────────────┘
                                        │ /metrics (scraped directly)
                                        ▼
                     ┌────────────────┐    ┌──────────────────┐
                     │ PROMETHEUS     ├───►│ GRAFANA  :3000   │
                     │ :9090, 5s      │    │ engine health    │
                     └────────────────┘    └──────────────────┘
```

**Two dashboards answering two different questions, deliberately built differently.**
`localhost:8080/dashboard` is server-rendered HTML answering *how is the product being used* — requests and tokens per key, remaining budget, request history. `localhost:3000` is Grafana answering *is the system healthy* — KV cache, prefix hit rate, TTFT/ITL, queue time, preemptions. Conflating infrastructure health with product usage is a common operational mistake; making them visibly different artifacts makes the distinction hard to lose.

**One command brings the whole stack up:**

```powershell
cd deployment/docker
docker compose up -d                              # engine + gateway
docker compose --profile observability up -d      # + Prometheus + Grafana
```

| URL | What |
| --- | --- |
| `http://localhost:8080/v1/chat/completions` | OpenAI-compatible API |
| `http://localhost:8080/chat` | Chat UI |
| `http://localhost:8080/dashboard` | Product usage dashboard |
| `http://localhost:3000` | Grafana — engine health |
| `http://localhost:9090/targets` | Prometheus scrape status |
| `http://localhost:8000/metrics` | vLLM metrics, raw |

---

## 2. The pinned stack

| Component | Pinned value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, compute 8.6 (Ampere) |
| Driver | **616.56**, CUDA UMD 13.4 |
| Host | Windows 11, Docker Desktop 4.89.0, WSL2, GPU under WDDM |
| Engine image | `vllm/vllm-openai:v0.11.0` — digest `sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c` |
| Model | `Qwen/Qwen2.5-3B-Instruct-AWQ` |
| Model revision | `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` |
| Kernel | `awq_marlin` (confirmed selected in logs) |
| Gateway deps | fastapi 0.115.6, uvicorn 0.34.0, httpx 0.28.1, jinja2 3.1.5 |

**Engine flags:** `--quantization awq_marlin --dtype float16 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.78 --enable-prefix-caching --swap-space 2`

**Resulting memory layout:** 1.9542 GiB weights, 0.98 GiB KV cache = **28,480 tokens** (1,780 blocks × 16), 6.94x maximum concurrency at full context.

The driver version is pinned deliberately: `vllm/vllm-openai:v0.11.0` declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8` and refuses to start below it. **An environment that cannot start the pinned image is as much a reproducibility break as an unpinned dependency.**

---

## 3. The four boundaries, demonstrated

### Boundary 1 — structured output silently dropped

**The failure:** a gateway that deserializes a request into a typed model and re-serializes it silently discards any field the model does not declare. `response_format` is the classic casualty. Nothing looks wrong — HTTP 200, a plausible response, no error anywhere. Structured output simply stops being structured.

**The defence, structural rather than disciplinary.** The gateway parses the body into an **untyped `dict`** and never into a schema. A dict has no allowlist, so nothing can be dropped — including fields that do not exist yet. (Stage 3 forwarded raw bytes, which was stronger; Stage 4 traded that away to inject `stream_options.include_usage` for streaming budget accounting. The trade is recorded in `DECISIONS.md`, and it is why the contract test became load-bearing.)

**The proof.** 22 contract tests run against a stub upstream that records exactly what the gateway sent. The decisive one is not `response_format`-specific:

```python
def test_unknown_future_field_survives(gateway, auth):
    payload = base_request(some_field_invented_in_2027={"nested": ["a", 1, None, True]})
```

Testing `response_format` only proves someone remembered that field. **This asserts the gateway has no allowlist at all** — a typed-model gateway passes every field-specific test while failing this one, which is precisely how the failure reaches production.

**Proven to bite.** A field allowlist was deliberately inserted into the gateway. Result: **14 failed, 8 passed** — `response_format`, plus `top_p`, `top_k`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `n`, `user`, `tools`, `tool_choice`, `guided_json`. Every request still returned 200. **A team debugging "why doesn't `seed` work?" would add `seed` to the allowlist, ship, and remain broken for the other eleven — which is how this failure mode survives for years, presenting as a series of unrelated small bugs.**

**An unplanned dividend.** Stage 9's load generator needed vLLM's `ignore_eos` extension, which is not part of the OpenAI schema. It reached the engine with no gateway change at all, because unmodelled fields pass through untouched. The Boundary 1 decision paid for itself in a place nobody anticipated when making it.

### Boundary 2 — unpinned versions invalidating benchmarks

**The failure:** a floating image tag or model name silently changes the thing under measurement, so before/after comparisons stop meaning anything with no visible signal.

**Pinned:** engine image by digest, model by commit hash `3559b226…` via `--revision` *and* `--tokenizer-revision` (separate fields in vLLM; an unpinned tokenizer would shift token counts and therefore budget accounting), and the driver version recorded as an environment constraint.

**Verified rather than assumed.** The server was restarted with the pins applied and the engine config confirmed live: `revision=3559b226…, tokenizer_revision=3559b226…`, KV cache identical at 28,432 tokens. A pin that silently fails is worse than no pin.

**Deviation recorded honestly:** the two models used in Stage 11's comparison were *not* revision-pinned. Acceptable for one-off measurements recorded as single data points; not acceptable for anything shipped. If either were adopted, pinning comes first.

### Boundary 3 — monitoring that drops the metrics that matter

**The failure:** a monitoring bridge faithfully forwards most of an engine's metrics while quietly dropping the two that would have warned you first — **KV cache utilisation** and **prefix cache hit rate**.

**The defence:** Prometheus scrapes vLLM's `/metrics` **directly**. No sidecar, no bridge, no translation layer. Nothing can be lost in between because there is no in-between.

**The dashboard leads with exactly those two metrics**, in the first row, under a heading naming the boundary. Both earned their place before the dashboard existed:

- KV cache at **18.5%** under load (Stage 2) is why Stage 9 hunted for a thermal/compute ceiling rather than a memory one — and was right.
- Prefix hit rate reaching **88%** across a chat conversation (Stage 6) explained why TTFT stayed flat as context grew 55x.
- Prefix hit rate spiking to **100%** at the exact moment of a preemption (Stage 10) made the recompute-after-eviction mechanism visible in two panels at once.

**Scope stated rather than faked:** GPU hardware metrics (temperature, clocks, power) are **not** in Grafana. vLLM exports engine state only, and dcgm-exporter expects Linux driver access patterns WSL2/WDDM does not reliably provide. `benchmarks/watch_gpu.ps1` covers hardware state instead. Given that this project's single largest performance finding — an 11.6x loss — was invisible in every engine metric and only visible in `nvidia-smi`, pretending a dashboard covered hardware would have been worse than admitting it does not.

### Boundary 4 — engine preemption corrupting gateway state

**The failure:** when the engine runs out of KV cache it evicts running sequences. A request whose final usage chunk never arrives gets billed zero, and the budget drifts silently.

**How the eviction was forced.** `--max-num-seqs 8` was set *above* the measured 6.94x capacity in Stage 1 — deliberately overcommitted so this demonstration would be possible. Nine stages later: 8 concurrent requests of 476 + 3,500 = 3,976 tokens each demand **31,808 tokens against 28,480 available**.

**The engine did what it should:**

```
KV cache:     13.6%  →  98%  →  22%  →  98%     (the 98→22 drop IS the eviction)
Preemptions:  0  →  non-zero, two pulses
Running:      8  →  briefly 7  →  8
ITL p99:      spiked to ~1 s
Prefix hit:   spiked to 100% twice, aligned with the preemption pulses
```

**The gateway's accounting was exact:**

| | Before | After | Delta |
| --- | --- | --- | --- |
| Requests | 221 | 237 | **16** |
| Prompt tokens | 103,264 | 110,817 | **7,553** |
| Completion tokens | 10,538 | 66,538 | **56,000** |
| Errors | 0 | 0 | **0** |

**56,000 = 16 × 3,500 exactly. Not one token lost.** The reason is structural: the `finally` block in the gateway's streaming generator records the request whether the stream completes, errors, or is torn down. Evictions were entirely invisible to callers — requests simply took longer (E2E p50 117.9 s).

---

## 4. Baseline

Measured under the protocol in §6. GPU verified in P0. 512 prompt tokens in, 128 out.

| Metric | Single stream (conc 1) | Batched (conc 8) | Change |
| --- | --- | --- | --- |
| Total generated tokens | 4,096 (exact) | 16,384 (exact) | — |
| **Output throughput** | **33.39 tok/s** | **137.15 tok/s** | **4.11x** |
| Median TTFT | 654.13 ms | 758.76 ms | +16% |
| **Median ITL** | **26.13 ms** | **29.51 ms** | **+13%** |
| **P99 ITL** | **33.62 ms** | **637.62 ms** | **19x worse** |
| Median E2EL | 4,049 ms | 8,548 ms | 2.1x |
| Peak KV cache | ~2.4% | **18.5%** | — |
| Requests queued | 0 | 0 | — |

**Continuous batching is worth 4.11x throughput for a 13% median latency cost** — 51% scaling efficiency against 8x concurrency. The shortfall is expected: decode is memory-bandwidth-bound and eight sequences contend for one bus.

**The cost hides in the tail.** P99 ITL degrades 19x while the median moves 13%. Those are scheduler pauses when an arriving request's prefill preempts ongoing decode. **Mean TPOT (44.90 ms) smooths this away entirely** — only P99 ITL exposes it.

---

## 5. The concurrency ceiling

**SLO stated before measuring: p95 TTFT < 1.5 s.**

With unique prompts (no prefix-cache assistance), 512 in / 256 out:

| conc | tok/s | TTFT p95 | SLO | margin |
| ---: | ----: | -------: | :--: | -----: |
| 1 | 39.3 | 601 ms | PASS | **+60%** |
| 2 | 64.7 | 1,165 ms | PASS | **+22%** |
| 3 | 84.4 | 1,729 ms | FAIL | −15% |
| 4 | 108.1 | 2,269 ms | FAIL | −51% |

**Reliable ceiling: concurrency 2, at 64.7 tok/s.**

### Which resource ran out — the diagnostic

At the highest levels tested (up to concurrency 32):

| Signal | Value | Reading |
| --- | --- | --- |
| KV cache peak | **13.6%** | Not memory — nowhere close |
| Preemptions | **zero** | Nothing was ever evicted |
| Prefill time | **flat** across the whole ramp | Prefill did not get slower |
| Queue time | **→ ~12 s p50, ~15 s p99** | Requests waiting for admission |
| Running / waiting | pinned **7–8** / peaked **24** | At the `--max-num-seqs` cap |
| ITL p50 | 16.3 → 26.2 ms over 32x concurrency | Decode barely degraded |

**TTFT grew from 348 ms to 13,281 ms and essentially all of that growth is queueing, not work.** The binding constraint is `--max-num-seqs 8` — a value we chose in Stage 1 against a 4,096-token worst case that a 640-token workload never approaches. Not memory, not decode bandwidth.

**The margin column exists because a bare verdict misled us once.** Every 3B run at concurrency 4 clustered around the limit: 1,476 (PASS by 24 ms), 1,405, 2,115, 1,530 (FAIL by 30 ms), 1,766. A 24 ms pass and a 30 ms fail are the same measurement, with session drift (§6) far larger than the gap. **A binary verdict against a threshold turns noise into apparent fact when the measurement lands near the threshold.**

---

## 6. Measurement protocol — assembled from this project's own failures

Every rule exists because ignoring it produced numbers that looked like results and were not.

| Rule | The failure that produced it |
| --- | --- |
| **Verify the GPU is in P0 before recording anything** | A host power setting held the card at 210 MHz core / 405 MHz memory under 100% load, costing **11.6x throughput** — silently, with no error in any engine log |
| **Restart the engine between compared pairs** | Two *identical* runs differed by **16% throughput** and **45% on P99 ITL** from prefix-cache warming alone |
| **Measure pairs consecutively** | Sustained load drifts results by **~24%** over ~35 minutes — larger than most effects being measured. Recovers on cooling; it is thermal, not degradation |
| **Discard a warm-up run** | Boost clocks vs thermal steady state moved mean TTFT by **900%** (51.5 → 515.4 ms) |
| **Launch each variant once before measuring it** | The `torch.compile` cache is keyed on the vLLM config, so every flag change costs a one-off ~30–60 s compile |
| **`--ignore-eos` on every run** | Without it, an identical request produced 2,683 and then 2,823 tokens; a `max_tokens=128` ramp averaged ~42 |
| **≥ 40 samples per level** | With nearest-rank percentiles, p95 of 16 samples **is the maximum** — one cold-start outlier made p95 TTFT read 4,700 ms against 1,419 ms |
| **Report the margin, not just PASS/FAIL** | A 24 ms pass and a 30 ms fail were reported as different outcomes of the same measurement |
| **Quote the input/output ratio with every throughput figure** | The same hardware and config produced **89.5 tok/s** (512 in / 42 out) and **205.6 tok/s** (476 in / 3,500 out) |
| **Treat KV-cache differences under ±0.2% as noise** | vLLM profiles *free* VRAM at startup; 28,432 vs 28,480 tokens is compositor variance |

---

## 7. Optimization results

### 7.1 `--max-num-seqs` 8 → 24

| Metric | Before | After | Change |
| --- | --- | --- | --- |
| Peak output throughput | 256.1 tok/s | **344.2 tok/s** | **+34.4%** |
| TTFT p50 @ conc 24 | 17,207 ms | 2,820 ms | **−83.6%** |
| ITL p50 @ conc 24 | 21.3 ms | 39.9 ms | +87% |
| **ITL p99 @ conc 24** | **53.2 ms** | **1,283.3 ms** | **24x worse** |
| **Highest SLO-compliant level** | **4** | **4** | **no change** |

Validated by a control group: concurrency 1, 4 and 8 must be unchanged because the cap does not bind below itself. Concurrency 8 moved 0.8%.

**Under the stated SLO this change is worth nothing.** The SLO already fails at concurrency 8, and the cap only binds above it. **"+34% throughput" would have been a true and thoroughly misleading headline** — stating the SLO *before* measuring is what made that visible. The product keeps `--max-num-seqs 8`.

### 7.2 Shared prefix vs unique prompts

| Metric @ conc 8 | Unique | Shared | Change |
| --- | --- | --- | --- |
| **Highest SLO-compliant level** | fails at 4 | **passes at 8** | **2x+** |
| TTFT p50 | 2,512 ms | **451 ms** | **−82.0%** |
| Output throughput | 194.2 tok/s | 259.9 tok/s | +33.8% |
| **ITL p50** | 27.7 ms | 28.4 ms | **+2.5%** |
| Peak KV cache | ~18% | **~9.5%** | roughly halved |
| Mean prompt tokens | 471 | **541** | +15% |

**ITL did not move, and that is the confirming detail** — prefix caching accelerates prefill only; decode still streams all the weights per token regardless. Throughput at concurrency 1 was also unchanged, because a single request's wall time is decode-dominated. The benefit appears only when prefill *competes* with decode: +0%, +20.5%, +33.8%.

Two findings strengthen it: the shared prompts were **15% longer** (the winning arm did more nominal work), and **peak KV cache roughly halved**, because shared prefix blocks are stored once and referenced by every sequence. Prefix caching buys memory as well as speed.

### 7.3 Model comparison

**Unquantized 1.5B — does not fit.**

| | 3B-AWQ | 1.5B fp16 |
| --- | --- | --- |
| Parameters | 3.09B | 1.54B (**half**) |
| Weights | 1.954 GiB | **2.887 GiB (+47.7%)** |
| KV cache | 0.98 GiB | **0.07 GiB (−92.9%)** |
| Serves `--max-model-len 4096`? | Yes | **No — fails to start** |

**The model with half the parameters uses 48% more memory and gets 93% less KV cache.** KV cache is the *remainder* of the budget, so a 0.93 GiB increase in weights did not cost 0.93 GiB of cache — it consumed nearly all of it.

**1.5B-AWQ — the comparison that matters.**

| | 3B-AWQ | 1.5B-AWQ |
| --- | --- | --- |
| Weights | 1.9542 GiB | **1.1018 GiB** |
| KV cache tokens | 28,480 | **69,616** |
| Max concurrency @ 4096 | 6.94x | **17.00x** |
| **Reliable SLO ceiling** | conc **2** | conc **5** |
| **Throughput there** | **64.7 tok/s** | **278.8 tok/s** |

**4.3x more reliably-SLO-compliant throughput** — larger than every other optimization in this project combined.

**And the two models fail differently.** The 3B goes +22% → **−15%** margin between concurrency 2 and 3: a cliff. The 1.5B slopes: +50 → +35 → +19 → +7. A system that falls off a cliff needs a much larger safety factor than one that degrades gradually.

**What this does not measure: answer quality.** Every number favours the 1.5B; none capture why someone might still choose the 3B. The honest form is conditional — *if 1.5B-class quality is acceptable for the use case, it is 4.3x better on this hardware.* That judgement is not the benchmark's to make.

### 7.4 The scaling law that fell out

The model swap changed weights and parameter count by *different* ratios, because AWQ changes bytes-per-parameter without changing parameter count. Both phases were predicted correctly, from different quantities:

| Phase | Bound by | Scales with | Predicted | Measured |
| --- | --- | --- | --- | --- |
| Decode (ITL) | memory bandwidth | **weight bytes** | −39% | **−37.6%** |
| Prefill (TTFT) | compute | **parameter count** | ~−50% | **−50.7%** |

Fitting `ITL = a × weight_GiB + b` across both models:

```
3B:    25.0 ms = a × 1.9542 + b
1.5B:  15.6 ms = a × 1.1018 + b
       →  a = 11.03 ms/GiB,  b = 3.45 ms
```

**Decode ≈ 11.03 ms per GiB of weights, plus 3.45 ms of fixed overhead** (KV reads, sampling, kernel launches). The fixed term is 13.8% of the 3B's ITL. And `1 GiB / 11.03 ms` implies **~97 GB/s of effective memory bandwidth — roughly half the card's ~192 GB/s rating**, plausible for a thermally-limited workload.

**A model built in Stage 1 to explain a memory budget correctly predicted a model swap it was never derived from, to within 3%.**

---

## 8. Three incidents

### 8.1 Forced KV cache eviction

Covered in §3, Boundary 4. Engine evicted; gateway accounting exact to the token; zero errors; evictions invisible to callers.

### 8.2 Deliberately broken structured-output forwarding

Covered in §3, Boundary 1. A field allowlist inserted into the gateway produced **14 test failures**, including 12 fields nobody would have thought to check. Reverted, with a comment at the site warning against reintroducing field filtering.

### 8.3 vLLM killed mid-load-test

`docker compose kill vllm` (SIGKILL) during a running load.

| Behaviour | Outcome |
| --- | --- |
| In-flight requests | 12 of 16 failed; 4 completed before the kill |
| Gateway process | **survived** — never crashed or hung |
| New requests | honest **502**, not a hang |
| `/health` | reported `vllm: unreachable` |
| Recovery | **automatic**, no gateway restart |

Two upstream failures were correctly distinguished: `[Errno -2] Name or service not known` after the kill (Compose removes the container from the network, so DNS stops resolving) versus `All connection attempts failed` after restart (name resolves, nothing listening). A gateway flattening both to "upstream error" would lose real diagnostic information.

**A design flaw was found and fixed.** The gateway flipped to `unhealthy` ~45 s after the kill, because `/health` returned 503 whenever vLLM was unreachable and the container healthcheck treats non-200 as failure. **The gateway was marked unhealthy because something else died.** To an orchestrator a failing liveness probe means "restart this container" — which helps nothing and drops every in-flight request that was about to succeed when the engine returned.

Fixed by splitting the endpoints: **`/health` is liveness** (always 200 while the process runs, upstream status as information) and **`/ready` is readiness** (503 when the gateway cannot serve). Failing readiness removes a pod from service without killing it, so it rejoins automatically — exactly the recovery behaviour observed.

---

## 9. Honest limitations

**These numbers do not transfer to a Linux server unchanged.**

1. **WSL2 forces `pin_memory=False`** — logged at every startup. Host-to-device transfers are unpinned and slower than on native Linux.
2. **WDDM reserves ~0.79 GiB** of the 4 GiB card for the Windows compositor, capping `--gpu-memory-utilization` at ~0.80. The same GPU on Linux in TCC mode would have more KV cache.
3. **Thermal limiting is the binding constraint under sustained load.** The card boosts to 1,372 MHz, reaches 87 °C in ~3 seconds, and settles at **712 MHz** — a 48% clock reduction inside a single 21-second run. Memory clock does not throttle, which is why decode is insulated and prefill is not.
4. **Effective memory bandwidth ~97 GB/s**, about half the card's rating.
5. **The benchmark client runs in a container** reaching the server over `host.docker.internal`. Constant across all runs, so comparisons are unaffected.
6. **Answer quality was never measured.** Every performance conclusion involving the 1.5B is conditional on quality being acceptable, and that was not tested.
7. **`p95` on 40 samples is still coarse.** It is the 39th value of 40. Effects smaller than ~20% should not be trusted from a single run.

**Known security gaps, scoped out of v1 deliberately:** the product dashboard is unauthenticated, Grafana allows anonymous admin, and the chat UI holds the API key in `localStorage` where any script on the origin could read it. All bind to localhost. All would need addressing before facing a network.

---

## 10. What real scale would need

**Move off Windows.** WDDM's VRAM reservation and WSL2's unpinned memory are pure overhead. A Linux host with the same silicon would have more KV cache and faster host-device transfers — and would make dcgm-exporter viable, closing the hardware-metrics gap in Grafana.

**A GPU that does not thermally collapse.** A 48% clock reduction within 21 seconds makes every sustained measurement a measurement of the cooling system. Datacenter cards hold their clocks; this one does not, and it is why the protocol needs a thermal warm-up at all.

**Replace SQLite.** It is correct here — single writer, atomic increments, WAL for concurrent reads — but it assumes one gateway process. Horizontal scaling needs Postgres or Redis for budgets, because two gateway replicas against one SQLite file would corrupt the accounting the Boundary 4 test was built to protect.

**Reserve budget rather than deduct after.** The current design checks before and deducts after, permitting overshoot bounded by `(prompt + max_tokens) × requests_in_flight`. Measured at 0.8% single-stream; at concurrency 32 the bound is ~2,700 tokens. Fine for a demo, not for billing.

**Separate readiness from liveness at the orchestrator.** Already implemented as endpoints; Kubernetes should wire `/health` to `livenessProbe` and `/ready` to `readinessProbe`, so an engine outage removes a pod from service without restarting it.

**Real authentication.** API keys in a database with no hashing, no rotation, no scopes. A production system needs an auth provider, hashed key storage, and a revocation path.

**Decide the model question with quality data.** The 1.5B-AWQ is 4.3x better on every metric this project measured. Whether it is *good enough* requires an evaluation harness this project does not have — and that, not throughput, is the deciding input.

**Serve a shared system prompt.** The single highest-value finding here was free: a workload sharing a prefix more than doubled SLO-compliant concurrency at no latency cost. Any real deployment with a system prompt should verify it is actually being shared, and watch the prefix-cache hit rate as a first-class metric rather than a curiosity.

---

## 11. What this project got wrong, and what that taught

Recorded because the corrections are more instructive than the conclusions.

| Prediction | Reality | Lesson |
| --- | --- | --- |
| Driver CUDA 12.7 is "newer than vLLM ships" | Image required ≥ 12.8; container never started | The image's declared floor is the constraint, checked before your code runs |
| `--gpu-memory-utilization 0.90` fits | Only 3.21 of 4.0 GiB free; engine refused to start | vLLM checks **free** memory, not total. `nvidia-smi` showing 0 MiB used is not evidence |
| Weights ~2.2 GiB → KV ~15–17k tokens | 1.95 GiB → **28,480 tokens** | KV cache is the *remainder*; every other error lands on it |
| `hf-cache` volume covers the compile cache | It does not; every launch paid ~60 s | Verify a cache volume covers the path the cache uses |
| A benchmark client needs no GPU | vLLM's CLI dies building its argument parser without `libcuda` | A tool can need a resource for reasons unrelated to your task |
| Stub returning the right bytes is enough | `content=bytes` is *already read*; `aiter_raw()` raised | A stub must reproduce the **mode** of the real dependency, not only its payload |
| Healthcheck can call `python` | Ubuntu image has only `python3`; probe failed silently for 5 minutes | A healthcheck that cannot execute is indistinguishable from a service that is not ready |
| `waiting > 0` means saturation | It also counts requests merely awaiting the next scheduler step | Queue *time* is the instrument; queue *depth* is a transient |
| ITL grows noticeably with context | Only 2.4% across an 8.5x context increase | Decode is weight-dominated; attention is ~3% of the read at 1,709 tokens |
| Raising `--max-num-seqs` is a win | +34% throughput, **zero** SLO benefit, 24x worse ITL p99 | An optimization is only meaningful relative to a stated objective |
| Concurrency 4 passes the SLO | Passed by 24 ms once, failed by 30 ms later | Report the margin; a verdict near a threshold is noise |
| `compose up gateway` touches only the gateway | It reverted `vllm` to the base config; a whole ramp 404'd | Compose reconciles **every** service against the files given |

### Added during finalization and v2

| Prediction | Reality | Lesson |
| --- | --- | --- |
| Cool the card to ≤50 °C before measuring | Unreachable — it idles at 74 °C with clocks pinned — **and unnecessary**, since the warm-up already forces steady state | A threshold invented rather than measured will be chased for no benefit. Ask what the control is *for* before enforcing it |
| Build cache 3–8 GB, dangling images 0.5–2 GB | **21.37 GB** and **0 B** | `--build` retags in place, so rebuild debris goes to the *build* cache, not to dangling images. The intuition runs backwards |
| `vllm-cache` holds 0.3–1.5 GB | **17.32 MB** | Compiled graphs are metadata, not weights. Its value is the 40 s it saves, not disk |
| `hf-cache` ≈ 7 GB for 7 GB of models | **14.17 GB** — a second cache (`xet`, 6.0 GB) nobody had looked at | `docker system df` shows volume totals only. `du` inside the volume or the size is unexplained |
| A verified heat soak makes runs comparable | Run 3 came in **20–25% above** Run 1 despite both reaching 86–87 °C | Die temperature is not the whole thermal state. The heatsink saturates, so the clock the card *holds* at 87 °C falls — and `nvidia-smi` cannot see it. **This is the mechanism behind the "24% session drift" measured in Stage 11 and never explained** |
| Check `watch_gpu` for `0x20` before measuring | Checking took 66 s, the card shed 13 °C, and the measurement came in 13–18% high | **Verifying the soak destroys it.** Chain the commands and verify post-hoc |
| The second identical request reports `cached_tokens` | `prompt_tokens_details: null`, from vLLM directly | And working through it showed the *plan* was flawed anyway: `cached_tokens` arrives in the response, after the work is done, so it could never have driven admission control. Its absence costs observability, not the mechanism |
| Prefill scales linearly with prompt length | Attention is **43% of prefill at 22,586 tokens**; a fitted `n²` term matches all three measurements within 5% | A model that fits at one scale is not a law. The linear approximation was true at chat lengths and wrong by 2x at the context limit |
| The capacity curve will show a **knee** from prefix-cache collapse | **No knee.** Hit rate held 83–89% from 10 to 60 users | Admission control caps concurrency, which caps how many conversations are *progressing*, which keeps the working set inside the pool. **Load shedding protects the prefix cache as a side effect** — two independently designed mechanisms reinforcing |
| The alert rule's metric is `vllm:gpu_cache_usage_perc` | It is `vllm:kv_cache_usage_perc` | This project had *already recorded* that exact error. Reading the warning does not prevent repeating it; reading `/metrics` does |
| Simulated users can share one API key | It shed requests at **four users** — per-key limits are keyed on the key, so the "population" was one tenant correctly throttled | The limit was right and the experiment was wrong. A load generator that does not model identity measures the wrong limit |
| `--max-num-seqs 32` will cost some KV cache | 69,760 tokens at both 8 and 32 | Only `cudagraph_capture_sizes` grew. Batch-size caps cost graph memory, not cache |

**And the one that went right for a reason worth keeping:** the per-token KV arithmetic — `2 × layers × kv_heads × head_dim × 2 bytes` — predicted 36 KiB/token for the 3B before the engine ever started (measured 36.1), 28 KiB/token for the 1.5B (vLLM computed 28.0), and combined with weight-byte reasoning predicted a model swap's decode speedup to within 3%.

**A small amount of arithmetic done before measuring is worth more than a large amount of measurement done without it** — because it tells you when a measurement is wrong.

---

## 12. v2 — from an API to a service

v1 answered *"can this hardware serve an API, and is it hardened against four
named failure boundaries?"* v2 asks the harder question: **how many real people
can talk to it at once, and what happens at the edge of that?**

### 12.1 What was added

| Component | Why |
| --- | --- |
| **Admission control** (`gateway/admission.py`) | v1 accepted everything. Stage 9 measured TTFT climbing past 13 s and still rising - an unbounded queue turns "too much load" into "everyone has a bad time" |
| **Per-key fairness** | A global limit alone lets one caller take every slot |
| **Context policy** (`gateway/context.py`) | v1 had none: a long enough conversation hit a hard 400 mid-chat with no way forward |
| **Gateway metrics** (`/metrics`) | Queue depth, shed reasons, and client-observed TTFT are facts the engine cannot know, because they happen before a request reaches it |
| **Alert rules** | Selected to *predict* failure rather than report it |
| **`chat_sim.py`** | `ramp.py` measures the engine well and models chat badly: no think time, no accumulating context, no shared prefix |
| **Chat UI rework** | Answers were cut off mid-sentence and markdown rendered as literal asterisks |

### 12.2 The decision that shapes the whole design

**The gateway is the only limiter.** `--max-num-seqs` was deliberately raised to
32 - far above the gateway's limit of 6 - so the engine's own cap can never
silently shape traffic the gateway believes it is controlling.

This is not a throughput decision. Under the SLO, raising `--max-num-seqs` is
worth exactly nothing: the reliable ceiling is concurrency 6 at both 8 and 32,
and concurrency 12 fails by 36% either way. It buys peak throughput
(367 → 686 tok/s) that cannot be used without abandoning the latency target.
The reason to ship it is that **an invisible second limiter is how a capacity
model becomes wrong** - and this project already has a documented incident of a
config silently reverting underneath a measurement.

### 12.3 The queue depth arithmetic, which is the non-obvious part

A queued request's TTFT includes its queue wait. At the concurrency limit the
service completes roughly 1.4 requests/second, so **every queue slot adds ~0.7 s
to the TTFT of whoever occupies it**. Two slots consume the entire 1.5 s budget.

The honest consequence: **you cannot hold a tight TTFT SLO and also queue
deeply.** The queue exists to absorb sub-second arrival jitter, not to store
backlog, and the queue *timeout* - not the queue depth - is the term that bounds
p95 TTFT under overload:

```
worst-case client TTFT  ≈  queue_timeout + engine_TTFT
```

That is a design knob, and it was measured wrong first: at a 2.0 s timeout, p95
TTFT under overload settled at a beautifully flat **2,070 ms** - bounded, but
above the 1.5 s objective, because a request that waits two seconds has already
spent the whole budget before the engine starts.

### 12.4 The chat UI, and two bugs that were user-visible

**Answers were cut off mid-sentence.** `max_tokens` was pinned at 512 and
nothing read `finish_reason`. A model asked for a detailed answer simply
stopped, with no signal that it had been truncated rather than finished -
verified directly: `finish_reason: length`, ending inside a Python code fence.
Now the cap is higher, `finish_reason == "length"` triggers bounded automatic
continuation, and hitting the bound says so.

**Markdown rendered as literal punctuation.** The reply went into `textContent`,
so the model's headings, bold text, numbered lists and fenced code blocks
appeared as raw `#`, `**` and backticks. There is now a small renderer in the
page - written out rather than pulled from a CDN, because the page is served
same-origin by an authenticated gateway and a third-party script tag would undo
that for a formatting convenience.

It is covered by 25 tests that **extract the functions from the shipped file**
rather than copying them, so the test cannot pass while the real renderer is
broken - the failure mode Stage 5 already hit with a stub that reproduced the
wrong delivery mode. Five of those tests are security cases: model output is
untrusted input, and a prompt that makes the model echo `<script>` or a
`javascript:` URL must not reach `innerHTML` as markup.
