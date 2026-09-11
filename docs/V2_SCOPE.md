# v2 Scope — Multi-User Chat Service

v1 asked *"can this hardware serve an API, and is it hardened against four named failure boundaries?"* Answered: yes, with the evidence in `FINAL_REPORT.md`.

**v2 asks a harder question: how many real people can talk to it at once, and what happens at the edge of that?**

That is a different engineering problem. v1's load was synthetic, uniform, and arrived all at once. Real chat traffic is bursty, has long think time, accumulates context across turns, and includes users who behave badly. v1 has no admission control, no backpressure, no fairness, and no strategy for a conversation that outgrows the context window — so it would degrade unboundedly rather than gracefully.

**Finalized model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`** (revision to be pinned — see `FINALIZATION_PLAN.md` Phase 5.1).

---

## The capacity model this is built on

Derived from v1's measurements rather than assumed.

**Memory.** The 1.5B-AWQ holds **69,616 tokens** of KV cache (1.859 GiB at 28 KiB/token) with 1.1018 GiB of weights, at `--gpu-memory-utilization 0.78`.

**PagedAttention decouples context length from memory per request.** `--max-model-len` is a ceiling, not an allocation — blocks are allocated 16 tokens at a time as a sequence grows. So the context window can be raised without reserving memory:

| `--max-model-len` | Worst-case concurrency | Viable? |
| ---: | ---: | :-- |
| 4,096 (v1) | 17.0x | yes |
| 8,192 | 8.5x | yes |
| 16,384 | 4.2x | yes |
| **32,768** (model's native limit) | 2.1x | **yes** — 32,768 × 28 KiB = 896 MiB < 1,859 MiB available |

**Realistic concurrency, since conversations do not sit at full context:**

| Average conversation | Concurrent in-flight requests |
| ---: | ---: |
| 1,000 tokens | 69 |
| 2,000 tokens | 34 |
| 4,000 tokens | **17** |
| 8,000 tokens | 8 |

**Simultaneous users, accounting for think time.** A chat user sends a message every ~45 s; a request takes ~6 s. Duty cycle ≈ **12%**, so one concurrent slot serves roughly 8 users.

**But compute binds first.** Each user generates ~256 tokens per message every ~51 s ≈ **5 tok/s sustained**. Against measured throughput:

```
345 tok/s (measured at conc 8)  ÷ 5  =  ~69 users
400-550 tok/s (Phase 3 target)  ÷ 5  =  ~80-110 users
```

**First-pass estimate: ~80–110 simultaneous chat users, compute-bound.**

### CORRECTION — the first-pass estimate is wrong, and the reason is the interesting part

The figure above assumes each turn costs ~256 tokens of decode plus a *trivial* prefill, because prefix caching makes the conversation history free. **That assumption holds for one user and fails at scale.**

**The prefix cache is one shared pool of 69,616 tokens. It is not per-user.**

```
100 users × 4,000-token conversations  =  400,000 tokens of prefix wanted
KV cache available                     =   69,616 tokens
                                          -> only ~17 conversations fit
```

Blocks are evicted LRU. With 100 users at 45 s think time, ~2.2 requests arrive per second, so a conversation's cached prefix survives roughly **8 seconds** before other conversations push it out — against a 45 s think time. **Essentially every turn becomes a cache miss.**

And a miss is an order of magnitude more expensive:

| Per turn | Prefill work | GPU-seconds | Users supportable |
| --- | --- | ---: | ---: |
| **Cache hit** | ~50 new tokens | ~0.8 s | **~62** |
| **Cache miss** | 4,000 tokens re-prefilled | ~7.4 s | **~7** |

**Revised estimate: ~17–35 simultaneous users**, bounded by `KV_cache ÷ average_conversation_length` rather than by raw compute.

**This reframes v2's central question.** The binding constraint on a multi-user chatbot is not throughput and not memory-for-running-sequences — it is **whether the working set of active conversations fits in the shared prefix cache**. The system has a self-limiting equilibrium: capacity holds while the working set fits, then degrades as hit rate collapses, and the degradation is 9x rather than gradual.

Stage 6's finding — prefix caching keeping TTFT flat across a 55x context increase — **held because there was exactly one user.**

**Levers this opens, all to be measured rather than assumed:**

| Lever | Mechanism | Cost |
| --- | --- | --- |
| Shorter conversations | Truncation or summarisation shrinks the working set, so more users stay cached | Loses conversation history |
| More KV cache | Only available by shrinking weights further, or a bigger card | Model quality |
| CPU KV offload | `--swap-space` already reserves 2 GiB of host RAM; some vLLM builds can offload prefix blocks | PCIe transfer cost — must be measured, not assumed to help |
| Admit fewer users | Accept the ceiling and shed load above it | The honest option, and what admission control is for |
| Session affinity | If ever scaled to multiple replicas, routing a user to the replica holding their prefix | Complexity; irrelevant on one GPU |

**Both estimates above are hypotheses. Phases A and B exist to measure which is right — and the gap between 17 and 110 users is the single most consequential open question in v2.**

---

## What v1 lacks, and why each gap matters at 100 users

| Gap | v1 behaviour | What happens at scale |
| --- | --- | --- |
| **No admission control** | Unbounded queue | Stage 9 measured TTFT climbing to 13 s and still rising. Every user waits; nobody is served well |
| **No backpressure** | Accepts everything | Clients cannot distinguish "slow" from "overloaded" and will not back off |
| **No per-key fairness** | First-come-first-served | One user with a script starves everyone else |
| **No context management** | UI resends full history | A long conversation eventually exceeds `max-model-len` and hard-fails mid-chat |
| **No load shedding** | Degrades uniformly | Better to serve 80 users well and reject 20 than serve 100 badly |
| **No alerting** | Dashboards only | Nobody is watching Grafana at 3 a.m. |
| **Single gateway process** | SQLite, one writer | Cannot scale horizontally without corrupting budget accounting |

---

## Phases

Each phase follows v1's pattern: predict in writing, build, measure, record what was wrong.

---

## KV cache lifecycle for a conversation — the mechanism v2 is built on

**Two kinds of block, routinely confused.**

**Running-sequence blocks** are held while a request is in flight and freed the instant it completes. Never retained.

**Prefix-cache blocks** move, on completion, to a free list where they remain **content-addressed by hash**. A later request whose prompt starts with the same tokens reuses them. They are evicted **LRU** when the scheduler needs space.

**Therefore an idle or disconnected chat consumes no capacity.** Its blocks are already free and allocatable by anyone. Idle users never block new users — they only risk losing their cache. This is a much better property than it first appears and should be stated in the README, because the intuitive fear ("100 idle users will exhaust my GPU") is simply wrong.

**On return, vLLM hashes the prompt block by block and matches the longest cached prefix:**

| Cache state | Behaviour | Cost, 4,000-token conversation |
| --- | --- | --- |
| Fully cached | Prefill skips history; only new tokens processed | ~0.8 GPU-s |
| **Evicted** | **Entire history re-prefilled from scratch** | **~7.4 GPU-s** |
| Partially cached | Reuses the matching prefix, prefills the remainder | between |

**The hash is a chain** — block N's hash incorporates blocks 1…N−1 — so a hit requires a *contiguous prefix from the start*. Evicting one early block invalidates everything after it. Combined with LRU-by-recency, a conversation's blocks tend to be evicted together, making the outcome closer to all-or-nothing than to graceful partial reuse.

**Nothing surfaces this.** No error, no warning, no log line. The request simply costs 9x more. The only signal is the prefix cache hit rate — which is exactly the Boundary 3 metric this project already argued a monitoring wrapper would drop.

### Two design consequences

**1. Check whether vLLM reports per-request cache hits.**
Stage 4's output showed `prompt_tokens_details` present but empty — that is where OpenAI-compatible APIs place `cached_tokens`. If this build populates it, the gateway can record cache-hit ratio **per request**, the product dashboard can show it **per key**, and we would know precisely which users pay the miss penalty. **This would be the single most valuable metric v2 could add**, and it requires no new infrastructure. Verify early.

**2. Cache-aware admission control.**
A cache-miss request costs roughly 9x a hit. Admission control that counts requests will systematically under- or over-admit depending on cache state. If `cached_tokens` is available, the gateway can weight each request by its *uncached* prefix length and **admit by predicted GPU-seconds rather than by request count**. That converts admission control from a crude concurrency cap into a real cost model.

### What cannot be controlled

There is no per-user cache pinning in vLLM, and `--swap-space` governs preemption swapping rather than prefix retention. The available levers are architectural, not configuration:

- **Shorten conversations** (Phase E) — smaller working set, more conversations stay cached. This is the primary capacity lever.
- **Accept misses and size prefill capacity for them** — honest, and sets a much lower ceiling.
- **Session affinity** — only meaningful with multiple replicas; irrelevant on one GPU.

---

### Phase A — Realistic multi-user load generator

**The measurement that matters most is prefix cache hit rate as a function of user count.** Everything else in this scope depends on where that curve falls off. Instrument it first.


`ramp.py` sends uniform synchronous load. Real chat does not look like that.

**Build `load_testing/chat_sim.py`:** simulated users with **Poisson arrivals**, **log-normal think time** between turns, **multi-turn conversations that accumulate context**, and a mix of conversation lengths. Reports per-user experience (TTFT distribution as *users* see it) rather than per-request aggregates.

**This is the instrument for every phase that follows.** The capacity model above is untested until this exists.

**Predict before running:** the ~80–110 user figure, and that prefix caching will make multi-turn far cheaper than v1's unique-prompt benchmarks implied.

### Phase B — Find the real user ceiling

Ramp *simulated users*, not concurrent requests, against the 32k-context config. Find where p95 TTFT (measured per user turn) breaks the SLO.

**The key question, restated after the correction above:** the estimates differ by 6x (17 vs 110 users) depending entirely on whether conversations stay in the prefix cache. Ramp users while watching **prefix cache hit rate** alongside p95 TTFT.

**Predicted shape: a knee, not a slope.** Capacity should hold roughly flat while the working set fits, then fall sharply as hit rate collapses and every turn starts re-prefilling its whole history. **Finding that knee — and showing it is a cache-capacity effect rather than a compute effect — is v2's headline result.**

If the curve is instead a gentle slope, the model is wrong somewhere and that matters more than the number.

### Phase C — Admission control and backpressure

The single largest gap. A saturated service must **refuse work it cannot do well** rather than accept everything and serve it badly.

- Track in-flight requests and queue depth in the gateway
- Reject with **503 + `Retry-After`** past a configured threshold
- Threshold derived from Phase B's measured ceiling, not guessed
- Expose queue depth and rejection rate as metrics

**Measure:** with admission control on, does p95 TTFT stay bounded as offered load exceeds capacity? That is the whole point — the graph should go flat instead of climbing.

### Phase D — Per-key fairness

One key must not monopolise. Per-key in-flight limits, and a fair-queuing policy so a burst from one user does not starve others.

**Measure:** one abusive key at 50 concurrent alongside ten normal users. Do the normal users' p95 TTFTs stay within SLO?

### Phase E — Context management for long conversations

At 32k context a conversation eventually overflows. Options, to be chosen with reasoning: reject with a clear error, drop oldest turns (sliding window), or summarise older turns. Each has a different failure mode and a different cost.

**Measure:** conversation quality and TTFT across 50+ turn conversations. Watch prefix cache hit rate — a sliding window *breaks* the shared prefix and could destroy the caching benefit measured in Stage 6. That interaction is the interesting part.

---

## Test strategy — covering every traffic type without wasting days

Three dimensions, tested independently before being combined. **One simulator, many named scenarios, results to JSON, one command to run the matrix.**

### Dimension 1 — traffic shape

| Shape | What it exercises | Why it matters |
| --- | --- | --- |
| **Steady** (Poisson, fixed rate) | Baseline capacity | The number that goes in the README |
| **Ramp** (rate increases over time) | Where the knee is | Finds the ceiling in one run instead of many |
| **Burst / spike** (10x for 30 s) | Admission control | The graph that proves graceful degradation |
| **Thundering herd** (all users at once) | Cold-start + queue behaviour | What happens after an outage, when every client reconnects simultaneously |
| **Diurnal** (slow sine over minutes) | Recovery between peaks | Does it come back cleanly, or stay degraded? |
| **Long-tail mix** (mostly short, a few very long) | Head-of-line blocking | One 32k prompt among 20 short ones — the realistic worst case |
| **Adversarial** (one key hammering) | Fairness | Whether one user can starve the rest |

### Dimension 2 — conversation state

The cache lifecycle above makes this a first-class variable, not a detail:

| State | How to produce it | Expected cost |
| --- | --- | --- |
| **Cold** (new user) | Fresh conversation, unique opening | Full prefill of a short prompt |
| **Warm** (returning, cached) | Same conversation, short think time | ~0.8 GPU-s per turn |
| **Cold-return** (evicted) | Same conversation after enough other traffic to evict it | **~7.4 GPU-s** — the case that breaks naive capacity models |
| **Growing** (long multi-turn) | 20–50 turns accumulating context | Rising per-turn cost; watch prefix hit rate |
| **Overflowing** (exceeds window) | Turns until context is exceeded | Whatever policy Phase E chooses; never a crash |

**Cold-return is the scenario that matters most and the one a naive test suite would never generate**, because it only appears when *other* traffic evicts your conversation. It must be constructed deliberately: run user A's conversation, then drive enough other users to push A out of cache, then have A send another turn and measure.

### Dimension 3 — failure injection

The 14 cases in Phase F, each run against a *steady* baseline so the failure's effect is separable from load variation.

### The simulator

**`load_testing/chat_sim.py`** — the instrument for all of the above:

```
--users N                 simulated concurrent users
--pattern                 steady | ramp | burst | herd | diurnal | longtail | adversarial
--turns N                 conversation length before a user departs
--think-mean / --think-sigma   log-normal think time (real users are not uniform)
--conversation-mix        distribution of conversation lengths
--duration                wall-clock run time
--seed                    reproducible arrivals and prompts
--out results.json        per-turn records for later analysis
```

**It must report per-USER experience, not per-request aggregates.** A user cares about the p95 of *their* turns; averaging across all requests hides the user who got unlucky every time. Record per-turn: TTFT, ITL, E2E, prompt tokens, cached tokens (if available), HTTP status, and which turn number it was.

**`load_testing/scenarios/*.json`** — named, version-controlled scenario definitions, so "the burst test" means exactly one thing across runs and across machines.

**`load_testing/run_matrix.ps1`** — runs the suite and writes a summary table. The whole matrix should be one command, or it will stop being run.

### Running the matrix gracefully

Rules that keep a long test session from becoming a repair session:

1. **Order by destructiveness.** Read-only traffic shapes first, then failure injection, then anything touching the database or killing containers.
2. **Snapshot before destructive tests.** Copy `gateway-data` before the disk-full and DB-locked cases; those can leave state that needs manual repair.
3. **Every scenario has a hard timeout.** A hung run should fail, not sit overnight.
4. **One variable at a time.** Never change traffic shape and config in the same run — v1 has three separate incidents caused by exactly that.
5. **Fixed seeds.** Arrivals, think times and prompt content must be reproducible, or scenarios are not comparable across runs.
6. **Respect the 24% drift.** Compare arms measured back to back; insert cooling breaks between blocks if absolute numbers matter.
7. **Record the config with every result.** Every JSON output should embed the engine flags, model, revision and driver version that produced it — otherwise a results file six weeks from now is uninterpretable.
8. **Discard the first request of every scenario.** Cold-start cost was measured at 3,292 ms versus 310 ms in v1.

---

### Phase F — Systematic failure injection

v1 broke three things deliberately. v2 should work a matrix:

| Failure | Expected behaviour |
| --- | --- |
| Engine OOM / preemption storm | Graceful degradation, accurate accounting |
| Engine crash mid-conversation | 502, automatic recovery, conversation resumable |
| Gateway restart under load | In-flight requests fail cleanly; budgets intact |
| Slow client (reads one byte/s) | Does not hold a slot indefinitely |
| Client disconnects mid-stream | Slot released, usage recorded |
| Burst 10x over capacity | 503s, not 30-second TTFTs |
| Database locked / disk full | Fails closed on billing, not open |
| Thermal throttle mid-load | Detected and alerted, not silent |

### Phase G — Alerting

Prometheus alert rules on the metrics that predict failure rather than report it: KV cache > 85%, preemptions non-zero, queue time p95 > threshold, prefix hit rate collapse, GPU below expected clocks (the Stage 2 failure that cost 11.6x and was invisible in every engine metric).

### Phase H — Horizontal scale path

SQLite is one writer. Document — and ideally demonstrate — what changes for two gateway replicas: budgets in Postgres or Redis, atomic decrements, and a test proving the Boundary 4 accounting guarantee still holds under concurrent writers.

### Phase I — Publishable artifact

README with the capacity model, the measurement methodology, the reproduction steps, and the honest limitations. **The wrong-predictions table from `FINAL_REPORT.md` is the most valuable part of this repository for a reader** and should be prominent, not buried.

---

## What carries over unchanged from v1

Do not rebuild or re-litigate these:

- **Boundary 1** — untyped-dict forwarding, 22 contract tests, proven to bite (14 failures under sabotage)
- **Boundary 2** — pinning discipline (extend to the 1.5B's revision)
- **Boundary 3** — Prometheus scraping vLLM directly, 13-panel dashboard
- **Boundary 4** — `finally`-block accounting, verified exact under forced eviction
- **`/health` vs `/ready`** — liveness/readiness split
- **The measurement protocol** — all ten rules in `optimization_results.md`
- **The scaling law** — decode ∝ weight bytes, prefill ∝ parameter count
- **Session drift ~24%** — compare only consecutive pairs
- **Compose override trap** — always carry `-f` flags
- **Model-name resolution** — never hardcode; ask `/v1/models`
