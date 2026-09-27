# Failure Matrix

Fourteen ways this service can be hurt, each with the behaviour **expected before
testing** and the behaviour **measured after**.

A case that behaves badly is a finding, not a failure. The purpose is to know
what happens, not to assert that something convenient happens — a matrix where
every row says PASS is usually a matrix whose expectations were written
afterwards.

**Reproduce:**

```powershell
python load_testing/failure_matrix.py --case all
.\benchmarks\run_evidence_suite.ps1              # everything, including 2, 5, 6, 9
```

The generated tables for the latest run are in [RESULTS.md §6](RESULTS.md#6-failure-injection).

Container-level cases (5, 6) are driven from the shell and shown inline below.

---

## The configuration under test

| | |
| --- | --- |
| Engine | `vllm/vllm-openai:v0.11.0`, `Qwen/Qwen2.5-1.5B-Instruct-AWQ` @ `3ecffa0c…` |
| Context window | 32,768 tokens |
| KV cache | 69,760 tokens (~1.86 GiB at 28.0 KiB/token) |
| Engine scheduler | `--max-num-seqs 32` (deliberately above the gateway's limit), `--max-num-batched-tokens 2048`, `--long-prefill-token-threshold 512` |
| Gateway admission | 10 in flight, 12 queued, 1.0 s queue timeout, dynamic per-key share (cap 5), cost weighted by the uncached part of the prompt |
| Output bound | `max_tokens` default 1,024 if absent, ceiling 2,048 |
| SLO | p95 TTFT < 1.5 s **for chat-length prompts (~500–1,000 tokens)** |

The SLO qualifier is not pedantry. A 22,586-token prompt costs 13 s of prefill
on this hardware; no admission policy makes that fit a 1.5 s target, and a
matrix that reported it as a failure would be reporting arithmetic.

---

## Results

**14 pass, 1 documented limitation (13, thermal), 0 unexplained failures.**
Measured on the final configuration in the evidence run of 2026-09-27.

| # | Case | Expected | Measured | |
| --- | --- | --- | --- | :---: |
| 1 | **Burst 10x over capacity** (100 simultaneous) | admitted latency bounded; excess refused with 503 + `Retry-After` | 12 served, **88 refused**, `Retry-After: 2`; served p95 TTFT **1,441 ms** | ✅ |
| 2 | **One abusive key at 50 concurrent** | other users stay within SLO | normal users: **94% of messages within the SLO** from first send, retries included; abuser refused **2,014 of 2,079 (96.9%)** | ✅ |
| 3 | **Single ~20k-token prompt** | short requests keep flowing | short requests **99 ms → 636 ms** while the 20k prompt prefilled (it was 11–13 s before `--long-prefill-token-threshold`); the big prompt itself served, 20,448 tokens | ✅ |
| 4 | **Conversation outgrows the window** | 200 with oldest turns dropped and reported; never a hard 400 | HTTP **200**, `X-Context-Trimmed-Messages: 27`; ~95,488 tokens sent → engine saw **9,726** | ✅ |
| 4b | **A single message larger than the window** | refused before it costs a slot | **413** before admission, decided on the engine's exact token count (`/tokenize`), never on a character estimate; a body over 4 MiB is refused before it is even parsed | ✅ |
| 5 | **Engine crash mid-stream** | 502, auto-recovery, gateway stays Running (liveness) while going NotReady (readiness) | crash detected **1.2 s**, automatic recovery **79.4 s**; `/health` **200 throughout**, `/ready` 503→200; 149×502, 3×500, 4×200; **156 of 156** requests in the ledger, **drift 0** | ✅ |
| 6 | **Gateway restart under load** | in-flight fail cleanly, budgets intact | back in **4.8 s**; 5 in-flight requests dropped their connection (no hang, no silent truncation), 4 served; ledger and budgets **survived** | ✅ |
| 7 | **Client disconnects mid-stream** | slot released, usage still recorded | in-flight returned to **0**; ledger row written from the generator's `finally` | ✅ |
| 8 | **Slow client (~1 chunk/s)** | cannot hold a slot indefinitely; others unaffected | **5/5** concurrent normal requests succeeded, p50 TTFT **127 ms**; bounded by `GATEWAY_STREAM_TIMEOUT_S` | ✅ |
| 9 | **Prefix cache thrash at high user count** | measured, not assumed | hit rate held **63–95%** across every run, 10 to 80 users. **No collapse**: see the capacity model for why | ✅ |
| 10 | **KV pressure / preemption** | accounting exact | 12 requests × 700 tokens: ledger **+12 rows, +2,100 completion tokens** for 3 served (exact); preemptions 0; KV peaked ~57% during cold 8k-token openings | ✅ |
| 11 | **Budget exhausted mid-conversation** | clean 429, conversation resumable | **429** with `X-Budget-Remaining: 0`; same conversation on a funded key → **200**. Refusing costs ~8 ms vs ~5,000 ms to serve | ✅ |
| 12 | **Ledger unwritable (disk full / permissions)** | fails **closed** on billing | `200,200,200` then `503,503,503` — **3 unbilled** (the configured threshold), then the breaker trips. 503 during cooldown, **200 after** | ✅ |
| 13 | **Thermal throttle under sustained load** | detected and alerted, not silent | detected by `watch_gpu.ps1`; **not alertable in Prometheus** — see below | ⚠️ |
| 14 | **Cold start after restart** | first-request cost known and excluded | **3,292 ms vs ~310 ms** warm; every tool discards one request before recording | ✅ |

**Reproduce cases 5 and 6:**

```powershell
python load_testing/kill_test.py --case engine    # crash the EngineCore worker
python load_testing/kill_test.py --case gateway   # restart the gateway under load
```

### Three ways of "killing the engine", two of which measure nothing

Worth recording, because the first two look like they work:

* **`docker compose kill vllm`** — container exits 137, but `restart: unless-stopped`
  **does not restart it**. Measured `restarts=0`. Docker treats an operator kill
  as intentional. It is easy to assume the policy covers this; it does not.
* **`docker exec vllm kill -9 1`** — silently ignored. The kernel does not
  deliver signals to a PID namespace's init from *inside* that namespace.
* **`pkill -9 -f EngineCore`** — the realistic crash. The process doing the GPU
  work dies, the API server exits, and the restart policy *does* apply.

And the timing needs care: the first version of this test reported a recovery
time of **0.8 s**, because it probed `/ready` before the API server had noticed
its worker was gone — timing the gap between issuing a kill and the kill taking
effect, and calling it recovery. The test now confirms the outage exists before
starting the clock on it ending.

---

## Case 3 — written off as unfixable, then fixed with one flag

A single ~20,000-token prompt pushed concurrent short requests from ~100 ms to
**10–20 s** of TTFT. Nothing errored and nothing was shed; the service was
simply unavailable to everyone else for the length of one prefill.

**The cause** is the V1 scheduler's order of work. Running requests are
scheduled first, and a long prompt mid-prefill takes the whole 2,048-token step
budget until it is done, so a new arrival gets nothing. Chunked prefill does
not prevent this, which was the assumption going in.

**Why it was written off.** vLLM documents this trio for the problem:

```
--max-num-partial-prefills=4 --max-long-partial-prefills=1 --long-prefill-token-threshold=2048
```

They were accepted, logged `Concurrent partial prefills enabled`, and then the
server died on `assert envs.VLLM_USE_V1`, **79 restarts before it was caught**.
The conclusion recorded was "the engine-side fix exists and cannot be used".

**What was actually true.** Only `--max-num-partial-prefills > 1` forces the V0
fallback. `--long-prefill-token-threshold` on its own is a plain per-step cap in
the V1 scheduler, and it was never tested alone. (At 2,048 it could not have
helped anyway: that is the whole step budget.) Ablated with a drift control:

| threshold | short requests behind the 20k prompt | the 20k prompt itself | 20-user chat, SLO attainment |
| ---: | ---: | ---: | ---: |
| 0 (vLLM default) | 12,942 ms / 11,116 ms (control) | 13.3 s / 11.4 s | 86.6% / 97.3% (control) |
| 1024 | 547 ms | 12.1 s | 97.3% |
| **512 (shipped)** | **378 ms** | 12.3 s | 90.4% |

The long prompt keeps making progress every step; it just no longer takes the
whole step. Ordinary traffic was measured alongside and stayed inside the
run-to-run spread (the two identical threshold-0 runs are 86.6% and 97.3%), so
the only effect larger than the noise is head-of-line blocking. 512 over 1024
because it gives newcomers more of each step; that difference is also small. **Lesson: when several flags fail together, test them one at a time
before writing off the problem.**

**Two false passes along the way**, both worth knowing about:

1. A fixed prompt string hit the prefix cache on the second run (13,159 ms →
   370.9 ms) and reported PASS while measuring nothing. The prompt now starts
   with a per-run nonce; it has to be at the **front**, because vLLM's prefix
   hash is a chain from block 0.
2. When the gateway's first 413 guard wrongly refused the 20k prompt (it
   assumed 4.5 characters per token; plain English measured 8.3), the short
   requests got *faster*, and the case kept passing while no long prefill ran
   at all. The verdict now requires the big prompt to be served.

**Weighted admission is still worth having.** The threshold stops one long
prefill from blocking newcomers, but only one prefill runs at a time, so two
cold 20k prompts still serialise. The gateway charges a cold long prompt up to
8 of 10 slots, which keeps a burst of them from queueing everyone else.

---

## Notes on the cases that are not simple passes

### 13. Thermal throttling — detected, but not by Prometheus

This is the most consequential failure this project ever hit and the hardest to
alert on. A host power setting once cost **11.6x throughput**, silently, with
no error in any engine log. Finalization Phase 2 then measured a further **42%
swing between two runs of an identical command**, caused only by the
temperature the card started from.

Neither is visible in vLLM's metrics, because both are hardware facts. Seeing
them needs GPU telemetry; on Linux that is `dcgm-exporter`, which expects
driver access patterns WSL2 under WDDM does not reliably provide. Rather than
ship an alert rule that silently never fires, the split is stated explicitly:

* **Prometheus** covers engine and gateway state.
* **`benchmarks/watch_gpu.ps1`** covers hardware state.
* Every recorded measurement carries the **idle temperature it started from**.

`alerts.yml` carries a proxy rule — sustained throughput below the measured
floor while the queue is busy and the cache is healthy — which infers a
hardware slowdown from software symptoms. It is a substitute, and labelled as
one.

### 14. Cold start — known, and excluded rather than hidden

The first request after an engine start costs **3,292 ms against ~310 ms**
warm, from one-time kernel autotuning, first tokenization and chat-template
detection. Every measurement tool in this repo discards one request before
recording: `ramp.py` via its warm-up level, `chat_sim.py` via an explicit
throwaway turn.

Full engine start on this machine is **~60–90 s** with a warm `torch.compile`
cache and ~156 s cold. Compose's `service_healthy` dependency means the gateway
does not accept traffic until the engine answers, so a cold start is slow
rather than broken.

**`docker compose up -d` returning is not the engine being ready** — it returns
in about 6 seconds. Wait for `docker compose ps` to read `healthy`. This cost a
failed measurement during Phase 3.
