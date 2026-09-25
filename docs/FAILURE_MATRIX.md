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
.\load_testing\run_matrix.ps1 -Only scenarios      # cases 1, 2, 9
```

Container-level cases (5, 6) are driven from the shell and shown inline below.

---

## The configuration under test

| | |
| --- | --- |
| Engine | `vllm/vllm-openai:v0.11.0`, `Qwen/Qwen2.5-1.5B-Instruct-AWQ` @ `3ecffa0c…` |
| Context window | 32,768 tokens |
| KV cache | 69,760 tokens (~1.86 GiB at 28.0 KiB/token) |
| Engine concurrency cap | `--max-num-seqs 32` — deliberately above the gateway's limit |
| Gateway admission | 6 in flight, 12 queued, 0.6 s queue timeout, dynamic per-key share (cap 3), prompt-weighted cost |
| Output bound | `max_tokens` default 1,024 if absent, ceiling 2,048 |
| SLO | p95 TTFT < 1.5 s **for chat-length prompts (~500–1,000 tokens)** |

The SLO qualifier is not pedantry. A 22,586-token prompt costs 13 s of prefill
on this hardware; no admission policy makes that fit a 1.5 s target, and a
matrix that reported it as a failure would be reporting arithmetic.

---

## Results

| # | Case | Expected | Measured | |
| --- | --- | --- | --- | :---: |
| 1 | **Burst 10x over capacity** (60 simultaneous) | p95 bounded; excess refused with 503 + `Retry-After` | 6 served, **54 shed**, `Retry-After: 2`; served p95 TTFT **263 ms** | ✅ |
| 2 | **One abusive key at 50 concurrent** | other users stay within SLO | normal users p95 **536 ms** vs 191 ms baseline (**+181%**, still 64% inside SLO); **0/10 over SLO**; abuser shed **4,669 of 4,749 (98.3%)** | ✅ |
| 3 | **Single ~20k-token prompt** | short requests keep flowing; TTFT rises but stays in low seconds | short requests **108 ms → 19,698 ms** during the big prefill. All 8 completed, 0 shed | ⚠️ |
| 4 | **Conversation outgrows the window** | 200 with oldest turns dropped and reported; never a hard 400 | HTTP **200**, `X-Context-Trimmed-Messages: 27`; ~95,488 tokens sent → engine saw **9,726** | ✅ |
| 4b | **A single message larger than the window** | same policy | **Gap found by audit, now closed.** Trimming never drops the current question, so one 333k-token message passed through untrimmed and came back as the engine's 400 *after* taking a slot. The gateway now returns **413** before admission; the engine is never called | ✅ |
| 5 | **Engine crash mid-stream** | 502, auto-recovery, gateway stays Running (liveness) while going NotReady (readiness) | crash detected **1.4 s**, automatic recovery **83.1 s**; `/health` **200 throughout**, `/ready` 503→200; **155×502**, 4×500, 3×200; **162 of 162** requests in ledger, **drift 0** | ✅ |
| 6 | **Gateway restart under load** | in-flight fail cleanly, budgets intact | back in **10.4 s**; 12 in-flight failed with a dropped connection (no hang, no silent truncation), 4 served; ledger 17,854→17,858 rows and 4,976,558→4,977,083 tokens — **budgets survived** | ✅ |
| 7 | **Client disconnects mid-stream** | slot released, usage still recorded | in-flight returned to **0**; ledger row written from the generator's `finally` | ✅ |
| 8 | **Slow client (~1 chunk/s)** | cannot hold a slot indefinitely; others unaffected | **5/5** concurrent normal requests succeeded, p50 TTFT **124 ms**; bounded by `GATEWAY_STREAM_TIMEOUT_S` | ✅ |
| 9 | **Prefix cache thrash at high user count** | measured, not assumed | hit rate held **74–97%** from 10 to 60 users. **No collapse** — see the capacity model for why | ✅ |
| 10 | **KV exhaustion / preemption storm** | accounting exact | 12 requests × 700 tokens: ledger **+12 rows, +1,400 completion tokens** for 2 served (exact); preemptions 0; KV never above ~17% | ✅ |
| 11 | **Budget exhausted mid-conversation** | clean 429, conversation resumable | **429** with `X-Budget-Remaining: 0`; same conversation on a funded key → **200**. Refusing costs ~8 ms vs ~5,000 ms to serve | ✅ |
| 12 | **Ledger unwritable (disk full / permissions)** | fails **closed** on billing | `200,200,200` then `503,503,503` — **3 unbilled** (the configured threshold), then the breaker trips. 503 during cooldown, **200 after** | ✅ |
| 13 | **Thermal throttle under sustained load** | detected and alerted, not silent | detected by `watch_gpu.ps1`; **not alertable in Prometheus** — see below | ⚠️ |
| 14 | **Cold start after restart** | first-request cost known and excluded | **3,292 ms vs ~310 ms** warm; every tool discards one request before recording | ✅ |

**13 pass, 2 documented limitations, 0 unexplained failures.**

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

## Case 3 — the one that does not pass, and why the fix is not available

A single ~20,000-token prompt pushes concurrent short requests from **108 ms to
19,698 ms** of TTFT. They all complete, none are shed, and nothing errors — but
for twenty seconds the service is effectively unavailable to everyone else.

**This was measured twice after a false pass was caught.** The first re-test
reported 370 ms and looked fixed. It was not: the harness used a fixed prompt
string, so the second run got a complete prefix-cache hit and measured nothing.
Adding a per-run nonce **at the front** of the prompt — vLLM's prefix hash is a
chain from block 0, so a nonce at the end would leave every preceding block
cached — restored the real behaviour: **19,995 ms and 20,833 ms** on two
consecutive runs.

**The cause** is in `SchedulerConfig`: `max_num_partial_prefills = 1`. Only one
request may be mid-prefill at a time, so every short request queues behind all
ten chunks of the long one. Chunked prefill does *not* prevent this, which was
the assumption going in.

**The engine-side fix exists and cannot be used.** vLLM has exactly the right
controls:

```
--max-num-partial-prefills=4 --max-long-partial-prefills=1 --long-prefill-token-threshold=2048
```

They are accepted, they log `Concurrent partial prefills enabled`, and then the
server dies on `assert envs.VLLM_USE_V1` — **79 restarts before it was caught.**
Setting `max_num_partial_prefills > 1` falls back to the V0 engine, and vLLM
0.11.0's OpenAI API server requires V1. A feature that configures cleanly and
then makes the server unstartable is worse than one that rejects the flag.

**What was done instead — weighted admission.** The gateway now charges a
request admission slots in proportion to its prefill chunks
(`cost = 1 + prompt_tokens / 2048`, capped at 4 of 6). This does not fix
head-of-line blocking — that is engine-side — but it **bounds the damage**, and
that bound is measured: three concurrent 20k prompts produce
`[200 (cost 4), 503 queue_timeout, 200 (cost 4)]`. Without weighting all three
would have been admitted, and the blocking would have compounded to a minute.

It also implements what `V2_SCOPE.md` asked for — admitting by predicted GPU
cost rather than request count — just for a different reason than expected.

**Honest options if this mattered more:** cap prompt size at the gateway and
return 413 (a product decision, and it removes the 32k window's value); run a
vLLM build where the V1 scheduler supports concurrent partial prefills; or
accept it and document the envelope, which is what is done here.

---

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
