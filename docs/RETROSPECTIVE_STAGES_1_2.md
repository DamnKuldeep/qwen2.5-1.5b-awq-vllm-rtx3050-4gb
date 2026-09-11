# Retrospective — Stages 1 & 2

Written at the close of Stage 2. Narrative synthesis of how the engine was stood up and how the baseline was measured: what was predicted, what happened, what caused it, what approach changed, and what each metric actually tells you.

`PROGRESS_LOG.md` holds the chronological record and `DECISIONS.md` holds the reasoning per choice. This file is the story that connects them, plus the generalizable method.

---

# Stage 1 — Standing up a tuned engine

## The goal

Get vLLM running with model, kernel, and memory flags chosen deliberately *before* first launch rather than tuned afterward. Acceptance criteria were narrow on purpose: Marlin selected, nonzero KV cache, a reported concurrency figure, server up.

## The method: predict, then measure

Before writing the launch script, the memory budget was derived from first principles and the predictions written down. That matters — **an unrecorded prediction cannot be wrong, and being wrong is where the learning is.**

| Quantity | How it was derived | Predicted | Actual |
| --- | --- | --- | --- |
| Weights | 3.09B params, minus 311M fp16 embeddings, rest at ~4.5 bits | ~2.2 GiB | **1.95 GiB** |
| KV per token | `2 (K,V) × 36 layers × 2 KV heads × 128 head_dim × 2 bytes` | 36 KiB | **36.1 KiB** ✓ |
| KV cache total | budget − weights − overhead | ~15–17k tokens | **28,432** |
| Max concurrency | KV total ÷ max-model-len | ~3.5–4x | **6.94x** |

**The lesson in that table:** the per-token cost was exact; the total was off by nearly 2x. Because **KV cache is the remainder** — budget minus weights minus activations minus overhead — every error in the other three terms lands entirely on it. A 250 MiB weights error became a ~13,000-token KV error.

> **Generalizable:** when a quantity is computed as a leftover, its error bar is the sum of every other term's error bar. Size the big terms carefully or do not trust the remainder.

## Where each flag came from

Nothing was a default. Each was derived from the 4096 MiB confirmed in Stage 0.

**`--max-model-len 4096`** against a native 32,768. One 32k sequence would demand 32,768 × 36 KiB = 1.15 GiB — more KV cache than exists after weights; vLLM would refuse to start. The cap is what makes concurrency possible at all.

**`--max-num-seqs 8`** against a real capacity of 6.94. **Deliberately overcommitted.** Real requests rarely reach max length, so the overcommit buys throughput; when they collectively do, vLLM preempts and evicts rather than crashing. That eviction path *is* Boundary 4, which Stage 10 must trigger. Sizing conservatively at 4 would have made the Stage 10 demonstration impossible.

**`--quantization awq_marlin`** — AWQ is the weight *format*, Marlin is the *kernel* that reads it. Separate choices, and vLLM will silently pick a slow kernel. Naming it converts a silent 11x performance loss into a loud startup error.

**`--max-num-batched-tokens 2048`**, below default. Activation memory scales with tokens-per-step; every MiB saved becomes KV cache. Trades peak single-request throughput for concurrency headroom — correct when memory, not compute, is binding.

## Three failures, and what each taught

### Failure 1 — CUDA floor

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8
```

Driver 566.07 provided CUDA 12.7. The vLLM image declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, enforced by the container runtime in a *prestart hook* — so the failure was `runc create failed`, before any Python existed.

The Stage 0 assessment had recorded 12.7 as "newer than what vLLM ships." **That was wrong** — vLLM moved its images to CUDA 12.8 for Blackwell support.

> **Generalizable:** "my driver is CUDA 12.x" is not the constraint. The *image's declared floor* is, and it is checked before your code runs.

Three options existed: update the driver, set `NVIDIA_DISABLE_REQUIRE=1` (would probably have worked, via CUDA minor-version compatibility), or pin an older image. **The driver update was chosen** — the override would have carried an unsupported configuration through twelve more stages, so any strange behaviour in Stage 9 or 11 would have had two candidate explanations instead of one. On a project whose output is trustworthy measurements, that cost lands exactly where it hurts.

### Failure 2 — memory sizing

```text
Free memory on device (3.21/4.0 GiB) is less than desired GPU memory
utilization (0.9, 3.6 GiB)
```

The budget was computed against **total**. vLLM checks against **free**. Only 3.21 of 4.0 GiB was free once a CUDA context existed — WDDM reserves VRAM for the Windows compositor, plus the WSL2 context cost.

> **Generalizable:** `nvidia-smi` reading `0MiB / 4096MiB` is not evidence that 4 GiB is usable. The reservation only materialises once a CUDA process exists, so Stage 0's reading *could not* have predicted it.

Fixed by lowering to 0.78 and pulling it into a `$GPU_MEM_UTIL` variable so Stage 11 can revisit it cheaply.

### Failure 3 (not really a failure) — the compile cache

After pinning the model revision, `torch.compile` re-ran for 61 seconds. The cache directory changed: `6ee1e77343` → `33c2863007`.

**The compile cache is keyed on the vLLM config.** Adding two flags changed the key.

> **Generalizable, and it matters for Stage 11:** every optimization variant changes a flag, so every variant pays a one-off ~60 s compile. Launch each variant once to warm its cache *before* measuring, or that minute contaminates the result.

## What Stage 1 produced

| | |
| --- | --- |
| Kernel | `awq_marlin` confirmed in logs |
| Attention / sampler | Flash Attention (V1 engine) / FlashInfer |
| Weights on GPU | 1.9542 GiB |
| KV cache | 0.98 GiB = **28,432 tokens** (1777 blocks × 16) |
| Max concurrency @ 4096 | 6.94x |
| Engine pinned | `vllm/vllm-openai:v0.11.0`, digest `sha256:014a95f2…` |
| Weights pinned | commit `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` |
| Driver recorded | 616.56 / CUDA UMD 13.4 |

**Boundary 2 complete** — engine, weights and driver all pinned with written reasoning. The difference between "I wrote down what I used" and "someone else can reproduce this."

---

# Stage 2 — The baseline

## The approach change, made before measuring

`PROJECT_PLAN.md` named `vllm bench latency` and `vllm bench throughput`. That was overridden:

1. **`PRODUCT_SPEC.md` requires TTFT and ITL**, and those only exist on a *serving* path. An offline batch job has no notion of "time to first token for a streamed request."
2. **An offline benchmark instantiates its own engine.** The server holds 3.12 GiB of a 4 GiB card; a second copy does not fit.

The deciding argument: `bench serve` measures **the code path this product actually ships**. That is what Stage 11's optimizations must be compared against for the comparison to mean anything.

> **Generalizable:** benchmark the thing you ship. An easier measurement of a different thing is not a substitute.

## Four rounds of iteration

### Round 1 — the client would not start

```text
ImportError('libcuda.so.1: cannot open shared object file')
RuntimeError: Failed to infer device type
```

The benchmark client was run without `--gpus`, on the reasoning that a benchmark client needs no GPU. Correct about the *workload*, wrong about the *CLI*: the `vllm` entrypoint builds every subparser at import, including `serve`, whose parser instantiates `VllmConfig` defaults, which demand a resolvable device.

> **Generalizable:** a tool can require a resource for reasons unrelated to the work you are asking it to do. The failure was at argument-parsing time, not benchmark time — read *where* in the stack the traceback originates.

### Round 2 — the runs were not comparable to each other

Both runs requested 32 × 128 = 4096 output tokens. They produced **2,683** and **2,823** — with an identical `--seed`.

Two compounding causes:

- Without `--ignore-eos`, the model stops at EOS. `--random-output-len` was a *ceiling*, not a quantity. The random dataset feeds gibberish; the model answers briefly.
- With no explicit temperature, the server applied Qwen's HF generation config (`temperature 0.7, top_p 0.8, top_k 20`) — **the exact non-determinism flagged at the end of Stage 1, coming due.** Stochastic sampling put EOS somewhere different each run.

Fixed with `--ignore-eos --temperature 0`. The concurrent run was also raised to 128 prompts: 32 prompts at concurrency 8 is only four waves, so ramp-up and drain dominate the average — visible directly as `50.19` average against `72.00` peak.

> **Generalizable, and the single most important benchmarking rule: make the work identical before comparing anything.** A benchmark whose workload varies between runs measures the variance, not your change.

Deliberately *not* fixed server-side. Qwen's defaults are right for the product and the Stage 6 chat UI. The benchmark is the special case, so the benchmark overrides per-request — keeping the measured thing identical to the shipped thing in every respect except the one controlled on purpose.

### Round 3 — correctness revealed a catastrophe

With the harness fixed, throughput **collapsed**:

| | Round 2 (invalid) | Round 3 (correct harness) |
| --- | --- | --- |
| Mean ITL | 43.16 ms | **223.86 ms** |
| Mean TTFT | 453.99 ms | **275.50 ms** |

**Decode 5x slower. Prefill slightly faster.** That split was the entire diagnosis, and the reasoning chain is the most valuable thing in this document:

- **Prefill is compute-bound.** Big matrix multiplies, high arithmetic intensity.
- **Decode is memory-bandwidth-bound.** Every generated token streams all 1.95 GiB of weights out of VRAM.
- A change that leaves prefill healthy while destroying decode is therefore **not a compute problem. It is bandwidth.**

Second clue — the **distribution shape**. Round 3 reported mean 223.86 / median 225.53 / P99 227.08: three numbers within 1% across fifteen minutes. That flatness is the signature of a **hard cap**, not contention. Contention produces skew, as Round 2's 43 mean / 18.5 median / 105 P99 did.

Third clue, confirming it — batching scaling *improved*, 2.4x → 4.9x. If bandwidth is the binding constraint, batching amortises one weight-read across more sequences, so **scarcer bandwidth makes batching help more.** The constraint tightened and the curve responded as theory says it should.

### Round 4 — instrument the layer below

Rather than guess, the hardware was sampled with `benchmarks/watch_gpu.ps1`:

```text
clocks.sm  clocks.mem  power   temp  util   pstate  throttle
210 MHz    405 MHz     14.3 W  73 C  100 %  P8      0x24
```

**P8 — the idle power state — at 100% utilisation.** Memory at 405 MHz against a rated ~5500: roughly **1/15th of rated bandwidth**, which explains a 12x decode collapse precisely.

The throttle bitmask decoded to `0x04` (SwPowerCap) `| 0x20` (SwThermalSlowdown). **SwThermalSlowdown at 73 °C was the anomaly** — this silicon throttles for real in the high 80s. A software thermal limit tripping in the low 70s is a *configured policy*, not silicon protecting itself.

**The fix:** Windows Power mode → Best performance, and NVIDIA Control Panel → Power management mode → Prefer maximum performance.

| | P8 | P0 | |
| --- | --- | --- | --- |
| Memory clock | 405 MHz | 5501 MHz | 13.6x |
| Power | 14 W | 32–46 W | |
| Mean ITL | 224.09 ms | 20.42 ms | 11x |
| Output throughput | 4.15 tok/s | 47.99 tok/s | **11.6x** |

**11.6x throughput from a host power setting.** No model, engine or flag change.

The detail worth remembering most: **the charger was not the problem.** NVIDIA's *default* adaptive power policy alone held the card in P8. That policy is a graphics heuristic, and a continuous stream of small memory-bound kernel launches — exactly what LLM decode is — does not trip it into a performance state.

> **Generalizable:** nothing in the vLLM logs indicated any problem and the output was entirely plausible. **A silent 11.6x loss is only visible if you instrument the layer beneath the one you are measuring.**

### Round 5 — a new constraint appears

With P0 restored, the sampler showed something else:

```text
16:48:26   1372 MHz, 43.42 W, 85 C, 0x04  (SwPowerCap)
16:48:31   1012 MHz, 39.35 W, 87 C, 0x20  (SwThermalSlowdown)
16:48:43    712 MHz, 32.45 W, 87 C, 0x20   <- settles
```

The card boosts to 1372 MHz, hits 87 °C in three seconds, and settles at **712 MHz — a 48% clock reduction inside a single 21-second run.** This is genuine silicon thermal limiting.

Critically: **memory clock held at 5501 MHz throughout. It does not throttle.**

That predicts a specific asymmetry — decode (bandwidth-bound) insulated, prefill (compute-bound) collapsing. Tested by comparing the discarded warm-up against the measured run:

| | Warm-up (22 s, boosting) | Run A (123 s, steady state) |
| --- | --- | --- |
| Mean ITL | 21.57 ms | 25.92 ms (**+20%**) |
| Mean TTFT | 51.54 ms | 515.36 ms (**+900%**) |

**Prediction confirmed**, and it establishes a rule: **results depend on run length**, so every measurement needs a discarded warm-up to reach thermal steady state. Without it, Stage 11 would be comparing thermal mass rather than optimizations.

One honest loose end: TTFT degraded 10x while clock only halved. Clock alone does not explain the magnitude. Flagged for Stage 9 rather than explained away.

---

# The baseline we finally have

**Run A — single stream** (32 requests, one at a time, 4096 tokens exact):

| Metric | Value |
| --- | --- |
| Output throughput | 33.39 tok/s |
| Median TTFT | 654.13 ms |
| Median ITL | 26.13 ms |
| P99 ITL | 33.62 ms |
| Median E2EL | 4049.01 ms |
| KV cache usage | ~2.4% |

**Run B — batched at 8** (128 requests, 16384 tokens exact):

| Metric | Value | vs Run A |
| --- | --- | --- |
| Output throughput | **137.15 tok/s** | **4.11x** |
| Median TTFT | 758.76 ms | +16% |
| Median ITL | 29.51 ms | **+13%** |
| **P99 ITL** | **637.62 ms** | **19x worse** |
| Median E2EL | 8548.21 ms | 2.1x |
| Peak KV cache usage | **18.5%** | |
| Requests queued | **0** | |

## Three findings that shape the rest of the project

**1. Batching is worth 4.11x throughput for a 13% median latency cost.** 51% scaling efficiency against 8x concurrency. The shortfall is expected — decode is bandwidth-bound and eight sequences contend for the same memory bus.

**2. The cost hides in the tail.** P99 ITL degrades **19x** while the median moves 13%. Those are scheduler pauses when an arriving request's prefill preempts ongoing decode. Chunked prefill bounds the damage; it cannot eliminate it. **Mean TPOT (44.90 ms) smooths this away completely** — only P99 ITL exposes it. This is the exact signature Stage 10 should hunt for when forcing an eviction.

**3. KV cache is not the constraint.** 18.5% peak, zero queued requests. **Stage 9's ceiling will be thermal/compute, not memory** — the opposite of what a 4 GiB card intuitively suggests, and a direct consequence of `--max-model-len 4096` keeping per-request demand small.

## One caveat found afterward, in the server logs

```text
Prefix cache hit rate: 69.5% ... 74.6%
```

Far too high for a dataset with no shared prefix. The hits are **cross-run**: `--seed 42` generates identical prompts every invocation, prefix caching is on, and the cache persists in the server process between benchmark runs.

**Consequence: recorded TTFT is optimistic.** ITL is unaffected — prefix caching only accelerates prefill. **Rule: restart the vLLM server between measured comparison pairs**, or the second run inherits the first's cache and looks better for reasons unrelated to the change under test.

---

# What each metric actually tells you

| Metric | Definition | Bound by | What a change means |
| --- | --- | --- | --- |
| **TTFT** | Time to first token — prefill + queueing | **Compute** (SM clock) | Rises with GPU load, queue depth, prompt length. Falls with prefix cache hits. The user-perceived "did it hear me" latency. |
| **ITL** | Gap between consecutive streamed tokens | **Memory bandwidth** | Rises with batch size (bus contention). Largely immune to thermal throttling here, since memory clock does not throttle. |
| **TPOT** | `(E2EL − TTFT) / (output_tokens − 1)` | same as ITL | Same physical quantity as ITL, averaged per request. **Smooths out individual stalls.** |
| **E2EL** | Total request wall time | both | What a non-streaming client experiences. |
| **Output throughput** | Total generated tokens ÷ duration | both | Your capacity number. What batching improves. |
| **Peak vs average throughput** | max instantaneous vs overall | — | A large gap means ramp-up/drain dominate — the run is too short. |
| **KV cache usage %** | Fraction of PagedAttention blocks in use | — | How close to memory-driven preemption you are. |
| **Prefix cache hit rate** | Prefill tokens served from cache | — | Boundary 3 metric. High values across independent runs = contamination. |
| **Running / Waiting** | Scheduler queue depth | — | `Waiting > 0` means `--max-num-seqs` has been exceeded. |

**Why median, mean and P99 are all reported:**

- **Median** — the typical experience, robust to outliers.
- **Mean** — pulled by outliers. When mean ≫ median, something is stalling.
- **P99** — the tail, where users feel pain and where SLOs live.

Run B is a textbook case: median ITL 29.51, mean 44.73, P99 637.62. Reading only the median, batching looks nearly free. Reading P99, the stalls appear. **Report all three or you are choosing what to hide from yourself.**

---

# The method — generalizable beyond this project

1. **Predict before you measure, in writing.** An unrecorded prediction cannot be wrong. The 2.2 vs 1.95 GiB miss taught more than a correct guess would have.
2. **Make the work identical before comparing anything.** `--ignore-eos` and `--temperature 0` were not polish — without them every number measured dice rolls.
3. **Verify the environment is in a known state, and instrument the layer below.** vLLM logged nothing wrong during an 11.6x loss.
4. **Split the metric by its bottleneck.** "Prefill fine, decode broken" narrowed the search from everything to memory bandwidth in one step. Knowing which metrics are compute-bound and which are bandwidth-bound is the highest-leverage diagnostic available.
5. **Read the distribution, not the average.** Flat mean/median/P99 = a hard cap. Wide skew = contention. The shape names the cause.
6. **Change one thing at a time**, and re-verify.
7. **Prefer structural fixes to disciplinary ones.** Stage 3's raw-byte forwarding does not *remember* to preserve `response_format` — it makes forgetting inexpressible.
8. **Record everything with reasoning, and never rewrite history.** The CUDA-floor mistake is more instructive than the correct answer.
9. **Design your constraints for what you will need later.** `--max-num-seqs 8` against a 6.94 capacity is overcommitted on purpose, so Stage 10 can force an eviction. Conservative sizing would have looked safer and made the demonstration impossible.
