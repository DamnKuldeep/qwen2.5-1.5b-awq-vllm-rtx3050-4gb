# Progress Log

Append-only. One entry per confirmed stage (or per notable event within a stage — a bug hit and fixed counts as its own entry). Never delete or rewrite past entries — if something later turns out to be wrong, add a new entry correcting it, don't erase the old one. This file is the project's memory across separate Claude Code sessions: always read it before assuming where we are.

Use this format per entry:

```
## Stage N — <short title> — <date>

**What we did:**
<one or two lines>

**Command(s) run:**
<the exact command(s) the user ran>

**Result:**
<what actually happened — summarize what they reported back>

**Issues hit / fixed:**
<or "none">

**Status:** ✅ confirmed working / ⚠️ working with caveats / ❌ blocked
```

---

## Stage 0 — Environment re-verification — 2026-09-07

**What we did:**
Confirmed Docker Desktop is running and that `--gpus all` GPU passthrough into a container still works, before building anything on top of it. No files created.

**Command(s) run:**

```powershell
docker version
docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi
```

**Result:**
Docker client and server both 29.7.2 (Docker Desktop 4.89.0, containerd v2.3.3, runc 1.4.3), server `linux/amd64` via the `desktop-linux` context — WSL2 backend healthy.
`nvidia-smi` ran inside the container and reported: NVIDIA-SMI 565.65, **Driver 566.07, CUDA 12.7**, one GPU — NVIDIA GeForce RTX 3050 — at **0MiB / 4096MiB**, 0% utilization, 57C, P8, no running processes.

These are the numbers Stage 1 is sized against: 4096 MiB total VRAM, entirely free, compute capability 8.6 (Ampere) — above Marlin's ≥8.0 requirement, so the AWQ + `awq_marlin` decision in `DECISIONS.md` is valid on this actual card.

**Issues hit / fixed:**
None. Nothing regressed since the course.

**Status:** ✅ confirmed working

---

## Stage 1 — Blocked on first launch: CUDA version floor — 2026-09-07

**What we did:**
Created `deployment/docker/run_vllm.ps1` and `run_vllm.sh` with the tuned flag set, and attempted the first launch.

**Command(s) run:**

```powershell
.\deployment\docker\run_vllm.ps1
```

**Result:**
The `vllm/vllm-openai:v0.11.0` image pulled successfully (digest `sha256:014a95f2…`), then the container failed to start:

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8,
please update your driver to a newer version, or use an earlier cuda container
```

**Issues hit / fixed:**
The vLLM v0.11.0 image is built against the CUDA 12.8 toolkit and declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8`. The installed driver (566.07) provides CUDA 12.7 — just under the line. The NVIDIA container runtime enforces this in a prestart hook, so the failure is a runc error before any Python runs; nothing was wrong with the launch flags or the pin.

Note: the Stage 0 assessment recorded CUDA 12.7 as "newer than the CUDA runtime the vLLM images ship" — that was wrong. vLLM moved its published images to CUDA 12.8 for Blackwell support. **Lesson worth keeping: a host driver being CUDA 12.x is not sufficient; the image's `NVIDIA_REQUIRE_CUDA` floor is the actual constraint, and it is checked before startup, not at runtime.**

Resolution chosen (of three considered — driver update, `NVIDIA_DISABLE_REQUIRE=1` override, or pinning an older CUDA 12.4-era vLLM image): **update the Windows NVIDIA driver.** Reasoning in `DECISIONS.md`.

**Status:** ❌ blocked — pending driver update, then re-run Stage 1

---

## Stage 1 — Driver updated, CUDA floor cleared; new block on memory sizing — 2026-09-07

**What we did:**
Updated the Windows NVIDIA driver via the NVIDIA App, reset WSL, re-verified GPU passthrough on both sides of the container boundary, and relaunched vLLM.

**Command(s) run:**

```powershell
wsl --shutdown
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi
.\deployment\docker\run_vllm.ps1
```

**Result:**
Driver now **616.56, KMD 616.56, CUDA UMD 13.4** — comfortably past the `cuda>=12.8` floor. Host and container `nvidia-smi` agree, so the WSL driver mapping refreshed correctly. GPU still reads `0MiB / 4096MiB`, now under `Driver-Model: WDDM`.

vLLM got much further. Three of the four Stage 1 acceptance criteria passed before it stopped:

- `[awq_marlin.py:119] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.` — Marlin engaged, the ~11x kernel decision confirmed on real hardware
- `[scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=2048.`
- `Initializing a V1 LLM engine (v0.11.0)` with `enable_prefix_caching=True`, `quantization=awq_marlin`, `dtype=torch.float16`, `max_seq_len=4096`

Then engine core init failed:

```text
ValueError: Free memory on device (3.21/4.0 GiB) on startup is less than desired
GPU memory utilization (0.9, 3.6 GiB). Decrease GPU memory utilization or reduce
GPU memory used by other processes.
```

**Issues hit / fixed:**
`--gpu-memory-utilization` is checked against **free** memory at startup, not **total**. Only 3.21 of 4.0 GiB is free once a CUDA context exists: WDDM reserves VRAM for the Windows desktop compositor, and the WSL2 CUDA context itself costs several hundred MiB. The Stage 0 reading of `0MiB / 4096MiB` could not have predicted this, because the reservation only materialises once a CUDA process exists.

Hard ceiling is therefore ~0.80. **Fixed by lowering the dial to 0.78**, pulled out to a `$GPU_MEM_UTIL` variable at the top of both launch scripts so Stage 11 can revisit it cheaply. Reasoning appended to `DECISIONS.md`; the WDDM mechanism added to `docs/CONCEPTS_EXPLAINED.md`.

Also noted for later, not acted on: `WARNING [interface.py:381] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.` — a real but unavoidable WSL2 cost that belongs in the Stage 13 "these numbers don't transfer to a server" section. And `revision=None` in the engine config confirms the model revision pin is still outstanding.

**Status:** ⚠️ in progress — relaunching at 0.78

---

## Stage 1 — vLLM server up and tuned — 2026-09-07

**What we did:**
Relaunched with `--gpu-memory-utilization 0.78`. The engine initialised fully and the OpenAI-compatible server came up on port 8000.

**Command(s) run:**

```powershell
.\deployment\docker\run_vllm.ps1
```

**Result — the reference numbers for this machine:**

| Metric | Value |
| --- | --- |
| Model weights on GPU | **1.9542 GiB** |
| Available KV cache memory | **0.98 GiB** |
| GPU KV cache size | **28,432 tokens** |
| `num_gpu_blocks` | **1777** (× 16 tokens/block = 28,432 ✓) |
| Max concurrency @ 4096 tokens/request | **6.94x** |
| Bytes per KV token (derived) | **~36.1 KiB** |
| Cold start, model already cached | ~72 s engine init (54 s of it `torch.compile`) |

All four Stage 1 acceptance criteria met: `awq_marlin` kernel selected, nonzero KV cache block count, maximum-concurrency figure reported, `Application startup complete`.

Also engaged: **Flash Attention** backend on the V1 engine, **FlashInfer** for top-k/top-p sampling, CUDA graph capture (0.05 GiB, 2 s). `torch.compile` output is cached to `/root/.cache/vllm/torch_compile_cache` inside the `hf-cache` named volume, so restarts skip the 54 s compile.

**Estimate vs. actual — worth keeping:**
The per-token KV cost was predicted exactly (36 KiB from `2 × 36 layers × 2 KV heads × 128 head_dim × 2 bytes`; actual 36.1 KiB). Weights were over-estimated at ~2.2 GiB against an actual 1.95 GiB — AWQ packs the embedding layer better than assumed. Because **KV cache is the remainder of the budget, it absorbs every other estimation error**: a 250 MiB weights error became a ~13,000-token KV cache error, nearly doubling the real figure versus the estimate. On a small card, size the weights carefully or expect the KV figure to be wrong.

The `--max-num-seqs 8` choice landed as intended: capacity is 6.94 full-length sequences against a permitted 8 — a mild deliberate overcommit, enough for Stage 10 to force a preemption without normal traffic thrashing.

**Issues hit / fixed:**
None on this run. The two blocks that preceded it (CUDA floor, memory sizing) are logged as their own entries above.

Flagged, not yet acted on: `WARNING [model.py:1389]` — Qwen's HF generation config overrides vLLM's sampling defaults with creative settings (`temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05`). Any request not setting sampling params explicitly is therefore non-deterministic. Relevant to Stage 2 benchmark comparability and to the Stage 5 contract test; to be handled explicitly rather than inherited silently.

**Status:** ✅ confirmed working — model revision pin outstanding, closing it next

---

## Stage 1 — Smoke test passed, model revision pinned (Boundary 2 complete) — 2026-09-07

**What we did:**
Ran one real generation against the live server, and read the exact model commit hash out of the downloaded snapshot to pin it.

**Command(s) run:**

```powershell
docker run --rm -v hf-cache:/cache --entrypoint ls vllm/vllm-openai:v0.11.0 /cache/hub/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots
.\deployment\docker\smoke_test.ps1
```

**Result:**
Snapshot hash: **`3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`** — now pinned via `--revision` and `--tokenizer-revision` in both launch scripts.

Generation succeeded: 44 prompt tokens, 38 completion tokens, 82 total, HTTP 200. The server logged the request and reported:

```text
Avg prompt throughput: 4.4 tokens/s, Avg generation throughput: 3.8 tokens/s,
Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

**This log line is Boundary 3 made concrete.** `GPU KV cache usage` and `Prefix cache hit rate` — precisely the two metrics the reading identifies as being silently dropped by monitoring wrappers — are emitted natively by the engine, both here and on `/metrics`. Confirms the `DECISIONS.md` choice to have Prometheus scrape vLLM directly. Both read 0.0% only because the request completed inside the 10-second logging interval.

Note the throughput figures here are **not** a baseline: a single short request measured across a 10 s averaging window says nothing useful. Stage 2 produces the real numbers.

Answer quality observation: the model described the KV cache in generic caching terms rather than correctly describing attention key/value reuse. Expected for a 4-bit 3B model; worth remembering when reviewing Stage 6 chat UI output.

**Boundary 2 is now fully demonstrated:** engine pinned to `vllm/vllm-openai:v0.11.0` (digest `sha256:014a95f2…`), weights pinned to commit `3559b226…`, driver version recorded (616.56 / CUDA UMD 13.4), and the reasoning for each written down in `DECISIONS.md`.

**Issues hit / fixed:**
None.

**Status:** ✅ confirmed working — pending one restart to verify the pinned revision launches cleanly

---

## Stage 1 — Pin verified; torch.compile cache is config-keyed — 2026-09-07

**What we did:**
Restarted the server with `--revision` / `--tokenizer-revision` set, to verify the pin launches cleanly rather than assuming it.

**Command(s) run:**

```powershell
.\deployment\docker\run_vllm.ps1
```

**Result:**
Pin confirmed live in the engine config: `revision=3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd, tokenizer_revision=3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`. KV cache identical at **28,432 tokens**, `num_gpu_blocks: 1777`, max concurrency 6.94x — byte-for-byte the same configuration, so nothing non-deterministic is in play.

Model loading dropped from **393.4 s to 6.3 s** — the `hf-cache` named volume eliminated the re-download as intended.

**Unexpected finding worth keeping — `torch.compile` recompiled anyway (61.6 s):**
The compile cache directory changed between launches:

```text
before:  /root/.cache/vllm/torch_compile_cache/6ee1e77343/rank_0_0/backbone
after:   /root/.cache/vllm/torch_compile_cache/33c2863007/rank_0_0/backbone
```

**The compile cache is keyed on the vLLM configuration.** Adding the two revision flags changed the config and therefore the cache key, invalidating the previous compile. Not a fault — the next launch with these exact flags will hit `33c2863007` and skip it.

Direct consequence for Stage 11: **every optimization variant changes a flag, so every variant pays a fresh ~60 s compile on its first launch.** That time must be excluded from any benchmark measurement, and each variant should be launched once to warm its cache before being measured.

Total restart wall time ~2.5 min, of which ~62 s was the one-off recompile.

**Issues hit / fixed:**
None.

**Status:** ✅ Stage 1 complete and confirmed working

---

## Stage 2 — First benchmark attempt: harness bugs found, numbers discarded — 2026-09-07

**What we did:**
Built `benchmarks/run_bench_serve.ps1` and `benchmarks/baseline.md`, and ran the first two measurements. Chose `vllm bench serve` over the plan's `latency`/`throughput` subcommands — reasoning in `DECISIONS.md`; short version is that TTFT/ITL only exist on a serving path, and an offline benchmark's second engine will not fit alongside the server on a 4 GiB card.

**Command(s) run:**

```powershell
.\benchmarks\run_bench_serve.ps1
.\benchmarks\run_bench_serve.ps1 -Concurrency 8
```

**Issue 1 — client container failed before benchmarking (fixed):**
First invocation omitted `--gpus`, on the reasoning that a benchmark client needs no GPU. Correct about the workload, wrong about the CLI:

```text
Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file')
RuntimeError: Failed to infer device type
```

Without `--gpus`, Docker does not map the driver libraries in, vLLM resolves `UnspecifiedPlatform`, and the `vllm` entrypoint dies while *building its argument parser* — it constructs every subparser including `serve`, whose parser instantiates `VllmConfig` defaults, which require a resolvable device. Fixed by adding `--gpus all` purely for device detection; the client loads no model and allocates no KV cache.

**Issue 2 — the runs were not comparable to each other (fixed, numbers discarded):**
Both runs requested 32 × 128 = 4096 output tokens. Run A produced **2,683**; Run B produced **2,823** — with an identical `--seed`.

Two compounding causes. Without `--ignore-eos`, the model stops at EOS, so `--random-output-len` is a ceiling rather than a quantity, and the random dataset's gibberish prompts draw short answers. And with `temperature=None` sent by the client, the server applied Qwen's HF generation config (`temperature 0.7, top_p 0.8, top_k 20`) — **the exact non-determinism flagged at the end of Stage 1, now come due** — so EOS landed differently each run.

Fixed by adding `--ignore-eos --temperature 0` to the harness. Deliberately not fixed server-side: Qwen's defaults are correct for the product and the Stage 6 chat UI; the benchmark is the special case and overrides per-request.

**Issue 3 — too few prompts for the concurrent run (fixed):**
32 prompts at concurrency 8 is four waves; ramp-up and drain dominate. Visible directly in the output: `Output token throughput: 50.19` against `Peak output token throughput: 72.00`. Concurrent runs will use 128+ prompts.

**Preliminary numbers — recorded for history, NOT the baseline:**

| Metric | Run A (conc 1) | Run B (conc 8) |
| --- | --- | --- |
| Benchmark duration (s) | 130.33 | 56.25 |
| Generated tokens (requested 4096) | 2,683 | 2,823 |
| Output token throughput (tok/s) | 20.59 | 50.19 |
| Peak output token throughput (tok/s) | 57.00 | 72.00 |
| Total token throughput (tok/s) | 146.07 | 340.94 |
| Median TTFT (ms) | 368.53 | 276.86 |
| P99 TTFT (ms) | 732.55 | 514.57 |
| Median TPOT (ms) | 18.53 | 122.78 |
| Median ITL (ms) | 18.50 | 119.79 |
| P99 ITL (ms) | 105.77 | 252.96 |
| Median E2EL (ms) | 2107.65 | 11299.94 |

**What is still trustworthy in the above:** the medians and the overall shape. **Median ITL 18.50 ms ≈ 54 tok/s single-stream decode** is a healthy figure for a 3B AWQ model on a 35 W laptop card and is good evidence Marlin is working. Batching behaved as designed — 2.4x aggregate throughput for 6.5x worse per-request ITL. Median TTFT *improving* under load (368 → 277 ms) is mildly counterintuitive, plausibly chunked prefill packing prefills more efficiently; to be confirmed on a clean run before it is believed.

**Status:** ⚠️ harness corrected, re-measuring for the real baseline

---

## Stage 2 — Harness correct, but decode collapsed: suspected GPU power state — 2026-09-07

**What we did:**
Re-ran both benchmarks with `--ignore-eos --temperature 0`.

**Command(s) run:**

```powershell
.\benchmarks\run_bench_serve.ps1
.\benchmarks\run_bench_serve.ps1 -Concurrency 8 -NumPrompts 128
```

**Result — the harness fix worked, and revealed a hardware problem:**
`Total generated tokens` was **exactly 4096** (Run A) and **exactly 16384** (Run B). Determinism achieved; `ignore_eos=True, temperature=0.0` confirmed in both namespace dumps.

But throughput collapsed versus the earlier (invalid) attempt:

| Metric | Prev. Run A (invalid) | Clean Run A | Clean Run B (conc 8, 128 prompts) |
| --- | --- | --- | --- |
| Benchmark duration (s) | 130.33 | 925.75 | 754.07 |
| Generated tokens | 2,683 | 4,096 | 16,384 |
| Output token throughput (tok/s) | 20.59 | **4.42** | **21.73** |
| Peak output token throughput (tok/s) | 57.00 | 6.00 | 39.00 |
| Mean TTFT (ms) | 453.99 | **275.50** | 8018.94 |
| Median TTFT (ms) | 368.53 | 274.90 | 9128.45 |
| P99 TTFT (ms) | 732.55 | 285.31 | 19248.36 |
| Mean ITL (ms) | 43.16 | **223.86** | 305.37 |
| Median ITL (ms) | 18.50 | 225.53 | 263.54 |
| P99 ITL (ms) | 105.77 | 227.08 | 2432.69 |
| Median E2EL (ms) | 2107.65 | 28929.17 | 51233.33 |

**Diagnosis in progress — three pieces of evidence point away from the engine and at the hardware:**

1. **Prefill is fine; decode collapsed.** Mean TTFT actually *improved* (454 → 276 ms) while mean ITL degraded ~5x (43 → 224 ms). Prefill is compute-bound; decode is memory-bandwidth-bound, streaming all ~1.95 GiB of weights per generated token. A change that spares prefill and destroys decode is a bandwidth problem, not a compute or engine problem.
2. **The ITL distribution went flat.** Clean Run A reports mean 223.86 / median 225.53 / P99 227.08 — within 1% of each other across 15 minutes. That is the signature of a hard cap holding the GPU at a fixed rate, not of contention or scheduling (which produce skew, as the earlier run's 18.5 median / 105 P99 did).
3. **Batching scaling *improved*,** 2.4x before → 4.9x now (4.42 → 21.73 tok/s). Expected if memory bandwidth is the binding constraint: batching amortizes one weight-read across more sequences, so scarcer bandwidth makes batching help *more*. The constraint got tighter, and the scaling curve responded accordingly.

**Leading hypothesis:** the GPU memory clock has dropped into a power-saving state — laptop on battery, a balanced/quiet Windows power profile, or the NVIDIA driver's adaptive power management. Laptop GPUs can drop memory clock by close to an order of magnitude in these states, which would hit decode roughly this hard while barely touching prefill.

Created `benchmarks/watch_gpu.ps1` to sample `clocks.sm`, `clocks.mem`, `power.draw`, `pstate` and `clocks_throttle_reasons.active` once per second under load, to confirm or kill the hypothesis with data rather than act on a guess.

**Neither run is recorded as the baseline.** `benchmarks/baseline.md` stays unfilled until the GPU is in a known, stable, reproducible power state — a baseline measured under an unexplained hardware cap would silently invalidate every Stage 11 comparison built on it.

**Status:** ⚠️ blocked — diagnosing GPU power/clock state before recording any baseline

---

## Stage 2 — Root cause confirmed: GPU pinned in P8 under full load — 2026-09-07

**What we did:**
Sampled GPU clocks, power, temperature and throttle reasons once per second with `benchmarks/watch_gpu.ps1` while running a short benchmark.

**Command(s) run:**

```powershell
.\benchmarks\watch_gpu.ps1            # third terminal
.\benchmarks\run_bench_serve.ps1 -NumPrompts 8
```

**Result — conclusive:**

```text
timestamp, clocks.sm, clocks.mem, power.draw, temperature.gpu, utilization.gpu, pstate, throttle
16:36:24,  210 MHz,   405 MHz,    13.94 W,    70,              100 %,           P8,     0x0000000000000024
```

Sustained across the entire generation phase: **P8 at 100% GPU utilization.** P8 is the *idle* performance state. 210 MHz core and 405 MHz memory are idle clocks — memory is running at roughly **1/15th of this card's rated speed**. Power draw ~14 W against a 35 W cap.

Throttle bitmask `0x24` decodes to two flags OR'd together:

| Bit | Meaning |
| --- | --- |
| `0x04` | SwPowerCap — driver capping clocks to hold a power limit |
| `0x20` | SwThermalSlowdown — driver capping clocks to hold a *software* thermal target |

`SwThermalSlowdown` active at **64–73 °C** is the anomaly. This silicon throttles for real in the high 80s. A software thermal limit tripping in the low 70s is a configured policy — an OEM quiet/eco profile, battery operation, or driver power management — not the GPU protecting itself.

Benchmark under these conditions: mean ITL **224.09 ms**, mean TTFT **2165.56 ms**, output throughput **4.15 tok/s**, 1024 generated tokens (exact, so the harness itself is correct).

**Why this matched the earlier diagnosis exactly:** decode streams all ~1.95 GiB of weights per generated token and is memory-bandwidth-bound; prefill is compute-bound with high arithmetic intensity. A 15x memory clock reduction therefore devastates ITL while leaving TTFT comparatively intact — precisely the split observed. The flat ITL distribution (mean 224.09 / median 225.74 / P99 227.65) was the hard clock cap.

**Note:** Stage 0's very first `nvidia-smi` reported `P0` at 15 W, so P0 is reachable on this machine. Something moved it into a state it is not leaving.

**Status:** ❌ blocked — GPU must reach P0 under load before any baseline is recorded

---

## Stage 2 — Power state fixed: 11.6x throughput recovered — 2026-09-07

**What we did:**
Corrected the host power/thermal configuration and re-measured with the clock sampler running.

**Result — resolved:**

| Metric | Throttled (P8) | Corrected (P0) | Change |
| --- | --- | --- | --- |
| `clocks.mem` | 405 MHz | **5501 MHz** | 13.6x |
| `clocks.sm` (boost → sustained) | 210 MHz | 1372 → 712 MHz | — |
| `power.draw` | ~14 W | 32–46 W | — |
| `pstate` under load | P8 | **P0** | — |
| Mean ITL (ms) | 224.09 | **20.42** | 11.0x faster |
| Mean TTFT (ms) | 2165.56 | **53.53** | 40x faster |
| Output token throughput (tok/s) | 4.15 | **47.99** | 11.6x |
| Benchmark duration, 8 prompts (s) | 246.79 | 21.34 | 11.6x |

**11.6x throughput from a host power setting alone** — no change to the model, engine, or any vLLM flag. Worth stating plainly in the Stage 13 report: on laptop hardware, verifying the GPU power state is a precondition for any benchmark being meaningful, and the failure is silent. Nothing in the vLLM logs indicated a problem; only `nvidia-smi` clock sampling revealed it.

**New finding — thermal throttling is now the binding constraint, and it varies within a single run:**

```text
16:48:26   1372 MHz, 43.42 W, 85 C, 0x04  (SwPowerCap)
16:48:29   1327 MHz, 44.15 W, 86 C, 0x20  (SwThermalSlowdown)
16:48:31   1012 MHz, 39.35 W, 87 C, 0x20
16:48:36    765 MHz, 32.64 W, 86 C, 0x20
16:48:43    712 MHz, 32.45 W, 87 C, 0x20   <- steady state
```

The card boosts to 1372 MHz, reaches 87 °C within ~3 seconds, and settles at **712 MHz — a 48% clock reduction — inside a single 21-second run.** This is real silicon thermal limiting at 87 °C, unlike the earlier software policy tripping at 73 °C.

Two consequences:

1. **Memory clock does not throttle** (held at 5501 MHz throughout). Decode is bandwidth-bound, so ITL is largely protected — consistent with its tight distribution (median 21.12, P99 27.99). Prefill is compute-bound, so **TTFT is the metric that will degrade under sustained load.**
2. **Results now depend on run length.** A short run measures boost clocks; a long run measures thermally-limited steady state. Comparing runs of different durations would measure thermal mass rather than any optimization — the exact trap Stage 11 must avoid.

**Action:** `benchmarks/baseline.md` gains a mandatory thermal protocol — a discarded warm-up run to reach steady state, then the measured run — applied identically to every future comparison.

**Status:** ✅ root cause resolved — recording the real baseline next, under the new thermal protocol

---

## Stage 2 — Baseline recorded — 2026-09-07

**What we did:**
Recorded the real baseline under the thermal protocol (discarded warm-up, then measured runs back to back), with GPU state verified in P0 throughout via `watch_gpu.ps1`.

**Command(s) run:**

```powershell
.\benchmarks\run_bench_serve.ps1 -NumPrompts 8          # warm-up, discarded
.\benchmarks\run_bench_serve.ps1                        # Run A
.\benchmarks\run_bench_serve.ps1 -Concurrency 8 -NumPrompts 128   # Run B
```

**Result:** full numbers in `benchmarks/baseline.md`. Headlines:

| Metric | Run A (conc 1) | Run B (conc 8) | Change |
| --- | --- | --- | --- |
| Total generated tokens | 4096 (exact) | 16384 (exact) | — |
| Output token throughput (tok/s) | 33.39 | **137.15** | **4.11x** |
| Median ITL (ms) | 26.13 | 29.51 | +13% |
| P99 ITL (ms) | 33.62 | **637.62** | **19x worse** |
| Median TTFT (ms) | 654.13 | 758.76 | +16% |
| Mean TPOT (ms) | 26.12 | 44.90 | +72% |

**The thermal protocol justified itself immediately.** Warm-up (22 s, boost clocks) vs Run A (123 s, steady state), identical config: mean ITL 21.57 → 25.92 ms (+20%), mean TTFT 51.54 → 515.36 ms (+900%). Precisely the predicted compute-bound/bandwidth-bound split — memory clock holds at 5501 MHz so ITL barely moves, SM clock falls ~1455 → 712 MHz so prefill collapses. Recording the warm-up as the baseline would have turned every future TTFT comparison into a measurement of GPU run duration.

**GPU state during Run B** (sampled): `P0` throughout, `clocks.mem` steady at **5501 MHz**, `clocks.sm` settling at **712 MHz**, 27–31 W, 87–89 °C, throttle `0x20` (SwThermalSlowdown) sustained. Thermal limiting is now the binding constraint, and it is genuine silicon limiting at ~88 °C rather than the earlier bogus 73 °C policy.

**Three findings worth carrying forward:**

1. **Batching is worth 4.11x throughput for a 13% median ITL cost** — 51% scaling efficiency against 8x concurrency. The shortfall is expected: decode is memory-bandwidth-bound and eight sequences contend for the same bus.
2. **The cost hides in the tail.** P99 ITL degrades 19x while the median moves 13%. Those are scheduler pauses when an arriving request's prefill preempts ongoing decode. Mean TPOT (44.90 ms) smooths it away entirely — **only P99 ITL exposes it.** This is the exact signature Stage 10 should look for when forcing a KV-cache eviction.
3. **KV cache is not the constraint at this concurrency** — 8 requests × ~640 tokens ≈ 5,120 of 28,432 available (~18%, estimated; to be confirmed with the Stage 8 Grafana panel). **Stage 9's concurrency ceiling will therefore be compute/thermal, not KV memory** — counterintuitive for a 4 GiB card, and a direct consequence of `--max-model-len 4096` keeping per-request cache demand small.

**Deferred:** the supplementary offline benchmark (`vllm bench latency` / `throughput`). It would only quantify HTTP and scheduling overhead; `PRODUCT_SPEC.md`'s required TTFT/ITL/throughput are all satisfied by the serving benchmark, measured on the path the product actually ships. Reasoning recorded in `baseline.md`.

**Engine-side metrics from the server logs (Run B):** steady **15.4–18.5% GPU KV cache usage** with `Running: 8 reqs, Waiting: 0 reqs` throughout — the scheduler never queued. This measures what was previously estimated at ~18% and confirms finding 3 above.

**Caveat found in the server logs — prefix cache contamination across runs:** the engine reported a **63–75% prefix cache hit rate** during Run B, far above what a dataset with `random_prefix_len 0` should produce. The hits are cross-run: `--seed 42` generates identical prompts every invocation, prefix caching is enabled, and the cache persists in the server process between benchmark runs. Run B partly reused KV blocks computed during Run A and the warm-up.

Consequence: **recorded TTFT is optimistic** (some prefills partly cached); ITL is unaffected, since prefix caching only accelerates prefill. **New rule recorded in `baseline.md`: restart the vLLM server between measured comparison pairs**, or the second run inherits the first's cache and looks better for reasons unrelated to the change under test. Critical for Stage 11's prefix-caching experiment, which needs a cold cache as its control.

**The fix, recorded for reproducibility** (now in `baseline.md`) — two settings, worth 11.6x throughput between them:

1. Windows → Settings → System → Power & battery → **Power mode → Best performance**
2. NVIDIA Control Panel → Manage 3D settings → Global Settings → **Power management mode → Prefer maximum performance**

The charger was **not** the cause, and the vendor thermal profile was not touched. **NVIDIA's default "adaptive/optimal power" setting alone was enough to hold the card in P8 under a full CUDA compute workload.** That is the transferable lesson: the adaptive policy is a graphics-oriented heuristic, and a continuous stream of small memory-bound kernel launches — which is exactly what LLM decode looks like — does not trip it into a performance state. Anyone benchmarking inference on a consumer NVIDIA GPU under Windows should set this explicitly and verify P0 before trusting a single number.

**Status:** ✅ Stage 2 complete — baseline recorded in `benchmarks/baseline.md`

---

## Stage 3 — Gateway skeleton: auth + streaming proxy — 2026-09-07

**What we did:**
Built `gateway/main.py` (FastAPI), `gateway/requirements.txt`, `gateway/test_gateway.ps1`. API-key auth plus a transparent streaming proxy to vLLM. No usage tracking, budgets, or database — deliberately deferred to Stage 4 so a failure here has exactly one possible cause.

**Command(s) run:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r gateway\requirements.txt
uvicorn gateway.main:app --port 8080 --reload
.\gateway\test_gateway.ps1
```

**Result — all five checks passed:**

1. `/health` → `gateway: ok, vllm: ok`
2. Valid key → completion returned ("gateway works", prompt=38 completion=3)
3. Invalid key → 401
4. Missing Authorization header → 401
5. Streaming → 37 chunks with steadily increasing timestamps

**Streaming timings (the result this stage existed to produce):**

```text
chunk  1 at  88 ms      <- TTFT
chunk  3 at 100 ms
chunk  4 at 117 ms      <- gaps of 11, 17, 17 ms
chunk  5 at 134 ms
chunk 20 at 382 ms      <- 16.5 ms/chunk from chunk 5
37 chunks over 664 ms   <- ~16 ms/chunk overall
```

Timestamps increase steadily across all 37 chunks — **streaming is not buffered.** Had it been, all chunks would have arrived together at ~664 ms.

**Predicted vs actual:**

| Metric | Predicted | Actual | Verdict |
| --- | --- | --- | --- |
| First chunk (TTFT) | 50–650 ms | **88 ms** | ✅ boost-clock end of range |
| Gap between chunks (ITL) | 20–30 ms | **~16 ms** | ⚠️ faster than predicted |
| Gateway overhead | < 10 ms | not isolated | ⚠️ unverified |

**Why the ITL prediction was wrong — the useful part:**
16 ms against a 26.13 ms baseline, for two compounding reasons.

1. **Cold card.** The baseline was thermal steady state at 712 MHz; the GPU had been idle so this ran at boost. Baseline's own warm-up comparison showed the same effect (21.57 ms boosting vs 25.92 ms steady).
2. **Much shorter context** — the factor not accounted for in the prediction. Decode cost is weight streaming (constant, ~1.95 GiB/token) **plus attention reading the entire KV cache so far, which grows with context.** Baseline used 512-token prompts; this test used 38.

**Carry forward: ITL is not a constant of the model — it grows with context length.** A 4,000-token conversation will show measurably worse ITL than a 40-token one on identical hardware. Directly relevant to Stage 6's chat UI, where conversation history accumulates, and to interpreting Stage 9's ramp.

The same reasoning explains TTFT (88 ms vs 654 ms baseline): prefill is compute-bound and scales with prompt length — ~13.5x fewer prompt tokens plus roughly 2x the clock. **The numbers are internally consistent with the model of the system built across Stages 1 and 2**, which is the substantive result.

**Not measured:** gateway overhead in isolation. Would require the identical request against `:8000` directly for comparison. Claimed <10 ms; unverified.

**Issues hit / fixed:**
None. First run passed.

**Status:** ✅ Stage 3 complete and confirmed working

---

## Stage 4 — Usage tracking, budgets, product dashboard — 2026-09-07

**What we did:**
Added SQLite-backed request logging and per-key token budgets, plus the server-rendered `/dashboard` (Jinja2). New files: `gateway/db.py`, `gateway/seed_db.py`, `gateway/__init__.py`, `gateway/templates/dashboard.html`, `gateway/test_budget.ps1`; `gateway/main.py` rewritten.

**Command(s) run:**

```powershell
pip install -r gateway\requirements.txt
python -m gateway.seed_db
uvicorn gateway.main:app --port 8080 --reload
.\gateway\test_budget.ps1
```

**Issue hit and fixed — PowerShell encoding trap:**
The first version of `test_budget.ps1` failed to parse with a cascade of misleading "missing closing brace" errors pointing at lines 81, 114, 131. Root cause was visible only in the mojibake `â€"`: the file was written as UTF-8, but **Windows PowerShell 5.1 reads `.ps1` files as Windows-1252 unless they carry a UTF-8 BOM.** A UTF-8 em dash (`E2 80 94`) decodes as three cp1252 characters whose last byte `0x94` is U+201D, a curly closing quote — which PowerShell accepts as a string delimiter. A dash inside a string literal therefore terminated the string early.

Earlier scripts were unaffected because their em dashes appeared only inside `#` comments, where a stray quote is harmless. **Rule adopted: `.ps1` string literals stay pure ASCII.**

Also fixed: `Invoke-WebRequest` without `-UseBasicParsing` prompts interactively with a security warning on every call under PS 5.1, which would block any unattended run.

**Result — all four checks passed:**

1. Streamed request carried a usage chunk: `prompt_tokens: 38, completion_tokens: 80, total_tokens: 118` — proving the injected `stream_options.include_usage` works
2. `response_format: json_object` returned parseable JSON (`{"city": "Paris", "country": "France"}`)
3. Budget exhaustion: 6 requests served, **429 on request 7**
4. `X-Budget-Remaining` decreased monotonically: 2000 → 1664 → 1328 → 992 → 656 → 320 → 429

**Predicted vs actual:**

| Metric | Predicted | Actual | Verdict |
| --- | --- | --- | --- |
| Requests before 429 | 6 or 7 | **429 on request 7** | ✅ exact |
| Tokens per request | ~325 | 336 (36 prompt + 300 completion) | ✅ |
| Final `tokens_used` | slightly over 2,000 | **2,016 (100.8%)** | ✅ |
| Streamed usage recorded | yes | yes | ✅ |
| `response_format` survives | yes | yes | ✅ |

**Dashboard confirmed:** `dev-key-beta` at 100.8%, remaining cell reading "exhausted", red consumption bar; the 429 logged in Recent requests with 0 tokens. Totals: 9 requests, 2,196 tokens, 1 error.

**Three findings worth carrying forward:**

1. **The overshoot is bounded and quantifiable.** Request 6 passed the check at `used=1680 < 2000`, then consumed 336, landing at 2,016 — an overshoot of exactly 16 tokens (0.8%). The general bound is **`(prompt_tokens + max_tokens) × requests_in_flight`**. At concurrency 1 that is one request's worth. **Under Stage 9's load test at concurrency 8, up to eight requests could pass the check simultaneously, so overshoot could reach ~2,700 tokens.** Worth verifying there rather than assuming.
2. **Rejecting is ~640x cheaper than serving.** The 429 took **8 ms**; served requests took ~5,100 ms. This is why the budget check runs *before* the upstream call — protecting the GPU from work it should not do is the actual point of a budget, not merely accounting for it afterwards.
3. **Latency remains internally consistent with earlier stages.** Served requests took ~5,205 ms for 300 completion tokens ≈ **17.35 ms/token**, matching Stage 3's ~16 ms with similarly short prompts (36 tokens here). The same model of the system continues to predict its own behaviour across four stages.

**Status:** ✅ Stage 4 complete and confirmed working

---

## Stage 5 — Contract test (Boundary 1) — 2026-09-07

**What we did:**
Built a two-layer test suite: 22 contract tests against a stub upstream (`httpx.MockTransport`) and 1 integration test against the real stack. New files: `gateway/tests/conftest.py`, `gateway/tests/test_structured_output.py`, `pytest.ini`, `gateway/requirements-dev.txt`.

**Command(s) run:**

```powershell
pip install -r gateway\requirements-dev.txt
pytest -v
pytest -m integration -v
```

**Result:** `22 passed, 1 deselected` in 6.07 s; `pytest -m integration` → `1 passed`.

**Issue hit and fixed — the stub reproduced the wrong delivery mode:**
First run: 20 passed, **2 failed** — both streaming tests, with `httpx.StreamConsumed` raised inside the gateway's `tee()`.

The stub built its SSE response with `content=<bytes>`. httpx treats a `Response` constructed from bytes as **already read**, so there is no stream left and `aiter_raw()` raises immediately. Real vLLM returns a genuine stream, so **the gateway was correct and the stub was wrong** — same bytes, wrong delivery mode. Fixed by making the stub body an async generator.

**Lesson worth keeping: a stub must reproduce the MODE of the real dependency, not merely its payload.** A stub returning the right bytes the wrong way exercises a different code path than production takes — which is how a suite goes green while the real system is broken. Here it failed loudly, which is the good outcome.

**Secondary lesson — reading async tracebacks:** the failure produced ~90 lines of `ExceptionGroup` wrapping from anyio/starlette task groups, and every useful word was in the last four lines of the innermost sub-exception. **Read async tracebacks bottom-up, innermost-first**; the `+-+------ 1 ------` marker is where the real error starts.

**Predicted vs actual:**

| Metric | Predicted | Actual | Verdict |
| --- | --- | --- | --- |
| Contract tests | 22 passed | 22 passed | ✅ |
| Integration | 1 deselected by default | 1 deselected | ✅ |
| Runtime | < 2 s | **6.07 s** | ⚠️ under-estimated |
| First-run outcome | all green | 2 failed (stub bug) | ⚠️ |

Runtime miss explained: the `gateway` fixture is function-scoped, so each of the 22 tests runs the app's full lifespan and builds a fresh SQLite database — roughly 0.27 s each. Accepted deliberately. Stage 4 showed how quickly a 2,000-token budget is consumed; a shared database would let one test's spending fail an unrelated later test, producing order-dependent failures.

**The test that carries the stage:** `test_unknown_future_field_survives` sends `some_field_invented_in_2027` and asserts it arrives intact. Field-specific tests only prove someone remembered that field; **this one asserts the gateway has no allowlist at all**, which is the actual Boundary 1 guarantee. A typed-model gateway passes every field-specific test while failing this one — exactly how the failure reaches production.

**Status:** ✅ Stage 5 complete and confirmed working — revisited in Stage 10, where these tests are deliberately broken to prove they bite

---

## Stage 6 — Minimal chat UI — 2026-09-07

**What we did:**
Built `ui/index.html` (plain HTML/CSS/JS, no framework, no build step) and added a `/chat` route to the gateway that serves it. The page streams responses, and displays TTFT, tokens/sec, reply size, context size and remaining budget live.

**Design decision not in the plan:** the page is **served by the gateway** rather than opened as a file. A `file://` page has origin `null`, so the browser blocks its response from `localhost:8080` under the same-origin policy even though the request succeeds. The alternative — CORS middleware — would mean deciding which origins may call an authenticated API, and permissive CORS on a keyed API is a genuine security mistake. Same-origin avoids the decision entirely. Reasoning in `DECISIONS.md`.

**Result:** streaming rendered progressively, the dashboard recorded every turn, and the budget counter decremented live.

| Turn | Context sent | TTFT | Speed | Reply |
| --- | --- | --- | --- | --- |
| 1 | 31 tok | 555 ms | 52.8 tok/s | 13 tok |
| ~6 | **1,709 tok** | **591 ms** | 49.5 tok/s | 512 tok |

**Finding 1 — prefix caching keeps TTFT flat across a conversation.**
Context grew **55x** while TTFT moved 555 → 591 ms. Prefill is compute-bound and should scale with prompt length; 1,709 tokens at the measured ~300–900 tok/s prefill rate would cost seconds. It did not, because **each turn's prompt is the previous turn's prompt plus new text**, so the entire prior conversation is a prefix-cache hit and only the genuinely new tokens are prefilled. `--enable-prefix-caching` (Stage 1) doing exactly the job it was chosen for. Multi-turn chat is its ideal workload. Stage 11 measures this deliberately; here it appeared by accident, which is stronger evidence.

**Finding 2 — the context effect on decode is real but small, and the Stage 1 arithmetic predicts it.**
Three replies of exactly 512 tokens at growing context:

| Request | Context | Duration | Effective speed |
| --- | --- | --- | --- |
| #13 | 202 tok | 10,345 ms | 49.5 tok/s |
| #14 | 755 tok | 10,456 ms | 48.9 tok/s |
| #16 | 1,709 tok | 10,608 ms | 48.3 tok/s |

Context grew 8.5x; speed fell **2.4%**. The prediction from Stage 1's memory model:

```text
KV read at 1,709 context = 1,709 x 36.1 KiB  ~= 62 MB
Weight read per token                        ~= 1,950 MB
ratio                                        ~= 3.1%
```

**Predicted ~3%, measured 2.4%.**

**Refinement to the Stage 3 lesson (which over-sold this):** decode on this model is **weight-bandwidth-dominated, not attention-dominated.** Even at the full 4,096-token limit the KV read is only ~7% of the weight read. Doubling ITL would need roughly **56,000 tokens** of context — far beyond this deployment. Context growth becomes first-order only on models with much larger KV per token (MHA rather than GQA) or 100k+ context windows.

**Predicted vs actual:**

| Metric | Predicted | Actual | Verdict |
| --- | --- | --- | --- |
| TTFT, first short turn | 60–200 ms | **555 ms** | ❌ wrong layer + warm card |
| Speed | 50–60 tok/s | 52.8 tok/s | ✅ |
| Context by turn ~6 | several hundred | 1,709 tok | ✅ |
| Speed degradation with context | "measurably lower" | only 2.4% | ⚠️ over-sold |

**TTFT miss explained:** partly the card being thermally settled at 712 MHz rather than cold at boost, but mainly that **the UI measures TTFT at a different layer** — time to the first chunk containing actual content (vLLM's first SSE chunk is an empty role delta, which the UI skips), including browser fetch setup and JS parsing. **Client-side TTFT is always larger than server-side TTFT, and it is the one the user feels.** State which layer a TTFT number was measured at whenever quoting it.

**Status:** ✅ Stage 6 complete and confirmed working

---

## Correction to Stage 1 — the torch.compile cache was never persisted — 2026-09-07

Found while writing the Stage 7 Compose file, and correcting the earlier entry rather than editing it (this file is append-only).

The Stage 1 entry stated that `torch.compile` output "is cached to `/root/.cache/vllm/torch_compile_cache` inside the `hf-cache` named volume, so restarts skip the 54 s compile." **That was wrong.** The volume mounts `/root/.cache/huggingface`; the compile cache lives at `/root/.cache/vllm`, which is *outside* it — in the container's writable layer, destroyed by `--rm` on every run.

So every launch has paid the full ~60 s compile. The Stage 1 explanation that the cache key changed when the revision flags were added is still true — the directory hash did change from `6ee1e77343` to `33c2863007` — but it was not the only reason, and on its own it would have been a one-off. The recompile was in fact unavoidable every time.

**Fixed** by adding a `vllm-cache:/root/.cache/vllm` volume to `docker-compose.yml`, `run_vllm.ps1` and `run_vllm.sh`. This matters beyond convenience: Stage 10 kills and restarts vLLM mid-load-test, and Stage 11 launches once per optimization variant. A minute of compile per launch would contaminate both.

**Lesson:** mounting a cache volume only helps if it covers the path the cache actually uses. Verify with `docker exec <container> ls <path>` rather than assuming a parent directory covers a child.

---

## Stage 7 — Containerized stack, one command — 2026-09-07

**What we did:**
Created `gateway/Dockerfile`, `.dockerignore` and `deployment/docker/docker-compose.yml` wiring vLLM + gateway together, with Prometheus and Grafana defined behind a Compose profile for Stage 8.

**Command(s) run:**

```powershell
cd deployment\docker
docker compose up --build
docker compose up -d
```

**Issue hit and fixed — a healthcheck that could not execute:**
First `compose up` left vLLM at `health: starting` for over five minutes and the gateway never started at all, because it waits on `condition: service_healthy`.

Cause: the healthcheck invoked `python`, but the vLLM image is Ubuntu-based and has no `python` alias — visible in the container's own command line, `python3 -m vllm.ent…`. The probe failed every 15 s with "not found". **During `start_period`, failing probes do not mark a container unhealthy — they leave it at `starting`, so the failure was completely silent.**

Doubly ironic: the same file carries a comment about not depending on whether `curl` exists in a given image tag, and then made the identical assumption about `python`.

**Two habits that would have caught it immediately:**

1. Verify the interpreter exists before depending on it — `docker exec vllm python3 -c "print(1)"` takes two seconds.
2. **`docker inspect --format="{{json .State.Health}}" <container>`** is the only place a failing probe's stderr surfaces. First thing to run whenever a container sits at `starting` longer than expected.

**Result after the fix — everything green:**

- `docker compose up -d` → vLLM `Healthy` in **63.8 s**, gateway `Started` immediately after
- `docker compose ps` → both `healthy`
- `/health` → `gateway: ok, vllm: ok, vllm_url: http://vllm:8000` — container-to-container DNS confirmed, not a fallback to a published host port
- Chat UI works at `http://localhost:8080/chat`

**The compile-cache correction validated:**

```text
Directly load the compiled graph(s) for dynamic shape from the cache, took 2.412 s
torch.compile takes 7.79 s in total          (was 33.35 s on the cold run)
```

vLLM reached `Application startup complete` in **48 s** vs 87 s cold. Predicted ~50 s. The `vllm-cache` volume fixes the error found earlier in this log.

**The external `hf-cache` volume worked:** `Model loading took 1.9542 GiB and 3.96 seconds` — no re-download. Compose created `inference-product_vllm-cache` and `inference-product_gateway-data` but not an hf-cache, because `external: true` pointed it at the existing volume.

**Build was 25.3 s** (predicted 60–120 s) — the Dockerfile's layer ordering put `pip install` in a cached layer above the source copies.

**Three findings worth carrying forward:**

1. **KV cache size is not bit-identical across runs.** 28,432 → **28,480 tokens**; `num_gpu_blocks` 1777 → **1780**; concurrency 6.94x → 6.95x. Same image, flags and model — vLLM **profiles free VRAM at startup**, and Windows compositor usage varied by a few MB. Three blocks ≈ 48 tokens ≈ 1.7 MB. **For Stage 11: a ±0.2% swing in KV cache size is run-to-run noise, not a result.**
2. **The first request after startup costs far more.** Chat UI showed TTFT **3,292 ms** on the first message (35 tokens of context) versus **310 ms** later at *389* tokens of context — one-time kernel autotuning, first tokenization and chat-template detection. **Stage 9 must discard the first request**, and in Stage 10 the first request after a vLLM restart will look alarming without being a problem.
3. **Healthchecks are now background traffic.** vLLM's self-check plus the gateway's `/health` (which proxies to vLLM) each fire every 15 s. Harmless, but non-zero baseline load that will appear in Stage 8's Prometheus counters and Stage 9's measurements.

Also reproduced in-container: prefix cache hit rate climbing **67.9% → 81.5% → 88.2%** across a conversation, matching the Stage 6 finding.

**Status:** ✅ Stage 7 complete and confirmed working

---

## Stage 8 — Prometheus + Grafana (Boundary 3) — 2026-09-07

**What we did:**
Created `observability/prometheus/prometheus.yml` (scraping vLLM's `/metrics` directly), Grafana datasource and dashboard provisioning, and a 13-panel dashboard `vllm-engine.json`. Started via the Compose profile added in Stage 7.

**Command(s) run:**

```powershell
docker compose --profile observability up -d
(Invoke-RestMethod http://localhost:9090/api/v1/targets).data.activeTargets | Select-Object scrapeUrl, health
.\benchmarks\run_bench_serve.ps1 -Concurrency 8 -NumPrompts 64
```

**Result:** both scrape targets `up` (`http://vllm:8000/metrics`, `http://localhost:9090/metrics`). Grafana loaded and normalised the provisioned dashboard (re-serialised at `schemaVersion 40` with panel IDs and `pluginVersion 11.5.1`), datasource variable resolved to a real UID.

**Issue hit and fixed — wrong metric name:**
The dashboard's KV cache panel used `vllm:gpu_cache_usage_perc`. The engine actually exports **`vllm:kv_cache_usage_perc`** — renamed somewhere in the V1 line, and written from memory rather than from the endpoint. The panel would have shown "No data".

Caught by listing every exported metric rather than grepping for guesses:

```powershell
(Invoke-WebRequest http://localhost:8000/metrics -UseBasicParsing).Content -split "`n" | Select-String "^# TYPE"
```

**Lesson: enumerate what a system exports before writing queries against it.** A first attempt grepped for four guessed names with `-First 12`, which truncated the output and proved nothing either way — a worse diagnostic than asking the open question.

**Two metrics found in the listing and added to the dashboard:**

1. **`vllm:inter_token_latency_seconds`** — a *separate* histogram from `time_per_output_token_seconds`. vLLM exports both, which is the Stage 2 ITL-vs-TPOT distinction built into the engine: TPOT averages across a request and smooths stalls away, ITL is per-token and exposes them. Given Stage 2 measured P99 ITL degrading 19x while mean TPOT moved 72%, ITL got its own panel.
2. **`vllm:request_queue_time_seconds`** — separates "the GPU got slower" from "requests are waiting behind others", which is precisely the ambiguity Stage 9 has to resolve when TTFT rises.

**Finding — the prefix cache contamination effect, now quantified.**
Two runs of the identical command, same seed, six minutes apart, nothing changed between them:

| Metric | 07:49 | 07:55 | Change |
| --- | --- | --- | --- |
| Duration | 51.27 s | 44.17 s | −13.9% |
| Output throughput | 159.78 tok/s | **185.45 tok/s** | **+16.1%** |
| Median TTFT | 2,583.57 ms | 2,190.65 ms | −15.2% |
| Median ITL | 25.64 ms | 20.45 ms | −20.2% |
| P99 ITL | 52.07 ms | **28.36 ms** | **−45.5%** |

The only difference is a warmer prefix cache on the second run. **Two identical runs differ by ~16% throughput from cache state alone.** This gives the `baseline.md` rule a number: in Stage 11, any optimization claiming under ~16% improvement, measured without a server restart, is indistinguishable from cache warming.

**Second-order effect worth naming:** P99 ITL improved **45%**. Tail ITL spikes are scheduler stalls where an arriving request's prefill interrupts ongoing decode; cheaper (cached) prefill means shorter stalls. **Prefix caching therefore improves the decode tail, not only prefill** — not obvious, and it explains part of the gap to the Stage 2 baseline's 637 ms P99.

**Also noted:** these runs are *not* comparable to the Stage 2 baseline (64 prompts vs 128, and different cache state). TTFT of ~2,200–2,600 ms here versus 759 ms in the baseline is closer to the honest cost of a cold 512-token prefill; the baseline figure was flattered by cache reuse, exactly as its caveat recorded.

**Scope decision:** GPU hardware metrics (temperature, clocks, power) are deliberately **not** in Grafana — vLLM exports engine state only, and dcgm-exporter expects Linux driver access patterns WSL2/WDDM does not provide reliably. `benchmarks/watch_gpu.ps1` remains the tool for hardware state. Given that Stage 2's 11.6x loss was invisible in every engine metric and only visible in `nvidia-smi`, pretending a dashboard covered it would be worse than saying it does not. Recorded in `DECISIONS.md`.

**Panel confirmation — all 13 render with data:**

- **GPU KV cache usage** peaked at **17.1%**, matching the Stage 2 measurement of 18.5%
- **Prefix cache hit rate** showing both series
- **Requests running vs waiting**: running peaked at 8, **waiting peaked at 3**
- **Inter-token latency**: p99 visibly above p50
- **Preemptions**: flat at zero
- No panel showed "No data"

**Correction to guidance I gave — `num_requests_waiting` is not purely a saturation signal.**
The panel description originally said "waiting > 0 sustained means demand exceeds concurrency." That is too simple. **Waiting also counts requests that have arrived but not yet been admitted to the running batch in the current scheduler step**, so with continuous arrivals a small transient count is normal — observed at 3 while running was pinned at 8 and the client's concurrency limit was 8.

The real saturation signal is waiting staying *elevated* while running sits at max. **The Queue time panel is the better instrument**, measuring how long requests actually waited rather than how many were momentarily unscheduled. Panel description corrected.

**The Queue time panel decomposed TTFT immediately:**

```text
queue p50 ~0.2 s  +  prefill p50 ~1.5 s  ≈  TTFT p50 ~1.7-2.2 s
queue p99 ~2.0 s
```

**Prefill dominates.** The ~2,190 ms median TTFT is not queueing — it is genuinely ~1.5 s to prefill 512 tokens at the thermally-limited 712 MHz. The compute-bound prediction from Stage 2, now measured directly rather than inferred.

**Prefix cache panel reading (cumulative vs 1m):** cumulative sat at ~8% because it covers all history since engine start, diluted by the benchmark's 32,708 prompt tokens. The **1-minute rate** is the one to read: run 1 peaked ~13%, run 2 peaked ~50% — the +16% throughput difference between identical runs, visible as its cause.

**Dashboard readability fixes applied** after the user noted the axes were unusable: KV cache lost its hard 0–100 max (`axisSoftMax: 25`), TPOT and ITL switched to **log scale** so a 2 s spike no longer flattens a 25 ms baseline, and Preemptions got `axisSoftMax: 5` so a flat-zero series does not render against a 0–100 axis.

**Status:** ✅ Stage 8 complete and confirmed working — Boundary 3 demonstrated

---

## Stage 9 — Concurrency ramp and the ceiling — 2026-09-07

**What we did:**
Built `load_testing/ramp.py` — a custom asyncio load generator (per `DECISIONS.md`: no new framework, every measured thing visible) that ramps concurrency against **the gateway**, not vLLM directly, so auth, budget checks, SQLite writes and the streaming proxy are all in the measured path.

**SLO stated before measuring: p95 TTFT < 1.5 s.**

**Design decision carried from Stage 8:** unique prompts per request by default. Stage 8 measured two identical runs differing by 16% throughput purely from prefix-cache warming; a ramp with repeated prompts would measure the cache, not the hardware. A `--shared-prefix` flag inverts this for Stage 11's prefix-caching experiment.

**Command run:**

```powershell
python load_testing\ramp.py
```

**Result:**

| conc | ok/n | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p99 | E2E p50 | SLO |
| ---: | :--: | ----: | -------: | -------: | ------: | ------: | ------: | :--: |
| 1 | 16/16 | 41.3 | 348 | 373 | 16.3 | 20.1 | 1.05 | PASS |
| 2 | 16/16 | 59.2 | 392 | 630 | 18.1 | 313.2 | 1.47 | PASS |
| 4 | 16/16 | 73.9 | 452 | **1379** | 19.7 | 363.3 | 2.22 | PASS |
| **8** | 16/16 | **89.5** | 1152 | **2773** | 22.1 | 704.6 | 3.60 | **FAIL** |
| 16 | 32/32 | 87.3 | 4313 | 6574 | 22.5 | 732.8 | 7.50 | FAIL |
| 24 | 48/48 | 79.4 | 8525 | 10439 | 25.2 | 749.4 | 11.74 | FAIL |
| 32 | 64/64 | 80.2 | 13281 | 14985 | 26.2 | 797.5 | 16.31 | FAIL |

Zero failed requests at every level.

**THE SLO FIRST BREAKS AT CONCURRENCY 8.** (Predicted 4 or 8.)

**Which resource ran out — the evidence:**

| Signal | Value | Reading |
| --- | --- | --- |
| KV cache peak | **13.6%** | Not memory — nowhere close |
| Preemptions | **flat zero** | Nothing was ever evicted |
| Prefill p50 | **flat** across the whole ramp | Prefill did not get slower |
| Queue p50 / p99 | **→ ~12 s / ~15 s** | Requests waiting for admission |
| Running / waiting | pinned **7–8** / peaked **24** | At the `--max-num-seqs` cap |
| ITL p50 | 16.3 → 26.2 ms over 32x concurrency | Decode barely degraded |

**TTFT grew from 348 ms to 13,281 ms and essentially all of it is queueing, not work.** The Queue time panel decomposes it directly — queue exploding while prefill stays flat.

**Verdict: the binding constraint is `--max-num-seqs 8`** — a Stage 1 value chosen against a 4,096-token worst case that this 640-token workload never approaches. Not memory, not decode bandwidth. **Stage 2's prediction that the ceiling would be "thermal/compute rather than memory" was half right: it correctly ruled out memory, but the actual limit is our own configuration, sitting below the hardware limit.**

**The throughput knee — the operationally useful answer:**
Throughput peaks at **89.5 tok/s at concurrency 8** and then *declines* (87.3, 79.4, 80.2). Past the knee, added concurrency buys only latency — textbook saturation.

**Concurrency 4 is the highest level meeting the SLO**, delivering **73.9 tok/s — 83% of peak throughput at roughly half the p95 TTFT** (1,379 ms vs 2,773 ms). That is the configuration to run at under this SLO.

**Caution recorded for Stage 11:** the engine never ran more than 8 sequences, so this data **cannot** say whether the hardware has headroom behind the cap. Memory says yes — 28,480 tokens ÷ ~640 per request ≈ **44 concurrent requests would fit**. But throughput plateauing at 8 hints compute may already be near its limit, and there is a workload reason to suspect so: with unique prompts each request is 512 prompt tokens and only ~42 generated, making this test **prefill-dominated**, and prefill is compute-bound. The Token throughput panel confirms it — prompt tok/s peaked near **900** while generation sat around **150**. Raising `--max-num-seqs` and re-measuring is the clean experiment.

**Flaw found in the harness (fixed):** `max_tokens=128` but replies averaged **~42 tokens** — the model hit EOS early, the same defect that invalidated the first Stage 2 benchmark, reappearing. Harmless for measuring queueing, fatal for Stage 11's before/after comparison. Added an `--ignore-eos` flag.

**Worth noting why that needed no gateway change:** `ignore_eos` is a vLLM extension, not part of the OpenAI schema. It reaches the engine only because the gateway forwards unmodelled fields untouched — **the Boundary 1 decision from Stage 3 paying for itself in a place we did not anticipate when we made it.**

**Status:** ✅ Stage 9 complete and confirmed working

---

## Stage 10(a) — Forced KV cache eviction (Boundary 4) — 2026-09-07

**What we did:**
Forced the engine to exhaust KV cache and preempt running sequences, then checked whether the gateway's own request accounting survived it.

The arithmetic that makes it happen: 476 prompt + 3,500 output ≈ 3,976 tokens per request; 8 concurrent = **31,808 tokens against 28,480 available** — over by ~12%. This is only possible because `--max-num-seqs 8` was deliberately set *above* the measured 6.94x capacity in Stage 1, specifically so this demonstration could exist. That decision paid off nine stages later.

**Command run:**

```powershell
python load_testing\ramp.py --levels 8 --max-tokens 3500 --ignore-eos --no-warmup
```

**Engine-side result — eviction achieved:**

```text
KV cache:     13.6%  ->  98%  ->  22%  ->  98%      (the 98->22 drop IS the eviction)
Preemptions:  0  ->  non-zero, two pulses
Running:      8  ->  briefly 7  ->  8
ITL p99:      spiked to ~1 s
Prefix hit:   spiked to 100% twice, aligned with the preemption pulses
```

**The prefix-cache spikes are the mechanism made visible.** vLLM preempts by *discarding* a sequence's KV cache and recomputing it later; on recompute the surviving blocks are hits, so a 100% hit-rate spike at the exact moment of preemption is the recomputation happening. The eviction and its cost are legible in two panels simultaneously.

**Gateway-side result — the Boundary 4 check, and it is exact:**

| | Before | After | Delta |
| --- | --- | --- | --- |
| Requests | 221 | 237 | **16** |
| Prompt tokens | 103,264 | 110,817 | **7,553** |
| Completion tokens | 10,538 | 66,538 | **56,000** |
| Errors | 0 | 0 | **0** |

- **16 requests** — exactly what the ramp sent, all recorded
- **56,000 completion tokens = 16 × 3,500 exactly.** Not one token lost
- **7,553 prompt tokens ≈ 16 × 472**, matching the 468–482 range in the request log
- **Zero errors** — every client received a 200

**Not one request lost its usage chunk to the eviction.** That is precisely the failure this test exists to detect: a preempted request whose final usage chunk never arrives, silently billing zero and letting budgets drift. The reason it did not happen is structural — the `finally` block in the gateway's `tee()` generator records the request whether the stream completes normally, errors, or is torn down.

**Evictions were entirely invisible to callers.** They only took longer: E2E p50 of **117.9 seconds**.

**Unexpected finding — throughput is a property of the workload, not the system.**
This run measured **205.6 tok/s** against Stage 9's peak of **89.5 tok/s** — same hardware, same config, same concurrency 8. **A 2.3x difference from workload shape alone.** Stage 9 sent 512 in / ~42 out (prefill-dominated, compute-bound); this sent 476 in / 3,500 out (decode-dominated, bandwidth-bound).

**Consequence for Stage 13: every throughput figure in the report must state its input/output ratio, or it misleads.** "This system does 89.5 tok/s" is not a fact about the system. Both numbers are correct and describe different things.

Secondary: ITL p50 rose to 30.7 ms from Stage 9's 22.1 ms — partly the Stage 6 context effect (sequences reaching ~4,000 tokens), partly eviction recompute.

**Status:** ✅ Stage 10(a) complete — Boundary 4 demonstrated

---

## Stage 10(b) — Deliberately broke `response_format` forwarding — 2026-09-07

**What we did:**
Inserted a field allowlist into `gateway/main.py` reproducing the exact Boundary 1 failure — a gateway that rebuilds the request from a known set of fields and silently drops the rest, which is what a typed request model does implicitly. Allowlist kept only `model, messages, max_tokens, temperature, stream, stream_options`.

**Result: 14 failed, 8 passed** — exactly the predicted count.

Failures: `test_response_format_reaches_upstream`, `test_unknown_future_field_survives`, and 12 of the 14 parametrized `test_request_field_survives` cases (`temperature` and `max_tokens` passed because they were on the allowlist).

**The sabotage broke nothing visible.** Requests still returned 200, responses were still valid completions, no error appeared in any log. `response_format` simply never reached vLLM and structured output quietly stopped being structured. That silence is the entire failure mode.

**Two things worth keeping from this run:**

1. **The failure message earned its length.** `AssertionError: response_format was DROPPED by the gateway. This is the Boundary 1 failure: the client asked for structured output and vLLM never heard about it.` A test failing with `assert 'response_format' in {...}` says *what* broke; this says *what it means*, to someone who was not there when it was written.

2. **The 12 parametrized failures are the real lesson.** `top_p`, `top_k`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `n`, `user`, `tools`, `tool_choice`, `guided_json` — all silently dropped. A team debugging "why doesn't `seed` work?" would add `seed` to the allowlist, ship, and remain broken for the other eleven. **This is how the failure mode survives for years: it presents as a series of unrelated small bugs rather than one systemic one.**

**Sabotage reverted**, with a comment left at the site warning against reintroducing field filtering.

**Status:** ✅ Stage 10(b) complete — the contract test is proven to bite

---

## Stage 10(c) — Killed vLLM mid-load-test — 2026-09-07

**What we did:**
Started a load, killed the engine outright with SIGKILL mid-flight, observed the gateway's behaviour, then restarted and verified recovery.

**Command(s) run:**

```powershell
python load_testing\ramp.py --levels 4 --max-tokens 1500 --ignore-eos --no-warmup
docker compose kill vllm          # while the load was running
Invoke-RestMethod http://localhost:8080/health ; docker compose ps
docker compose start vllm
```

**Result — every prediction held:**

| Prediction | Outcome |
| --- | --- |
| In-flight requests fail | **12 of 16 failed**, 4 completed before the kill |
| Gateway process survives | `Up 2 hours` throughout — never crashed or hung |
| `/health` reports upstream down | `{"gateway":"ok","vllm":"unreachable",...}` |
| Recovery without touching the gateway | ✅ automatic once vLLM was healthy |
| **Gateway marked `unhealthy`** | ✅ — **and this was the design flaw we were looking for** |

**Two different upstream errors, correctly distinguished:**

```text
after kill:   "error": "[Errno -2] Name or service not known"
after start:  "error": "All connection attempts failed"
```

`docker compose kill` removes the container from the network, so `vllm` stops resolving in DNS. After `start`, the name resolves but nothing is listening yet. A gateway that flattened both into "upstream error" would lose real diagnostic information.

**THE DESIGN FLAW — liveness vs readiness (fixed).**
The gateway flipped to `(unhealthy)` roughly 45 s after the kill (15 s interval × 3 retries), because `/health` returned 503 whenever vLLM was unreachable and the container healthcheck treats non-200 as failure.

**The gateway was marked unhealthy because something else died.** To an orchestrator, a failing *liveness* probe means "restart this container" — which helps nothing here, and would drop every in-flight request that was about to succeed the moment the engine returned.

**Fix applied:** split the endpoints.

- **`/health` — liveness.** Always 200 while the process runs. Upstream status included as *information*, not as a status code.
- **`/ready` — readiness.** 503 when the gateway cannot serve. This is what a load balancer should route on, and Stage 12's `readinessProbe`. Failing readiness removes a pod from service without killing it, so it rejoins automatically — exactly the recovery behaviour observed here.
- Dockerfile healthcheck now documents why it probes `/health` and not `/ready`.

**A Stage 7 prediction came true.** Two runs at identical settings:

| | Before kill | After recovery |
| --- | --- | --- |
| p95 TTFT | 1,419 ms — PASS | **4,700 ms — FAIL** |

Stage 7 recorded: *"in Stage 10 the first request after a vLLM restart will look alarming and won't be."* The restarted engine paid its cold-start cost on request one, and `--no-warmup` put it inside the measurement.

**Which exposed a real flaw in the ramp methodology (fixed).** With nearest-rank percentiles, **p95 of 16 samples is s[15] — the maximum**:

```text
k = round(0.95 × 16 + 0.5) − 1 = 15   ->  the largest of 16
```

A single outlier therefore *defines* p95. Every "p95" in the Stage 9 ramp at low concurrency was really "the slowest request". **This does not invalidate Stage 9's conclusion** — the queue-time evidence was unambiguous and the effect sizes were order-of-magnitude — but it is a limitation that belongs in the record rather than being rediscovered by someone else.

Fixes: minimum samples per level raised from 16 to **24** (p95 of 24 is the 23rd value, not the largest), a printed warning whenever n < 40, and the `pct()` docstring now states the limitation.

**Second bug found and fixed in `ramp.py`:** the failure summary printed `lr.results[0].error`, but `results[0]` is usually a *success* — so the kill test printed `12 failed at concurrency 4:` with an empty reason, hiding the actual error. Now reports the first genuinely failed result, with its status code.

**Status:** ✅ Stage 10 complete — all three deliberate failures demonstrated, two real design flaws found and fixed

---

## Stage 11, Experiment 1 — Raise `--max-num-seqs` — 2026-09-07

**What we did:**
Created `benchmarks/optimization_results.md` with a measurement protocol assembled entirely from failures found earlier in this project, then ran the first before/after pair. Parameterised `--max-num-seqs` in `docker-compose.yml` as `${VLLM_MAX_NUM_SEQS:-8}` so the comparison is reproducible from the command line rather than from memory.

**This experiment is not in `PROJECT_PLAN.md`.** It was added because Stage 9 produced direct evidence for it, and running the planned experiments while ignoring our own findings would have been strange. Full table in `benchmarks/optimization_results.md`.

**Workload:** 512 prompt in / 256 generated out (2:1) — deliberately between Stage 9's prefill-heavy 12:1 and Stage 10's decode-heavy 1:7, since Stage 10(a) showed throughput varying 2.3x by shape.

**Control group validated the comparison first.** Concurrency 1, 4 and 8 must be unchanged, because below the cap only the client's limit binds:

| conc | Before | After | Δ |
| ---: | -----: | ----: | --: |
| 1 | 55.4 | 52.7 | −4.9% |
| 4 | 159.9 | 158.7 | −0.8% |
| **8** | **256.1** | **254.0** | **−0.8%** |

Concurrency 8 is the strongest control and moved 0.8%. Comparison valid.

**Result — outcome 1: the cap was the binding constraint.**

| Metric | `--max-num-seqs 8` | `--max-num-seqs 24` | Change |
| --- | --- | --- | --- |
| Peak output throughput | 256.1 tok/s | **344.2 tok/s** | **+34.4%** |
| Throughput at conc 16 | 255.0 | 313.4 | +22.9% |
| Throughput at conc 24 | 252.7 | 344.2 | +36.2% |
| TTFT p50 at conc 24 | 17,207 ms | 2,820 ms | **−83.6%** |
| TTFT p95 at conc 24 | 18,739 ms | 7,548 ms | −59.7% |
| ITL p50 at conc 24 | 21.3 ms | 39.9 ms | +87% |
| **ITL p99 at conc 24** | **53.2 ms** | **1,283.3 ms** | **24x worse** |
| Peak KV cache | ~20% | ~60% | — |
| Peak running | 8 | 24 | — |
| Preemptions | 0 | 0 | — |

Engine-side predictions all held: running reached 24, KV cache reached ~60% (predicted 50–65%), preemptions stayed at zero. Throughput gain of +34.4% slightly exceeded the predicted +10–30%.

**The prediction I got badly wrong: ITL degraded 24x on the tail**, not merely "worse". At `--max-num-seqs 24` and concurrency 24, individual users see pauses of over a second between tokens while the median still reads a healthy 40 ms. **Stage 2's lesson recurring at a different layer** — the median says "modest cost", the mean says "somewhat worse", and only P99 reveals that the streaming experience broke.

**The conclusion that matters, and it is uncomfortable: under the stated SLO this optimization is worth nothing.**
The SLO still first breaks at concurrency 8, and concurrency 4 remains the highest passing level at ~159 tok/s — identical before and after. The cap only binds above 8 concurrent requests, and the SLO already fails there.

Honest conditional statement:

- **Holding p95 TTFT < 1.5 s:** `--max-num-seqs` is irrelevant. Concurrency 4, ~159 tok/s.
- **Relaxing to p95 TTFT < 5 s:** `--max-num-seqs 24` reaches **313 tok/s at concurrency 16** (p95 4,819 ms), which the old configuration could not reach at *any* concurrency. **+22.9% capacity from one config value** — if the ITL tail is acceptable.

**An optimization is only meaningful relative to a stated objective.** "+34% throughput" would have been a true and thoroughly misleading headline. Stating the SLO *before* measuring is what makes the difference visible.

**Left untested, worth testing:** `--max-num-seqs 16`. The ITL p99 jump between concurrency 16 (64.9 ms) and 24 (1,283.3 ms) suggests the cliff sits between them, so 16 may capture most of the throughput with far less tail damage.

**Status:** ✅ Stage 11 Experiment 1 complete

---

## Stage 11, Experiment 2 — Shared prefix vs unique prompts — 2026-09-07

**What we did:**
Measured `--enable-prefix-caching`'s effect deliberately, having observed it indirectly three times (Stages 6, 8, 10a). Two arms run consecutively with `--max-num-seqs` reverted to the default 8, vLLM force-recreated before each for a cold cache. Full tables in `benchmarks/optimization_results.md`.

Added a `prompt` column to `ramp.py`'s output first, because the comparison is only valid if prompt lengths are comparable — the shared preamble is a fixed string while unique prompts are generated to a word count, so equality could not be assumed.

**Result — the strongest finding in the project.**

**The SLO holds to concurrency 8 with a shared prefix, versus failing at concurrency 4 without one.** Same hardware, same config, same model; only the prompt structure differs.

| Metric @ conc 8 | Unique | Shared | Change |
| --- | --- | --- | --- |
| **Highest level meeting SLO** | fails at 4 | **passes at 8** | **2x+ SLO-compliant concurrency** |
| TTFT p50 | 2,512 ms | **451 ms** | **−82.0%** |
| TTFT p95 | 3,561 ms | **648 ms** | **−81.8%** |
| Output throughput | 194.2 tok/s | 259.9 tok/s | **+33.8%** |
| ITL p50 | 27.7 ms | 28.4 ms | **+2.5%** |
| ITL p99 | 57.1 ms | 56.4 ms | −1.2% |
| Peak KV cache | ~18% | **~9.5%** | roughly halved |
| Prefix cache hit rate | near 0% | **~95% sustained** | — |
| Mean prompt tokens | 471 | **541** | +15% |

**ITL did not move, and that is the confirming detail.** Prefix caching accelerates *prefill only* — decode still streams all ~1.95 GiB of weights per token regardless of what is cached. The mechanism behaved exactly as the memory model built in Stage 1 predicts.

**Throughput at concurrency 1 was also unchanged** (41.5 → 41.1 tok/s). Correct: with 256 output tokens, one request's wall time is dominated by decode, so removing prefill work barely helps. The benefit appears only when prefill *competes* with decode for the GPU — which is why it scales with concurrency: +0%, +20.5%, +33.8%.

**Two findings that strengthen the result:**

1. **The shared prompts were 15% longer** (541 vs 471 tokens) — the winning arm did *more* nominal prefill work. The comparison is conservative. The `prompt` column added for this check earned itself immediately.
2. **Peak KV cache usage roughly halved** (~18% → ~9.5%). Shared prefix blocks are stored **once** and referenced by every sequence rather than duplicated per request. Prefix caching therefore buys *memory* as well as speed, so more concurrent requests fit. Not predicted; it compounds the benefit.

**THE CAVEAT THAT MATTERS MORE THAN THE RESULT — session drift.**
Arm A's numbers sit **~24% below Experiment 1's baseline** for an identical configuration and workload: 194.2 vs 256.1 tok/s at concurrency 8, with ITL at concurrency 1 drifting 17.0 → 23.4 ms. Nothing changed except roughly 35 minutes of additional sustained load.

**Measurements drift ~24% across a session.** Arm A and Arm B are comparable because they ran consecutively; neither is comparable to a baseline recorded half an hour earlier.

This is the strongest argument yet for the protocol rule that compared pairs must be measured back to back — **a 24% drift would swamp most real optimizations, including Experiment 1's +34%.** Every cross-session comparison in the Stage 13 report must be qualified accordingly.

**Operational conclusion:** a shared system prompt is worth more than any configuration tuning attempted in this project. It doubles SLO-compliant concurrency and costs nothing but structuring the workload to share a prefix.

**Status:** ✅ Stage 11 Experiment 2 complete

---

## Stage 11, Experiment 3 — Qwen2.5-1.5B unquantized vs 3B-AWQ — 2026-09-07

**What we did:**
Pre-downloaded `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB, 7 min) *before* starting the comparison, so the two arms would be separated only by a container restart — forced by the 24% session drift measured in Experiment 2. Created `deployment/docker/docker-compose.1_5b.yml`, a Compose override that replaces only the vLLM `command`, leaving the shipping config untouched.

**Prediction, recorded before running:**

```text
Budget at --gpu-memory-utilization 0.78   =  3.12 GiB
Qwen2.5-1.5B fp16: 1.54B x 2 bytes        =  2.87 GiB
Activations + overhead                    ~  0.30 GiB
Left for KV cache                         ~  0.00 GiB   -> predicted startup failure
```

**Result — prediction confirmed to three decimal places:**

```text
Model loading took 2.8871 GiB          (predicted 2.87)
Available KV cache memory: 0.07 GiB    (predicted ~0)

ValueError: To serve at least one request with the models's max seq len (4096),
0.11 GiB KV cache is needed, which is larger than the available KV cache memory
(0.07 GiB). Based on the available memory, the estimated maximum model length is 2624.
```

The engine entered a restart loop and never served a request.

| | Qwen2.5-**3B**-AWQ | Qwen2.5-**1.5B** fp16 |
| --- | --- | --- |
| Parameters | 3.09B | 1.54B (**half**) |
| **Weights on GPU** | **1.954 GiB** | **2.887 GiB (+47.7%)** |
| **KV cache available** | **0.98 GiB** | **0.07 GiB (−92.9%)** |
| KV cache tokens | **28,480** | ~2,624 (−90.8%) |
| Max concurrency @ 4,096 | 6.94x | **fails to start** |

**The model with half the parameters uses 48% more memory and gets 93% less KV cache.** The mechanism is the Stage 1 lesson at its sharpest: **KV cache is the remainder of the budget**, so a 0.93 GiB increase in weights did not cost 0.93 GiB of cache — it consumed nearly all of it. Weights and cache do not trade linearly when the budget is small.

**The per-token arithmetic held a third time.** Predicted 28 KiB/token for the 1.5B (28 layers × 2 KV heads × 128 head_dim × 2 for K+V × 2 bytes). vLLM derived a maximum model length of 2,624 from 0.07 GiB, back-solving to **28.0 KiB/token** — the same calculation that gave 36.1 KiB/token for the 3B in Stage 1 and 36 KiB predicted before ever launching it.

**Why "just lower `--max-model-len`" was not pursued.** vLLM suggests 2,624, which would start — and yield a **maximum concurrency of 1.28x**: effectively a single-user server with a context window 36% below what `PRODUCT_SPEC.md` specifies. That answers a different question. The question asked was whether the unquantized 1.5B can serve *this product's configuration* on this card, and it cannot.

**Not subject to session drift**, because this is a memory fact rather than a throughput measurement. The 3B's startup figures have been byte-identical across every restart in this project, so no fresh arm-A ramp was required.

**Stage 0 chose AWQ by reasoning from a published benchmark. Stage 11 turned that into something this project proved on its own hardware.**

**Status:** ✅ Stage 11 Experiment 3 complete

---

## Stage 11, Experiment 3c — 3B-AWQ vs 1.5B-AWQ — 2026-09-07

**Why this experiment exists — the user caught a weakness in the plan.**
`PROJECT_PLAN.md` specifies "unquantized 1.5B vs AWQ 3B", which tests whether *dropping quantization* helps. Experiment 3 answered that decisively (no). But the user pointed out that `Qwen/Qwen2.5-1.5B-Instruct-AWQ` exists — and `DECISIONS.md` had named it as the fallback from Stage 0 onward. **The fp16 variant was a strawman for the product decision**: nobody choosing between these models would take fp16 when an official AWQ checkpoint exists.

Same quantization on both sides makes this a comparison of model size against model quality, and unlike the fp16 arm it **runs at the shipping `--max-model-len 4096`** — no configuration had to be weakened to make the comparison possible. Created `deployment/docker/docker-compose.1_5b-awq.yml`.

**Result — the 1.5B-AWQ wins decisively.**

| conc | 3B-AWQ tok/s | 1.5B-AWQ tok/s | Δ |
| ---: | -----------: | -------------: | --: |
| 1 | 46.1 | **67.8** | **+47.1%** |
| 4 | 127.8 | **213.2** | **+66.8%** |
| 8 | 200.7 | **345.0** | **+71.9%** |

| | Highest SLO-compliant level | Throughput there |
| --- | --- | --- |
| 3B-AWQ | concurrency **1** | 46.1 tok/s |
| **1.5B-AWQ** | concurrency **4** | **213.2 tok/s** |

**4.6x more SLO-compliant throughput** — larger than every other optimization in this project combined. Experiment 1's `--max-num-seqs` change was worth nothing under the SLO; Experiment 2's shared prefix was worth 2x.

**THE RESULT THAT MATTERS MOST — two different scaling laws, measured simultaneously:**

| Phase | Bound by | Scales with | Predicted | Measured |
| --- | --- | --- | --- | --- |
| Decode (ITL p50) | memory bandwidth | **weight bytes** | −39% | **−37.6%** |
| Prefill (TTFT p50) | compute | **parameter count** | ~−50% | **−50.7%** |

These are *different percentages from the same model swap*, because AWQ changes bytes-per-parameter without changing parameter count.

Back-solving the ITL result: if ITL is proportional to weight bytes, `1.954 GiB × (15.6 / 25.0) = 1.219 GiB` of weights for the 1.5B-AWQ. **The prediction made beforehand by counting parameters was 1.19 GiB — agreement within 3%.**

**The compute-bound/bandwidth-bound distinction that began as an explanation in Stage 2 has become a quantitative law that correctly predicted a model swap it was never derived from.** That is the strongest validation of the mental model built across this project.

**What this does not measure: answer quality.** Every number favours the 1.5B; none capture why someone might still choose the 3B. The honest form of the result is conditional — *if 1.5B-class quality is acceptable for the use case, it is 4.6x better on this hardware.*

**Memory numbers, read from the log:**

| | 3B-AWQ | 1.5B-AWQ | Predicted |
| --- | --- | --- | --- |
| Weights | 1.9542 GiB | **1.1018 GiB** | 1.19 (within 8%) |
| KV cache tokens | 28,480 | **69,616** | ~62,000 (within 12%) |
| Max concurrency @ 4096 | 6.94x | **17.00x** | ~15x |

**A better decode model falls out of two measurements.** With ITL known for two weight sizes, fit `ITL = a × weight_GiB + b`:

```text
3B:    25.0 ms = a × 1.9542 + b
1.5B:  15.6 ms = a × 1.1018 + b
       ->  a = 11.03 ms/GiB,  b = 3.45 ms
```

**Decode ≈ 11.03 ms per GiB of weights plus 3.45 ms of fixed overhead** (KV reads, sampling, kernel launches — the parts that do not scale with model size). That fixed term is 13.8% of the 3B's ITL, which is why the pure-proportional prediction was slightly off. And `1 GiB / 11.03 ms` gives **~97 GB/s of effective memory bandwidth**, roughly half the RTX 3050's ~192 GB/s rating — plausible for a thermally-limited workload, and a number this project never set out to measure.

**Status:** ✅ Stage 11 Experiment 3c complete

---

## Stage 11 correction — SLO ceilings re-measured with margins — 2026-09-08

**Why:** the user asked why the 3B failed the SLO at concurrency 4 when it had passed earlier. Checking the record, every 3B run at concurrency 4 clustered around the 1,500 ms limit: 1,476 (PASS by 24 ms), 1,405, 2,115, 1,530 (FAIL by 30 ms), 1,766. **A 24 ms pass and a 30 ms fail are the same measurement**, with the 24% thermal drift far larger than the gap.

I had flagged the 1,476 ms result as "uncomfortably close to the line" when it appeared and then reported PASS/FAIL as if it were a fact anyway. **A binary verdict against a threshold turns noise into apparent fact when the measurement lands near the threshold.**

**Fix:** `ramp.py` now reports the **margin** against the SLO alongside the verdict, and both models were re-measured at every level.

**3B-AWQ**

| conc | tok/s | TTFT p95 | SLO | margin |
| ---: | ----: | -------: | :--: | -----: |
| 1 | 39.3 | 601 | PASS | **+60%** |
| 2 | 64.7 | 1,165 | PASS | **+22%** |
| 3 | 84.4 | 1,729 | FAIL | −15% |
| 4 | 108.1 | 2,269 | FAIL | −51% |

**1.5B-AWQ**

| conc | tok/s | TTFT p95 | SLO | margin |
| ---: | ----: | -------: | :--: | -----: |
| 4 | 271.8 | 757 | PASS | **+50%** |
| 5 | 278.8 | 977 | PASS | **+35%** |
| 6 | 295.5 | 1,208 | PASS | +19% |
| 7 | 321.2 | 1,394 | PASS | +7% |

**Applying a margin discipline of ≥ +20%** (outside the measured 24% drift band):

| | Reliable ceiling | Throughput there |
| --- | --- | --- |
| 3B-AWQ | concurrency **2** | **64.7 tok/s** |
| **1.5B-AWQ** | concurrency **5** | **278.8 tok/s** |

**4.3x**, replacing the earlier 4.6x estimate which rested on a bare PASS.

**The two models fail differently, and that shape matters as much as the ceiling.** The 3B goes +22% → **−15%** between concurrency 2 and 3 — a cliff. The 1.5B slopes: +50 → +35 → +19 → +7. A system that falls off a cliff needs a much larger safety factor than one that degrades gradually, and a bare PASS/FAIL table hides that distinction completely.

**Also found:** a Compose trap that cost a full wasted ramp. `docker compose up -d --build gateway` **without** the `-f` override flags silently recreated `vllm` with the base file's model, because **Compose reconciles every service against the files it was given, not just the one named**. 320 requests then returned 404. Omitting `-f` is not "leave the others alone", it is "revert the others".

Two fixes: `$env:COMPOSE_FILE="a.yml;b.yml"` sets it once per shell, and `ramp.py` now queries `/v1/models` before starting and refuses to run against the wrong engine — turning a multi-minute silent failure into a one-second specific error.

**Also fixed:** `ui/index.html` hardcoded the model name, so the chat UI broke with `The model Qwen/Qwen2.5-3B-Instruct-AWQ does not exist` whenever the engine served anything else. It now asks the gateway via `/v1/models` and displays the served model in the header. **A client that hardcodes what the server is serving breaks on every deployment change** — and the gateway already proxied `/v1/models` precisely so it would not have to.

**Outstanding:** the answer-quality comparison between the two models is still unresolved. An initial impression of "not much difference" was probably taken against the 3B, since the gateway rebuild had reverted the engine at that moment.

**Status:** ✅ Stage 11 complete — Experiment 4 (speculative decoding) and Stage 12 (Kubernetes) remain optional

---

## Stage 13 — Final report — 2026-09-08

**What we did:**
Assembled `docs/FINAL_REPORT.md` from `PROGRESS_LOG.md`, `DECISIONS.md`, `benchmarks/baseline.md` and `benchmarks/optimization_results.md`.

**Contents:** architecture and URLs; the pinned stack; the four boundaries each with the evidence that demonstrates it; the Stage 2 baseline; the concurrency ceiling with the resource diagnostic; the measurement protocol and the failure that produced each of its ten rules; four optimization results with before/after tables; the derived scaling law; three incident write-ups; honest limitations including the security gaps scoped out of v1; a "what real scale would need" section; and a table of **twelve wrong predictions with what each taught**.

**Definition of done from `PRODUCT_SPEC.md` — all met:**

| Requirement | Evidence |
| --- | --- |
| Boundary 1 — contract test catching a real regression | 22 contract tests; sabotage produced 14 failures |
| Boundary 2 — pinned versions with written reasoning | image digest, model commit, driver version; pins verified live |
| Boundary 3 — dashboard on real engine metrics incl. KV-% and prefix-hit-rate | Prometheus scraping vLLM directly; both metrics in the first row |
| Boundary 4 — request state surviving engine preemption | eviction forced; 56,000 tokens recorded exactly, zero errors |
| Benchmark report with baseline, ceiling, before/after | `baseline.md`, `optimization_results.md`, `FINAL_REPORT.md` |
| One command, documented URLs | `docker compose up -d`, URL table in the report |

**The finding the report leads on, and it was not the goal when the project started:** decode scales with weight *bytes* while prefill scales with *parameter count*. Both were predicted before the measurement, from arithmetic built in Stage 1 to explain a memory budget — and the decode prediction was right to within 3% for a model swap it was never derived from.

**Status:** ✅ Stage 13 complete — v1 delivered

---

## ▶ RESUME HERE — 2026-09-08

**v1 is complete.** Stages 0–13 done, all four boundaries demonstrated, `docs/FINAL_REPORT.md` written.

**Two documents define what comes next:**

1. **`docs/FINALIZATION_PLAN.md`** — 8 phases of closing work, **written but NOT yet executed**. Clean slate and resource reclamation, canonical cold-card numbers for both models, the 1.5B's absolute maximum-throughput sweep, full regression suite, boundary re-verification on the finalized model, speculative decoding, Kubernetes, report update.
2. **`docs/V2_SCOPE.md`** — the next project: turning the API into a **multi-user chat service**, with admission control, backpressure, per-key fairness, context management for long conversations, systematic failure injection, and alerting.

**v2's central open question, and it is a 6x uncertainty.** A first-pass capacity model gave ~80–110 simultaneous users (compute-bound). It is probably wrong. **The prefix cache is one shared pool of 69,616 tokens, not per-user** — so 100 users with 4,000-token conversations want 400,000 tokens of prefix against 69,616 available, only ~17 conversations fit, and LRU eviction means a conversation's cache survives ~8 seconds against a 45-second think time. Essentially every turn becomes a miss, and a miss costs ~7.4 GPU-seconds against ~0.8 for a hit.

**Revised estimate: ~17–35 users**, bounded by `KV_cache ÷ average_conversation_length`. Stage 6's finding that prefix caching keeps TTFT flat across 55x context growth **held because there was exactly one user.**

**Both figures are hypotheses.** V2 Phase A/B measures prefix cache hit rate as a function of user count and looks for the predicted knee. That curve is v2's headline result, and it matters more than the number.

**Config inherited without re-derivation, found 2026-09-08:** `--max-model-len 4096` was sized for the **3B** (a 32,768-token sequence would need 1.15 GiB against 0.98 GiB of KV cache — vLLM would refuse to start). The 1.5B needs only 896 MiB against **1,859 MiB**, so it fits, and raising the ceiling costs ~512 KB of block table rather than memory. **We are using ~12% of the model's supported context.** `FINALIZATION_PLAN.md` Phase 2.5 verifies the model's real limit, confirms KV cache size is unchanged at 16k and 32k, and measures whether long requests behave as the linear-prefill model predicts. The real costs are latency (a 32k prompt is 33–55 s of prefill, monopolising the GPU) and prefix-cache pressure (one 32k conversation is 48% of the shared cache) — not memory.

**Finalized model going forward: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`** — 4.3x more reliably-SLO-compliant throughput than the 3B (278.8 tok/s at concurrency 5 versus 64.7 at concurrency 2). Its revision still needs pinning; that is `FINALIZATION_PLAN.md` Phase 5.1 and is required for Boundary 2 to hold on the shipped model.

**Open question that no measurement here can settle:** answer quality of the 1.5B versus the 3B. Every performance conclusion favouring the 1.5B is conditional on it. One informal impression of "not much difference" was probably taken against the 3B, because a gateway rebuild had reverted the engine at that moment. `FINALIZATION_PLAN.md` Phase 4.6 is where this gets decided.

**Three things blocking later phases, to gather when convenient:**

- The 1.5B-AWQ's snapshot hash (Phase 5.1, Boundary 2)
- `minikube status` and whether the node still advertises `nvidia.com/gpu` (Phase 7)
- `vllm serve --help | Select-String speculative` from this exact build (Phase 6 — the flag schema changes across releases, and guessing is how the `gpu_cache_usage_perc` metric-name error happened)

**Tooling fixed in preparation, not yet exercised:** every test script now resolves the served model from `/v1/models` instead of hardcoding it; `ramp.py` gained `--min-requests`, `--echo-task`, an SLO margin column, a served-model preflight check, and model auto-detection; `seed_db` gained `--reset`; `--max-num-batched-tokens` is parameterised for the throughput sweep.

---

## Finalization Phase 0 — Clean slate, revision pin, verified architecture — 2026-09-08

**What we did:**
Cleared stale shell state, brought the stack down, reclaimed disk, deleted the two models we will never run again, cleared the compile cache, pinned the finalized model's revision, and read the model architecture directly from `config.json` instead of relying on the value Stage 11 back-solved from an error message.

**Result — disk:**

| Item | Before | After | Reclaimed |
| --- | --- | --- | --- |
| Build cache | 21.69 GB | **315 MB** | **21.37 GB** |
| `hf-cache` | 14.17 GB | **10.72 GB** | 3.45 GB |
| `inference-product_vllm-cache` | 17.32 MB | deleted | 17 MB |
| Dangling images | — | — | **0 B** |

**Result — the 1.5B-AWQ is pinned.** Snapshot hash `3ecffa0ceb27851800f45519bab9c457a04405e1`, exactly one snapshot present, written into `docker-compose.1_5b-awq.yml` as `--revision` / `--tokenizer-revision`. This closes the deviation recorded in Stage 11 and makes **Boundary 2 hold on the finalized model**. Pinned *before* Phase 2 measures rather than after, because adding the flags changes the vLLM config and therefore the `torch.compile` cache key — measuring first would have produced canonical numbers for a config we do not ship, and dropped a recompile into a measurement.

**Result — the capacity model now rests on verified architecture:**

```text
config.json: num_hidden_layers 28, num_key_value_heads 2, num_attention_heads 12,
             hidden_size 1536, max_position_embeddings 32768, rope_scaling null

head_dim = 1536 / 12 = 128
KV/token = 2 (K,V) x 28 x 2 x 128 x 2 bytes = 28,672 B = 28.0 KiB
```

Independently reproduces the 28 KiB/token figure Stage 11 back-solved out of a vLLM startup error. Two consequences follow as arithmetic rather than recollection: 69,616 tokens x 28,672 B = **1.859 GiB** of cache (matches the measured value), and one full 32,768-token sequence costs **896 MiB against 1,859 MiB available**, so the model's native maximum context fits. **`rope_scaling: null` means 32,768 is a hard ceiling**, not a YaRN-extendable one — Phase 2.5's decision is bounded between 4,096 and 32,768 with nothing above it.

**Predicted vs actual — four of five wrong:**

| Item | Predicted | Actual | Verdict |
| --- | --- | --- | --- |
| Build cache reclaimed | 3-8 GB | **21.37 GB** | ❌ 3x under |
| Dangling images | 0.5-2 GB | **0 B** | ❌ |
| `hf-cache` size | ~7 GB | **14.17 GB** | ❌ 2x under |
| `vllm-cache` size | 0.3-1.5 GB | **17.32 MB** | ❌ ~50x over |
| `config.json` values | exact list | exact match | ✅ |

**What each miss taught:**

1. **Dangling images were 0 B because `--build` retags in place.** Every `docker compose up --build` overwrote `inference-product-gateway:latest`, leaving no `<none>` orphans. The debris went to the *build* cache instead, where two entries from four days ago (11.9 GB and 9.06 GB) were 96% of the total. The intuition that rebuilds leave dangling images runs backwards on Compose.
2. **The compile cache is ~1.4 MB per config, not ~100 MB.** Compiled graphs are metadata and serialized artifacts, not weights. Deleting `vllm-cache` reclaimed nothing meaningful. It was still correct to delete — a dozen stale config keys are clutter — but the justification is hygiene, not disk. **The compile cache's entire value is the ~40 s it saves, and its entire cost is measurement contamination if that 40 s lands in the wrong place.**
3. **`hf-cache` was twice its contents because of a second cache nobody had looked at.** `du` inside the volume: `hub` 4.1 GB, **`xet` 6.0 GB** — the Hugging Face Xet download-chunk cache. `docker system df -v` reports only the volume total, so the split was invisible. Now the largest reclaimable item in the project.

**The finding that outgrew the phase — `--gpu-memory-utilization` is reopened.**
`nvidia-smi` with the entire stack down showed **290 MiB of VRAM held by ordinary desktop applications** (Edge WebView2, WhatsApp, Phone Link, Start Menu) at 1% utilization. Stage 1 attributed the 3.21-of-4.0 GiB free-memory shortfall entirely to the WDDM compositor; part of it was ordinary user software. With those closed the card reports **0 MiB in use**.

`FINALIZATION_PLAN.md` Phase 3 states the dial *"cannot move — this is a Windows constraint, not a choice."* That claim is now falsifiable. The dial is a fraction of *total* memory, so closing apps does not enlarge KV cache by itself — it enlarges the headroom against the startup free-memory check that forced 0.78 down in the first place. Weights are fixed at 1.1018 GiB, so **every additional MiB of budget becomes KV cache** at 28 KiB/token. **Since the shared prefix cache is the binding constraint on v2's whole capacity question, this is a lever on the headline result rather than a config tweak.** Deliberately not changed yet: Phase 2 measures the shipping 0.78 and reads free memory out of vLLM's own startup log, then the dial becomes a separate arm.

**Also established — the ≤50 °C cooldown target in `FINALIZATION_PLAN.md` was invented, and is probably unreachable.** With zero GPU processes and 0% utilization, the card still sits at **P0, 1057 MHz core, 5501 MHz memory, 16.53 W, 74 °C**. That is the Stage 2 fix doing its job: *Prefer maximum performance* pins the clocks permanently, including at idle, so the card has no low-power state to fall into and dumps 16.5 W into a thin chassis continuously. The protocol rule should be **"cool to the measured idle floor and record it"**, not a number chosen in advance.

**Issues hit / fixed:**
A `rm -rf` typo (`models--distillgpt2`, double L) silently did nothing — `rm -rf` on a missing path exits 0 and prints no output, so it looked successful. Re-run with the correct spelling. **A delete that cannot fail is a delete that cannot be verified; check the listing afterwards rather than the exit code.**

**Status:** ✅ confirmed working — thermal floor and the `xet` reclaim outstanding

**Phase 0 closed — 2026-09-08.** `/cache/xet` deleted and verified by listing: `/cache` holds only `hub`, and `hub` still carries both AWQ models. `hf-cache` 14.17 GB → ~4.1 GB. Total Phase 0 reclaim: **~27.8 GB** (21.37 build cache + 3.45 dead models + 6.0 xet − overlap), against a predicted 6-11 GB.

**Decided during Phase 0, and recorded because the reasoning generalizes: hardware-accelerated GPU scheduling stays ON.** It is not the per-app GPU-assignment control it is often mistaken for, it is a global driver-level change to a WDDM path this project's GPU-PV passthrough depends on, and — decisively — **it requires a reboot, so two arms could never be measured back to back.** With a ~24% session-drift band larger than any plausible effect, the setting is *untestable under this project's own protocol*. Changing it would be a guess that could never be verified, which is the exact class of change the protocol exists to prevent.

**Status:** ✅ Finalization Phase 0 complete

---

## Finalization Phase 2 — 1.5B-AWQ canonical, and decode turns out to be core-clock-gated — 2026-09-08

**What we did:**
Brought the full stack up on the pinned 1.5B-AWQ override, verified the engine config, and ran the ramp twice — once heat-soaked, once accidentally from a cooled card with `watch_gpu.ps1` sampling.

**Startup — three predictions exact:**

| | Predicted | Actual |
| --- | --- | --- |
| Weights | 1.1018 GiB | **1.1018 GiB** |
| KV cache | 69,616 tokens | **69,616 / 69,760** |
| Max concurrency @ 4,096 | 17.00x | **17.00x / 17.03x** |
| Kernel | awq_marlin | confirmed |
| Revision pin | 3ecffa0... | **live in engine config** |

Boundary 2 now holds on the shipped model. The two KV figures come from two startups; 144 tokens is +0.21%, at the noise threshold the protocol already defines.

**THE HEADLINE — two runs of an identical command differed by 42%, and the cause rewrites a v1 claim.**

| conc | Run 1 (heat-soaked) | Run 2 (from 58 C) | Run 1 ITL p50 | Run 2 ITL p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 59.0 | **89.0** | 15.8 | **10.1** |
| 2 | 110.5 | **157.4** | 15.8 | **11.2** |
| 4 | 188.0 | **257.2** | 17.0 | **12.4** |
| 6 | 238.4 | **330.9** | 18.1 | **13.2** |
| 8 | 293.1 | **416.7** | 18.8 | **13.9** |

Identical config, seed, model and command. 471 prompt / 256 output tokens throughout. The difference is entirely thermal: Run 2 began at 58 C because an interruption let the card cool, and the instruction to skip the warm-up ("already at steady state") was asserted rather than checked.

**GPU trace, Run 2:**

```text
23:36:19  load starts     930-1342 MHz   64 C   0x04 SwPowerCap
23:37:43  transition                     86 C   0x20 SwThermalSlowdown
23:39-40  steady state   1000-1200 MHz   86-87 C  0x20 sustained
```

**Finding 1 — sustained clock is ~1,100-1,200 MHz on this model, not 712 MHz.** Stage 2 recorded 712 MHz steady state while serving the 3B. Exactly one sample across a 4.5-minute run touched 712. The 1.5B draws less power per unit work, so it holds a far higher sustained clock. **Every figure derived from 712 MHz is stale**, including the "~97 GB/s effective bandwidth, 50.5% of the card rating" calculation in FINALIZATION_PLAN.md.

**Finding 2 — decode is CORE-CLOCK-GATED, not memory-bandwidth-bound, and this was a standing untested prediction.**

```text
ITL ratio    15.8 / 10.1  = 1.56
clock ratio  ~1300 / ~830 = 1.57
```

clocks.mem was pinned at **5501 MHz in both runs and never varied**. If decode were purely bandwidth-bound, ITL could not swing 56% while the memory clock held constant. It tracked the *core* clock almost exactly.

FINALIZATION_PLAN.md's appendix hypothesised precisely this — "at 712 MHz the SMs cannot issue memory requests fast enough to saturate 192 GB/s" — and recorded it as **"a testable prediction this project cannot test."** It was tested by accident here, and it holds. This does not overturn *decode is proportional to weight bytes*, which was measured across two models at comparable thermal state; it adds a second term. Decode time behaves as max(bandwidth_time, issue_time) and on this card **issue time wins**, so the memory bus is never saturated.

**Finding 3 — a ramp on a heating card confounds concurrency with temperature.** Run 2 climbs 58 to 87 C while stepping 1 to 8, so low concurrency was measured cool and high concurrency hot. ITL rises **+38%** across Run 2 levels but only **+19%** across heat-soaked Run 1 — the excess is thermal, not contention. **Run 1 is the internally valid ramp; Run 2 looks better and is contaminated.**

**Finding 4 — closing desktop applications gained no KV cache, as predicted.** The second startup followed killing Edge WebView2, WhatsApp and Phone Link, freeing 290 MiB. KV cache moved 69,616 to 69,760 tokens: noise. `--gpu-memory-utilization` is a fraction of *total* memory, so freeing VRAM buys only headroom against the startup free-memory check. Converting that headroom into cache requires moving the dial itself.

**Predicted vs actual on the ramp — three of four wrong, all in the same direction:**

| Level | Predicted | Run 1 (canonical) | Verdict |
| --- | --- | --- | --- |
| conc 4 | PASS +45-55% | **PASS +26%** | wrong, over |
| conc 6 | PASS +15-25% | **FAIL -6%** | wrong side |
| conc 8 | marginal | **FAIL -46%** | far worse |
| Throughput climbing at 8 | yes | yes (293.1) | correct |

The systematic over-prediction came from anchoring on Stage 11 numbers, which are now visibly cool-card figures. **Reliable ceiling under sustained load, at the >= +20% margin discipline: concurrency 4 at 188.0 tok/s (471 in / 256 out).**

**Also fixed:** `docker compose logs vllm | Select-String` returned nothing because **vLLM logs to stderr and docker compose logs preserves the source stream**, so piping stdout carried an empty stream. Redirecting all streams to a file with `*>` and then using `Select-String -Path` works. And /v1/models through the gateway requires the API key — correct behaviour, not a bug.

**Unidentified:** throttle mask 0x400 appears at load transitions with high SM clock and low utilisation. Not in the documented NVML reason set we have; recorded rather than guessed.

**Status:** in progress — Run 1 stands as canonical but has no telemetry; one confirmation run outstanding

---

## Finalization Phase 2 (cont.) — Run 3 with verified heat soak: session drift explained, and a prompt-index bug found — 2026-09-09

**What we did:**
Re-ran the ramp with a deliberately long warm-up (80 requests at concurrency 8, max_tokens 512 — 89 s of load) and `watch_gpu.ps1` sampling throughout, to certify the thermal state the canonical numbers were taken in.

**Heat soak confirmed.** The trace shows load starting 23:50:05, `SwPowerCap` (0x04) until 23:51:05, then sustained `SwThermalSlowdown` (0x20) at 86-87 C. The measured ramp ran 23:51:39 to 23:57:00 entirely inside that state, with a 5-second gap after the warm-up.

**Three runs, and they order by STARTING temperature even though all three reach the same die temperature:**

| conc | Run 1 | Run 2 | Run 3 (verified soak) |
| ---: | ---: | ---: | ---: |
| 1 | 59.0 | 89.0 | **70.0** |
| 2 | 110.5 | 157.4 | **147.8** |
| 4 | 188.0 | 257.2 | **230.7** |
| 6 | 238.4 | 330.9 | **294.8** |
| 8 | 293.1 | 416.7 | **367.0** |
| idle temp before run | ~78 C | 58 C | **69 C** |

Predicted Run 3 would reproduce Run 1 within 10%; it came in **20-25% above**. Wrong, and the reason is the finding.

**THE MECHANISM BEHIND SESSION DRIFT, observed since Stage 11 and never explained.**
Die temperature is capped at 86-87 C in all three runs. What differs is **the clock the card can hold at that temperature**: Run 2 sustained ~1,100-1,200 MHz, Run 3 ~950-1,050 MHz. As the heatsink and chassis saturate, less delta-T is available, so the card must drop clocks further to hold the same die temperature. **`nvidia-smi` reports die temperature, not heatsink base temperature — so this second thermal variable is invisible in every metric available to this project.** Run 2 to Run 1 is −30%, the same order as Stage 11's measured ~24% drift.

**The core-clock law now holds across three independent runs:**

| Run | ITL p50 | Clock implied by ITL | Clock observed |
| --- | ---: | ---: | ---: |
| 2 | 10.1 ms | ~1,250 MHz | 1,250-1,350 |
| 3 | 13.2 ms | **957 MHz** | **~1,010 (within 6%)** |
| 1 | 15.8 ms | ~799 MHz | no telemetry |

ITL predicts SM clock to within 6% across a 56% range. `clocks.mem` was 5,501 MHz in every sample of every run.

**BUG FOUND — a warm-up run poisons the measured run's early levels through the prefix cache.**
Run 3 reported **TTFT p50 of 77 ms at concurrency 2 against 229 ms at concurrency 1**. Latency cannot improve threefold by adding concurrency, so the number is impossible rather than merely surprising.

Cause, in `ramp.py`: prompts are seeded from the request index, `next_index` advances across levels within a run, **but resets to 0 on every invocation**. The warm-up used `--min-requests 80`, taking indices 0-79. The measured ramp's level 1 (0-39) and level 2 (40-79) fell entirely inside that range and were served from the prefix cache the warm-up had just populated. Levels 4, 6 and 8 (indices 80+) were cold.

The three runs confirm the mechanism exactly:

| Run | Warm-up size | conc 1 TTFT | conc 2 TTFT | Monotonic? |
| --- | ---: | ---: | ---: | --- |
| 3 | 80 | 229 | **77** | **no — both cached** |
| 1 | 16 | 283 | 348 | yes — barely touched |
| 2 | none | 201 | 237 | yes — clean |

The bug only fires when the warm-up is large enough to cover the measured levels, which is precisely what the "make the warm-up longer" instruction caused. **Every ramp in this project that used a warm-up has optimistically biased low-concurrency TTFT figures.** The SLO ceiling lands at concurrency 6-8, in the uncontaminated region, so that conclusion survives.

**Fixed** by adding `--index-offset` to `ramp.py`. The measured run takes `--index-offset 1000`, past any plausible warm-up, so its prompts are genuinely cold. Levels within a single run were already unique and remain so.

**CANONICAL NUMBERS — Qwen2.5-1.5B-Instruct-AWQ, shipping config, 471 in / 256 out:**

| | Value |
| --- | --- |
| Reliable SLO ceiling (margin >= +20%) | **concurrency 6 at 294.8 tok/s, +24%** |
| Under deep chassis heat soak | degrades to **concurrency 4 at 188.0 tok/s** |
| Peak throughput measured | 416.7 tok/s (conc 8, cool start) |
| KV cache at concurrency 8 | 8 x ~730 = 5,840 of 69,760 tokens = **8.4%** |

**The binding constraint is compute, and it is not close.** KV cache sits at 8.4%, so preemption is arithmetically impossible and memory is nowhere near the limit. `SwThermalSlowdown` was active for the entire measured ramp. This is a thermal ceiling, and `--max-num-seqs 8` plus prefill capacity are the levers — exactly what Phase 3 targets.

**The honest form of the capacity statement:** capacity on this hardware is not a number but a range bounded by chassis thermal saturation. **Use concurrency 4 as the conservative bound for v2's capacity model**, since a chat service runs continuously.

**Status:** Finalization Phase 2 complete

---

## Finalization Phase 2.5 — context window: prefill turns out to be quadratic — 2026-09-09

**What we did:**
Raised `--max-model-len` to 16,384 and then 32,768, and measured genuinely long prompts. Phase 2.5.1 (architecture verification) was already completed during Phase 0.

**The plan's Phase 2.5 was cut down deliberately.** It called for full ramps at 16k and 32k to check that short requests are unaffected. Phase 2 established the run-to-run thermal noise floor at **+/-25%**, and the predicted effect is *zero* — a measurement that cannot resolve what it is looking for is not worth 25 minutes. The claim it tests is settled exactly and noise-free by the startup log instead.

**32,768 is live and works.** A 22,586-token prompt was served successfully, which is proof on its own: that request would be rejected outright at `--max-model-len` 16,384, let alone 4,096.

**THE FINDING — prefill is not linear in prompt length, and the project has been assuming it is since Stage 2.**

| prompt tokens | TTFT p50 | implied prefill rate |
| ---: | ---: | ---: |
| 471 | 283 ms | — |
| **6,774** | **2,717 ms** | 2,493 tok/s |
| **22,586** | **13,032 ms** | **1,733 tok/s** |

The rate *falls 30%* as the prompt grows, which no linear model can produce. Fitting `TTFT = c + a*n + b*n^2`:

```text
TTFT ~= 0.13 s  +  n / 3,070  +  n^2 x 1.11e-8
         fixed      linear        attention (quadratic)
```

| n | Predicted | Measured |
| ---: | ---: | ---: |
| 471 | 0.29 s | 0.28 s |
| 6,774 | 2.85 s | 2.72 s |
| 22,586 | 13.16 s | **13.03 s** |

Within 5%, and within 1% on the largest. The term breakdown is the substance:

| prompt | attention share of prefill |
| ---: | ---: |
| 471 | 1.6% |
| 4,096 | 12% |
| 22,586 | **43%** |
| 32,768 | **53%** (extrapolated) |

**"Prefill scales linearly with prompt length" is a short-prompt approximation** — true at chat lengths, wrong by a factor of two at the context limit. Extrapolated cost of a full 32,768-token prompt: **~23 s of TTFT**. The plan estimated 33-55 s from 3B-derived prefill rates; a purely linear 1.5B estimate gave 16 s; the quadratic model gives 23 s and fits every measured point.

**Predicted vs actual:** ~8k predicted 3.5-5.5 s, measured 2.72 s at 6.8k (scaling to ~3.0 s at 8k) — **slightly under the range**. ~26k predicted 11-17 s, measured 13.03 s at 22.6k — **inside the range, but for the wrong reason**: the total was roughly right from a model that is structurally wrong.

**SECOND FINDING — long context degrades decode far more on the smaller model.**
ITL rose **12.8 to 18.2 ms (+42%)** between the two prompts. Stage 6 measured only 2.4% context degradation and concluded decode was "weight-bandwidth-dominated, not attention-dominated" — but that was the 3B across 1,709 tokens. Both results follow from the same ratio:

```text
KV read / weight read
  3B  @ 1,709 tok:   62 MB / 2,000 MB =  3.1%   (Stage 6 measured 2.4%)
1.5B  @ 1,709 tok:   47 MB / 1,183 MB =  4.0%
1.5B  @ 22,586 tok: 632 MB / 1,183 MB = 53%
```

**The lighter the model, the more context costs** — a direct consequence of the weight-bytes law, since KV per token falls more slowly than weight bytes do. Stage 6's "doubling ITL would need ~56,000 tokens of context" was computed for the 3B and does not transfer.

**THIRD FINDING — the SLO as written is not well-defined.**
Both runs report FAIL at −267% and −844%. That is not information: no hardware meets a 1.5 s TTFT on a 22,586-token prompt. **`p95 TTFT < 1.5 s` is only meaningful with a stated prompt-size envelope, and it currently has none.** Every SLO verdict in this project has implicitly assumed ~471-token prompts. The capacity model must carry the envelope explicitly.

**Decision — ship `--max-model-len 32768`.** Reasoning: it is a **ceiling, not an allocation**. PagedAttention allocates 16-token blocks on demand and KV cache size is set by leftover memory regardless, so the only GPU cost is the block table (~512 KB). Capping at 16,384 would not prevent long-conversation cache pressure; it would only remove the capability while leaving the pressure to be managed by context policy either way.

**But the measurement sharpens what that policy has to do.** A long conversation is expensive in three ways at once, not one: quadratic prefill on a cache miss, +42% ITL, and **47% of the entire shared prefix pool for a single 32k conversation** (32,768 of 69,760 tokens). That is the strongest argument yet for V2 Phase E, and it means the context *policy* matters far more than the context *ceiling*.

**Status:** Phase 2.5 complete pending the KV-cache-size confirmation from the 32,768 startup log

**Phase 2.5 confirmed and shipped — 2026-09-09.**

```text
Using max model len 32768
GPU KV cache size: 69,616 tokens
Maximum concurrency for 32,768 tokens per request: 2.12x
```

**69,616 tokens at 32,768 — identical to the figure at 4,096.** Raising the ceiling eightfold costs exactly zero KV cache, and the reported max-concurrency figure is the only thing that moves (predicted 2.13x from 69,616 / 32,768, reported 2.12x). Both predictions exact. `--max-model-len=32768` is now the default in `docker-compose.1_5b-awq.yml`, with the derivation written into the file so the next person does not inherit an unexamined number the way we inherited 4,096 from the 3B.

**`prompt_tokens_details` is `null`, not populated — the prediction was wrong and it matters.**
Two identical ~930-token requests through the gateway both returned `"prompt_tokens_details": null`. The prediction was `cached_tokens` around 1,150 on the second call.

This is the field `V2_SCOPE.md` identified as "the single most valuable metric v2 could add". Without it there is no per-request cache-hit ratio, so **admission control cannot weight requests by predicted GPU-seconds** and must fall back to counting requests — which systematically mis-admits, since a miss costs roughly 9x a hit.

Not yet concluded: the gateway was not ruled out as the cause before recording this. Asking vLLM directly on port 8000, plus checking the `prefix_cache` counters on `/metrics` to confirm the second request really was a hit, is outstanding. Asserting that the gateway forwards raw rather than verifying it is the precise error that has cost this project three separate failures.

**`cached_tokens` resolved, and the gateway is innocent — 2026-09-09.**
vLLM on port 8000, bypassing the gateway entirely, returns `"prompt_tokens_details": null`. **Boundary 1 holds** — the untyped forwarding is not dropping the field, because there is no field to drop. The control confirms the premise: `prefix_cache_queries_total` 121,162 tokens against `prefix_cache_hits_total` 2,912, roughly three full hits of the repeated 930-token prompt, so cached requests demonstrably occurred and vLLM still reported nothing.

**Working through the consequence exposed a flaw in `V2_SCOPE.md`'s plan, independent of the field's absence.** The scope says the gateway could "weight each request by its uncached prefix length and admit by predicted GPU-seconds rather than by request count" if `cached_tokens` were available. But **`cached_tokens` arrives in the response, after the work is done** — admission control needs the estimate *before* dispatch. Even with the field populated it could only have seeded a predictive model. Its absence therefore costs **per-request observability** (per-key dashboards, exact attribution), not the admission mechanism.

**The mechanism v2 needs is available anyway, and is a better design.** The gateway sees each conversation's full message array, so it knows every conversation's token length and when it last spoke. Against a 69,616-token LRU pool, a conversation's prefix survives while *tokens admitted since its last turn < 69,616* — evaluable at admission time, and **falsifiable against `prefix_cache_hits_total / prefix_cache_queries_total`**. A prediction that can be checked beats a number that can only be read afterwards.

**And the aggregate hit rate is precisely what v2's headline measurement requires** — prefix cache hit rate as a function of user count. The capacity curve is fully supported by the metrics this build already exposes.

Also observed: the four 22,586-token prompts (90,344 tokens against a 69,616-token pool) evicted the small repeated prompt's blocks, which is why five calls produced roughly three hits rather than four. LRU behaving exactly as `V2_SCOPE.md` describes, seen incidentally.

**Status:** Finalization Phase 2.5 complete

---

## Finalization Phase 3 — the engine cap, and a protocol rule that defeats itself — 2026-09-09

**What we did:**
Reframed the plan's four-config throughput sweep into a single arm answering the question v2 actually needs: **at `--max-num-seqs 8`, is concurrency 8 failing the SLO because of the hardware or because of the engine's own cap?** Four configs at a +/-25% noise floor would mostly have measured temperature. Levels 4, 6 and 8 were included as an in-run control against Phase 2's Run 3.

**THE CONTROL FAILED, and the telemetry says exactly why.**

```text
00:32:54 -> 00:33:51   warm-up runs, card reaches 84 C
00:33:52 -> 00:34:58   IDLE 66 s - card falls 84 C -> 71 C
00:34:59 -> 00:37:14   measured ramp
```

Levels 4 and 6 ran under `0x04` (power-capped, ~1,300-1,450 MHz); `0x20` did not engage until 31 s into the ramp. Levels 4/6/8 came in **+13-18% above Run 3** (272.5 / 331.8 / 420.8 against 230.7 / 294.8 / 367.0) rather than the predicted within-10%.

**The instruction caused it, and the lesson generalises: verifying the heat soak destroys it.** The rule written after Phase 2 said to check `watch_gpu` for sustained `0x20` and a plateaued temperature *before* starting the measured run. But checking takes time, and the card sheds 13 C in 66 s of idle. **The verification cannot precede the measurement; it has to happen during the warm-up, with the two commands chained so no gap exists, and the trace read afterwards to decide whether to keep the result.**

**The headline prediction held anyway, because it is robust to the confound:**

| | `--max-num-seqs 8` (Run 3) | `--max-num-seqs 32` |
| --- | --- | --- |
| conc 6 margin | PASS **+24%** | PASS **+33%** |
| conc 8 margin | FAIL −7% | PASS +7% |
| conc 12 | capped | **FAIL −36%** |
| conc 16 | capped | **FAIL −63%** |

At the >= +20% margin discipline both give **concurrency 6**, and concurrency 12 fails by 36% even in the *favourable* thermal state, so the ceiling is below 12 regardless of which run is trusted. **Removing the cap bought throughput above the SLO and nothing at or below it** — the same shape as Stage 11 Experiment 1, where +34% peak throughput was worth exactly zero SLO-compliant capacity.

**New peak throughput for the project: 686.3 tok/s** at concurrency 16, **471 in / 512 out**. The plan predicted 400-550 and I predicted 430-490 for the ramp; both under. The ratio matters and is quoted deliberately — the measured ramp's 517.5 tok/s at concurrency 16 was 475 in / 256 out, a more prefill-heavy shape.

**Predicted vs actual:**

| | Predicted | Actual | |
| --- | --- | --- | --- |
| SLO ceiling stays at 6 | yes | **yes** | correct |
| Levels 4/6/8 reproduce Run 3 within 10% | yes | +13-18% | wrong (thermal, explained) |
| conc 16 throughput | 430-490 | 517.5 | wrong, under |
| conc 16 TTFT p95 | 2.5-4 s | 2.448 s | wrong, marginally under |

**DECISION — ship `--max-num-seqs 32`, and not for throughput.**
Under the SLO it is worth nothing. The reason is architectural: v2 puts admission control in the gateway at roughly 6 concurrent, and **the engine cap must sit well above that so it can never be the binding constraint**. One limiter, in the gateway, where the policy and the metrics live. A cap of 8 would silently shape traffic the gateway believes it is controlling — an invisible second limiter is exactly the kind of thing that makes a capacity model wrong. At concurrency 16 the cost is nil: ITL p99 38.5 ms, KV cache ~17%, zero preemptions.

**Canonical capacity, stated with its thermal range rather than as a single number:**

| | |
| --- | --- |
| Reliable SLO ceiling | **concurrency 6** |
| Throughput there | **295-332 tok/s** (471 in / 256 out), range is thermal |
| Peak throughput, SLO abandoned | **686.3 tok/s** (471 in / 512 out, conc 16) |
| Binding constraint | **compute/thermal** — KV at 8-17%, preemptions zero |

**Outstanding:** the startup log for `--max-num-seqs 32` was captured before the engine finished loading (`docker compose up -d` returns in ~6 s; vLLM needs ~60-90 s), so the KV-cache-size line is still unconfirmed at this setting. The ramp that followed also failed once with `vllm: unreachable` for the same reason. **`docker compose up -d` returning is not the engine being ready** — wait for `docker compose ps` to read healthy before reading logs.

**Status:** Finalization Phase 3 complete

---

## v2 build — admission control, fairness, context policy, chat UI — 2026-09-09

**Mode change:** the user handed over execution at this point. Everything below was run rather than instructed.

**Phase 4 regression on the shipped config — all green.**

| Check | Result |
| --- | --- |
| Startup: weights / KV / concurrency | 1.1018 GiB, **69,760 tokens**, 2.13x @ 32,768 |
| `max_model_len` / `max_num_seqs` defaults | **32768 / 32**, both from the compose file, no env vars |
| Contract tests | **22 passed**, 1 deselected |
| Integration test | **1 passed** |
| `test_gateway.ps1` | 5/5, streaming unbuffered (38 chunks over 447 ms) |
| `test_budget.ps1` | 429 on request **7**, exactly as Stage 4 |
| Markdown renderer | **25/25** (`node ui/test_markdown.mjs`) |

**`--max-num-seqs 32` costs no KV cache**, confirming the Phase 3 prediction: 69,760 tokens at `max_num_seqs` 8 and 32 alike. The only visible change is `cudagraph_capture_sizes` growing from `[16,8,4,2,1]` to `[64,56,...,2,1]`.

**ANSWER QUALITY — the last unresolved question in the project, now resolved.**
Asked the Stage 6 control question ("explain the KV cache in LLM inference"), the 1.5B answers with a **generic key-value store**: hash lookups, a Python `dict` subclass, cache eviction. It never mentions attention keys and values, or reuse across decode steps. That is **the same error Stage 6 recorded from the 3B**, which "described the KV cache in generic caching terms rather than correctly describing attention key/value reuse."

**On this project's own control, the two models fail identically.** The 4.3x SLO-compliant throughput advantage is therefore not being paid for in quality on this task. One data point on one question, and it is a question where both models are simply wrong - but it is the comparison that was on the record, and it no longer blocks the model decision.

**What was built:**

| Component | What it does |
| --- | --- |
| `gateway/admission.py` | Bounded in-flight (6), shallow queue (12), 2 s queue timeout, per-key cap (3), stream timeout, Prometheus exposition |
| `gateway/context.py` | Sliding-window context policy preserving system messages and the current question |
| `gateway/main.py` | Wires both in; `/metrics`, `/admission`; fail-closed budget check, fail-soft ledger write |
| `ui/index.html` | Markdown rendering, bounded auto-continuation, 503/trim surfacing |
| `ui/test_markdown.mjs` | 25 tests including XSS, extracted from the shipped page so they cannot drift |
| `load_testing/chat_sim.py` | Poisson arrivals, log-normal think time, accumulating multi-turn conversations, per-user reporting |
| `load_testing/failure_matrix.py` | The worst cases, each with a stated expectation |
| `load_testing/run_matrix.ps1` | The whole matrix, one command, seeded |
| `observability/prometheus/alerts.yml` | 10 rules, selected to PREDICT failure rather than report it |
| `observability/grafana/dashboards/gateway-capacity.json` | The v2 dashboard |

**Two bugs found by writing the tests, both worth keeping:**

1. **The alert rule used `vllm:gpu_cache_usage_perc`. The real metric is `vllm:kv_cache_usage_perc`.** Caught by reading `/metrics` instead of trusting the name - which is the exact error this project already recorded once ("guessing is how the `gpu_cache_usage_perc` metric-name error happened"). Writing the rule from memory reproduced the same mistake within an hour of having read the warning.

2. **The simulator shed requests at FOUR simulated users**, which looked like a capacity disaster and was not. Per-key admission limits are keyed on the API key, and all simulated users were presenting `dev-key-alpha` - so the "population" was one tenant, correctly throttled to its 3-slot share. **The limit was right and the experiment was wrong.** Fixed by seeding a 48-key pool (`dev-user-00..47`) and giving each simulated user its own identity, which is also what real chat users have.

**Status:** v2 components built and verified; measurement in progress

---

## v2 measurement — the capacity curve, and four bugs the tests found — 2026-09-09

**THE HEADLINE: degradation is bounded, and the queue timeout decides where the flat line sits.**

The same sweep run twice, changing one variable:

| users | qt 2.0s: p95 / shed / over SLO | qt 0.6s: p95 / shed / over SLO |
| ---: | --- | --- |
| 10 | 260 ms / 0% / 0-of-10 | 191 ms / 0% / **0-of-10** |
| 20 | 1,137 ms / 0% / 1-of-20 | 434 ms / 8.0% / **0-of-20** |
| 30 | 2,070 ms / 10.0% / 21-of-30 | 706 ms / 25.1% / **0-of-30** |
| 40 | 2,071 ms / 22.5% / 32-of-40 | 717 ms / 38.9% / **0-of-40** |
| 60 | 2,116 ms / 47.1% / 49-of-58 | **778 ms** / 56.9% / **0-of-58** |

At a 2.0 s queue timeout, p95 TTFT rose **2.2% while offered load doubled** — a textbook flat line. But it was flat at 2,070 ms, *above* the 1.5 s objective, and 49 of 58 users breached. **Bounded is not the same as acceptable**, and that distinction is the whole finding.

The fix was arithmetic: worst-case TTFT ≈ queue timeout + engine TTFT, so a 1.5 s target minus ~0.8 s of engine time leaves ~0.6 s of queue budget. **Ten percentage points more shedding bought zero SLO breaches at every level.**

**The control validates the sweep:** the 10-user level re-run last gave 266 ms against 260 ms, and 83.9% against 83.4% cache hit. Thermal drift did not dominate.

**THE PREDICTED KNEE DID NOT APPEAR, and the reason is a better result than the knee.**
Prefix cache hit rate held **74-97% across every run** — 10 to 60 users, and 550- to 2,500-token conversations. Three reasons, and the first was not anticipated:

1. **Load shedding protects the prefix cache.** Admission control caps how many conversations are *progressing*, so the working set never outgrows the 69,760-token pool. Two mechanisms designed independently reinforce each other.
2. Within a conversation the prefix is reused by construction — turn *n* resends turns computed seconds ago, still the most recently used blocks.
3. The shared system prompt is a hit for every user on every turn, putting a floor under the aggregate.

**The 6x uncertainty this project set out to resolve — ~17 users if cache-bound, ~110 if compute-bound — resolves to ~22 users, admission-bound.** The cache is not the binding constraint in any regime reachable while admission control works.

**FOUR BUGS, EACH FOUND BY A TEST, THREE OF THEM IN THE FIXES:**

**1. Fixed per-key caps are quotas, not fairness.** The adversarial scenario measured **10/10 normal users over SLO at p95 4,825 ms**. A fixed cap of 3 against 6 slots hands one tenant half the service regardless of how many other tenants exist. Replaced with a dynamic share — divide the pool by the number of *contending* keys — and the abusive key's share fell from 50% to ~14%. Result: **0/10 users over SLO, 4,669 of 4,749 abusive requests shed (98.3%)**, normal p95 536 ms against a 191 ms baseline (**+181% degradation, still 64% inside the SLO**).

**2. The load generator was the bottleneck, not the server.** The first fairness re-test reported p95 10,779 ms for normal users while the gateway's own histogram showed **all 105 served requests under 1.0 s**. Fifty abusive coroutines had starved ten normal coroutines of scheduler time *inside the simulator's event loop*, and the harness reported its own scheduling delay as server latency — a ~10x error, in the direction that makes a working mechanism look broken. Fixed with `--abusive-only`, so the attacker runs in its own process.

**3. `max_num_partial_prefills > 1` silently disables the V1 engine.** Worst case #3 measured a ~20k-token prompt pushing concurrent short requests from 82 ms to **12,201 ms** of TTFT. vLLM has exactly the right controls, they are accepted, they log `Concurrent partial prefills enabled` — and then the server dies on `assert envs.VLLM_USE_V1` in a crash loop. **79 restarts before it was caught.** Reverted; the mitigation moved to the gateway as weighted admission, where a long prompt costs slots in proportion to its prefill chunks (measured: three concurrent 20k prompts, one shed with `cost=4`).

**4. A circuit breaker with no half-open state latches forever.** The billing breaker (added because WAL mode lets the budget *read* succeed while the *write* fails, so the service was serving unbilled work) reset only on a successful write — but refused every request before a write could be attempted. Restoring the database left the service returning 503 permanently. Fixed with a cooldown and a probe:

```text
healthy                     200
ledger unwritable           200, 200, 200   <- bounded unbilled loss (threshold 3)
                            503, 503, 503   <- fail CLOSED
immediately after restore   503             <- still cooling, correctly
after 10 s cooldown         200             <- probe succeeded, breaker closed
next request                200             <- recovered
```

**Status:** v2 measured; two residual findings recorded in docs/FAILURE_MATRIX.md

---

## v2 complete — failure matrix closed, everything verified — 2026-09-09

**Final verification on the shipped configuration:**

| Check | Result |
| --- | --- |
| Contract tests (Boundary 1) | **22 passed**, 1 deselected |
| Integration test | **1 passed** |
| Markdown renderer (incl. 5 XSS cases) | **25 passed, 0 failed** |
| Gateway behaviour | 5/5; streaming unbuffered, 38 chunks over 459 ms |
| Smoke test | 200, model resolved from `/v1/models` |
| Failure matrix | **12 pass, 2 documented limitations, 0 unexplained failures** |

**Worst cases 5 and 6, measured rather than inherited from v1:**

```text
case 5 - engine crash mid-stream
  crash detected            1.4 s
  automatic recovery        83.1 s      (restart: unless-stopped, restarts=2)
  gateway /health           200 THROUGHOUT   <- liveness green, never restarted
  gateway /ready            503 -> 200       <- readiness failed and recovered
  requests                  155x502, 4x500, 3x200
  ledger                    162 rows for 162 requests, DRIFT 0

case 6 - gateway restart under load
  back after                10.4 s
  in-flight                 12 failed with a dropped connection, no hangs
  ledger                    17,854 -> 17,858 rows; budgets intact across restart
```

**THREE WAYS TO KILL AN ENGINE, TWO OF WHICH MEASURE NOTHING.** Worth recording because the first two look like they work:

1. `docker compose kill vllm` — container exits 137, but **`restart: unless-stopped` does not restart it** (`restarts=0`). Docker treats an operator kill as intentional.
2. `docker exec vllm kill -9 1` — **silently ignored**. The kernel does not deliver signals to a PID namespace's init from inside that namespace.
3. `pkill -9 -f EngineCore` — the realistic crash, and the restart policy does apply.

The timing needed care too: the first version reported **0.8 s of "recovery"** because it probed `/ready` before the API server had noticed its worker was gone — timing the gap between issuing a kill and the kill landing, and calling it recovery.

**A FALSE PASS CAUGHT, AND IT NEARLY SHIPPED.**
Worst case #3 re-ran after the weighted-admission fix and reported the big prompt's TTFT at **370.9 ms against 13,159 ms** before. It looked fixed. It was not: the harness used a fixed prompt string, so the second run was a **complete prefix-cache hit** and measured nothing. This is the cross-run cache contamination Stage 2 already documented, reappearing in a new place.

Adding a per-run nonce **at the front** of the prompt — vLLM's prefix hash is a chain from block 0, so a nonce at the end would leave every preceding block cached — restored the real behaviour: **19,995 ms and 20,833 ms** on two consecutive runs. Case #3 is a documented limitation, not a pass.

**The chatbot fixes, verified end to end.** A prompt that truncates at 256 tokens now continues automatically: 4 rounds, 4,534 characters, seams joined mid-sentence without repetition, ending naturally ("This concludes our guide on Python decorators") rather than mid-word. Markdown renders as headings, lists, bold and fenced code.

**Answer quality, one more time:** the smoke test's KV-cache answer still describes a generic key-value store with caching of "frequently accessed data" — no mention of attention keys and values. Consistent across every observation, and the same error the 3B made.

**Status:** ✅ v2 complete. Stack stopped.

---

## ▶ RESUME HERE — 2026-09-09

**v2 is complete and the stack is stopped.** Bring it back with:

```powershell
cd deployment/docker
$env:COMPOSE_FILE="docker-compose.yml;docker-compose.1_5b-awq.yml"
docker compose --profile observability up -d
docker compose ps        # wait for vllm to read "healthy" - takes 60-90 s
```

**The shipped configuration**, all of it derived rather than inherited:

| | Value | Why |
| --- | --- | --- |
| Model | `Qwen2.5-1.5B-Instruct-AWQ` @ `3ecffa0c…` | 4.3x more SLO-compliant throughput than the 3B; no observable quality difference on the control question |
| Context window | **32,768** | Measured to cost zero KV cache; it is a ceiling, not an allocation |
| `--max-num-seqs` | **32** | Set above the gateway's limit so the engine can never be the hidden limiter |
| Admission | **6 in flight, 12 queued, 0.6 s timeout, dynamic per-key share** | Every value measured; see docs/CAPACITY_MODEL.md |
| Capacity | **~22 users** at 12 s think time | Derivation in the capacity model; it moves with think time |

**What is left undone, in priority order:**

1. **Head-of-line blocking from very long prompts** (failure matrix #3). Needs a vLLM build whose V1 scheduler supports concurrent partial prefills. The gateway bounds the damage; it cannot remove it.
2. **Answer quality** rests on one control question. A real eval set would change the model decision's confidence, not its direction.
3. **Kubernetes** (`deployment/k8s/` is still just a README). The liveness/readiness split is now measured under Compose, which is most of the value, but the pod-level demonstration is not done.
4. **Speculative decoding** — never attempted; `ramp.py --echo-task` exists for it.
5. **Multi-replica accounting.** SQLite has one writer. Postgres or Redis with atomic decrements, plus a test that Boundary 4 still holds under concurrent writers.
6. **`--gpu-memory-utilization` above 0.78.** Headroom exists but taking it makes startup depend on desktop VRAM state, which breaks "one command brings the stack up".

---

## Post-publication — the output bound the "22 users" question exposed — 2026-09-11

Asked precisely what the capacity figure bounds — context length, maximum output per call, what happens without an EOS — the answer turned up a gap. **vLLM's OpenAI server defaults an absent `max_tokens` to `max_model_len − input_length`** (read from `entrypoints/utils.py:get_max_tokens` in the image, not assumed). On the 32,768-token window a client that simply omits the field asks for ~30,000 tokens of generation, and a model that does not emit EOS delivers them: one of six admission slots held for seven-plus minutes, ended only by the 300 s stream timeout cutting the answer off.

An admission limit whose per-slot hold time is unbounded is a limit on *count*, not on *work*, and the capacity arithmetic does not hold under it.

**Fixed:** the gateway injects `max_tokens=1024` when the field is absent and clamps values above 2,048, reporting the clamp in `X-Max-Tokens-Clamped-From`. Both are additions, not removals — the same class of modification as `stream_options.include_usage`, so Boundary 1 is intact. Three contract tests added (absent → bounded, oversized → clamped and reported, sane value → untouched): **25 passed**.

**The envelope, stated.** README now carries a table of what "22 users" assumes on every axis — concurrency, think time, output per turn, conversation length, prompt size, window, output cap, EOS behaviour, stream bound, thermal state, SLO definition — and what happens outside each. `CAPACITY_MODEL.md` gains the output-length derivation: at 12 s think time, 192-token replies → ~22 users, 512 → ~12, 1,024 → ~8, 2,048 → ~6. **Reply length is the strongest capacity lever after concurrency**, and it is a product decision.

README reoriented around the inference engineering: a decision → before → after table for every change that moved a number, and a "what watches it" section for the predictive signals and the failure matrix.

**Status:** pushed
