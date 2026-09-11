# Baseline Benchmark

The reference numbers for this machine. **Every "improvement" claimed in Stage 11 is measured against this file.** Nothing here is an estimate — every figure is produced by a command recorded below, on the hardware and software stack pinned below.

---

## The stack these numbers describe

If any line in this table changes, these numbers stop being comparable and the baseline must be re-measured.

| Component | Pinned value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, compute capability 8.6 (Ampere) |
| Driver | 616.56 (KMD 616.56, CUDA UMD 13.4) |
| Host | Windows 11, Docker Desktop 4.89.0, WSL2 backend, GPU under WDDM |
| Engine image | `vllm/vllm-openai:v0.11.0` — digest `sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c` |
| Model | `Qwen/Qwen2.5-3B-Instruct-AWQ` |
| Model revision | `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` |
| Kernel | `awq_marlin` (confirmed selected in engine logs) |
| Attention backend | Flash Attention, V1 engine |
| Sampler | FlashInfer (top-k / top-p) |

**Engine configuration:** `--dtype float16 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.78 --enable-prefix-caching --swap-space 2`

**Resulting memory layout** (from Stage 1 startup):

| Quantity | Value |
| --- | --- |
| Model weights on GPU | 1.9542 GiB |
| Available KV cache memory | 0.98 GiB |
| GPU KV cache size | 28,432 tokens |
| `num_gpu_blocks` | 1777 (× 16 tokens/block) |
| Bytes per KV token (derived) | ~36.1 KiB |
| Max concurrency @ 4096 tokens/request | 6.94x |

---

## Known distortions in these numbers

Stated up front so the Stage 13 report doesn't have to walk them back. These figures are **honest for this machine** and will not transfer unchanged to a Linux server:

1. **WSL2 forces `pin_memory=False`** — logged at every startup: `Using 'pin_memory=False' as WSL is detected. This may slow down the performance.` Host-to-device transfers are unpinned and therefore slower than on a native Linux host.
2. **WDDM reserves ~0.79 GiB** of the 4 GiB card for the Windows compositor, capping `--gpu-memory-utilization` at ~0.80. A Linux host in TCC mode would give this project more KV cache on identical hardware.
3. **Laptop thermal and power limits** — the card runs at a 35 W cap. Sustained load will throttle in a way a desktop or datacenter card would not.
4. **The benchmark client runs in a container** talking to the server over `host.docker.internal`, adding a small amount of loopback networking overhead to every measurement. It is included in TTFT and end-to-end latency, and is identical across all runs, so it does not affect before/after comparisons.
5. **The client container is started with `--gpus all`** despite doing no GPU work. This is required for device detection only — see `run_bench_serve.ps1` for why. It loads no model and allocates no KV cache, so it does not take memory from the server.

---

## Methodology

**Workload:** synthetic, `--dataset-name random`, fixed input and output lengths, fixed `--seed 42`. Chosen over the default ShareGPT dataset so that run-to-run variance does not mix with the effect being measured — see `DECISIONS.md`. Realistic variable-length traffic is Stage 9's job, not this file's.

**Determinism — `--ignore-eos --temperature 0`:** both are required, and the first benchmark attempt proved it by producing 2,683 and then 2,823 output tokens for an identical request of 4,096 with an identical seed. Without `--ignore-eos` the model stops at EOS, making `--random-output-len` a ceiling rather than a quantity; without an explicit temperature the server applies Qwen's HF generation config (`temperature 0.7, top_p 0.8, top_k 20`) and samples stochastically. Together these guarantee every run generates **exactly** `num_prompts × output_len` tokens, so any measured difference is the change under test rather than the dice. Sampling is overridden for benchmarking only — the server retains Qwen's defaults, which are the right behaviour for the product.

**Thermal protocol — mandatory, and the reason is measured, not theoretical.** This GPU boosts to 1372 MHz, reaches 87 °C in about three seconds of sustained load, and settles at **712 MHz — a 48% clock reduction — within a single 21-second run.** A short run therefore measures boost clocks and a long run measures thermally-limited steady state, so two runs of different durations are not comparable even with identical everything else. Every measurement in this file and in `optimization_results.md` follows the same protocol:

1. Run a discarded warm-up pass (`-NumPrompts 8`) to bring the card to thermal steady state.
2. Immediately run the measured pass, without waiting for the card to cool.

Note that `clocks.mem` does **not** throttle — it holds 5501 MHz throughout. Since decode is memory-bandwidth-bound, **ITL is largely insulated from thermal throttling; TTFT is not**, because prefill is compute-bound and rides the SM clock down. Expect TTFT to be the noisier metric across runs, and treat a TTFT regression with more suspicion than an ITL regression.

**Warm-up (software):** `vllm bench serve` sends a single test request before measuring. The server is additionally launched once with its final flag set before any measurement, so the `torch.compile` cache is warm and the ~60 s compile is not inside any timing.

**Required host power configuration — reproduce these two settings before any measurement:**

1. **Windows → Settings → System → Power & battery → Power mode → Best performance**
2. **NVIDIA Control Panel → Manage 3D settings → Global Settings → Power management mode → Prefer maximum performance**

Notably, the charger was *not* the issue and the vendor thermal profile was not changed — the default NVIDIA "adaptive/optimal power" setting was sufficient on its own to hold the card in P8 under a CUDA compute workload. That is worth knowing: **the adaptive power policy does not recognise an LLM decode loop as work deserving a performance state.** It is a graphics-oriented heuristic, and continuous small kernel launches against a memory-bound workload apparently do not trip it.

**Required GPU power state.** Every measurement must be taken with the GPU in **P0**. This is not a formality: a single host power setting held the card in **P8 at 100% utilization** (210 MHz core, 405 MHz memory — roughly 1/15th of rated memory bandwidth), which cost **11.6x throughput** and produced entirely plausible-looking benchmark output with no error or warning anywhere in the vLLM logs. Verify with `benchmarks/watch_gpu.ps1` alongside any run whose numbers will be recorded — `pstate` must read `P0` and `clocks.mem` must read ~5500 MHz while `utilization.gpu` is at 100%.

**Two runs:**

- **Run A — `--max-concurrency 1`.** The clean latency baseline. One request in flight means TTFT and ITL contain no queueing delay, so they state what this hardware does per token and nothing else.
- **Run B — `--max-concurrency 8`, 128 prompts.** Matches `--max-num-seqs 8`. Shows what continuous batching buys in aggregate throughput, and what it costs in per-request latency. Uses 4x the prompts of Run A because at concurrency 8 a 32-prompt run is only four waves and ramp-up/drain dominate the average — the first attempt showed this directly, reporting 50.19 tok/s average against a 72.00 tok/s peak. Not a ceiling-finding exercise; that is Stage 9.

---

## Run A — single stream (`--max-concurrency 1`)

**Command:**

```powershell
docker run --rm `
  --gpus all `
  -v hf-cache:/root/.cache/huggingface `
  --entrypoint vllm `
  vllm/vllm-openai:v0.11.0 `
  bench serve `
  --backend openai-chat `
  --base-url http://host.docker.internal:8000 `
  --endpoint /v1/chat/completions `
  --model Qwen/Qwen2.5-3B-Instruct-AWQ `
  --dataset-name random `
  --random-input-len 512 `
  --random-output-len 128 `
  --num-prompts 32 `
  --max-concurrency 1 `
  --seed 42 `
  --ignore-eos `
  --temperature 0 `
  --percentile-metrics ttft,tpot,itl,e2el
```

Wrapped as `.\benchmarks\run_bench_serve.ps1` so that Stage 9 and Stage 11 re-run exactly this, differing only in explicit parameters.

**Results — measured 2026-09-07, GPU verified in P0, thermal protocol applied:**

| Metric | Value |
| --- | --- |
| Successful requests | 32 |
| Total generated tokens | **4096 (exact)** |
| Benchmark duration (s) | 122.67 |
| Request throughput (req/s) | 0.26 |
| Output token throughput (tok/s) | **33.39** |
| Peak output token throughput (tok/s) | 56.00 |
| Total token throughput (tok/s) | 166.71 |
| Mean TTFT (ms) | 515.36 |
| Median TTFT (ms) | **654.13** |
| P99 TTFT (ms) | 700.38 |
| Mean TPOT (ms) | 26.12 |
| Median TPOT (ms) | 26.72 |
| P99 TPOT (ms) | 29.68 |
| Mean ITL (ms) | 25.92 |
| Median ITL (ms) | **26.13** |
| P99 ITL (ms) | 33.62 |
| Mean E2EL (ms) | 3832.62 |
| Median E2EL (ms) | 4049.01 |
| P99 E2EL (ms) | 4443.83 |

**The thermal protocol justified itself on this run.** The discarded warm-up (8 prompts, 22 s, card still on boost clocks) versus Run A (32 prompts, 123 s, thermal steady state), identical configuration:

| Metric | Warm-up (boosting) | Run A (steady state) | Change |
| --- | --- | --- | --- |
| Mean ITL | 21.57 ms | 25.92 ms | +20% |
| Mean TTFT | 51.54 ms | 515.36 ms | +900% |

Exactly the predicted split: memory clock does not throttle so ITL barely moves, while SM clock falls ~1455 → 712 MHz and compute-bound prefill collapses with it. Recording the warm-up as the baseline would have made every future TTFT comparison a measurement of how long the GPU had been running.

Open question, deliberately not explained away: TTFT degraded ~10x while SM clock only halved. Clock alone does not account for the magnitude; sustained power delivery is the likely additional factor. Flagged for Stage 9.

---

## Run B — batched (`--max-concurrency 8`, 128 prompts)

**Command:** `.\benchmarks\run_bench_serve.ps1 -Concurrency 8 -NumPrompts 128`

**Results — measured 2026-09-07, immediately after Run A (card still at thermal steady state):**

| Metric | Run A (conc. 1) | Run B (conc. 8) | Change |
| --- | --- | --- | --- |
| Successful requests | 32 | 128 | — |
| Total generated tokens | 4096 (exact) | **16384 (exact)** | — |
| Benchmark duration (s) | 122.67 | 119.46 | — |
| Request throughput (req/s) | 0.26 | 1.07 | 4.1x |
| **Output token throughput (tok/s)** | **33.39** | **137.15** | **4.11x** |
| Peak output token throughput (tok/s) | 56.00 | 360.00 | 6.4x |
| Total token throughput (tok/s) | 166.71 | 684.57 | 4.11x |
| Mean TTFT (ms) | 515.36 | 1747.17 | 3.4x worse |
| Median TTFT (ms) | 654.13 | 758.76 | +16% |
| P99 TTFT (ms) | 700.38 | 3781.19 | 5.4x worse |
| Mean TPOT (ms) | 26.12 | 44.90 | +72% |
| Median TPOT (ms) | 26.72 | 44.37 | +66% |
| Mean ITL (ms) | 25.92 | 44.73 | +73% |
| **Median ITL (ms)** | **26.13** | **29.51** | **+13%** |
| **P99 ITL (ms)** | **33.62** | **637.62** | **19x worse** |
| Median E2EL (ms) | 4049.01 | 8548.21 | 2.1x |
| P99 E2EL (ms) | 4443.83 | 8963.98 | 2.0x |
| Peak GPU KV cache usage (%) | ~2.4% | **18.5%** | — |
| Requests queued (`Waiting`) | 0 | 0 | — |
| Prefix cache hit rate during run | 52–67% | 63–75% | see caveat below |

---

## Interpretation

**Continuous batching is worth 4.11x aggregate throughput for a 13% median latency cost.** 33.39 → 137.15 output tok/s while median ITL moves only 26.13 → 29.51 ms. That is 51% scaling efficiency against an 8x concurrency increase — the shortfall is expected, since decode is memory-bandwidth-bound and eight sequences still contend for the same 5501 MHz memory bus, but the trade is strongly favourable.

**The cost hides in the tail, not the median.** P99 ITL degrades 19x (33.62 → 637.62 ms) while median ITL moves 13%. Individual tokens occasionally stall over half a second — scheduler pauses when a newly-arrived request's prefill preempts ongoing decode. Chunked prefill (`--max-num-batched-tokens 2048`) bounds the damage but cannot eliminate it. Note that **mean TPOT (44.90 ms) smooths this away entirely**; only P99 ITL exposes it. This is the metric to watch in Stage 10, where a forced KV-cache eviction should produce the same signature.

**KV cache is not the constraint at this concurrency — measured, not estimated.** The engine reported a steady **15.4–18.5% GPU KV cache usage** with all 8 requests running, and `Waiting: 0 reqs` throughout: the scheduler never had to queue anything. **Implication for Stage 9: the concurrency ceiling on this machine will be compute/thermal, not KV cache memory.** That is the opposite of what a 4 GiB card intuitively suggests, and it is a direct consequence of `--max-model-len 4096` keeping per-request cache demand small. Stage 9 should ramp concurrency well past 8 and expect to hit thermal limits — or `--max-num-seqs` itself — before cache exhaustion.

**Caveat — prefix cache contamination across runs, and the rule it implies.** The engine logged a **63–75% prefix cache hit rate** during Run B. That is far higher than a dataset with `random_prefix_len 0` should produce, because the hits are *cross-run*: `--seed 42` generates identical prompts on every invocation, prefix caching is enabled, and the cache lives in the server process across benchmark runs. Run B was partly reusing KV blocks computed during Run A and the warm-up.

Consequence: **the TTFT figures here are optimistic**, because some prefills were partially cached. ITL is unaffected — prefix caching only accelerates prefill.

**Rule for every future comparison: restart the vLLM server between measured pairs.** Otherwise the second run of a pair inherits the first run's cache and appears faster for a reason unrelated to the change under test. This matters most for Stage 11's prefix-caching experiment, which requires a genuinely cold cache as its control.

**Reference figures for Stage 11 comparison** — any optimization is measured against these, using the identical harness and thermal protocol:

| Baseline metric | Value |
| --- | --- |
| Single-stream output throughput | 33.39 tok/s |
| Single-stream median ITL | 26.13 ms |
| Single-stream median TTFT | 654.13 ms |
| Batched (8) output throughput | 137.15 tok/s |
| Batched (8) median ITL | 29.51 ms |
| Batched (8) P99 ITL | 637.62 ms |

---

## Supplementary — offline upper bound

**Deferred, deliberately.** Would be run with the server stopped, since an offline benchmark instantiates its own engine and a second copy of the model does not fit alongside the server on a 4 GiB card.

`PRODUCT_SPEC.md` requires baseline TTFT/ITL/throughput, and the serving benchmark above provides all three from the code path this product actually ships. An offline run would only quantify HTTP and scheduling overhead — interesting, not load-bearing for any later stage. Revisit if Stage 11 produces a result that needs an absolute hardware ceiling to interpret.
