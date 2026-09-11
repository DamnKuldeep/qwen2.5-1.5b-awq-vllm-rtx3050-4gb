# Optimization Results — Stage 11

Before/after tables for every optimization attempted. Compared against `baseline.md` and against each other under a fixed protocol.

**An optimization only counts if it was measured under the protocol below.** Several rules here exist because ignoring them earlier produced numbers that looked like results and were not.

---

## The measurement protocol

Every rule is derived from something this project measured going wrong.

| Rule | Origin |
| --- | --- |
| **Restart vLLM between every compared pair** | Stage 8: two *identical* runs differed by **16% throughput** and **45% on P99 ITL**, from prefix-cache warming alone. Any claimed improvement below ~16% measured without a restart is indistinguishable from cache state. |
| **Launch each variant once before measuring it** | Stage 1: the `torch.compile` cache is keyed on the vLLM config, so every flag change costs a one-off ~30–60 s compile. That minute must not land inside a measurement. |
| **Warm up to VERIFIED full heat soak, then measure without pausing** | Stage 2: boost clocks vs thermal steady state moved mean TTFT by **900%** (51.5 ms → 515.4 ms) on otherwise identical runs. **Strengthened in Finalization Phase 2**, where two runs of an identical command differed by **42% throughput and 56% on ITL** because one started heat-soaked and the other from 58 °C. The old 25 s warm-up is not enough — telemetry shows the card needs **~85 s under load** to move from `SwPowerCap` to sustained `SwThermalSlowdown`. Warm up until `watch_gpu.ps1` shows sustained `0x20` and a plateaued temperature, and treat **any interruption between warm-up and measurement as invalidating the warm-up**. |
| **Chain warm-up and measurement with no gap — verifying the soak destroys it** | Finalization Phase 3: the Phase 2 rule said to check `watch_gpu` for sustained `0x20` before measuring. Checking took 66 seconds, during which the card shed **13 °C (84 → 71)**, and the measured levels came in 13–18% high. The card cools far faster than a human can read a trace. **Chain the commands (`cmd1; cmd2`), then read the trace afterwards to decide whether to keep the result.** Verification is post-hoc or it is self-defeating. |
| **A ramp on a heating card measures temperature, not concurrency** | Finalization Phase 2: levels run in ascending order while the card heats, so low concurrency is measured cool and high concurrency hot. ITL rose **+38%** across a cool-start ramp against **+19%** across a heat-soaked one — the excess was thermal, masquerading as contention. This bias is systematic, not noise, and it flatters low concurrency. |
| **`--ignore-eos` on every run** | Stages 2 and 9: without it the model stops at EOS and output volume varies run to run — 2,683 vs 2,823 tokens for an identical request in Stage 2, ~42 tokens against a requested 128 in Stage 9. |
| **At least 40 samples per level** | Stage 10(c): with nearest-rank percentiles, p95 of 16 samples *is the maximum*. A single cold-start outlier made p95 TTFT read 4,700 ms against 1,419 ms on an otherwise identical run. |
| **Quote the input/output ratio beside every throughput figure** | Stage 10(a): the same hardware and config produced **89.5 tok/s** (512 in / 42 out, prefill-dominated) and **205.6 tok/s** (476 in / 3,500 out, decode-dominated). "Tokens per second" is meaningless without the shape. |
| **Treat KV-cache-size differences under ±0.2% as noise** | Stage 7: vLLM profiles *free* VRAM at startup, and desktop compositor usage varies. 28,432 vs 28,480 tokens is not an effect. |
| **Verify the GPU is in P0 before recording anything** | Stage 2: a host power setting cost **11.6x throughput**, silently, with no error in any engine log. `benchmarks/watch_gpu.ps1` alongside every recorded run. |
| **Offset the measured run's prompt indices past the warm-up's** | Finalization Phase 2: `ramp.py` seeds prompts from the request index and resets that counter to 0 on every invocation, so an 80-request warm-up handed the measured run's first 80 requests straight back out of the prefix cache. It reported **TTFT p50 of 77 ms at concurrency 2 against 229 ms at concurrency 1** — latency improving as concurrency rises, which is impossible. Use `--index-offset 1000` on the measured run. |
| **Compare only runs that started from the same idle temperature** | Finalization Phase 2: three runs of an identical command gave 293.1 / 367.0 / 416.7 tok/s at concurrency 8, ordering exactly by the temperature the card idled at beforehand (~78 / 69 / 58 °C) — **despite all three reaching the same 86–87 °C die temperature**. The heatsink and chassis saturate, so less ΔT is available and the card holds a lower clock at the same die temperature. `nvidia-smi` reports die temperature only, so this variable is invisible. **This is the mechanism behind the ~24% session drift** measured in Stage 11. Record the idle temperature beside every result. |

**Standard command form:**

```powershell
# 1. restart the engine (cold prefix cache, config applied)
docker compose up -d --force-recreate vllm

# 2. warm the compile cache AND reach verified thermal steady state, discard the result
#    Watch watch_gpu.ps1: do not proceed until throttle reads 0x20 and temperature has
#    plateaued. ~85 s of load is the observed minimum; a 25 s warm-up is not enough.
python load_testing\ramp.py --levels 8 --max-tokens 512 --ignore-eos --no-warmup --min-requests 80

# 3. measure - note the index offset, or step 2 poisons step 3 via the prefix cache
python load_testing\ramp.py --levels 1,2,4,6,8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40 --index-offset 1000
```

Record the **idle temperature before step 2** with every result. Two runs that differ by more than a few degrees there are not comparable, whatever their die temperatures say during the run.

---

## Experiment 1 — Raise `--max-num-seqs`

**Not in the original plan.** Added because Stage 9 produced direct evidence for it and it would be strange to run planned experiments while ignoring our own findings.

**Hypothesis.** Stage 9 found the concurrency ceiling was `--max-num-seqs 8` — a value chosen in Stage 1 against a 4,096-token worst case. The evidence: KV cache peaked at **13.6%**, preemptions stayed at **zero**, prefill time stayed **flat**, while queue time climbed to **~12 s** and running was pinned at 8 with 24 waiting. Memory arithmetic says ~44 requests of that size would fit.

**Prediction.** Raising the cap should increase aggregate throughput and reduce queue time, until either compute saturates or KV cache genuinely fills. Stage 9's throughput plateau (89.5 tok/s at concurrency 8, *declining* beyond) suggests compute may already be close — so the honest possibilities are:

1. Throughput rises meaningfully → the cap was the constraint, and Stage 1's sizing was too conservative for this workload.
2. Throughput barely moves while KV cache climbs → compute was already the constraint and the cap was incidental.
3. Preemptions appear → we have moved the bottleneck onto memory.

All three are informative. Outcome 2 would mean Stage 9's conclusion needs qualifying.

**Workload:** 512 prompt tokens in, 256 generated out (2:1). Deliberately between Stage 9's prefill-heavy 12:1 and Stage 10's decode-heavy 1:7, since throughput varies 2.3x by shape.

### Before — `--max-num-seqs 8`

Measured 2026-09-07 under protocol: vLLM force-recreated (cold prefix cache), warm-up run discarded, `--ignore-eos`, 40 samples per level.

| conc | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (s) | SLO |
| ---: | ----: | ------------: | ------------: | -----------: | -----------: | ----------: | :--: |
| 1 | 55.4 | 334 | 389 | 17.0 | 34.6 | 4.69 | PASS |
| 4 | 159.9 | 1344 | **1476** | 19.5 | 39.9 | 6.39 | PASS |
| **8** | **256.1** | 1836 | 2688 | 21.0 | 43.6 | 7.95 | FAIL |
| 16 | 255.0 | 9077 | 10668 | 20.7 | 45.1 | 15.96 | FAIL |
| 24 | 252.7 | 17207 | 18739 | 21.3 | 53.2 | 24.16 | FAIL |

Engine state at the top levels: **KV cache peak ~20%**, **preemptions 0**, waiting peaked at **16**, queue time ~15 s p50 / 20 s p99, prefill time flat.

**Throughput is dead flat past concurrency 8** (256.1 → 255.0 → 252.7) while TTFT rises 9.4x and **ITL barely moves** (17.0 → 21.3 ms across 24x concurrency). Everything beyond concurrency 8 is pure queueing.

Note that concurrency 4 passes the SLO at p95 = 1,476 ms — **24 ms under the limit**. Close enough that the "after" column should be read with that margin in mind.

### After — `--max-num-seqs 24`

Measured 2026-09-07, identical protocol and workload.

| conc | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (s) | SLO |
| ---: | ----: | ------------: | ------------: | -----------: | -----------: | ----------: | :--: |
| 1 | 52.7 | 363 | 409 | 17.9 | 36.2 | 4.99 | PASS |
| 4 | 158.7 | 1343 | **1405** | 19.8 | 40.4 | 6.44 | PASS |
| 8 | 254.0 | 1856 | 2713 | 21.2 | 44.4 | 8.06 | FAIL |
| 16 | **313.4** | 3130 | 4819 | 27.4 | 64.9 | 12.22 | FAIL |
| 24 | **344.2** | 2820 | 7548 | 39.9 | **1283.3** | 17.77 | FAIL |

### Control group — the validity check

Concurrency 1, 4 and 8 must be unchanged: below the cap, only the client's limit binds, so `--max-num-seqs 8` and `24` are indistinguishable there. Concurrency 8 is the strongest control of the three.

| conc | Before | After | Δ |
| ---: | -----: | ----: | --: |
| 1 | 55.4 | 52.7 | −4.9% |
| 4 | 159.9 | 158.7 | −0.8% |
| **8** | **256.1** | **254.0** | **−0.8%** |

Within noise. Concurrency 1's 4.9% is small-sample variance on 40 requests, far inside the ±16% cache-state band established in Stage 8. **The comparison is valid.**

### Before / after summary

| Metric | `--max-num-seqs 8` | `--max-num-seqs 24` | Change |
| --- | --- | --- | --- |
| **Peak output throughput** (512 in / 256 out) | 256.1 tok/s | **344.2 tok/s** | **+34.4%** |
| Concurrency at peak | 8 | 24 | — |
| Throughput at conc 16 | 255.0 | 313.4 | **+22.9%** |
| Throughput at conc 24 | 252.7 | 344.2 | **+36.2%** |
| **Highest level meeting SLO** | **4 (159.9 tok/s)** | **4 (158.7 tok/s)** | **no change** |
| TTFT p50 at conc 24 | 17,207 ms | 2,820 ms | **−83.6%** |
| TTFT p95 at conc 24 | 18,739 ms | 7,548 ms | −59.7% |
| E2E p50 at conc 24 | 24.16 s | 17.77 s | −26.4% |
| ITL p50 at conc 24 | 21.3 ms | 39.9 ms | +87% |
| **ITL p99 at conc 24** | **53.2 ms** | **1,283.3 ms** | **24x worse** |
| Peak KV cache | ~20% | **~60%** | — |
| Peak running | 8 | 24 | — |
| Preemptions | 0 | **0** | — |

### Verdict

**Outcome 1 confirmed: the cap was the binding constraint**, not compute and not memory. Raising it converted queueing into throughput.

**But under the stated SLO the change is worth nothing.** The SLO still first breaks at concurrency 8, and concurrency 4 remains the highest passing level at ~159 tok/s — identical before and after. The cap only binds above 8 concurrent requests, and the SLO already fails there.

The honest statement is therefore conditional:

- **Holding p95 TTFT < 1.5 s:** `--max-num-seqs` is irrelevant. Run at concurrency 4, ~159 tok/s.
- **Relaxing to p95 TTFT < 5 s:** `--max-num-seqs 24` delivers **313 tok/s at concurrency 16** (p95 4,819 ms). The old configuration cannot reach that throughput at *any* concurrency. A **+22.9% capacity gain from one config value** — if the ITL tail is also acceptable.

**The ITL tail is the part that would be easy to miss.** p50 degraded 87% (annoying), p99 degraded **24x** (a visibly stuttering response — pauses over a second between tokens). Median streaming still looks fine at 40 ms. This is Stage 2's lesson recurring at a different layer: report p50, mean and p99 or you are choosing what to hide from yourself.

**An optimization is only meaningful relative to a stated objective.** "+34% throughput" would be a true and misleading headline.

**Not tested, worth testing:** `--max-num-seqs 16` may capture most of the throughput gain with far less tail damage — the ITL p99 jump between conc 16 (64.9 ms) and conc 24 (1,283.3 ms) suggests the cliff is between them.

---

## Experiment 2 — Shared prefix vs unique prompts (plan item b)

**Hypothesis.** `--enable-prefix-caching` has been on since Stage 1 and its effect has been observed indirectly three times: chat TTFT staying flat across a 55x context increase (Stage 6), a 16% throughput swing between identical benchmark runs (Stage 8), and 100% hit-rate spikes during eviction recompute (Stage 10a). This measures it deliberately.

**Design.** Two arms, run consecutively, `--max-num-seqs` reverted to the shipping default of 8 to isolate the prefix effect. vLLM force-recreated before each arm for a cold cache. Warm-up run in each arm uses the *same* prompt mode as its measurement, because a shared system prompt being cached after the first request is the realistic steady state.

### Arm A — unique prompts

| conc | prompt tok | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (s) | SLO |
| ---: | ---------: | ----: | ------------: | ------------: | -----------: | -----------: | ----------: | :--: |
| 1 | 471 | 41.5 | 479 | 589 | 23.4 | 47.4 | 6.53 | PASS |
| 4 | 470 | 121.1 | 1803 | 2115 | 25.6 | 51.8 | 8.40 | **FAIL** |
| 8 | 471 | 194.2 | 2512 | 3561 | 27.7 | 57.1 | 10.63 | **FAIL** |

### Arm B — shared prefix

| conc | prompt tok | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (s) | SLO |
| ---: | ---------: | ----: | ------------: | ------------: | -----------: | -----------: | ----------: | :--: |
| 1 | 541 | 41.1 | **122** | **202** | 24.2 | 49.0 | 6.38 | PASS |
| 4 | 541 | 145.9 | **231** | **406** | 26.2 | 52.3 | 7.00 | **PASS** |
| 8 | 542 | 259.9 | **451** | **648** | 28.4 | 56.4 | 7.84 | **PASS** |

### Before / after

| Metric @ conc 8 | Unique | Shared | Change |
| --- | --- | --- | --- |
| **Highest level meeting SLO** | **4 → fails** (passes only at 1) | **8, still passing** | **2x+ SLO-compliant concurrency** |
| TTFT p50 | 2,512 ms | 451 ms | **−82.0%** |
| TTFT p95 | 3,561 ms | 648 ms | **−81.8%** |
| Output throughput | 194.2 tok/s | 259.9 tok/s | **+33.8%** |
| ITL p50 | 27.7 ms | 28.4 ms | +2.5% |
| ITL p99 | 57.1 ms | 56.4 ms | −1.2% |
| Peak KV cache | ~18% | **~9.5%** | **roughly halved** |
| Prefix cache hit rate | near 0% | **~95% sustained** | — |
| Mean prompt tokens | 471 | 541 | +15% (shared arm did *more* work) |

### Verdict

**The single most valuable change measured in this project.** A shared system prompt more than doubles SLO-compliant concurrency — from failing at 4 to passing at 8 — while raising throughput 33.8%. It requires no configuration change at all, only a workload that shares a prefix.

**ITL did not move (+2.5%), which is the confirming detail.** Prefix caching accelerates *prefill only*; decode still streams all ~1.95 GiB of weights per token regardless of what is cached. The mechanism behaved exactly as the memory model predicts.

**Throughput at concurrency 1 is also unchanged** (41.5 → 41.1). Correct, and worth understanding: with 256 output tokens a single request's wall time is dominated by decode, so removing prefill work barely helps. The benefit only appears when prefill *competes* with decode for the GPU, which is why it scales with concurrency: +0%, +20.5%, +33.8%.

**Two findings that strengthen the result beyond the headline:**

1. **The shared prompts were 15% longer** (541 vs 471 tokens, visible in the `prompt` column added for exactly this check). The winning arm did more nominal prefill work. The comparison is conservative.
2. **Peak KV cache usage roughly halved**, ~18% → ~9.5%. Shared prefix blocks are stored **once** and referenced by every sequence rather than duplicated per request. Prefix caching therefore buys *memory* as well as speed — meaning more concurrent requests fit. Not predicted; it compounds the benefit.

### The caveat that matters more than the result — session drift

**Arm A's numbers sit ~24% below Experiment 1's baseline** for an identical configuration and workload: 194.2 vs 256.1 tok/s at concurrency 8, with ITL at concurrency 1 drifting 17.0 → 23.4 ms. Nothing changed except roughly 35 minutes of additional sustained load on the card.

**Measurements drift ~24% across a session.** Arm A and Arm B are comparable to each other because they ran consecutively; neither is comparable to a baseline recorded half an hour earlier.

This is the strongest argument yet for the protocol rule that pairs must be measured back to back — **a 24% drift would swamp most real optimizations**, including Experiment 1's +34%. Any cross-session comparison in this project's reports must be qualified accordingly.

---

## Experiment 3 — Qwen2.5-1.5B-Instruct (unquantized) vs 3B-AWQ (plan item a)

**The counterintuitive framing.** The *smaller* model uses *more* memory, because 4-bit weights beat fewer fp16 ones. The question is whether smaller-but-unquantized can beat larger-but-quantized on this card once that trade is paid.

**Prediction, stated before running — the arithmetic suggests a decisive negative:**

```text
Budget at --gpu-memory-utilization 0.78   =  3.12 GiB
Qwen2.5-1.5B fp16: 1.54B x 2 bytes        =  2.87 GiB
Activations + CUDA/non-Torch overhead     ~  0.30 GiB
                                             --------
Left for KV cache                         ~  0.00 GiB
```

Against the 3B-AWQ's measured 1.95 GiB of weights and **0.98 GiB of KV cache**.

Qwen2.5-1.5B has 28 layers x 2 KV heads x 128 head_dim = **28 KiB/token**, so a single 4,096-token sequence needs ~112 MiB. **Predicted outcome: it fails to start**, with vLLM refusing because `--max-model-len 4096` exceeds what the KV cache can hold.

**If that is what happens it is not a failed experiment — it is a decisive validation of the Stage 0 model choice.** "The quantized 3B fits with 0.98 GiB of KV cache; the unquantized 1.5B does not fit at all" is a stronger statement than any throughput comparison would have produced.

**Protocol note.** The 1.5B weights were downloaded *before* the comparison began, so the two arms are separated only by a container restart (~1–2 min) rather than by a ~7-minute download. Forced by the 24% session drift measured in Experiment 2 — a 15-minute gap between arms would have swamped the effect being measured.

**Method:** `deployment/docker/docker-compose.1_5b.yml` overrides only the vLLM `command`. Every other setting is held identical so the comparison isolates the model.

### Result — the prediction held exactly, and the answer is decisive

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
| Serves `--max-model-len 4096`? | Yes | **No** |

### Verdict

**The model with half the parameters uses 48% more memory and gets 93% less KV cache.**

The mechanism is the Stage 1 lesson at its sharpest: **KV cache is the remainder of the budget**, so a 0.93 GiB increase in weights did not cost 0.93 GiB of cache — it consumed nearly all of it. Weights and cache do not trade linearly when the budget is small.

**The per-token arithmetic held again.** Predicted 28 KiB/token for the 1.5B (28 layers × 2 KV heads × 128 head_dim × 2 for K+V × 2 bytes). vLLM derived a maximum model length of 2,624 from 0.07 GiB, which back-solves to **28.0 KiB/token** — the same calculation that predicted 36.1 KiB/token for the 3B in Stage 1.

**"Just lower `--max-model-len`" is not the answer.** vLLM suggests 2,624, which would start — and produce a **maximum concurrency of 1.28x**: effectively a single-user server, with a context window 36% below what `PRODUCT_SPEC.md` specifies. That answers a different question. The question asked was whether the unquantized 1.5B can serve *this product's configuration* on this card. It cannot.

**This result is not subject to session drift**, because it is a memory fact rather than a throughput measurement. The 3B's startup figures (1.9542 GiB weights, 0.98 GiB KV cache, 28,480 tokens) have been byte-identical across every restart in this project. No fresh arm-A ramp was needed.

**Stage 0 chose AWQ by reasoning from a published benchmark. Stage 11 turned that into something this project proved on its own hardware.**

---

## Experiment 3c — 3B-AWQ vs **1.5B-AWQ** (the comparison that actually matters)

**Why this supersedes Experiment 3b.** `PROJECT_PLAN.md` specifies "unquantized 1.5B vs AWQ 3B", which tests whether *dropping quantization* helps. Experiment 3 answered that decisively — no. But it is a strawman for the product decision: nobody choosing between these models would take the fp16 variant when an official AWQ checkpoint exists. `DECISIONS.md` named this model as the fallback from the very start.

**Same quantization on both sides** makes this a comparison of model size against model quality, rather than of a format that does not fit. And unlike the fp16 arm, **it runs at the shipping `--max-model-len 4096`** — no configuration has to be weakened to make the comparison possible.

**Prediction, from the same arithmetic used since Stage 1.** Qwen2.5-1.5B has 1.54B parameters, of which the tied embedding (151,936 vocab × 1,536 hidden ≈ 233M) stays fp16: ~1.31B at ~4.5 bits ≈ 0.75 GiB, plus ~0.44 GiB of embeddings ≈ **1.19 GiB of weights**.

| | 3B-AWQ (measured) | 1.5B-AWQ (predicted) |
| --- | --- | --- |
| Weights | 1.954 GiB | **~1.19 GiB (−39%)** |
| KV cache | 0.98 GiB | **~1.7 GiB (+74%)** |
| KV tokens | 28,480 | **~62,000 (2.2x)** |
| Max concurrency @ 4096 | 6.94x | **~15x** |
| ITL p50 @ conc 1 (bandwidth ∝ weight bytes) | ~18 ms | **~11 ms (−39%)** |
| Runs at shipping config? | yes | **yes** |

**The cost this harness cannot measure: answer quality.** A 1.5B model is meaningfully weaker than a 3B one, and no throughput number captures that. The Stage 6 chat UI already showed the 3B giving a generic, partly-wrong description of the KV cache; the 1.5B would be worse. That trade has to be stated, not measured away.

### Results — both arms run consecutively, shipping config, 512 in / 256 out

| conc | 3B-AWQ tok/s | 1.5B-AWQ tok/s | Δ |
| ---: | -----------: | -------------: | --: |
| 1 | 46.1 | **67.8** | **+47.1%** |
| 4 | 127.8 | **213.2** | **+66.8%** |
| 8 | 200.7 | **345.0** | **+71.9%** |

| conc | 3B ITL p50 | 1.5B ITL p50 | Δ |
| ---: | ---------: | -----------: | --: |
| 1 | 21.2 ms | 14.5 ms | −31.6% |
| 4 | 25.0 ms | **15.6 ms** | **−37.6%** |
| 8 | 27.1 ms | **16.9 ms** | **−37.6%** |

| conc | 3B TTFT p50 | 1.5B TTFT p50 | Δ |
| ---: | ----------: | ------------: | --: |
| 1 | 389 ms | 240 ms | −38.3% |
| 4 | 1,631 ms | **788 ms** | **−51.7%** |
| 8 | 2,335 ms | **1,150 ms** | **−50.7%** |

### Measured memory (from the startup logs)

| | 3B-AWQ | 1.5B-AWQ | Predicted for 1.5B |
| --- | --- | --- | --- |
| Weights | 1.9542 GiB | **1.1018 GiB** | 1.19 (within 8%) |
| KV cache tokens | 28,480 | **69,616** | ~62,000 (within 12%) |
| Max concurrency @ 4096 | 6.94x | **17.00x** | ~15x |

### The SLO ceiling, measured properly with margins

The coarse ramps above reported bare PASS/FAIL, which turned noise into apparent fact. Re-measured at every level with the margin against the 1,500 ms limit:

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

**Applying a margin discipline of ≥ +20%** (comfortably outside the measured 24% thermal drift band):

| | Reliable ceiling | Throughput there |
| --- | --- | --- |
| 3B-AWQ | concurrency **2** | **64.7 tok/s** |
| **1.5B-AWQ** | concurrency **5** | **278.8 tok/s** |

**4.3x more reliably-SLO-compliant throughput.** Larger than every other optimization in this project combined.

**The two models fail differently, and the shape matters.** The 3B goes +22% → **−15%** between concurrency 2 and 3 — a cliff. The 1.5B degrades gracefully: +50 → +35 → +19 → +7. A system that falls off a cliff needs a much larger safety factor than one that slopes, and a bare PASS/FAIL table hides that distinction entirely.

**The 1.5B is not saturated at 7.** Throughput was still climbing (271.8 → 321.2) and it holds 69,616 KV tokens with 17x concurrency headroom — so `--max-num-seqs 8` caps it far below its memory limit. Experiment 1's finding applies here with much more room than it had for the 3B, and is the obvious next experiment.

### Two different scaling laws, measured simultaneously

**ITL fell 37.6%. Predicted −39%, derived from weight bytes** (1.954 → ~1.19 GiB).

Back-solving: if ITL is proportional to weight bytes, `1.954 × (15.6 / 25.0) = 1.219 GiB`. **The prediction from parameter counting was 1.19 GiB — agreement within 3%.**

**TTFT fell 50.7%. Predicted "faster", derived from FLOPs** — 1.54B against 3.09B parameters is almost exactly half the compute, and prefill halved.

Those are *different percentages from the same model swap*, because AWQ changes bytes-per-parameter but not parameter count:

| Phase | Bound by | Scales with | Predicted | Measured |
| --- | --- | --- | --- | --- |
| Decode (ITL) | memory bandwidth | **weight bytes** | −39% | **−37.6%** |
| Prefill (TTFT) | compute | **parameter count** | ~−50% | **−50.7%** |

The compute-bound/bandwidth-bound distinction that began as an *explanation* in Stage 2 has become a quantitative law that correctly predicted a model swap it was never derived from.

### What this does not measure

**Answer quality.** A 1.5B model is meaningfully weaker than a 3B one. Every number above favours the 1.5B; none of them capture the reason someone might still choose the 3B. **The recommendation depends on a judgement this harness cannot make**, and the honest form of the result is: *if 1.5B-class quality is acceptable for the use case, it is 4.6x better on this hardware.*

---

## Experiment 3b — fp16 1.5B head-to-head at `--max-model-len 2048` (superseded, optional)

Kept for completeness. Superseded by Experiment 3c, which makes the same mechanistic point (decode is bandwidth-bound on weight bytes) without halving the context window to make the comparison possible.

Experiment 3 answered "can the unquantized 1.5B serve *this product's configuration*?" — no. It did **not** answer `PROJECT_PLAN.md`'s actual question, "which model wins on this card". This does, at the largest context length where a comparison is possible.

**Both arms run at `--max-model-len 2048`.** Reduced context is a real cost and this is explicitly not the shipping configuration; the point is a fair comparison, not a product recommendation.

**Prediction, stated before running — the counterintuitive one:**

| | 3B-AWQ | 1.5B fp16 | Reasoning |
| --- | --- | --- | --- |
| Weight bytes streamed per token | 1.954 GiB | **2.887 GiB** | 4-bit on more parameters beats fp16 on fewer |
| **ITL (decode)** | ~18–22 ms | **~27–33 ms — worse** | Decode is **bandwidth**-bound on weight bytes; 48% more bytes ≈ 48% slower |
| **TTFT (prefill)** | baseline | **faster** | Prefill is **compute**-bound on FLOPs; half the parameters ≈ half the work |
| KV cache @ 2048 | 28,480 tok → 13.9x concurrency | 2,624 tok → **1.28x** | |
| Throughput above concurrency ~2 | scales | **cannot batch** | KV-starved |

**Expected verdict: the 1.5B wins on TTFT and loses on everything else.** This is the compute-bound/bandwidth-bound split from Stage 2 separating two *models* rather than two phases — and it is why "smaller model = faster" is wrong in a way that matters.

**Results:** _(pending)_

---

## Thermal drift is reversible — recorded 2026-09-07

A 3B run taken after ~20 minutes of idle, identical config and workload to Experiment 2's arm A:

| conc | Exp 1 baseline (cold) | Exp 2 arm A (hot) | After cooling |
| ---: | --------------------: | ----------------: | ------------: |
| 1 | 55.4 | 41.5 | **51.1** |
| 4 | 159.9 | 121.1 | **154.5** |
| 8 | 256.1 | 194.2 | **245.3** |

**The 24% session drift recovered once the card cooled.** It is thermal and reversible, not degradation. This both confirms the drift diagnosis and strengthens the protocol rule: pairs must be measured consecutively, and a long session should include cooling breaks if absolute numbers matter.

---

## Experiment 4 — n-gram speculative decoding (plan item c, optional)

**Results:** _(pending)_
