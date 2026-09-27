# Capacity Model

**How many people can talk to this thing at once, and what decides the answer.**

Everything below is derived from measurements on one machine — a laptop RTX
3050 with 4 GiB of VRAM — and every number states how it was obtained. Where a
figure is inherited from an earlier stage rather than re-measured, it says so.

> **The answer: 20 concurrent chat users**, with 97% of messages getting a first
> token within 1.5 s of first send, retries included. The knee is sharp: 25
> users measured 67–78%. This document explains *why*; every number is also
> regenerated from the result files in [RESULTS.md](RESULTS.md).

---

## 1. The facts the model is built on

All measured, none assumed.

| Quantity | Value | How it was obtained |
| --- | --- | --- |
| Weights on GPU | **1.1018 GiB** | vLLM startup log, identical across every launch |
| KV cache | **69,760 tokens** | vLLM startup log |
| KV per token | **28.0 KiB** | `2 × 28 layers × 2 kv_heads × 128 head_dim × 2 bytes`, from `config.json` — and it back-solves exactly from the cache size |
| Context window | **32,768** | model's `max_position_embeddings`, `rope_scaling: null` so it is a hard ceiling |
| Concurrency at SLO, **cold prompts** | **6 requests** | `ramp.py`, every request a cold ~512-token prefill: p95 TTFT 1,005–1,135 ms with +24…33% margin (~69 °C idle) |
| Concurrency at SLO, cold prompts, deep heat-soak | **4 requests** | Same ramp with the card idling at ~78 °C beforehand: 6 fails by 7%. This is the thermal range, not measurement noise — see §6 |
| Concurrency that fails, cold prompts | **12 requests** | p95 TTFT 2,043 ms, −36% margin |
| Concurrency at SLO, **chat** | **10 requests** | `chat_sim.py` with retrying clients: paired back-to-back runs, 20-user SLO attainment 85–92% at 6 → 95.5% at 8 → 99% at 10 (later runs at 10 ranged 87–97%), admitted p95 no worse; throughput plateaus near 300 tok/s at 8–10 (`benchmarks/ablate_gateway.ps1`) |
| Prefill rate | **≈3,070 tok/s** linear term | fitted across 471 / 6,774 / 22,586-token prompts |
| Decode | **13–19 ms/token** | varies with sustained SM clock, see §6 |

**To be unambiguous about which number is shipped:** the gateway admits **10**.
The ramp's 6 is the right ceiling when every request is a cold prefill, and
the wrong one for chat, where each turn resends history the engine serves from
its prefix cache (60–90% hit rate in every run). Cold prompts are still
charged by their uncached size (`1 + uncached_tokens / 2,048` slots), which is
what keeps a burst of them inside the regime the ramp measured. Thermal state
still moves every number here (§6); the `ThroughputBelowThermalFloor` alert is
what tells an operator which state they are in.

### The context window costs nothing, and that is not obvious

`--max-model-len` is a **ceiling, not an allocation**. PagedAttention allocates
16-token blocks on demand and the KV cache is sized from whatever memory is
left after weights, regardless of the window. Measured directly:

```
--max-model-len  4096   ->  GPU KV cache size: 69,616 tokens
--max-model-len 32768   ->  GPU KV cache size: 69,616 tokens
```

Identical. Raising the window 8x cost zero memory. The only thing that moved
was vLLM's reported max-concurrency figure (17.00x → 2.12x), which describes a
worst case where every sequence runs to the full window — not an allocation.

> **Why two KV numbers appear in this repository — 69,616 and 69,760.**
> Both are real. vLLM *profiles free VRAM at startup*, and the Windows desktop
> compositor's usage varies by a few MB between launches, so the pool lands
> within ±0.2% of itself run to run. 144 tokens is 9 blocks, about 4 MB.
>
> The pair above is quoted as 69,616/69,616 because those two launches happened
> back to back: that is what makes "identical" a measurement rather than a
> coincidence. **69,760 is the figure from the shipped configuration** and is
> the one used everywhere capacity is computed. The measurement protocol already
> names this: differences under ±0.2% in KV cache size are noise, not an effect.

The real costs of a large window are **latency and cache pressure**, and both
are consequences of how long conversations actually get, not of the ceiling.

---

## 2. Prefill is quadratic, and the project assumed it was linear

Measured at three prompt sizes on the shipped config:

| prompt tokens | TTFT p50 | implied rate |
| ---: | ---: | ---: |
| 471 | 283 ms | — |
| 6,774 | 2,717 ms | 2,493 tok/s |
| 22,586 | 13,032 ms | **1,733 tok/s** |

The rate *falls* as the prompt grows, which no linear model produces. Fitting
`TTFT = c + a·n + b·n²`:

```
TTFT ≈ 0.13 s  +  n / 3,070  +  n² × 1.11e-8
        fixed      linear        attention (quadratic)
```

| n | predicted | measured |
| ---: | ---: | ---: |
| 471 | 0.29 s | 0.28 s |
| 6,774 | 2.85 s | 2.72 s |
| 22,586 | 13.16 s | **13.03 s** |

Within 5%, within 1% on the largest. The attention term is **1.6% of prefill at
471 tokens, 12% at 4,096, 43% at 22,586, and an extrapolated 53% at 32,768**.

*"Prefill scales linearly with prompt length"* — carried since Stage 2 — is a
short-prompt approximation that is wrong by a factor of two at the context
limit. A full 32,768-token prompt costs **≈23 s of TTFT**.

**Consequence for the SLO:** `p95 TTFT < 1.5 s` is only meaningful with a
stated prompt-size envelope. It has always implicitly meant ~500-token prompts.
It is now stated.

---

## 3. Long context also costs decode, and more on the smaller model

ITL rose **12.8 → 18.2 ms (+42%)** between a 6,774-token and a 22,586-token
conversation. Stage 6 measured only 2.4% degradation and concluded decode was
"weight-bandwidth-dominated, not attention-dominated" — but that was the 3B
across 1,709 tokens. Both follow from one ratio:

```
KV read / weight read
  3B  @  1,709 tok:   62 MB / 2,000 MB =  3.1%   (Stage 6 measured 2.4%)
1.5B  @  1,709 tok:   47 MB / 1,183 MB =  4.0%
1.5B  @ 22,586 tok:  632 MB / 1,183 MB = 53%
```

**The lighter the model, the more context hurts** — KV per token falls more
slowly than weight bytes do when you shrink a model. Stage 6's "doubling ITL
would need ~56,000 tokens" was computed for the 3B and does not transfer.

---

## 4. From concurrency to users

A chat user is not a concurrent request. They send a message, wait for a reply,
then think for a while.

```
duty cycle      = service_time / (service_time + think_time)
users supported = concurrency_limit / duty_cycle × utilisation_target
```

With ~4 s of service per turn under load (192 tokens at ~20 ms) and 10 slots,
12 s of think time gives a 25% duty cycle, so 10 slots would be fully busy at
~40 users. **The measured capacity is 20** — half that — and the gap is the
lesson of this section.

Duty-cycle arithmetic assumes arrivals are smooth, and chat arrivals are not.
They are bursty (Poisson), and with a tight queue budget (1.0 s) what decides
latency is the chance that a new message finds every slot busy. By Erlang C
that chance climbs steeply long before 100% utilisation. A refused user also
comes back 2–3 s later, adding load at exactly the wrong moment. So the usable
point sits near 50% utilisation, not 75–100%: measured with retrying clients,
20 users kept 97% of messages inside the SLO, and 25 users 67–78%.

Scaling that measured anchor by duty cycle (an extrapolation, not a
measurement):

| think time | duty cycle | users at ≥90% attainment |
| ---: | ---: | ---: |
| 12 s | 25% | **20** (measured) |
| 30 s | 12% | ~40 |
| 45 s | 8% | ~55 |

**This is why "how many users" has no single answer.** It is a function of think
time, and think time is a property of the product, not the hardware.

### Output length is the other half of the duty cycle

Service time is mostly decode (~20 ms per output token at ten concurrent),
so the reply length the product allows moves capacity as much as think time
does. Holding think time at 12 s and scaling the measured anchor by duty cycle:

| reply length | service time | duty cycle | users at ≥90% attainment |
| ---: | ---: | ---: | ---: |
| 192 tokens *(measured)* | ~4 s | 25% | **20** |
| 512 tokens | ~10 s | 46% | ~11 |
| 1,024 tokens *(gateway default)* | ~20 s | 63% | ~8 |
| 2,048 tokens *(gateway ceiling)* | ~41 s | 77% | ~6 |

Two consequences that were not obvious before this table existed:

1. **The `max_tokens` a client sends is a capacity decision, not a UX one.**
   A chat product that lets replies run to 1,000 tokens has a third of the
   capacity of one that keeps them at 200, on identical hardware.

2. **An absent `max_tokens` is the worst case, not a neutral one.** vLLM's
   OpenAI server defaults it to `max_model_len − input_length` — on a 32k
   window, ~30,000 tokens. A model that does not emit EOS holds a slot for
   seven minutes. The gateway therefore injects a default (1,024) when the
   field is missing and clamps anything above a ceiling (2,048), reporting
   the clamp in `X-Max-Tokens-Clamped-From`. Without that bound the "10 in
   flight" limit is a limit on *count*, not on *work*, and the capacity
   arithmetic above does not hold.

There is no priority or preemption for long generations. Continuous batching
gives every running sequence one token per scheduler step, and a sequence
holds its admission slot for the whole stream. What *could* starve others is a
long **prefill**: one 20k-token prompt held every short request for 11–13 s until
`--long-prefill-token-threshold=512` capped its share of each scheduler step
(now 0.4–0.6 s; case 3 in [FAILURE_MATRIX.md](FAILURE_MATRIX.md)).

---

## 5. The other constraint: the shared prefix cache

The KV cache is **one shared pool of 69,760 tokens, not a per-user allocation**.
Blocks are content-addressed by hash and evicted LRU. The hash is a *chain* —
block N incorporates blocks 1…N−1 — so a hit requires a contiguous prefix from
the very start, and evicting one early block invalidates everything after it.

```
working set = active users × average conversation length

  20 users ×  1,500 tok =  30,000   fits comfortably
  20 users ×  3,300 tok =  66,000   right at the edge
  20 users ×  6,300 tok = 126,000   ~1.8x over
  47 users ×  1,500 tok =  70,500   at the edge
```

A miss is not a small penalty. A fully cached 4,000-token turn prefills only the
new tokens; a fully evicted one re-prefills all 4,000 — roughly **9x the GPU
time**. Nothing surfaces this: no error, no log line, the request simply costs
more. The only signal is prefix cache hit rate.

### The crossover, and which constraint binds first

Both constraints are real, and which one you hit depends on think time:

```
admission binds when:  users > concurrency_limit / duty_cycle
cache binds when:      users > 69,760 / conversation_length
```

At 12 s think time and ~1,000-token conversations: admission binds at ~20
users, cache at ~70. **Admission binds first, by 3x.**

At 45 s think time and 4,000-token conversations: admission binds at ~55 users,
cache at ~17. **Cache binds first, by 3x.**

This is the single most important structural result here, and it was not
obvious in advance: **the binding constraint changes identity depending on how
people use the product.** A capacity model that reports one number is hiding
which regime it was measured in.

---

## 6. Thermal state is a first-class variable

Three runs of an identical command, differing only in the temperature the card
idled at beforehand:

| idle before run | conc 1 | conc 4 | conc 8 | ITL p50 |
| ---: | ---: | ---: | ---: | ---: |
| 58 °C | 89.0 | 257.2 | **416.7** tok/s | 10.1 ms |
| 69 °C | 70.0 | 230.7 | 367.0 tok/s | 13.2 ms |
| ~78 °C | 59.0 | 188.0 | 293.1 tok/s | 15.8 ms |

**All three reached the same 86–87 °C die temperature during the run.** What
differs is the clock the card can *hold* at that temperature: ~1,100–1,200 MHz
from cold, ~950–1,050 MHz once the chassis is heat-soaked. `nvidia-smi` reports
die temperature, not heatsink base temperature, so this second thermal variable
is invisible in every metric available.

This is the mechanism behind the "~24% session drift" this project measured in
Stage 11 and never explained.

And it produced a clean physical result. `clocks.mem` was pinned at 5,501 MHz in
every sample of every run, yet ITL swung 56% — tracking `clocks.sm` to within
6%:

| run | ITL p50 | clock implied by ITL | clock observed |
| --- | ---: | ---: | ---: |
| cold | 10.1 ms | ~1,250 MHz | 1,250–1,350 |
| warm | 13.2 ms | **957 MHz** | **~1,010** |

**Decode on this card is core-clock-gated, not memory-bandwidth-bound.** The SMs
cannot issue memory requests fast enough at thermally-limited clocks to
saturate the 192 GB/s bus. The v1 scaling law (`decode ∝ weight bytes`) still
holds across models at comparable thermal state; it is a special case of
`decode_time = max(bandwidth_time, issue_time)` where issue time wins.

---

## 7. The measured curve

All runs: `chat_sim.py`, Poisson arrivals, log-normal think time (mean 12 s),
6-message conversations that accumulate context, 192-token replies, fixed seed
42, 120 s per run, heat-soaked card, clients that retry a `503` like the OpenAI
SDK. **SLO attainment** is the share of messages whose first token arrived within
1.5 s of the first send, retries included. Every table here is regenerated in
[RESULTS.md](RESULTS.md) from the result files.

| users | SLO attainment | served first try | admitted p95 | output tok/s | cache hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | **100%** | 100% | 854 ms | 92 | 79% |
| 20 | **97%** | 97% | 840 ms | 182 | 88% |
| 25 | 67% / 78% | 71% / 81% | 1,437 / 1,315 ms | 195 / 184 | 74% / 93% |
| 30 | 64% | 71% | 1,823 ms | 218 | 87% |
| 40 | 35% | 41% | 1,633 ms | 230 | 85% |
| 60 | 24% | 27% | 2,190 ms | 236 | 81% |
| 10 *(control, re-run last)* | **100%** | 100% | 892 ms | 91 | 81% |

**The knee sits between 20 and 25 users, and it is sharp.** Admitted p50 roughly
triples between them, which is the engine saturating; output throughput then
plateaus in the low 200s tok/s while offered load keeps rising, and what gives
is the share of messages served on the first try. Past the knee, repeat runs
disagree (25 users: 67% and 78%): refused users come back 2–3 s later, so
saturated load feeds on itself. Below it, the control reproduced exactly.

### 7.1 The queue timeout, from the user's side

Worst-case admitted TTFT is roughly `queue timeout + engine TTFT`, which suggests
`1.5 s − ~0.9 s ≈ 0.6 s` of queue. That bound is right for admitted requests and
wrong for users, because it treats a refusal as free: a refused client waits
for `Retry-After: 2` and then queues again. Measured with retrying clients
(RESULTS §5b), 1.0 s gave the best attainment at 40 users (51% against 35% at
0.6 s), while 2.0 s pushed admitted p95 to ~2 s, past the SLO for the requests
that were served. **1.0 s is shipped.**

### 7.2 The prefix-cache knee: predicted, and it did not appear

The prediction on the record was a **knee**: capacity holding while the working
set fits the shared pool, then falling sharply as LRU eviction makes every turn
re-prefill its history.

**Hit rate never collapsed.** It held **63–95% across every run**, from 10 users
to 80. Conversation length, holding users at 20:

| opening context | mean prompt | SLO attainment | admitted p95 | cache hit |
| ---: | ---: | ---: | ---: | ---: |
| ~300 tokens | 588 | 88% | 1,494 ms | 82% |
| ~2,000 tokens | 1,808 | 60% | 2,914 ms | 74% |
| ~8,000 tokens | 6,244 | 27% | 5,350 ms | 70% |

Long conversations destroy attainment, but the cache holds. **Why the knee is
absent, and it is not a null result:**

1. **Admission control caps how many conversations are *progressing*.** The
   working set of the conversations actually advancing never outgrows the
   69,760-token pool, so eviction never becomes the binding effect. **Load
   shedding protects the prefix cache as a side effect**: two mechanisms
   designed independently that reinforce each other.

2. **Within a conversation, the prefix is reused by construction.** Turn *n*
   resends turns 1…*n*−1, computed seconds ago and still the most recently used
   blocks in the pool. The expensive event is the **cold first turn**: every
   user in the 8k run opens by pasting ~8,000 tokens, and twenty cold, quadratic
   prefills are close to a minute of GPU time on their own.

3. **The shared system prompt is always a hit**, for every user on every turn,
   which puts a floor under the aggregate hit rate.

So on this hardware **the cache knee is unreachable while admission control is
doing its job**: the capacity question the project set out to resolve (~17 users
if cache-bound, ~110 if compute-bound) resolves to **admission-bound, 20 users at
≥90% attainment**.

*Caveat:* prefix-cache counters are cumulative per engine process, and the
sweep's runs share one engine, so a later run can inherit blocks an earlier one
computed (25 users: 74% and 93% hit rate in two runs). The *shape* of the result
(no collapse) holds across every arm; the absolute percentages lean optimistic.

### 7.3 Traffic shapes

| shape | users | SLO attainment | served first try | admitted p95 | cache hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| **diurnal** (slow sine, peak 40) | 40 | **87%** | 87% | 1,233 ms | 95% |
| **ramp** (population grows over the run) | 60 | 54% | 54% | 1,378 ms | 84% |
| **burst** (baseline, then 4x spike) | 80 | 11% | 24% | 2,983 ms | 73% |
| **thundering herd** (all connect at once) | 60 | 9% | 12% | 3,230 ms | 63% |

The realistic shape is the gentle one: a daily cycle peaking at twice the
capacity point still keeps 87%, because the peak is brief and the cache stays
warm. A burst or a herd far past capacity fails on attainment but not on
correctness: nothing errors, every refusal carries `Retry-After`, and admitted
requests are still served in about 3 s. The herd is the worst case, because
every client arrives before any request has completed, and its cold openings
are the only time the cache hit rate dipped to 63%.

### 7.4 Fairness, quantified

One key issuing 50 concurrent requests continuously, against ten normal users
on their own keys: the normal users kept **94% SLO attainment** (7 of 10 with
every message inside it), while the abusive key had **2,014 of 2,079 requests
refused (96.9%)**.

The mechanism is a **dynamic** share: `ceil(10 / contending keys)`, capped at
half the pool. A fixed cap of half the pool was measured too, and it is not
fairness but a quota: with a fixed 3 of 6 slots, one abusive tenant pushed
**10/10 normal users past the SLO**. Dividing by the number of *contending*
keys drops the abuser's share to a tenth of the engine the moment ten other
tenants appear, and gives it back when they leave.

---

## 8. The levers, in order of effect

| Lever | Effect | Cost |
| --- | --- | --- |
| **Shared system prompt** | Doubles SLO-compliant concurrency (Stage 11 Exp 2: 2x, TTFT −82%) | None. Structure the workload to share a prefix |
| **Shorter conversations** | Directly shrinks the working set; the primary cache lever | Loses history; a trim is also a guaranteed cache miss |
| **Smaller model** | 4.3x more SLO-compliant throughput (3B → 1.5B) | Answer quality — measured as *no observable difference* on this project's control question, but that is one data point |
| **Admission limit measured on the real workload** | 10 in flight on chat, where a cold-prompt ramp says 6 | A burst of cold long prompts must still be charged by size (cost on uncached tokens) |
| **`--long-prefill-token-threshold 512`** | One long prefill no longer blocks everyone (11–13 s → 0.4 s for short requests) | The long prompt's own prefill takes more scheduler steps |
| **Queue timeout** | Bounds admitted TTFT under overload; 1.0 s maximises attainment with retrying clients | Longer waits break the SLO for admitted work; shorter ones turn waits into ≥2 s retries |
| **Better cooling** | Decode scales with sustained SM clock, ~1.5x observed range | Hardware |
| **`--max-num-seqs`** | Raises peak throughput (367 → 686 tok/s) | Nothing at or below the SLO — it buys throughput you cannot use |
| **`--gpu-memory-utilization`** | Every extra MiB becomes KV cache at 28 KiB/token | Startup becomes dependent on desktop VRAM state; see §9 |

---

## 9. Honest limitations

**Not measured:**

* Answer quality beyond a single control question. Every performance conclusion
  favouring the 1.5B is conditional on a judgement made on one prompt.
* Multi-replica behaviour. SQLite has one writer; budget accounting across two
  gateway processes is documented as a gap, not solved.
* Real user traffic. Think times, conversation lengths and topic mix are
  simulated from plausible distributions, not sampled from a live product.
* `--gpu-memory-utilization` above 0.78. Headroom demonstrably exists — desktop
  applications were holding 290 MiB that a headless machine would not — but
  raising it makes engine startup depend on what else is running on the
  desktop, which breaks the "one command brings the stack up" property. Left at
  0.78 deliberately, with the finding recorded.
* Speculative decoding, and Kubernetes self-healing.

**Measured but not transferable:**

* Every throughput figure is thermally bounded on a 35 W laptop part that
  throttles from ~1,500 MHz to ~950–1,200 MHz sustained. A desktop 3050 with
  the same memory bandwidth and real cooling should be substantially faster,
  and §6 predicts roughly linearly with sustained clock. **This project cannot
  test that.**
* WSL2 forces `pin_memory=False`, a real but unquantified cost.
* Windows WDDM reserves VRAM the engine cannot use, which is why
  `--gpu-memory-utilization` is 0.78 rather than the 0.9 vLLM's own guidance
  suggests.

**The honest form of the headline:** this is a characterisation of one stack on
one machine, with the derivation of why it behaves as it does and a record of
every wrong turn taken to get there. It is not a benchmark of the RTX 3050, and
it is not comparable to published figures that run larger models on datacenter
GPUs without thermal limits.
