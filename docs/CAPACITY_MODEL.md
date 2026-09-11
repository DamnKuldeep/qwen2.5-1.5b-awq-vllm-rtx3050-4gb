# Capacity Model

**How many people can talk to this thing at once, and what decides the answer.**

Everything below is derived from measurements on one machine — a laptop RTX
3050 with 4 GiB of VRAM — and every number states how it was obtained. Where a
figure is inherited from an earlier stage rather than re-measured, it says so.

---

## 1. The facts the model is built on

All measured, none assumed.

| Quantity | Value | How it was obtained |
| --- | --- | --- |
| Weights on GPU | **1.1018 GiB** | vLLM startup log, identical across every launch |
| KV cache | **69,760 tokens** | vLLM startup log |
| KV per token | **28.0 KiB** | `2 × 28 layers × 2 kv_heads × 128 head_dim × 2 bytes`, from `config.json` — and it back-solves exactly from the cache size |
| Context window | **32,768** | model's `max_position_embeddings`, `rope_scaling: null` so it is a hard ceiling |
| Concurrency at SLO | **6 requests** | `ramp.py`, p95 TTFT 1,005–1,135 ms with +24…33% margin, on a card at its normal warm idle (~69 °C) |
| Concurrency at SLO, deep heat-soak | **4 requests** | Same ramp with the card idling at ~78 °C beforehand: 6 fails by 7%. This is the thermal range, not measurement noise — see §6 |
| Concurrency that fails | **12 requests** | p95 TTFT 2,043 ms, −36% margin |

**To be unambiguous about which number is shipped:** the gateway admits **6**.
That holds the SLO with margin in the thermal state a laptop reaches after a
few minutes of use. After an hour of sustained load the same command measures
a ceiling of 4, and the 6-slot limit will then let p95 drift ~7% over target
before shedding catches it. The honest product statement is "6, degrading to 4
when the chassis is saturated" — and the alert on sustained throughput below
floor (`ThroughputBelowThermalFloor`) is what would tell an operator which
state they are in.
| Prefill rate | **≈3,070 tok/s** linear term | fitted across 471 / 6,774 / 22,586-token prompts |
| Decode | **13–19 ms/token** | varies with sustained SM clock, see §6 |

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

With a measured service time of ~3 s per turn and a concurrency limit of 6:

| think time | duty cycle | users at 100% | users at ~75% (usable) |
| ---: | ---: | ---: | ---: |
| 12 s | 20% | 30 | **~22** |
| 30 s | 9.1% | 66 | **~49** |
| 45 s | 6.3% | 96 | **~72** |

The 75% figure is not arbitrary: queueing delay grows without bound as
utilisation approaches 1, and the queue timeout converts that delay into 503s.
Running a latency-sensitive service above ~75% utilisation trades a large
amount of tail latency for a small amount of throughput.

**This is why "how many users" has no single answer.** It is a function of think
time, and think time is a property of the product, not the hardware.

### Output length is the other half of the duty cycle

Service time is mostly decode — ~14 ms per output token at six concurrent —
so the reply length the product allows moves capacity as much as think time
does. Holding think time at 12 s:

| reply length | service time | duty cycle | usable users |
| ---: | ---: | ---: | ---: |
| 192 tokens *(measured)* | ~3.0 s | 20% | **~22** |
| 512 tokens | ~7.5 s | 38% | ~12 |
| 1,024 tokens *(gateway default)* | ~14.6 s | 55% | ~8 |
| 2,048 tokens *(gateway ceiling)* | ~29 s | 71% | ~6 |

Two consequences that were not obvious before this table existed:

1. **The `max_tokens` a client sends is a capacity decision, not a UX one.**
   A chat product that lets replies run to 1,000 tokens has a third of the
   capacity of one that keeps them at 200, on identical hardware.

2. **An absent `max_tokens` is the worst case, not a neutral one.** vLLM's
   OpenAI server defaults it to `max_model_len − input_length` — on a 32k
   window, ~30,000 tokens. A model that does not emit EOS holds a slot for
   seven minutes. The gateway therefore injects a default (1,024) when the
   field is missing and clamps anything above a ceiling (2,048), reporting
   the clamp in `X-Max-Tokens-Clamped-From`. Without that bound the "6 in
   flight" limit is a limit on *count*, not on *work*, and the capacity
   arithmetic above does not hold.

There is no priority or preemption for long generations. Continuous batching
gives every running sequence one token per scheduler step, and a sequence
holds its admission slot for the whole stream. What *can* starve others is a
long **prefill** (case 3 in [FAILURE_MATRIX.md](FAILURE_MATRIX.md)), which is a different mechanism.

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

At 12 s think time and ~1,000-token conversations: admission binds at ~22
users, cache at ~70. **Admission binds first, by 3x.**

At 45 s think time and 4,000-token conversations: admission binds at ~72 users,
cache at ~17. **Cache binds first, by 4x.**

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

All runs: `chat_sim.py`, Poisson arrivals, log-normal think time (median 12 s),
6-turn conversations that accumulate context, fixed seed 42, 120 s, discarded
warm-up, heat-soaked card. Engine config embedded in every result file.

### 7.1 Degradation is bounded — and the queue timeout decides where the ceiling sits

The same sweep run twice, changing one variable: `GATEWAY_QUEUE_TIMEOUT_S`.

**Queue timeout 2.0 s:**

| users | shed % | p95 TTFT | worst user p95 | users over SLO | cache hit |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 10 | 0.0 | 260 ms | 592 ms | 0/10 | 83.4% |
| 10 *(control)* | 0.0 | **266 ms** | 834 ms | 0/10 | 83.9% |
| 20 | 0.0 | 1,137 ms | 1,570 ms | 1/20 | 88.8% |
| 30 | 10.0 | **2,070 ms** | 2,201 ms | 21/30 | 86.8% |
| 40 | 22.5 | **2,071 ms** | 2,211 ms | 32/40 | 88.1% |
| 60 | 47.1 | **2,116 ms** | 2,417 ms | 49/58 | 83.9% |

**Queue timeout 0.6 s (shipped):**

| users | shed % | p95 TTFT | worst user p95 | users over SLO | cache hit |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 10 | 0.0 | 191 ms | 215 ms | **0/10** | 96.6% |
| 20 | 8.0 | 434 ms | 1,090 ms | **0/20** | 85.8% |
| 30 | 25.1 | 706 ms | 801 ms | **0/30** | 83.0% |
| 40 | 38.9 | 717 ms | 920 ms | **0/40** | 83.5% |
| 60 | 56.9 | **778 ms** | 898 ms | **0/58** | 81.8% |

**Two things happen here, and the second is the more useful one.**

First, degradation is bounded in both configurations. At a 2.0 s timeout, p95
TTFT rose **2.2% while offered load doubled** (30 → 60 users). That flat line is
admission control working: excess load is refused with `503` + `Retry-After`
instead of being absorbed into a queue that makes everyone slower.

Second — and this is the part that matters — **bounded is not the same as
acceptable.** At 2.0 s the flat line sits at ~2,070 ms, *above* the 1.5 s
objective, and **49 of 58 users breached the SLO at 60 users**. The service
degraded gracefully into a state that still failed its promise.

The fix is arithmetic, not tuning:

```
worst-case client TTFT  ≈  queue_timeout + engine_TTFT
1.5 s objective − ~0.8 s engine TTFT  ≈  0.6 s of queue budget
```

At 0.6 s, the flat line drops to **778 ms at 60 users, and 0 of 58 users breach
the SLO** — for **10 percentage points more shedding** (47.1% → 56.9%).

> **That trade is the whole thesis of admission control**: refusing one request
> in ten more, so that every request you *do* accept is served within its
> latency target. A 503 the client can retry is a better product than a
> success nobody wanted to wait for.

### 7.2 The prefix-cache knee: predicted, and it did not appear

The prediction on the record was a **knee** — capacity holding while the
working set fits the shared pool, then falling sharply as LRU eviction makes
every turn re-prefill its history.

**Hit rate never collapsed.** It held **74–97% across every run**, from 10 users
to 60, and from 550-token conversations to 2,500-token ones.

The conversation-length ablation, holding users at 20 and growing conversations:

| target opening | mean prompt | shed % | p95 TTFT | **worst user p95** | users over SLO | cache hit |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 300 | 553 | 10.3 | 560 ms | 916 ms | 0/20 | 79.0% |
| 1,000 | 868 | 19.7 | 747 ms | 867 ms | 0/20 | 76.2% |
| 2,000 | 1,312 | 15.4 | 912 ms | 1,473 ms | 0/20 | 82.1% |
| 4,000 | 1,731 | 26.7 | 1,844 ms | **3,153 ms** | 6/20 | 83.7% |
| 8,000 | 2,541 | 28.2 | **3,380 ms** | **9,621 ms** | 6/20 | 86.7% |

Long conversations *do* destroy the tail — worst-user p95 goes from 916 ms to
**9,621 ms**, a 10x degradation. But hit rate went **up**, not down.

**Why the knee is absent, and it is not a null result:**

1. **Admission control caps how many conversations are *progressing*.** With six
   slots and 57% shedding at 60 users, only a couple of dozen conversations
   ever advance. The working set never outgrows the 69,760-token pool, so
   eviction never becomes the binding effect. **Load shedding protects the
   prefix cache as a side effect** — two mechanisms designed independently that
   turn out to reinforce each other.

2. **Within a conversation, the prefix is reused by construction.** Turn *n*
   resends turns 1…*n*−1, which were computed seconds ago and are still the most
   recently used blocks in the pool. The expensive event is not a mid-conversation
   miss; it is the **cold first turn**, which for an 8,000-token opening costs
   ~3.4 s of prefill by the model in §2. That is what the worst-user tail is
   made of.

3. **The shared system prompt is always a hit**, for every user on every turn,
   which puts a floor under the aggregate hit rate that no amount of eviction
   removes.

So the honest revision to the capacity model: **on this hardware the cache knee
is unreachable while admission control is doing its job.** The 6x uncertainty
this project set out to resolve — ~17 users if cache-bound, ~110 if
compute-bound — resolves to **~22 users, admission-bound**, and the cache is not
the binding constraint in any regime we can actually reach.

*Caveat, stated because it cuts against the result:* prefix-cache counters are
cumulative per engine process, and these runs share one. A later run inherits
blocks the earlier ones computed — the 96.6% at 10 users, measured immediately
after other runs, is visibly inflated relative to the 83.4% measured earlier
from a colder cache. Isolating this properly needs an engine restart between
arms, which was not done. The *shape* of the result (no collapse) is robust
because it holds across every arm; the absolute percentages are optimistic.

### 7.3 Traffic shapes

| shape | users | shed % | p95 TTFT | worst user p95 | users over SLO |
| --- | ---: | ---: | ---: | ---: | :---: |
| **burst** (baseline, then 4x spike) | 80 | 67.4 | 994 ms | 1,058 ms | **0/68** |
| **ramp** (population grows over the run) | 60 | 47.5 | 988 ms | 1,308 ms | **0/51** |
| **thundering herd** (all connect at once) | 60 | 63.9 | 1,064 ms | 2,257 ms | 5/55 |
| **adversarial** (1 abusive key @ 50 concurrent) | 10 | 10.7 | 536 ms | 818 ms | **0/10** |

The herd is the only shape that breaches, and only for 5 of 55 users. That is
the expected weak point: every client arriving in the same instant means the
queue fills before any request has completed, so the first cohort waits the
full queue timeout. It recovers within one think-time cycle.

### 7.4 Fairness, quantified

One key issuing 50 concurrent requests continuously, against ten normal users:

| | p95 TTFT (normal users) | worst user | users over SLO |
| --- | ---: | ---: | :---: |
| No abuser (baseline) | 191 ms | 215 ms | 0/10 |
| **With abuser** | **536 ms** | 818 ms | **0/10** |

**Degradation: +181%, and every user still 64% inside the SLO.** The abusive key
attempted 4,749 requests and had **4,669 shed (98.3%)**.

This required a fix. The first implementation used a **fixed** per-key cap of 3
against 6 slots, which hands one tenant half the service regardless of how many
other tenants exist — and it measured **10/10 normal users over SLO**. The
shipped version divides the pool by the number of *contending* keys, so the
abusive key's share falls from 50% to ~14% the moment other tenants appear.

---

## 8. The levers, in order of effect

| Lever | Effect | Cost |
| --- | --- | --- |
| **Shared system prompt** | Doubles SLO-compliant concurrency (Stage 11 Exp 2: 2x, TTFT −82%) | None. Structure the workload to share a prefix |
| **Shorter conversations** | Directly shrinks the working set; the primary cache lever | Loses history; a trim is also a guaranteed cache miss |
| **Smaller model** | 4.3x more SLO-compliant throughput (3B → 1.5B) | Answer quality — measured as *no observable difference* on this project's control question, but that is one data point |
| **Queue timeout** | Bounds p95 TTFT under overload | Higher 503 rate; the trade is explicit |
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
