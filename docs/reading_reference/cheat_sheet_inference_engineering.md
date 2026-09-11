# Inference Engineering Cheat Sheet
*Synthesized from 14 sources: The AI Engineer's Hardware & Inference course, bbycroft.net/llm, Horace He's "Brrrr" post, and Aleksa Gordić's vLLM internals deep-dive.*

---

## 1. Foundations — why inference costs what it costs

**Inference has two phases with opposite performance profiles, and almost everything else in this sheet is a consequence of that one fact.**

- **Prefill** — the whole prompt processed in one parallel pass. Produces the first token and fills the KV cache. **Compute-bound**: the GPU's math units are the limit.
- **Decode** — one token at a time, each step pulling the *entire* model's weights out of memory to compute one new token. **Memory-bandwidth-bound**: the GPU spends most of its time hauling weights, not computing with them.

**The KV cache** is why decode doesn't get slower with every token. Each token leaves behind a Key and Value vector per layer; instead of recomputing attention over the whole prefix at every step, the model looks up cached K/V from all previous tokens. It grows by one entry per token, per request, and lives in the same VRAM as the model weights — so it's the thing that actually runs out first, not the weights.

**The economics, concretely:**
- Output tokens cost **3–10x** more than input tokens (one trip through the model *per token* vs. one trip for the whole prompt).
- Before vLLM's PagedAttention, servers reserved a worst-case-sized contiguous memory block per request — **60–80% of GPU memory sat empty**, wasted on space that was reserved but unused. PagedAttention (paging the KV cache into small fixed-size blocks, like OS virtual memory) dropped that waste to under 4% and doubled-to-quadrupled tokens/sec on the same hardware.
- This is why **memory bandwidth, not raw compute, is the bottleneck the industry actually fights** — OpenAI's inference spend ($8.4B in 2025, projected $14.1B in 2026) dwarfs training spend, and the "memory wall" hits small models exactly as hard as large ones — it just scales down with you.

**Four ways teams misdiagnose this** (from *What is Inference?*):
1. Buying a faster/bigger GPU for a memory-bound problem — the arithmetic units were never the bottleneck.
2. Benchmarking one request at a time — production traffic is concurrent; a solo-request tokens/sec number is fiction.
3. Confusing `max_model_len` (what you *allow*) with the length traffic *actually sends* — the ceiling costs nothing until real long requests show up and eat your cache.
4. Optimizing the phase that isn't your bottleneck — chat is decode-heavy, classification is prefill-heavy; fix the wrong one and the bill doesn't move.

**The transformer itself:** bbycroft.net/llm is worth sitting with directly rather than just reading about — it's an interactive 3D walkthrough of a GPT-style model's literal forward pass: token embedding → positional encoding → stacked blocks of self-attention (Q/K/V) and MLP → final layer norm → softmax to next-token logits. If any attention explanation ever feels abstract, this is the fastest way to make it concrete again — you can watch the actual matrices light up.

---

## 2. GPU & hardware — what determines the bottleneck

**CPU vs GPU is a latency-vs-throughput trade, not a "better/worse" one.** A CPU has a handful of powerful, generalist cores built for fast sequential/branchy work. A GPU has thousands of simple cores (SIMT — single instruction, multiple threads) built to do the *same* operation across massive data simultaneously — the exact shape of matrix multiplication, which is nearly all a neural network does. Tensor Cores go further: dedicated hardware that does a whole small matrix multiply in one operation, and they — not the general CUDA cores — do most of a modern GPU's real AI throughput.

**Three specs that actually matter, in priority order:**
1. **Memory capacity (VRAM)** — does the model fit at all? Rule of thumb: **~2GB per billion parameters** for inference at FP16 (~0.5GB/B at 4-bit).
2. **Memory bandwidth** — how fast weights move from VRAM to the compute units. This is decode's bottleneck. (H100: 3.35 TB/s vs. a laptop's ~50GB/s DDR5 — a ~67x gap.)
3. **Compute throughput (TFLOPS)** — matters most for prefill and training, least for single-request decode.

**When a GPU is *not* the answer:** small classifiers, light/single-request traffic, or anything that isn't matrix math (data cleaning, business logic). The PCIe transfer between CPU and GPU memory has real cost — for a single small request, the round trip can cost more than the compute it enabled.

**H100 vs. H200 vs. B200** — the decision tree is *always* capacity → bandwidth → compute → interconnect:
| GPU | Memory | Bandwidth | Best for | Skip if |
|---|---|---|---|---|
| H100 | 80GB HBM3 | 3.35 TB/s | Fits in 80GB, $/token matters most, deepest software support | Need >80GB or FP4 |
| H200 | 141GB HBM3e | 4.8 TB/s | Memory-bandwidth-bound decode (most decode is), 70–100B models | Compute-bound (identical compute to H100) |
| B200 | 192GB HBM3e | 8 TB/s | Native FP4, rack-scale inference/training | Hopper-only stack, can't handle 1000W TDP |

**Note the honest theme across every hardware and engineering article here:** most teams don't actually have a compute problem. They have a memory problem they're trying to solve by buying more compute.

**Multi-GPU training failure modes** (only relevant once you scale past one GPU — good to recognize, not yet needed on a single RTX 3050): all-reduce is a barrier, so every GPU waits for the slowest one; a hang almost never comes from the GPU that reports the timeout; one GPU can OOM while its neighbors have room, because memory pressure is per-rank; and a checkpoint that saves only weights (not optimizer state, scheduler position, dataloader position, RNG state) can't actually resume a run. The article's own conclusion is worth keeping: **a single-GPU job sidesteps every one of these failure classes entirely** — which is exactly your situation right now.

---

## 3. The performance mental model that ties it all together

**Every deep learning performance question reduces to three regimes** (Horace He, "Making Deep Learning Go Brrrr"):
- **Compute-bound** — the math units are the limit. Fix: bigger/faster GPU, use Tensor Cores.
- **Memory-bandwidth-bound** — moving data between GPU memory and compute units is the limit. Fix: **operator fusion** (do several operations back-to-back without writing intermediate results back to memory and re-reading them — the single most important deep learning compiler optimization) or, for inference specifically, quantization (less data to move).
- **Overhead-bound** — Python/framework dispatch overhead dominates because the actual GPU work is tiny. Fix: tracing, avoid eager-mode Python, batch bigger.

**How to tell which regime you're in:**
- Compute-bound: achieved FLOPS is close to the GPU's peak FLOPS.
- Overhead-bound: double your batch size — if runtime barely moves, you weren't using the GPU, you were waiting on Python.
- `nvidia-smi`'s GPU-Util is roughly "what fraction of the time *any* kernel was running" — a good first signal of whether the GPU is starved by overhead.

**This one framework explains why prefill and decode behave oppositely**, why quantization helps decode specifically (fewer bytes = less bandwidth-bound waiting), and why batching helps throughput (amortizes the same memory-bound weight-read across more useful work) — everything downstream in this sheet is really a special case of this three-way model.

---

## 4. Optimization — quantization

**The core idea (JPEG for weights):** FP16 (2 bytes/weight) → INT8/INT4 (1 byte / 0.5 bytes). Map each weight's value range into a small number of bins (256 for INT8, 16 for INT4), store the bin index plus a scale factor. A 70B model: **140GB at FP16 → ~35GB at 4-bit** — the difference between needing 2 GPUs and needing part of one.

**PTQ vs. QAT:** Post-Training Quantization (GPTQ, AWQ, GGUF — quantize an already-trained model, no retraining, the default in practice) vs. Quantization-Aware Training (simulates low precision *during* training, better at extreme 2–3 bit but needs the full training infrastructure most teams don't have).

**The three formats, and why the choice is really about your *runtime*, not the algorithm:**

| Format | What it does | Best runtime | Gotcha |
|---|---|---|---|
| **GPTQ** | Quantizes column-by-column, uses Hessian info to compensate error in later columns | vLLM/GPU, with **Marlin kernel** | Column-wise error compounds — weaker on code/reasoning |
| **AWQ** | Protects the ~1% most important weight channels (by activation magnitude) with a scale factor before rounding | vLLM/GPU, with Marlin | Needs a published AWQ checkpoint; Turing-generation GPU or newer for Marlin |
| **GGUF** | A *file format*, not an algorithm — block+sub-block scales (Q4_K_M ≈ 4.5 effective bits) | llama.cpp / Ollama / CPU / Apple Silicon | **A trap inside vLLM**: no fast GPU kernel for its layout, ~93 tok/s vs. hundreds |

**The single most important fact in this whole section:** the *same* AWQ weights ran at **68 tok/s on vLLM's default kernel and 741 tok/s on the Marlin kernel** — same weights, same GPU, ~11x difference. The format matters far less than whether your serving stack has a fast kernel for it. **Match the format to the runtime, not the runtime to the format.**

**Practical rule of thumb:** don't quantize a small model to save a few more GB — smaller models have less redundancy and degrade faster per bit removed. Quantize the *largest* model that fits your latency budget instead. And: math/code/multi-step reasoning degrades before general Q&A does, at any given bit-width — test on your actual task, not a generic benchmark.

---

## 5. Inference engines — matching the tool to the workload

Every engine below reads/writes an OpenAI-compatible API, so switching between them in your own application code is usually a one-line base-URL change.

| Engine | Core trick | Best for | Ceiling |
|---|---|---|---|
| **Ollama** | Simplicity — one command, no config | Local dev, prototyping, single-user | No request batching at all — throughput flatlines regardless of load |
| **vLLM** | **PagedAttention** (paged KV cache, near-zero memory waste) | General production default, multi-hardware, broadest ecosystem | Doesn't reuse KV cache across requests with shared context as aggressively as SGLang |
| **SGLang** | **RadixAttention** (a radix tree caching KV by shared prefix across *different* requests) | Chat, RAG, agents — anywhere many requests share a system prompt/context (75–95% cache reuse in practice) | No advantage when every prompt is unique |
| **TensorRT-LLM** | Ahead-of-time, hardware-specific compiled engine | Long-lived, high-throughput, NVIDIA-only, stable-model deployments | 28+ minute compile per model change; 1–2 weeks setup; NVIDIA lock-in |
| **TGI** | — | *Officially in maintenance mode; HuggingFace recommends vLLM/SGLang* | Migrate away |

**The decision, in one line:** setup-speed dominates → Ollama. Shared context across requests dominates → SGLang. Peak NVIDIA throughput on a model that won't change → TensorRT-LLM. Everything else, or you want to keep your options open → vLLM (the safe, broad default).

---

## 6. Inside vLLM — what's actually happening under `vllm serve`

*(This section goes deeper than the others deliberately — it's the mechanism behind most of the sheet above.)*

**The engine loop, every step:** (1) **Schedule** — decide which requests run this step, prioritizing already-running decode requests over new prefill ones; (2) **Forward pass** — run the model, flattening every active sequence into one long "super-sequence" so continuous batching needs no padding; (3) **Postprocess** — append sampled tokens, check stop conditions, free KV blocks for finished requests.

**The KV cache manager** holds a pool of fixed-size blocks (16 tokens each, by default). A request needing 17 new tokens needs `ceil(17/16) = 2` blocks. When the pool runs low, vLLM either evicts a lower-priority request (**preemption** — recomputing it later) or holds off scheduling — this is the literal mechanism behind PagedAttention.

**Prefix caching** is the same block mechanism turned into a cache: each 16-token block gets hashed, and if a new request's opening tokens hash-match a block that's already computed (a shared system prompt, a repeated document), vLLM reuses it instead of recomputing — this is what SGLang's RadixAttention specializes further into a full tree structure.

**Chunked prefill** caps how many prefill tokens get processed per step, so one enormous prompt can't monopolize an entire engine step and starve every other request's decode — a scheduling fairness fix, not a memory fix.

**Speculative decoding:** a small/cheap method (n-gram lookup, or lightweight models like EAGLE/Medusa in vLLM's actual implementation — not a full separate LLM) proposes several tokens; the big model verifies all of them in one forward pass and accepts/rejects left to right. Statistically identical output distribution to normal decoding, just fewer expensive big-model passes.

**Disaggregated prefill/decode:** since prefill is compute-bound/bursty and decode is memory-bound/steady, dedicated prefill GPUs and dedicated decode GPUs can each be sized and scaled independently, with the KV cache transferred between them. (This needs multiple GPUs — the frontier-scale version of everything in this sheet, not something a single card does.)

**Scaling order:** tensor parallelism (shard weights across GPUs *within* one node, needs NVLink-class bandwidth) before pipeline parallelism (split model *across* nodes, needs less bandwidth but adds latency) before data-parallel replicas behind a load balancer.

**The metrics that actually matter** (formal definitions, matching your Module 6 exactly):
| Metric | Definition |
|---|---|
| TTFT | Time to first token — dominated by prefill + queueing |
| ITL | Time between consecutive tokens — dominated by decode + concurrent load |
| TPOT | Average ITL across a whole response |
| E2E latency | TTFT + sum of all ITLs |
| Throughput | Total tokens/sec across all requests |
| **Goodput** | Throughput counting *only* requests that met their SLO — the metric that actually reflects a usable server |

**The roofline model, precisely:** below a saturation batch size (`B_sat`), step time is nearly flat regardless of batch — you're bandwidth-bound, streaming weights in dominates. Above `B_sat`, step time grows with batch size — you're compute-bound, and every extra token in the batch adds real ITL. This is the exact mechanism behind "batching trades latency for throughput," now with a name for *where* the crossover happens.

`vllm bench {latency, throughput, serve}` are real, built-in CLI tools for measuring exactly these numbers, and vLLM ships an auto-tune script that searches configs to hit a target SLO (e.g., "maximize throughput while p99 latency stays under 500ms").

---

## 7. Should you self-host, and what breaks when you do

**The decision, in veto order** (each one overrides the ones after it):
1. **Sovereignty** — if data legally can't leave your network, self-host, full stop, regardless of cost.
2. **People** — no ops capacity → managed deployment (rent GPUs + serving stack, your data stays in your cloud tenancy). Have ops capacity → full self-hosting is viable.
3. **Volume** — under ~1M tokens/day, a hosted API is cheaper outright. ~2M/day is the real breakeven. Past 10M/day, owned hardware usually pays back in 6–12 months.
4. **Model access** — the strongest frontier models are closed-weight; that slice of work goes to an API no matter what the other three say.

Most real teams land on a **hybrid**: sensitive/high-volume/simple work stays local, rare hard-reasoning work goes to a frontier API. Reported savings: 40–70% vs. all-API.

**The part almost nobody accounts for going in:** self-hosting doesn't replace one vendor contract with zero maintenance — it replaces it with **four handoffs you now own**, and none of them show up in an engine benchmark:
1. **Your API layer can silently drop a field** (e.g. `response_format`) — the model answers in prose, nothing errors, and whoever's on call debugs the wrong layer. *Fix: parse response bodies in tests, don't just check status codes.*
2. **Version drift between your model manager and the engine** — e.g. a wrapper importing the engine's internal modules by name breaks on the engine's very next patch release. *Fix: pin both together, assert the exact import in CI.*
3. **Your monitoring only forwards a fraction of the engine's real metrics** — commonly missing exactly the two that would have warned you: **KV cache fullness %** and **prefix-cache hit rate**. *Fix: scrape the engine's own metrics endpoint directly; alert above ~90% sustained KV cache usage.*
4. **The engine can silently rewind a request** under memory pressure (evict-and-reschedule with a shorter token list) — any state your own code kept per-request (a streaming buffer, a token counter) can end up double-counting or corrupted. *Fix: detect a token-list shrink between steps and rebuild your state from scratch.*

The honest one-line summary of this entire source: **benchmarks measure inside components; they never measure the seams between them — and the seams are where self-hosted stacks actually break.**
