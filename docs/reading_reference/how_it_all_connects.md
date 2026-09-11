# How It All Connects
*Part 2 of the reading synthesis — the throughline, the alternatives map, the gaps, and the deployment pipeline as a whole.*

---

## The one throughline underneath all 14 sources

There's a single chain of cause-and-effect running through everything you read, even though the 14 pieces were written by different people for different purposes:

```
GPU hardware reality (memory bandwidth << compute speed)
        ↓ causes
prefill = compute-bound, decode = memory-bandwidth-bound
        ↓ which Horace He's 3-way model explains generally
compute / bandwidth / overhead — find which one you're actually starved on
        ↓ so the fix for decode is "move less data"
quantization (fewer bytes per weight → less bandwidth-bound waiting)
        ↓ but the fix only works if the kernel is fast
format (GPTQ/AWQ/GGUF) matched to a kernel (Marlin, etc.) matched to a runtime
        ↓ the runtime is a serving engine
vLLM/SGLang/TensorRT-LLM/Ollama — each optimizing a different part of "many requests, one GPU, don't waste memory"
        ↓ concretely, inside vLLM
PagedAttention + continuous batching + prefix caching + speculative decoding
        ↓ once this works on one GPU, scale it
tensor parallel → pipeline parallel → data parallel → disaggregated P/D
        ↓ and once you're running any of this yourself
you've taken on 4 operational boundaries a hosted API used to hide from you
```

Nothing in this list is a standalone fact you memorized — each one is the previous one's consequence. That's genuinely worth internalizing over memorizing the individual bullet points, because it's also how interview follow-up questions actually work: "why does quantization help" chains back to bandwidth-boundedness, which chains back to the prefill/decode split, which chains back to what a GPU physically is.

---

## Alternatives — things that compete for the same job

| Job | Options | What actually decides between them |
|---|---|---|
| Which GPU | H100 / H200 / B200 | Capacity → bandwidth → compute → interconnect, in that order |
| Quantization format | GPTQ / AWQ / GGUF / bitsandbytes | Which *runtime* you're serving on, not accuracy (they're all within ~6% of each other) |
| Serving engine | vLLM / SGLang / TensorRT-LLM / Ollama | Shared-context traffic? → SGLang. Setup speed? → Ollama. Max NVIDIA throughput, stable model? → TensorRT-LLM. Otherwise → vLLM |
| Speculative decoding proposer | n-gram / EAGLE / Medusa | n-gram is free but weak; EAGLE/Medusa need training a small head but propose better |
| Multi-GPU parallelism | Tensor parallel / Pipeline parallel / Data parallel | TP first (needs NVLink, stays in-node), PP when model still doesn't fit, DP to scale *replicas* once one copy fits |
| Where the model runs | Hosted API / Managed deployment / Full self-host | Sovereignty > ops capacity > volume > model access, in veto order |
| Fine-tuning a big model | Full fine-tune / QLoRA | QLoRA (frozen 4-bit base + trained LoRA adapters) turns a multi-GPU job into a single-GPU one |

---

## Complements — things that only work *together*, not instead of each other

It's easy to mentally file everything above as "pick one," but several pairs are not alternatives — they're a matched set, and using one without the other is the most common self-inflicted mistake in the source material:

- **A quantization format needs a kernel that reads it fast.** AWQ weights alone are not a speedup — AWQ *plus* the Marlin kernel is. Picking the format without checking kernel support is exactly how someone benchmarks GGUF inside vLLM and wrongly concludes "GGUF is slow."
- **PagedAttention and prefix caching are the same mechanism, applied twice.** Blocks that back one request's KV cache are the same blocks that get hash-matched and reused across requests with a shared prefix. Understanding one is most of the way to understanding the other.
- **Speculative decoding doesn't replace normal decoding — it drafts for it.** The big model still verifies every token; the small model just proposes candidates. It's an addition, not a swap.
- **Chunked prefill and continuous batching solve different problems that show up together.** Continuous batching lets new requests join without waiting for a batch boundary; chunked prefill stops one giant prompt from starving everyone else *within* that same batching system. You generally want both on, not one or the other.
- **Benchmarking and SLOs are a pair, not sequential steps.** A raw tokens/sec number without a latency target attached (goodput) can describe a server that's fast on average and unusable at the tail. You size the engine's config *to* an SLO, not just to a throughput number.
- **Self-hosting's four boundaries are a package deal.** You don't get to pick "I'll handle the metrics gap but skip the version-pinning problem" — they're independent failure modes that all come bundled with the decision to self-host at all.

---

## What to learn next — real gaps, not busywork

The 14 sources are unusually good at *why* and *what*, and they hand you real vocabulary and real decision frameworks. Here's what they don't yet give you, in order of how much it'll matter for your project and your target role:

1. **You haven't run vLLM's own bench tools yet.** `vllm bench latency/throughput/serve` and the auto-tune script are named in source 14 but you've only run manual timing scripts (Module 6) and basic concurrent requests (Module 7). Closing this gap is nearly free — it's the actual tool for the project below.
2. **You haven't measured your own roofline crossover.** The B_sat concept (below it you're bandwidth-bound, above it compute-bound) is precise and checkable on your own GPU — nobody has told you *your* RTX 3050's number yet. This is a genuinely good thing to go measure yourself rather than take on faith.
3. **Guided/structured decoding (JSON-mode) hasn't come up in the course yet**, but it's directly relevant to "building a product" — most real API products need reliable structured output, and it's a vLLM feature (`guided_decoding`) you haven't touched.
4. **Prefix caching specifically hasn't been a hands-on lab yet**, even though it's one of the highest-leverage features for a chat-style product (repeated system prompts get nearly free). Worth a dedicated small experiment.
5. **The four self-hosting boundaries are entirely unoperationalized.** You've read about the metrics gap and the silent-drop gap; you haven't yet built a system that actually tests for either. This is the single highest-value gap to close, and it's exactly what turns a "serving demo" into something that reads as production-minded in an interview.
6. **Disaggregated P/D and multi-GPU training debugging are genuinely out of scope for one GPU** — correctly so. Know the vocabulary (you now do), don't manufacture a fake reason to build it on hardware that can't represent it honestly.

---

## The deployment pipeline, step by step — how the whole thing actually gets handled

This is the order a real deployment decision tree runs in, compressed from every "how do I actually ship this" thread across the 14 sources:

1. **Pick the model and check it against your hardware's memory capacity first.** Not bandwidth, not compute — capacity. If it doesn't fit, nothing else matters yet.
2. **Pick a quantization format matched to your serving runtime**, not the other way around. Know your engine before you download a checkpoint.
3. **Pick the serving engine matched to your traffic shape.** Shared context (chat/RAG/agents) vs. unique-per-request (batch classification) is the single biggest fork.
4. **Set the engine's memory and scheduling knobs deliberately**: `max_model_len` (your real worst-case KV cache budget, not an arbitrary big number), `gpu_memory_utilization`, `max_num_seqs` (your batching-vs-latency dial), prefix caching on, chunked prefill threshold if prompts run long.
5. **Wrap the engine in the four boundary protections before calling it done**, not after something breaks in production: contract-test the API layer by parsing bodies, pin the engine version and assert imports in CI, scrape the engine's *own* full metrics endpoint (not a wrapper's subset) and alert on KV-cache-% and prefix-hit-rate specifically, and make any external state you keep tolerant of the engine rewinding a request.
6. **Benchmark under realistic concurrency, against an SLO, not a single request.** TTFT, ITL, throughput, and goodput — goodput is the one that tells you if the config you picked in step 4 actually works under real load.
7. **Decide the hosting model** — even a personal project benefits from explicitly reasoning through sovereignty → ops capacity → volume → model access, because it's exactly the framing a hiring team will ask you to defend.
8. **Only then think about scaling out** — tensor parallel, then pipeline parallel, then data-parallel replicas behind a load balancer, then (at real scale) disaggregated prefill/decode. Every one of these is solving a problem you don't have yet on a single GPU, and reaching for them early is how projects get needlessly complicated.

Steps 1–6 are entirely buildable on your RTX 3050. Step 7 is worth reasoning through on paper even though the answer for a solo learning project is obviously "self-host, for the experience." Step 8 is where a single consumer GPU honestly runs out of road — which is exactly the boundary the project below is designed to sit right up against, without pretending past it.
