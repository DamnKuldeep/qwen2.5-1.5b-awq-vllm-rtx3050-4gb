# Finalization Plan

Everything needed to close v1 out: reclaim resources, produce canonical numbers for both models on a cold card, find the 1.5B-AWQ's absolute throughput ceiling, re-verify every boundary and test against the finalized model, then speculative decoding and Kubernetes.

**Run the phases in order.** Several depend on the GPU being cold or on a specific engine config being live.

**Time estimate: ~3 hours**, most of it waiting on ramps. Phases 0–5 are the load-bearing ones; 6 and 7 are additive.

---

## Phase 0 — Clean slate and resource reclamation

**Goal:** free everything reclaimable, keep everything needed, and start from a cold GPU so the canonical numbers are not contaminated by the ~24% session drift measured in Stage 11.

### 0.1 Clear stale shell state

Leftover environment variables from the Stage 11 experiments will silently change the engine config. This is not hypothetical — it cost a wasted ramp already.

```powershell
cd D:\Projects\inference-product\deployment\docker
Remove-Item Env:\COMPOSE_FILE -ErrorAction SilentlyContinue
Remove-Item Env:\VLLM_MAX_NUM_SEQS -ErrorAction SilentlyContinue
Remove-Item Env:\VLLM_MAX_MODEL_LEN -ErrorAction SilentlyContinue
Remove-Item Env:\VLLM_MAX_NUM_BATCHED_TOKENS -ErrorAction SilentlyContinue
Get-ChildItem Env:VLLM_*, Env:COMPOSE_*
```

The last line should print nothing. **Better still: close the terminal and open a fresh one.**

### 0.2 Bring everything down

```powershell
docker compose --profile observability down
docker ps -a
```

`docker ps -a` should list no `inference-product` containers.

### 0.3 See what is actually on disk

```powershell
docker system df -v
```

Read the **VOLUME NAME / SIZE** section before deleting anything.

### 0.4 Reclaim safely

```powershell
docker builder prune -f
docker image prune -f
```

Build cache is regenerable and usually the largest reclaimable item. `image prune -f` (without `-a`) removes only dangling images — the layers orphaned by the several gateway rebuilds.

> **NEVER run `docker volume prune`.** With the stack down, every volume is "unused", so it would delete `hf-cache` — the models, ~7 GB, and a 7-minute re-download each.

### 0.5 Optional: drop the model we are not keeping

The unquantized `Qwen2.5-1.5B-Instruct` (~3 GB) served its purpose in Experiment 3 and will never be run again. List first, delete explicitly:

```powershell
docker run --rm -v hf-cache:/cache --entrypoint ls vllm/vllm-openai:v0.11.0 /cache/hub
```

Only if the output confirms the directory name:

```powershell
docker run --rm -v hf-cache:/cache --entrypoint rm vllm/vllm-openai:v0.11.0 -rf /cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct
```

**Keep** `models--Qwen--Qwen2.5-3B-Instruct-AWQ` and `models--Qwen--Qwen2.5-1.5B-Instruct-AWQ`.

### 0.6 Optional: clear the compile cache

`vllm-cache` now holds compiled graphs for every config tried during Stage 11 — a dozen or more, most never to be used again. Clearing it costs one ~40 s recompile per config we actually keep.

```powershell
docker volume rm inference-product_vllm-cache
```

Compose recreates it empty on next start. **Do this before Phase 1**, so the recompile lands during a startup rather than inside a measurement.

### 0.7 Let the GPU cool

**The single most important step in this phase.** Stage 11 measured ~24% drift from sustained load, fully recovered by cooling.

```powershell
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
```

**Wait until temperature is ≤ 50 °C.** From the ~66 °C idle observed after heavy use, expect 15–25 minutes. Do not start Phase 1 until it is there.

**Report back:** `docker system df -v` before and after, and the final `nvidia-smi` temperature line.

---

## Phase 1 — Canonical 3B-AWQ measurement

**Goal:** the reference numbers for the model this project shipped, taken on a cold card, for the final report.

```powershell
cd D:\Projects\inference-product\deployment\docker
docker compose --profile observability up -d
docker compose ps
```

Wait for both `vllm` and `gateway` to read `healthy` (expect ~90 s if the compile cache was cleared, ~50 s otherwise).

Verify the state before measuring anything:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod http://localhost:8080/ready
docker compose logs vllm | Select-String "Model loading took|KV cache size|Maximum concurrency|awq_marlin"
```

Then, from the project root, warm-up and measure back to back:

```powershell
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 4 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 1,2,3,4,6,8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

`--model` is no longer needed — the ramp auto-detects the served model.

**Predictions:** weights 1.9542 GiB, KV 28,480 tokens, 6.94x. Cold card should give slightly better than the drifted numbers: concurrency 1 around 45–55 tok/s, concurrency 2 passing the SLO with roughly +25–35% margin, concurrency 3 failing.

**Report:** the startup lines and the full ramp table.

---

## Phase 2 — Canonical 1.5B-AWQ measurement, shipping config

**Goal:** the same numbers for the model being finalized, at the *unmodified* shipping config, so Phase 3's tuning has an honest baseline.

```powershell
cd D:\Projects\inference-product\deployment\docker
$env:COMPOSE_FILE="docker-compose.yml;docker-compose.1_5b-awq.yml"
docker compose up -d --force-recreate vllm
docker compose logs vllm | Select-String "Model loading took|KV cache size|Maximum concurrency|awq_marlin"
```

`COMPOSE_FILE` is set once for the shell. **Every subsequent compose command in this terminal now carries both files** — which is the fix for the trap that reverted the engine mid-experiment.

```powershell
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 4 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 1,2,4,6,8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

**Predictions:** weights 1.1018 GiB, KV 69,616 tokens, 17.00x. Concurrency 5–6 should hold the SLO with ≥ +20% margin; throughput still climbing at 8 because `--max-num-seqs 8` caps it far below the memory limit.

**Report:** startup lines and the ramp table.

---

## Phase 2.5 — Context window: are we using what the model supports?

**We are not.** `--max-model-len 4096` against a model that supports (probably) 32,768 — about 12%. **And that value was derived for the 3B and inherited by the 1.5B without being re-derived.**

It was correct for the 3B: one 32,768-token sequence needs 32,768 × 36.1 KiB = **1.15 GiB**, against 0.98 GiB of KV cache. vLLM would refuse to start. 4,096 was the largest workable value.

It is not correct for the 1.5B: one 32,768-token sequence needs 32,768 × 28 KiB = **896 MiB**, against **1,859 MiB** available. It fits.

**Raising `--max-model-len` is nearly free.** It is a ceiling, not an allocation — PagedAttention allocates 16-token blocks on demand, and KV cache size is set by leftover memory regardless. The only GPU cost is the block table: `max_num_seqs × (max_model_len / 16) × 4 bytes` ≈ **512 KB** at 64 sequences and 32k context.

**The real costs are latency and cache pressure, not memory:**

| Prompt length | Prefill time ≈ TTFT floor (at ~600–1,000 tok/s) |
| ---: | :-- |
| 4,096 | 4–7 s |
| 8,192 | 8–14 s |
| 16,384 | 16–27 s |
| 32,768 | **33–55 s**, monopolising the GPU throughout |

And **one 32k conversation occupies 48% of the entire prefix cache** — two such users and the shared working set is gone.

### 2.5.1 Verify the model's actual limit and re-derive KV/token

```powershell
docker run --rm -v hf-cache:/cache --entrypoint python3 vllm/vllm-openai:v0.11.0 -c "import json,glob; p=glob.glob('/cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct-AWQ/snapshots/*/config.json')[0]; c=json.load(open(p)); print({k: c.get(k) for k in ['max_position_embeddings','rope_scaling','num_hidden_layers','num_key_value_heads','num_attention_heads','hidden_size','vocab_size']})"
```

**Expected:** `max_position_embeddings: 32768`, `num_hidden_layers: 28`, `num_key_value_heads: 2`, `hidden_size: 1536`, `num_attention_heads: 12`.

Those four architecture values let KV-per-token be re-derived independently: `2 × 28 layers × 2 kv_heads × (1536/12) head_dim × 2 bytes = 28,672 bytes = 28 KiB`. If it matches, the capacity model rests on verified numbers rather than recollection.

### 2.5.2 Measure at 16,384 and 32,768

```powershell
cd D:\Projects\inference-product\deployment\docker
$env:VLLM_MAX_MODEL_LEN=16384
docker compose up -d --force-recreate vllm
docker compose logs vllm | Select-String "KV cache size|Maximum concurrency|Using max model len"
```

Then the same at `32768`. **Confirm KV cache size is unchanged** (~69,616 tokens) — that is the claim that raising the ceiling costs no memory. Only the reported max-concurrency figure should move: 17.0x → 4.2x → 2.1x.

Then check that short requests are unaffected:

```powershell
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 4 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 1,4,8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

**Prediction: identical to Phase 2 within noise.** A longer ceiling should not slow short requests. If it does, something allocates per-`max_model_len` that this analysis missed — which would be worth knowing.

### 2.5.3 Measure a genuinely long request

```powershell
python load_testing\ramp.py --levels 1 --prompt-words 6000 --max-tokens 256 --ignore-eos --no-warmup --min-requests 8
```

~8,000 prompt tokens. **Prediction: TTFT 8–14 s**, confirming prefill scales linearly with prompt length.

**Recommendation to decide here:** ship **16,384** — 4x the window, 4.2x worst-case concurrency, tolerable worst-case TTFT — and manage typical conversation length with context policy (V2 Phase E) so the prefix-cache working set stays small. Take 32,768 instead if 2.5.2 shows no downside, since unused headroom costs nothing.

**Report:** the config values, KV cache size at each `max-model-len`, the short-request ramp, and the long-request TTFT.

---

## Phase 3 — Absolute maximum throughput for the 1.5B

**Goal:** find the highest sustainable throughput this hardware can produce with this model, and identify which resource finally stops it.

### The three levers, and why each is now available

| Lever | Shipping | Why it was constrained | Why it can move now |
| --- | --- | --- | --- |
| `--max-num-seqs` | 8 | Sized in Stage 1 against 4,096-token worst case with 28,480 tokens of cache | The 1.5B has **69,616 tokens** — 17x at full context, and ~95x at this workload's ~730 tokens/request |
| `--max-num-batched-tokens` | 2048 | Activation memory competed with KV cache when weights were 1.95 GiB | Weights are now **1.10 GiB**. This is the direct lever on prefill, and Stage 9 proved prefill/TTFT is the binding term. vLLM's own throughput guidance suggests values an order of magnitude higher |
| `--gpu-memory-utilization` | 0.78 | Only 3.21 of 4.0 GiB is free under WDDM | **Cannot move.** This is a Windows constraint, not a choice |

### The sweep

Four configs, each: restart → discarded warm-up → measured ramp. Run them consecutively; the card will be hot throughout, which is *correct* here because we want sustainable throughput, not a boost-clock peak.

```powershell
# Config A — shipping (Phase 2's numbers; no rerun needed)

# Config B — raise concurrency only
cd D:\Projects\inference-product\deployment\docker
$env:VLLM_MAX_NUM_SEQS=32
docker compose up -d --force-recreate vllm
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 8,16,24,32 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

```powershell
# Config C — raise concurrency AND prefill batch
cd D:\Projects\inference-product\deployment\docker
$env:VLLM_MAX_NUM_SEQS=32
$env:VLLM_MAX_NUM_BATCHED_TOKENS=8192
docker compose up -d --force-recreate vllm
docker compose logs vllm | Select-String "KV cache size|Maximum concurrency"
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 8,16,24,32 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

**Check the KV cache line on config C.** Raising batched tokens costs activation memory, which comes out of KV cache. If it drops below ~40,000 tokens the trade may not be worth it.

```powershell
# Config D — push concurrency to the memory wall
cd D:\Projects\inference-product\deployment\docker
$env:VLLM_MAX_NUM_SEQS=64
$env:VLLM_MAX_NUM_BATCHED_TOKENS=8192
docker compose up -d --force-recreate vllm
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 16 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16
python load_testing\ramp.py --levels 16,32,48,64 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40
```

**Watch Grafana throughout.** The three outcomes to distinguish:

| Signal | Meaning |
| --- | --- |
| **Preemptions go non-zero** | Memory wall found — back off `--max-num-seqs` one step |
| **Throughput plateaus, KV still low, queue time flat** | Compute saturated — the real hardware ceiling |
| **Throughput still climbing at 64** | Not yet found; extend the sweep |

**Memory arithmetic:** each request holds ~730 tokens (471 prompt + 256 output). 69,616 ÷ 730 ≈ **95 concurrent requests** before the cache is full. So config D at 64 should reach ~67% KV with no preemptions, and the wall is somewhere near 96.

**Prediction:** peak throughput lands between **400 and 550 tok/s**, with compute saturating before memory does. TTFT will be poor at these levels — this configuration maximises throughput, not latency, and the two are not the same product.

**Report:** all three ramp tables, peak KV cache %, whether preemptions ever appeared, and the config that produced the highest throughput.

### Then pick the finalized config

Two candidates, and they are different products:

- **Latency-optimised:** whatever config held the SLO with ≥ +20% margin at the highest concurrency (likely shipping, or config B at low concurrency).
- **Throughput-optimised:** the peak from the sweep, with the SLO explicitly abandoned.

**Write down which one is being finalized and why**, because every subsequent phase tests *that* config.

---

## Phase 4 — Full regression: no breaking points

**Goal:** prove the finalized 1.5B config passes everything the 3B ever passed. Run with the finalized config live.

### 4.1 Contract tests (no GPU needed)

```powershell
cd D:\Projects\inference-product
pytest -v
```

**Expect 22 passed, 1 deselected.** These use a stub upstream and are model-agnostic.

### 4.2 Integration test against the live engine

```powershell
pytest -m integration -v
```

**Expect 1 passed.** Now resolves the served model from `/v1/models` instead of hardcoding, so it works against either model.

### 4.3 Gateway behaviour

```powershell
.\gateway\test_gateway.ps1
```

Five checks: health, valid key, invalid key → 401, missing header → 401, streaming arrives incrementally. **Confirm it prints `Serving model: Qwen/Qwen2.5-1.5B-Instruct-AWQ`** at the top — that line is the fix for the hardcoded-model bug.

### 4.4 Budget enforcement

`dev-key-beta` is already exhausted from Stage 4, so reset it first:

```powershell
docker compose exec gateway python -m gateway.seed_db --reset
.\gateway\test_budget.ps1
```

**Expect:** streamed request carries a usage chunk, `response_format` returns valid JSON, and a 429 after N requests. **N will be larger than Stage 4's 7**, because the 1.5B is faster but bills the same tokens — the count depends on tokens, not speed, so it should be about the same. If it differs a lot, that is worth understanding.

### 4.5 Direct engine smoke test

```powershell
.\deployment\docker\smoke_test.ps1
```

### 4.6 Chat UI

Open **http://localhost:8080/chat**. Confirm the header shows **`Qwen2.5-1.5B-Instruct-AWQ`**, send several messages, and watch TTFT / speed / context in the footer.

**This is also the answer-quality check, which is still the one unresolved question in the whole project.** Ask three genuinely technical questions — the KV cache one from Stage 6 is a good control since you have seen the 3B's answer. Every performance conclusion about the 1.5B is conditional on this judgement, and it is yours to make.

### 4.7 Dashboards

- `http://localhost:8080/dashboard` — requests logged, token counts sane, budgets correct
- `http://localhost:3000` — all 13 panels showing data
- `http://localhost:9090/targets` — both targets UP

**Report:** pass/fail for each, and your quality impression with the model name confirmed.

---

## Phase 5 — Re-verify the four boundaries on the finalized model

### 5.1 Boundary 2 — pin the 1.5B's revision (REQUIRED)

**This is not optional if the 1.5B is being finalized.** It is currently unpinned, which is a recorded deviation acceptable for one-off measurements and not acceptable for anything shipped.

```powershell
docker run --rm -v hf-cache:/cache --entrypoint ls vllm/vllm-openai:v0.11.0 `
  /cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct-AWQ/snapshots
```

Send me the hash and I will write `--revision` / `--tokenizer-revision` into the compose override, exactly as was done for the 3B in Stage 1.

### 5.2 Boundary 4 — force an eviction on the 1.5B

The 1.5B has **69,616 tokens** of cache, so the Stage 10 recipe will not fill it: 8 × 3,976 = 31,808 is only 46%. Need more concurrent long requests:

```
24 concurrent × (471 + 3,500) = 95,304 tokens vs 69,616 available — over by 37%
```

```powershell
cd D:\Projects\inference-product\deployment\docker
$env:VLLM_MAX_NUM_SEQS=24
docker compose up -d --force-recreate vllm
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 24 --max-tokens 3500 --ignore-eos --no-warmup --min-requests 24
```

**Note the dashboard totals before and after**, then verify the delta is exact: 24 requests, 24 × 3,500 = **84,000** completion tokens, zero errors. That is the Boundary 4 assertion — the engine may evict, the gateway's accounting may not drift.

**Watch:** KV cache should reach ~100% and preemptions go non-zero.

### 5.3 Boundaries 1 and 3

Already covered by Phase 4.1/4.2 (contract tests) and 4.7 (dashboard). No extra work.

---

## Phase 6 — Speculative decoding (Experiment 4)

**Why the default workload cannot measure this.** N-gram speculation drafts tokens by finding the current suffix earlier in the context and proposing what followed. It helps only when output reproduces input — summarisation of a quoted passage, code editing, structured extraction. This project's random-word prompts share nothing with their answers, so speculation would draft garbage, fail verification, and *cost* time.

`ramp.py --echo-task` was added for this: it asks the model to repeat its input verbatim, which is the maximum-overlap case and the best possible showing for n-gram speculation.

**Control arm first — no speculation, echo task:**

```powershell
cd D:\Projects\inference-product\deployment\docker
Remove-Item Env:\VLLM_MAX_NUM_SEQS -ErrorAction SilentlyContinue
docker compose up -d --force-recreate vllm
cd D:\Projects\inference-product
python load_testing\ramp.py --levels 4 --max-tokens 256 --ignore-eos --no-warmup --min-requests 16 --echo-task
python load_testing\ramp.py --levels 1,4,8 --max-tokens 256 --ignore-eos --no-warmup --min-requests 40 --echo-task
```

**Then the speculative arm.** I will write the override file once you confirm the flag syntax this vLLM build accepts:

```powershell
docker run --rm --entrypoint vllm vllm/vllm-openai:v0.11.0 serve --help | Select-String -Pattern "speculative" -Context 0,6
```

vLLM 0.11 takes a JSON `--speculative-config`; the exact schema has changed across releases, so I would rather read it from your build than guess — that is how the `gpu_cache_usage_perc` metric-name error happened.

**Prediction:** on the echo task, meaningful ITL improvement (possibly 1.5–2x) because draft acceptance should be very high. On the normal random-prompt workload, **no improvement or a regression**. Both results are worth recording, and the second is the more useful one — it shows speculative decoding is a workload-dependent optimization, not a free win.

---

## Phase 7 — Kubernetes (Stage 12)

**What this demonstrates that Compose could not:** Stage 10(c) found that a failing liveness probe means "restart this container", and that conflating liveness with readiness would restart a healthy gateway whenever its upstream died. Compose has one healthcheck and cannot express the distinction. Kubernetes has both, so this is where the `/health` and `/ready` split earns itself.

**Before I write the manifests, I need to know:**

```powershell
minikube status
kubectl get nodes -o wide
kubectl describe node minikube | Select-String -Pattern "nvidia.com/gpu" -Context 1,1
```

Specifically: is the GPU-enabled Minikube cluster from the earlier course still running, and does the node still advertise `nvidia.com/gpu`? On a 4 GiB card only one vLLM pod can exist, so the self-healing demo is "delete it, time the recovery" rather than a rolling update.

**Then I will write** `deployment/k8s/`: vLLM Deployment + Service with a GPU resource request and a long `startupProbe`, gateway Deployment + Service with `livenessProbe: /health` and `readinessProbe: /ready`, a PVC for the SQLite database, and the model cache mounted from the host.

**The experiment:** run a load, `kubectl delete pod` the vLLM pod mid-flight, and measure recovery time — plus verify the gateway pod stays **Running but NotReady** rather than being restarted. That last part is the whole point, and it is the direct payoff of the Stage 10(c) fix.

---

## Phase 8 — Final report update

Once phases 1–7 are done I will update `docs/FINAL_REPORT.md` with:

- Canonical cold-card numbers for both models, replacing the drift-affected figures
- The maximum-throughput configuration and what limits it
- The finalized model and config, with the reasoning including answer quality
- The 1.5B's pinned revision (Boundary 2 closed for the shipped model)
- Boundary 4 re-verified on the finalized model
- Speculative decoding results, including the negative one
- The Kubernetes self-healing measurement
- The hardware-efficiency comparison below

---

## Appendix — How these numbers compare

### The card

| Spec | Value |
| --- | --- |
| Memory bandwidth | **192 GB/s** (128-bit GDDR6 @ 12 Gbps) |
| CUDA cores | 2,048 |
| VRAM | 4 GB |
| Power cap | 35 W (observed 27–48 W) |

### Our measured efficiency

Fitting `ITL = a × weight_GiB + b` across both models gave `a = 11.03 ms/GiB`, implying **~97 GB/s of effective memory bandwidth — 50.5% of the card's 192 GB/s rating.**

**And there is a striking coincidence worth investigating.** The card thermally throttles from ~1,500 MHz boost to a sustained **712 MHz — 47% of boost clock.** Our bandwidth efficiency is 50.5%. Those two numbers being nearly equal suggests the memory bus is not the limit at all: **at 712 MHz the SMs cannot issue memory requests fast enough to saturate 192 GB/s**, so the achievable bandwidth is being gated by the core clock.

If that is right, better cooling would improve decode roughly linearly with sustained clock, up to the point where the memory bus actually becomes the limit — and a desktop RTX 3050 (same bandwidth, far better cooling, 130 W) should be substantially faster at identical settings. **This is a testable prediction this project cannot test**, and it belongs in the report as such.

### Against published figures

Public benchmarks for this exact combination — vLLM, RTX 3050 4 GB, Qwen2.5 AWQ — do not appear to exist. The closest available reference points:

- **One reported RTX 3050 figure of 28.6 tok/s**, with the model and quantization unspecified. Our 3B-AWQ single-stream is 39–46 tok/s and the 1.5B-AWQ is 67.8 tok/s, but without knowing their model this is not a comparison.
- **General guidance that FP16 → Q4 yields roughly 3.5–4x**, which is consistent with the weight-bytes law derived here independently.
- **The widely-stated principle that LLM inference is memory-bandwidth-bound rather than compute-bound** — which this project measured rather than assumed, and then refined: *decode* is bandwidth-bound, *prefill* is compute-bound, and on this card the two scale with different quantities.
- **vLLM's own throughput guidance recommends `--max-num-batched-tokens` of 32,768+ and `--gpu-memory-utilization 0.9`.** We run 2,048 and 0.78 for reasons specific to a 4 GiB card under WDDM. Phase 3 tests how far toward that guidance the 1.5B's lighter weights allow us to move.

**The honest position:** these numbers are not comparable to published benchmarks, because almost all of those run larger models on datacenter GPUs without thermal limits. What this project can claim is narrower and more defensible — **a measured, reproducible characterisation of one specific stack on one specific machine, with the derivation of why it performs as it does and a documented account of every wrong turn taken to get there.**

---

## Sources

- [GeForce RTX 3050 Mobile specifications — gpu-monkey](https://www.gpu-monkey.com/en/gpu-nvidia_geforce_rtx_3050_mobile_laptop_gpu)
- [NVIDIA GeForce RTX 3050 Laptop GPU benchmarks and specs — Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3050-Laptop-GPU-Benchmarks-and-Specs.513790.0.html)
- [Local LLM performance: what tokens-per-second to expect — InventiveHQ](https://inventivehq.com/blog/local-llm-performance-what-to-expect)
- [vLLM throughput guide — PagedAttention and batching tips](https://blog.easecloud.io/ai-cloud/increase-throughput-with-vllm-serving/)
- [vLLM explained: PagedAttention and continuous batching — RunPod](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching)
- [Qwen2.5 speed benchmark — Qwen docs](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- [The definitive GPU ranking for LLMs — hardware-corner](https://www.hardware-corner.net/gpu-ranking-local-llm/)
