# Concepts Explained

A running glossary, added to the first time each concept actually appears in this project — not a pre-written reference. Short, plain-language, tied to *why it's here* in this specific build, not a generic textbook definition.

Format per entry:

```
## <Name> — introduced in Stage N

**What it is:** <one or two sentences>
**Why we're using it here:** <one or two sentences, specific to this project>
```

---

## `--quantization awq_marlin` — introduced in Stage 1

**What it is:** Tells vLLM which *kernel* to use to execute AWQ-quantized weights. AWQ is the weight format; Marlin is a hand-optimized CUDA kernel that reads that format. They are separate choices, and vLLM will happily pick a slow generic kernel for AWQ weights if you don't insist.
**Why we're using it here:** The same AWQ weights benchmark ~11x faster on Marlin than on vLLM's default AWQ path. Naming it explicitly turns a silent performance fallback into a loud startup error, so we find out at launch instead of in the Stage 2 benchmark. Requires compute capability ≥8.0 — this card is 8.6.

## `--dtype float16` — introduced in Stage 1

**What it is:** The floating-point type used for activations and for the parts of the model that stay unquantized (embeddings, layernorms).
**Why we're using it here:** Qwen's config defaults to bfloat16, but the AWQ Marlin kernel path is best-tested on fp16. Ampere supports both, and at this model size the numerical difference is negligible, so we take the well-trodden path.

## `--max-model-len` — introduced in Stage 1

**What it is:** The maximum total tokens (prompt + generated) in a single sequence. Caps context length below whatever the model natively supports.
**Why we're using it here:** Set to 4096 against a native 32,768. KV cache costs ~36 KiB/token for this model, so one 32k sequence would demand ~1.15 GiB — more KV cache than a 4GB card has left after weights. Capping at 4096 is what makes multiple concurrent requests possible at all.

## `--max-num-seqs` — introduced in Stage 1

**What it is:** The maximum number of sequences vLLM will run in a single batch — the concurrency ceiling of the engine's scheduler.
**Why we're using it here:** Set to 8. Deliberately overcommitted against available KV cache: if all 8 ran to full 4096 length simultaneously they'd need more cache than exists, and vLLM would preempt and evict. That's intentional — Stage 9 finds where it actually breaks and Stage 10 triggers the eviction on purpose (Boundary 4).

## `--max-num-batched-tokens` — introduced in Stage 1

**What it is:** How many tokens the scheduler may process in one engine step, across all sequences. With chunked prefill, a long prompt is split into chunks of this size rather than run in one shot.
**Why we're using it here:** Set to 2048 (below the usual default) because activation memory during the forward pass scales with this number, and every MiB it saves becomes KV cache. Smaller chunks mean slightly lower peak throughput but meaningfully more concurrency headroom — the right trade on a 4GB card.

## `--gpu-memory-utilization` — introduced in Stage 1

**What it is:** The fraction of *total* GPU memory vLLM is allowed to occupy. vLLM measures weights + activation peak + overhead, then claims everything remaining under this ceiling as KV cache — permanently, at startup.
**Why we're using it here:** 0.90 of 4096 MiB ≈ 3.7 GiB. The 10% headroom absorbs the CUDA context and display/compositor usage. Raising it toward 0.95 buys more KV cache and risks an OOM mid-run; this is the first dial to try in Stage 11.

## `--enable-prefix-caching` — introduced in Stage 1

**What it is:** Caches the KV blocks of prompt prefixes so that a second request sharing the same opening tokens skips recomputing them in prefill.
**Why we're using it here:** Every request through our gateway will eventually carry the same system prompt. On by default in vLLM V1, but stated explicitly because Stage 11 measures shared-prefix vs. no-shared-prefix workloads, and the prefix-cache hit rate is one of the two metrics (with KV-cache-%) that Boundary 3 says monitoring wrappers silently drop.

## `--ipc=host` — introduced in Stage 1

**What it is:** A Docker flag giving the container the host's IPC namespace, and with it, full-size shared memory.
**Why we're using it here:** Docker's default `/dev/shm` is 64 MB. PyTorch uses shared memory for inter-process tensor passing, and vLLM crashes with cryptic bus errors when it runs out. Standard for every PyTorch container.

## Named Docker volume (`-v hf-cache:/root/.cache/huggingface`) — introduced in Stage 1

**What it is:** A Docker-managed storage area, referenced by name rather than by host path, that outlives any single container.
**Why we're using it here:** The model is a ~2.2 GB download. Without this, `--rm` throws it away and every restart re-downloads it. A *named* volume rather than a bind mount specifically because Windows→WSL2 bind mounts bring path-translation and file-permission problems that a named volume simply doesn't have.

## `torch.compile` cache is config-keyed — introduced in Stage 1

**What it is:** vLLM compiles the model graph with `torch.compile` at startup (~60 s here) and caches the result under a directory named by a hash of the vLLM configuration — e.g. `torch_compile_cache/33c2863007/`. Change any config value and the hash changes, so the cache misses and it recompiles.
**Why we're using it here:** Observed directly: adding `--revision` and `--tokenizer-revision` changed the cache key from `6ee1e77343` to `33c2863007` and forced a full recompile. The consequence is for Stage 11 — every optimization variant changes a flag, so each one pays a one-off ~60 s compile on first launch. Launch each variant once to warm its cache *before* measuring it, or that minute contaminates the result.

## `vllm bench serve` vs. `latency` / `throughput` — introduced in Stage 2

**What it is:** Three benchmark subcommands built into the vLLM image. `latency` and `throughput` are *offline* — they instantiate their own engine in-process and measure batch work. `serve` is *online* — it sends HTTP requests to an already-running server and reports TTFT, ITL, TPOT and end-to-end latency percentiles.
**Why we're using it here:** Only `serve` produces TTFT/ITL, which `PRODUCT_SPEC.md` requires, because those are serving-path metrics that don't exist in a batch job. It's also the only option that fits: on a 4 GiB card an offline benchmark would need a second copy of the model on the GPU alongside our server's 3.12 GiB. Full reasoning in `DECISIONS.md`.

## TTFT / ITL / TPOT, as vLLM reports them — introduced in Stage 2

**What it is:** **TTFT** (time to first token) measures prefill plus queueing — how long before the user sees anything. **ITL** (inter-token latency) is the gap between consecutive streamed tokens, reported per-token so its distribution is visible. **TPOT** (time per output token) is the same idea averaged across a whole request: `(end_to_end − TTFT) / (output_tokens − 1)`.
**Why we're using it here:** ITL and TPOT measure the same physical thing but distribute differently — TPOT smooths over a request, ITL exposes individual stalls. A scheduler pause from a preemption shows up as an ITL p99 spike while barely moving mean TPOT, which is exactly the signature Stage 10 is trying to catch when it forces an eviction.

## FastAPI and ASGI — introduced in Stage 3

**What it is:** FastAPI is a Python web framework built on ASGI, the asynchronous server interface (the async successor to WSGI). Handlers are `async def` coroutines, so a handler that is waiting on network I/O yields the event loop to other requests instead of occupying a thread.
**Why we're using it here:** This gateway spends essentially all its time waiting for vLLM — the Stage 2 baseline measured 4 to 9 seconds of end-to-end latency per request. Under a synchronous framework each in-flight request would hold a worker thread for that entire duration, capping concurrency at the thread pool size. Async makes idle waiting nearly free, which is exactly the shape of this workload. FastAPI also generates OpenAPI docs at `/docs` for free.

## `httpx.AsyncClient` and why not `requests` — introduced in Stage 3

**What it is:** An async HTTP client with a connection pool, created once for the process lifetime in FastAPI's `lifespan` and reused across all requests.
**Why we're using it here:** `requests` is synchronous — a single call inside an async handler blocks the whole event loop and serialises every concurrent request through the gateway, throwing away the concurrency FastAPI provides. Reusing one long-lived client also keeps TCP connections to vLLM warm, removing a handshake from every request's TTFT.

## Streaming pass-through (`StreamingResponse` + `aiter_raw`) — introduced in Stage 3

**What it is:** Forwarding the upstream response body to the client incrementally, as bytes arrive, instead of collecting it fully and then sending it. `client.send(..., stream=True)` returns as soon as headers arrive; `aiter_raw()` yields undecoded bytes; a `BackgroundTask` closes the upstream response once the client finishes reading.
**Why we're using it here:** Buffering would convert time-to-first-token into end-to-end latency — for this project's measured baseline, roughly 654 ms becoming 4 seconds. `aiter_raw` rather than `aiter_text` keeps a server-sent-events stream byte-identical to what vLLM emitted. The `BackgroundTask` is not optional: without it, upstream connections are never released back to the pool and it is exhausted within minutes under load.

## Forwarding the request body as raw bytes — introduced in Stage 3

**What it is:** Reading the incoming request body with `await request.body()` and passing those bytes to the upstream unchanged, rather than parsing JSON into a model and re-serializing it.
**Why we're using it here:** This is the structural defence for **Boundary 1**. The named failure mode from the reading is a gateway that deserializes a request into a typed model and then re-serializes — silently dropping any field the model does not declare, `response_format` being the classic casualty. A field that is never modelled cannot be dropped. It converts "we remembered to forward that field" into "forgetting is not expressible". Stage 5's contract test proves the property still holds.

## Hop-by-hop headers — introduced in Stage 3

**What it is:** Headers meaningful only for a single connection rather than end-to-end — `Connection`, `Keep-Alive`, `Transfer-Encoding`, `Upgrade`, `TE`, `Trailers`, and the proxy-auth pair (RFC 9110). A proxy must consume them, not forward them.
**Why we're using it here:** We strip them in both directions. `Content-Length` is stripped too — not technically hop-by-hop, but forwarding the original value while re-framing the body produces a length mismatch and a corrupted response. `Host` is stripped so httpx sets it correctly for the upstream.

## CORS and the same-origin policy — introduced in Stage 6

**What it is:** Browsers refuse to let JavaScript on one origin read a response from a different origin unless the server explicitly opts in with CORS headers. An origin is scheme + host + port, and a page opened as `file://` has the origin `null`, which matches nothing.
**Why we're using it here:** It is why `ui/index.html` is served by the gateway at `/chat` rather than double-clicked. The failure is confusing the first time: the request *succeeds*, the gateway logs a 200, and the browser still hands the page an error — because the block is on reading the response, not on sending the request. Making the page same-origin sidesteps it entirely, and avoids having to decide which origins may call an authenticated API.

## Parsing server-sent events in the browser — introduced in Stage 6

**What it is:** Reading a streamed response with `res.body.getReader()`, decoding bytes with `TextDecoder`, buffering, and splitting on the `\n\n` event delimiter. Each event's payload lines begin with `data:`, and the stream ends with `data: [DONE]`.
**Why we're using it here:** The obvious mistake is assuming one network read equals one event. It does not — SSE events do not align with packet boundaries, so a single read can contain half an event, several events, or an event split across two reads. Buffering and splitting on the delimiter is what makes the parser correct. (The browser's built-in `EventSource` handles this, but only supports GET requests with no custom headers, so it cannot send our POST body or `Authorization` header.)

## Contract test (vs. smoke test) — introduced in Stage 5

**What it is:** A test asserting the **shape and content of what crosses a boundary**, rather than that a call succeeded. Here: what the gateway actually sent to vLLM, and what came back in the response body — not the status code.
**Why we're using it here:** Boundary 1's failure mode is silent. A dropped `response_format` still yields HTTP 200, a plausible response, and no error anywhere; the only symptom is that structured output quietly stops being structured. **A test asserting `status_code == 200` passes straight through the entire failure.** Asserting on the forwarded body is what makes the failure detectable at all.

## `httpx.MockTransport` and the stub upstream — introduced in Stage 5

**What it is:** A transport that intercepts requests inside httpx and returns responses from a Python function instead of sending them over the network. Swapped into the gateway's `app.state.client` after startup, so the shipping code runs unmodified while its upstream is a recorder.
**Why we're using it here:** It lets a test assert on the exact bytes vLLM *would have received*. That enables the only test that captures the real Boundary 1 property — **that a field nobody has heard of still survives.** Against real vLLM you can only observe the effects of fields vLLM understands; against a recording stub you can send `some_field_invented_in_2027` and prove it was not stripped. It also means the contract suite needs no GPU and runs in milliseconds, so it runs on every change rather than only when the full stack happens to be up.

## SQLite, WAL mode, and `asyncio.to_thread` — introduced in Stage 4

**What it is:** SQLite is a database that lives in a single file, with its driver (`sqlite3`) in the Python standard library — no server to install or run. WAL (write-ahead logging) is a journal mode where readers and writers do not block each other. `asyncio.to_thread` runs a blocking function on a worker thread so it does not stall the event loop.
**Why we're using it here:** SQLite means zero extra infrastructure to containerise in Stage 7, and the usage database can be opened by hand with any SQLite client while debugging. WAL matters because the dashboard is read while requests are being written — most visibly during Stage 9's load test, when watching the dashboard live is the whole point. `to_thread` matters because `sqlite3` is blocking: calling it directly inside an `async def` handler would stall every other in-flight request, reintroducing exactly the problem that choosing httpx over `requests` avoided.

## `stream_options: {"include_usage": true}` — introduced in Stage 4

**What it is:** An OpenAI-API request option that makes the server append a final SSE chunk carrying the `usage` object (prompt/completion/total tokens). Without it, a streamed response reports no token counts at all — intermediate chunks carry `"usage": null` and the stream simply ends with `data: [DONE]`.
**Why we're using it here:** It is the only way to bill a streaming request accurately. The gateway injects it into every streaming request, because clients will not send it and a budget that silently ignores streamed traffic is not a budget. This injection is what forced the Stage 4 decision to parse the request body — see `DECISIONS.md`.

## Jinja2 and server-rendered HTML — introduced in Stage 4

**What it is:** A templating engine. `{{ value }}` substitutes, `{% if %}` / `{% for %}` are control flow; FastAPI renders the template server-side and sends finished HTML. No frontend framework, no build step, no JavaScript required to display data.
**Why we're using it here:** It keeps v1 minimal, and it keeps a conceptual separation visible. Grafana (Stage 8) is the *infrastructure* dashboard answering "is the system healthy?"; this is the *product* dashboard answering "how is it being used?". Conflating those two questions is a real and common operational mistake, and building them as visibly different things — different tools, different technology — makes the distinction hard to lose.

## GPU performance states (P-states) and throttle reasons — introduced in Stage 2

**What it is:** `nvidia-smi` reports a performance state from **P0** (maximum) to **P8** (idle), plus a `clocks_throttle_reasons.active` bitmask in which the driver names why it is holding clocks down. The bits that matter here: `0x04` SwPowerCap, `0x08` HwSlowdown, `0x20` SwThermalSlowdown, `0x40` HwThermalSlowdown. They OR together, so `0x24` means SwPowerCap **and** SwThermalSlowdown are both active.

**Why we're using it here:** It invalidated an entire benchmark run. The card was sitting in **P8 at 100% utilization** — 210 MHz core, 405 MHz memory, roughly 1/15th of rated memory speed — which collapsed decode from ~18 ms/token to ~225 ms/token while barely touching prefill.

The general lesson for inference benchmarking: **decode is memory-bandwidth-bound and therefore extremely sensitive to memory clock, while prefill is compute-bound and much less so.** When TTFT looks fine and ITL is terrible, suspect the memory clock before you suspect the engine. And a `SwThermalSlowdown` firing in the low 70s C is a configured policy, not silicon protecting itself — real thermal limits on this class of GPU are in the high 80s.

## WDDM VRAM reservation (free vs. total memory) — introduced in Stage 1

**What it is:** On Windows, a consumer GPU runs under WDDM — the Windows Display Driver Model — which manages VRAM for the desktop compositor and holds back a portion that CUDA never reports as free. Combined with the cost of creating a CUDA context (several hundred MiB under WSL2), roughly 0.79 GiB of this 4 GiB card is unavailable before vLLM allocates anything.
**Why we're using it here:** It broke our second launch attempt. The key insight is that `--gpu-memory-utilization` is checked against **free** memory at startup, not **total** — so `nvidia-smi` reporting `0MiB / 4096MiB` does *not* mean 4 GiB is available to the engine. A Linux host with the card in TCC mode would not pay this tax, which is one concrete reason benchmark numbers from this machine won't transfer to a server unchanged — worth stating in the Stage 13 report.

## `NVIDIA_REQUIRE_CUDA` (the container CUDA floor) — introduced in Stage 1

**What it is:** An environment variable baked into CUDA-based images declaring the minimum CUDA version the host driver must provide (e.g. `cuda>=12.8`). The NVIDIA container runtime reads it and refuses to start the container in a *prestart hook* if the driver falls short — so the failure appears as a `runc create failed` error, not as a Python or CUDA error.
**Why we're using it here:** It blocked our first vLLM launch. The practical lesson: "my driver supports CUDA 12.x" is not the constraint — the image's declared floor is, and it is enforced before your code exists. It can be bypassed with `NVIDIA_DISABLE_REQUIRE=1` (CUDA minor-version compatibility often makes that work), but we chose to satisfy it properly instead — see `DECISIONS.md`.

## Model revision pinning — introduced in Stage 1

**What it is:** Passing an exact git commit hash of the Hugging Face model repo instead of trusting the repo name to keep meaning the same thing.
**Why we're using it here:** Boundary 2. `Qwen/Qwen2.5-3B-Instruct-AWQ` is a moving pointer — the owner can push re-quantized weights to it at any time. If that happens between our Stage 2 baseline and our Stage 11 optimization table, the comparison is silently invalid. We pin it at the end of Stage 1, once we can read the hash that actually landed on disk.

## `rope_scaling` and YaRN context extension — introduced in Finalization Phase 2.5

**What it is:** A field in a model's `config.json` describing whether its rotary position embeddings are stretched to cover a longer context than the model was trained on. YaRN is the most common such method. When it is `null`, `max_position_embeddings` is the model's real, hard ceiling; when it is set, the model claims a longer window achieved by interpolating positions, usually at some quality cost.
**Why it matters here:** Qwen2.5-1.5B-Instruct-AWQ reports `max_position_embeddings: 32768` with `rope_scaling: null`. That makes 32,768 the true limit for the context-window decision — there is no free headroom above it, and reaching for more would mean enabling YaRN deliberately and accepting a quality trade this project has not measured. Checking this field is what turns "the model supports 32k" from recollection into a verified fact.

## `docker builder prune` vs. `docker image prune` — introduced in Finalization Phase 0

**What it is:** Two different caches. The **build cache** holds intermediate BuildKit layers produced while building images; it is fully regenerable and invisible in `docker images`. **Dangling images** are tagged-then-retagged image layers orphaned by rebuilds, and they *do* appear in `docker images` as `<none>`.
**Why it matters here:** They are wildly different in size and the intuition runs backwards. On this machine the build cache was **21.37 GB** while dangling images were **0 B** — repeated `docker compose up --build` retags `inference-product-gateway:latest` in place rather than leaving orphans behind. `docker system df -v` is the only way to know which one is actually holding the disk, and it must be read *before* deleting rather than after.

## The Hugging Face Xet cache (`~/.cache/huggingface/xet`) — introduced in Finalization Phase 0

**What it is:** A second, separate cache used by newer `huggingface_hub` versions as a download backend. Xet stores content-defined *chunks* for deduplicated transfer, alongside the familiar `hub/` directory that holds the actual materialized model files. It is pure download-acceleration state — models load from `hub/`, not from `xet/`.
**Why it matters here:** On this machine `hub` held 4.1 GB of models while `xet` held **6.0 GB of chunks** — the download cache was larger than the thing it downloaded, and it accounted for most of a `hf-cache` volume that looked inexplicably twice the size of its contents. `docker system df -v` reports only the volume total, so the split is invisible until you `du` inside it. Worth checking on any machine where a model cache seems too large.
