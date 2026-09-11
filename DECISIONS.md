# Decisions Log

Every architectural choice made in this project, with the reasoning, in the order they were made. Append new entries at the bottom as the project progresses — never rewrite history here, even if a later decision reverses an earlier one (note the reversal explicitly instead).

---

## Made before Stage 0 (initial architecture)

**Model: `Qwen/Qwen2.5-3B-Instruct-AWQ`**
Qwen ships official AWQ checkpoints same-day as release — a strong signal the format is safe to depend on, not a community best-effort conversion. At 4-bit, weights are roughly 2GB, which leaves real headroom on a 4GB card for KV cache and concurrent requests — small enough to run comfortably, large enough to give genuinely good response quality, not a toy-scale model. Fallback if more KV-cache headroom is needed during concurrency testing: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`.

**Quantization + kernel: AWQ, explicitly with the Marlin kernel (`--quantization awq_marlin`)**
Benchmarked elsewhere, the *same* AWQ weights ran at 68 tok/s on vLLM's default kernel and 741 tok/s on Marlin — an ~11x difference from the kernel alone, not the format. AWQ is also the highest-accuracy 4-bit format in head-to-head testing against GPTQ and GGUF. Explicitly requesting `awq_marlin` avoids silently falling back to a slow path.

**Serving engine: vLLM**
Broadest hardware support, PagedAttention as the core memory mechanism, deepest documentation, and it's what the rest of this project (and the course before it) is already built around.

**Gateway framework: FastAPI**
Async-native, which matters because the gateway's whole job is proxying to another async service (vLLM) without blocking. Automatic OpenAPI docs for free. Minimal boilerplate. Python, matching existing skill level.

**Usage/budget storage: SQLite**
Zero extra infrastructure to install or run. Durable across restarts, unlike an in-memory dictionary. Simple enough to open and inspect by hand during debugging. A real product might graduate to Postgres or Redis — not needed for v1, and adding it now would be solving a scale problem this project doesn't have.

**Product usage dashboard: server-rendered HTML on the gateway itself (Jinja2 templates), not a separate service**
Keeps v1 minimal — no new frontend framework to learn yet. Also keeps the conceptual separation clean: Grafana is the *infrastructure* dashboard, this is the *product* dashboard, and they should visibly be different things built different ways, not two tabs of the same tool.

**Observability: Prometheus + Grafana, scraping vLLM's `/metrics` endpoint directly**
This directly addresses the exact failure mode named in the reading — a monitoring bridge that quietly forwards only a subset of an engine's real metrics, dropping the two (KV-cache-%, prefix-hit-rate) that would have warned you first. Scraping the engine directly means nothing gets lost between the engine and the dashboard.

**Deployment: Docker Compose first, Kubernetes (GPU-enabled Minikube) as an optional later stage**
Compose is the fastest path to "the whole system running together" on one command. Kubernetes is added afterward specifically to reproduce the Module 4 self-healing lab for real, against a real multi-service product instead of a single toy Pod — not because Compose is insufficient for this project's actual scale.

**Load testing: a small custom asyncio Python script, not Locust or k6**
No new framework to install or learn. Full transparency into exactly what's being measured and how — every line of the load generator is readable, matching the manual-timing approach already used in the course.

---

## Made during Stage 1 (vLLM launch configuration)

All of the following are consequences of one measured fact confirmed in Stage 0: **4096 MiB of VRAM, entirely free, compute capability 8.6.**

**The memory budget these flags are derived from**
Weights ≈ 2.2 GiB — 3.09B params, but the 311M-param token embedding matrix (151,936 vocab × 2048 hidden) stays fp16 because quantizing embeddings degrades quality badly, so it's ~2.78B at ~4.5 bits (~1.6 GiB) plus ~0.6 GiB of fp16 embeddings. CUDA context and non-Torch overhead ≈ 0.3 GiB. At `--gpu-memory-utilization 0.90` that leaves roughly 0.6–0.9 GiB for KV cache. At 36 KiB/token for this architecture (2 × 36 layers × 2 KV heads × 128 head_dim × 2 bytes), that's **~20,000 tokens of total KV cache**. Every number below falls out of that figure.

**`--max-model-len 4096`, against a native 32,768**
One full-length 32k sequence would need ~1.15 GiB of KV cache — more than exists after weights. vLLM would refuse to start. 4096 is chosen as the largest cap that still leaves room for several concurrent sequences: long enough for realistic prompts and for the shared-system-prompt prefix-cache test in Stage 11, short enough to not consume the entire cache with one request.

**`--max-num-seqs 8`, deliberately overcommitted**
8 × 4096 = 32k tokens of worst-case demand against ~20k of actual cache. This is not a mistake. Real requests rarely reach max length, so the overcommit buys throughput; and when they collectively do exceed it, vLLM preempts and evicts rather than OOMs. That eviction path is precisely Boundary 4, which Stage 10 triggers on purpose to prove the gateway's own request tracking survives it. Sizing conservatively here would have made that demonstration impossible.

**`--max-num-batched-tokens 2048`, below the usual default**
Activation memory during the forward pass scales with tokens-per-step. Every MiB not spent on activations becomes KV cache. Accepts slightly lower peak single-request throughput in exchange for concurrency headroom — the correct trade when KV cache, not compute, is the binding constraint on a 4GB card.

**`--gpu-memory-utilization 0.90`, revised during Stage 1 to `0.78`**
Originally set to 0.90 on the assumption that 10% headroom would cover the CUDA context and the Windows desktop compositor. **This was wrong and the first launch failed on it.** vLLM compares this fraction against *free* memory at startup, not *total*, and only 3.21 of 4.0 GiB is free once a CUDA context exists — Windows keeps the card in WDDM mode and reserves VRAM for the compositor, on top of the several hundred MiB the WSL2 CUDA context itself costs. That caps the dial at ~0.80; 0.78 leaves slack for the free figure drifting as the desktop redraws.

Two consequences worth recording. First, `nvidia-smi` showing `0MiB / 4096MiB` is not evidence that 4 GiB is usable — the reservation only materialises once a CUDA process exists, so the Stage 0 reading could not have predicted this. Second, **this ceiling is a property of Windows, not of the card**: the same GPU on a Linux host would not pay the WDDM tax and could run this dial higher. That belongs in the Stage 13 report as a stated reason these benchmark numbers do not transfer to a server unchanged.

Still the first dial to revisit in Stage 11, now pulled out to a `$GPU_MEM_UTIL` variable at the top of both launch scripts to make that revisit cheap.

**`--dtype float16` rather than the model config's bfloat16**
The AWQ Marlin path is best-tested on fp16. Ampere supports both and the accuracy difference is negligible at this scale, so we take the well-trodden path rather than debug a kernel edge case.

**Pinned image tag `vllm/vllm-openai:v0.11.0`, not `:latest` (Boundary 2)**
A floating tag means the engine silently changes underneath the benchmark numbers recorded in Stage 2, invalidating the Stage 11 before/after table without any visible signal. The pin is the entire point — it is what makes a measured "improvement" mean something. Model revision pinning is the same argument applied to the weights, and gets filled in at the end of Stage 1 once the downloaded commit hash is known.

**Host NVIDIA driver is part of the pinned stack, and has a floor of CUDA ≥ 12.8**
Discovered by failure, not by planning: `vllm/vllm-openai:v0.11.0` declares `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, and the NVIDIA container runtime enforces it in a prestart hook — the container never starts on a driver providing only CUDA 12.7. Three options were on the table: update the driver, set `NVIDIA_DISABLE_REQUIRE=1` to skip the check (CUDA minor-version compatibility means it would likely have worked), or pin an older CUDA 12.4-era vLLM image.

Chose the driver update. The override would have carried an unsupported configuration through all twelve remaining stages, and any strange kernel behaviour during Stage 9 load testing or Stage 11 optimization would then have had two candidate explanations instead of one — the cost lands precisely where this project is trying to produce trustworthy measurements. Pinning an older image would have meant an older engine and, worse, a pin chosen to dodge a driver rather than for reproducibility, which quietly undermines the Boundary 2 story we are trying to tell honestly.

Consequence for reproducibility: **the driver version belongs in the Stage 13 final report alongside the image tag and model revision.** An environment that can't start the pinned image is as much a reproducibility break as an unpinned dependency.

**Two launch scripts (`run_vllm.ps1` and `run_vllm.sh`) rather than one**
The `.ps1` is what actually runs on this Windows machine; the `.sh` exists because Stage 7 (Compose) and Stage 12 (Kubernetes) target Linux and translate this exact flag set into manifests. Duplication accepted deliberately, with a sync note in both files — the alternative (a bash-only script invoked through WSL) adds a path-translation layer to debug in the stage where the goal is to get the engine up cleanly.

---

## Made during Stage 2 (benchmark methodology)

**Benchmark with `vllm bench serve` as the primary baseline, not `vllm bench latency` / `throughput`**
`PROJECT_PLAN.md` Stage 2 names the latency and throughput subcommands. Both are *offline* benchmarks: they instantiate their own engine in-process rather than talking to a running server. Two reasons that is the wrong primary measurement here.

First, `PRODUCT_SPEC.md` requires baseline **TTFT and ITL**, and those metrics only exist on a serving path — an offline batch job has no notion of time-to-first-token for a streamed request. Only `vllm bench serve` reports them. Second, on a 4 GiB card an offline benchmark cannot coexist with the running server: our engine holds ~3.12 GiB, so a second engine instance simply will not fit and the benchmark would have to stop the thing it is meant to characterise.

The deciding argument is that `bench serve` measures the code path this product actually ships — HTTP in, streamed tokens out, through the scheduler under real request arrival. That is the number Stage 11's optimizations must be compared against for the comparison to mean anything. The offline pair is still run afterwards, with the server stopped, as a supplementary "what does this hardware do with no serving overhead at all" upper bound — useful context, not the baseline.

**Benchmark workload: `--dataset-name random` with fixed lengths and a fixed seed, not ShareGPT**
The bench tools default to a ShareGPT-derived dataset. Two problems for our purpose: it requires a download, and its prompts vary widely in length, so run-to-run variance mixes with whatever effect we are trying to measure. A synthetic random dataset with pinned input/output lengths and a pinned `--seed` makes every run comparable to every other run by construction. Given that this project's entire benchmark story is before/after tables, controlling the workload matters more than realism at this stage. Realistic variable-length traffic belongs in Stage 9's load test, where the goal is finding a ceiling rather than isolating a change.

**Benchmarks run with `--ignore-eos` and `--temperature 0`; the server keeps Qwen's sampling defaults**
Discovered by measurement, not planning. The first benchmark attempt asked for 32 × 128 = 4096 output tokens and got **2,683** on one run and **2,823** on the next, with an identical `--seed`. Two compounding causes: without `--ignore-eos` the model stops at EOS, so `--random-output-len` is a ceiling rather than a quantity, and the random dataset's gibberish prompts get short answers; and without an explicit temperature the server applies Qwen's HF generation config (`temperature 0.7, top_p 0.8, top_k 20`), making sampling stochastic so EOS lands somewhere different every run.

A baseline whose workload silently changes between runs cannot anchor a before/after table — a future run would differ from today's by an unknown mixture of the optimization and the dice. `--ignore-eos` forces exactly the requested token count, making the work identical by construction; `--temperature 0` removes the remaining randomness.

Deliberately **not** fixed at the server level (e.g. by launching with `--generation-config vllm`). Qwen's recommended sampling defaults are the right behaviour for the chat UI in Stage 6 and for the product generally — a user of this API should get good output without specifying sampling parameters. The benchmark is the special case, so the benchmark overrides per-request. This also keeps the thing being measured identical to the thing being shipped in every respect except the one we deliberately controlled.

**Concurrent benchmark runs use more prompts than the single-stream run**
32 prompts at concurrency 8 is four waves of batching, so ramp-up and drain dominate the reported average — visible in the first attempt as `Output token throughput: 50.19` against `Peak output token throughput: 72.00`. Concurrent runs use 128+ prompts so that steady-state batching, not the edges, determines the number.

**Baseline measured at `--max-concurrency 1` first**
A single in-flight request gives TTFT and ITL with no queuing delay mixed in — the cleanest possible statement of what this hardware does per token. Concurrency is then added as a separate, second measurement rather than conflated into the baseline, because at concurrency > 1 a rising TTFT can mean either the GPU is saturated or requests are waiting behind others, and the baseline should not have to distinguish those. Stage 9 does that work deliberately.

---

## Made during Stage 3 (gateway skeleton)

**The request body is forwarded as raw bytes and never parsed into a model**
This is the structural defence for Boundary 1, and it is a design choice rather than a coding convenience. The failure mode named in the reading is a gateway that deserializes a request into a typed model and re-serializes it, silently dropping any field the model does not declare — `response_format` being the classic casualty. The usual mitigation is a test plus discipline: remember to add each new field. That fails eventually, because it depends on someone remembering.

Forwarding opaque bytes removes the failure mode instead of defending against it. A field that is never modelled cannot be dropped, and the property holds for fields that do not exist yet. Stage 5's contract test then verifies the property rather than enumerating fields.

The cost is real and deferred deliberately: Stage 4 needs token counts for budget enforcement, which means reading a response body this stage merely pipes through. That tension gets solved in Stage 4 on its own terms rather than by compromising the proxy now.

**Responses are streamed, never buffered**
Buffering the upstream response before returning it would collapse time-to-first-token into end-to-end latency — against the Stage 2 baseline, roughly 654 ms becoming 4 seconds, with the streaming UX destroyed for the Stage 6 chat page. Implemented with `client.send(stream=True)` so headers are available before the body completes, `aiter_raw()` so a server-sent-events stream stays byte-identical, and a `BackgroundTask` to release the upstream connection once the client finishes reading. That last part is not optional: without it connections never return to the pool and it is exhausted within minutes under load.

**API keys in an environment variable for Stage 3, moving to SQLite in Stage 4**
There is nothing to store yet but the key itself. An environment variable keeps this stage to a single new concept (auth) rather than two (auth plus a database), which is the point of the stage boundary. Budgets arrive in Stage 4 and bring per-key state that genuinely needs a database, at which point the keys move with it.

**Errors are rendered in OpenAI's error envelope**
`PRODUCT_SPEC.md` promises that an existing OpenAI client library works by changing only the base URL. That promise covers failures as well as successes — a client that parses `error.message` on a 401 must keep working. A custom exception handler rewrites FastAPI's default `{"detail": ...}` into `{"error": {"message", "type", "code"}}`.

**`/health` is unauthenticated**
Stage 7's Docker Compose healthcheck and Stage 12's Kubernetes liveness probes both call it, and neither should need credentials to determine whether a container is alive. It reports this gateway's own status separately from vLLM reachability, so Stage 10's "kill vLLM mid-load-test" experiment can distinguish "the gateway died" from "the gateway is fine and its upstream is gone."

---

## Made during Stage 4 (usage, budgets, product dashboard)

**The raw-bytes guarantee is deliberately weakened to an untyped-dict round trip — and this is the most consequential decision in the gateway**
Stage 3 forwarded the request body as opaque bytes, which made Boundary 1's failure mode structurally impossible. Stage 4 breaks that, knowingly.

Budgets need token counts. Those live in the response `usage` object, and for *streaming* requests vLLM emits usage only when the request carries `stream_options: {"include_usage": true}`. Most clients will not send it, so the gateway must add it — which requires parsing the body it promised not to parse. Three options were considered: (a) do not enforce budgets on streaming requests, which guts the feature since the Stage 6 chat UI streams; (b) estimate by counting SSE chunks, which is approximate, wrong whenever a chunk carries multiple tokens, and drifts silently; (c) parse and modify.

Chose (c), with a precise mitigation. **Boundary 1's failure mode is specifically deserialization into a *typed* model** — a Pydantic class or dataclass — where any field the schema does not declare vanishes on re-serialization. We parse into an **untyped `dict`**, which has no schema, so unknown keys including `response_format` survive the round trip intact. Bodies that are not valid JSON are forwarded untouched rather than rejected, so we never invent an error vLLM would not have produced.

This is genuinely weaker than opaque bytes: the guarantee now rests on a property of `json.loads`/`json.dumps` rather than on never touching the field at all. **The consequence is that Stage 5's contract test changes character — it stops confirming something structurally unbreakable and becomes load-bearing, guarding a behaviour that can now regress.** That is precisely why Stage 5 exists and why Stage 10 deliberately breaks it: to prove the test actually bites.

**Budgets are checked before the request and deducted after it, accepting bounded overshoot**
A request's token count cannot be known before it is generated, so a pre-check against a running total is the only option that does not require a reservation. This permits overshoot: an in-flight request can push a key past its limit, and with concurrent requests several can.

The stricter alternative is reserving `max_tokens` upfront and refunding the unused portion. It never overshoots, but it rejects requests that would have fit, and `max_tokens` is frequently far above what a response actually uses. For a v1 with a stated per-key budget, bounded overshoot is the better trade and matches what most commercial APIs do. The dashboard displays consumption above 100% honestly rather than clamping it.

**Budget refusals (429) are recorded; invalid-key rejections (401) are not**
A dashboard showing only served requests hides the signal an operator opens it for — "this key keeps hitting its limit" is the thing worth seeing. A 429 has a valid key to attribute the record to. A 401 does not: the key is unknown by definition, so there is no row to attach it to, and recording arbitrary strings supplied by an unauthenticated caller would let anyone write unbounded data into our database.

**SQLite calls run on a worker thread via `asyncio.to_thread`**
`sqlite3` is blocking. Calling it directly inside an `async def` handler stalls the entire event loop for every other in-flight request — the same mistake the Stage 3 httpx-over-requests decision avoided, reintroduced through the back door. The calls are sub-millisecond either way, so this is about not establishing a habit that fails under Stage 9's load rather than about current performance. Connections are opened per call rather than pooled, because SQLite connections cannot safely move between threads and `asyncio.to_thread` gives no guarantee about which thread runs the call.

**WAL journal mode**
Lets the dashboard read while requests are being written. Under the default rollback journal, a dashboard refresh would contend with live traffic — precisely during Stage 9's load test, when watching the dashboard is the point.

**The token deduction is `tokens_used = tokens_used + ?` in SQL, not read-modify-write in Python**
Under Stage 9's load test many concurrent requests will bill the same key. A Python-side read-modify-write lets two requests read the same starting value and lose one another's increment, undercounting silently. The insert and the deduction also share one transaction: separately, a crash between them would either bill tokens never logged or log a request never billed, and the dashboard would then contradict the budget it is enforcing.

**The product dashboard is unauthenticated in v1**
`PRODUCT_SPEC.md` explicitly scopes out a real auth provider, and the service binds to localhost. Recorded as a known gap rather than an oversight: it would need protection before facing a network.

---

## Made during Stage 5 (contract test)

**Two test layers: contract tests against a stub upstream, integration tests against the real stack**
The contract layer swaps `httpx.MockTransport` into the gateway's client after startup, so the shipping code runs unmodified against a recorder. It asserts on the exact body vLLM *would have received*.

This split exists because the two layers can prove different things, and the more important property is only reachable from the stub. Against real vLLM you can only observe the effects of fields vLLM understands — so you can verify `response_format` works, but not that an *arbitrary unknown* field survives. Against a recording stub you can send `some_field_invented_in_2027` and assert it was not stripped. **That is the actual Boundary 1 guarantee: not "we remembered to forward response_format" but "this gateway has no allowlist at all."** A typed-model gateway passes every field-specific test while failing that one, which is precisely how the failure survives into production.

The integration test is still needed: forwarding a field to a server that ignores it is not a working feature. Both layers, each proving what the other cannot.

**Integration tests are excluded from the default `pytest` run**
`addopts = -m "not integration"`. The contract suite needs no GPU and finishes in milliseconds, so it can run on every change. **A test that only runs when the whole stack happens to be up is a test that stops being run** — and this suite's entire value is catching a regression that produces no other symptom. Integration tests run explicitly with `pytest -m integration`.

**Test dependencies live in a separate `requirements-dev.txt`**
So the Stage 7 container image does not ship a test framework it will never execute.

**A fresh temporary SQLite database per test (Stage 5)**
Budgets are stateful, and Stage 4 proved how quickly a 2,000-token budget is consumed. Sharing a database between tests would let one test's spending fail an unrelated later test, producing order-dependent failures — the kind that get "fixed" by re-running.

---

## Made during Stage 6 (chat UI)

**The chat page is served BY the gateway at `/chat`, not opened as a file**
`PROJECT_PLAN.md` says create `ui/index.html`, and it does exist there — but opening it directly would not work. A page loaded as `file://` has origin `null`; when its JavaScript calls `http://localhost:8080` the browser applies the same-origin policy and refuses to hand the response to the page, even though the request reaches the gateway and succeeds.

Two ways out: add CORS middleware, or make the page same-origin. **Chose same-origin.** Adding CORS to an *authenticated* API means deciding which origins may call it with a bearer token, and the tempting `allow_origins=["*"]` on a keyed API is a real security mistake rather than a configuration detail. Serving the page from the gateway means that decision never has to be made, and Stage 7's container ships the UI for free.

**The UI displays TTFT, tokens/sec and conversation token count**
Not decoration. These are the same metrics measured in the Stage 2 benchmark, and putting them on the page turns a number in a table into something felt. Specifically, the "Context sent" counter climbs every turn, which makes the Stage 3 finding — **ITL grows with context length, because every decode step's attention reads the whole KV cache** — directly observable as the assistant typing more slowly in a long conversation.

**The chat UI does not set sampling parameters**
So the server applies Qwen's recommended defaults (`temperature 0.7, top_p 0.8, top_k 20`). This is the deliberate other half of the Stage 2 decision: the benchmark overrides sampling to 0 for determinism, the product keeps the defaults because they produce better conversation. Same server, different callers, each getting what it needs.

**The API key is held in `localStorage` (Stage 6)**
Accepted for v1 with the caveat stated: any script on this origin can read it. `PRODUCT_SPEC.md` scopes out a real auth provider and this binds to localhost. A production system would use a session cookie and keep the provider key server-side. Recorded as a known gap, not an oversight.

---

## Made during Stage 7 (containerize and compose)

**`hf-cache` is declared `external: true`**
Compose prefixes volume names with the project name, so a plain `hf-cache:` entry would create `inference-product_hf-cache` — a different, empty volume — and vLLM would re-download the 2.7 GB model on first `compose up`. `external: true` points at the existing top-level volume. A cheap mistake to make and an expensive one to sit through.

**A separate `vllm-cache` volume for `/root/.cache/vllm`**
Correcting a Stage 1 error: the `torch.compile` cache is not inside the HuggingFace cache directory, so it was never persisted and every launch paid the full ~60 s compile. Mounting it properly matters for Stage 10 (kill and restart vLLM mid-load-test) and Stage 11 (one launch per optimization variant), where a minute of compile per start would contaminate the measurement.

**`depends_on` with `condition: service_healthy`, not bare `depends_on`**
Bare `depends_on` waits only for the container to *start*. vLLM then spends 2–3 minutes loading weights and compiling, during which the gateway would come up, fail health checks against a not-yet-listening engine, and report the stack broken while nothing is wrong. Startup ordering here is a health question, not a start question.

**`VLLM_BASE_URL: http://vllm:8000`, not localhost**
Inside the gateway container, `localhost` is the gateway itself. On a Compose network, service names are DNS names. This is the payoff for making it an environment variable back in Stage 3 rather than a constant — the code needed no change to move into Compose.

**Healthchecks use `python`, not `curl`**
The slim gateway image has no `curl`, and adding one would mean shipping a network tool in a production image purely for a check the runtime can perform itself. For vLLM, whether `curl` exists in a given image tag is not something a healthcheck should depend on.

**The gateway image runs as a non-root user and does not install `requirements-dev.txt`**
A container process that does not need root should not have it. And the image should not carry a test framework it will never execute.

**Prometheus and Grafana are defined now but behind a Compose profile**
`PROJECT_PLAN.md` asks for placeholders in Stage 7. Defining them behind `profiles: [observability]` means `docker compose up` will not try to start services whose config files do not exist yet, while the wiring — ports, volumes, dependencies — is already written and reviewable. Stage 8 adds the config and starts them with `--profile observability`.

**Database seeding runs in the container's `CMD` (Stage 7)**
`PRODUCT_SPEC.md` requires the whole stack to come up with one command, and a gateway with no API keys is not up in any useful sense. `seed_db` upserts without resetting `tokens_used`, so a key exhausted before a restart stays exhausted — the correct behaviour. Written as `sh -c` rather than a shell script file, because a `.sh` authored on Windows carries CRLF line endings and fails in Linux with an opaque "no such file or directory" pointing at the interpreter.

---

## Made during Stage 8 (observability)

**GPU hardware metrics are deliberately NOT in Grafana**
`PROJECT_PLAN.md` lists "GPU stats" among the panels. vLLM's `/metrics` exports engine state only — no temperature, clocks, power or utilisation. Getting those needs a separate exporter (dcgm-exporter), which expects Linux GPU driver access patterns that WSL2 under WDDM does not provide reliably.

Rather than ship a panel that might silently show nothing, the split is stated: **Grafana covers engine state, `benchmarks/watch_gpu.ps1` covers hardware state.** Given that Stage 2's single most consequential finding — an 11.6x throughput loss — was invisible in every engine metric and only visible in `nvidia-smi` clock sampling, pretending a dashboard covers hardware would be worse than admitting it does not. Revisit if the project ever moves to a Linux host, where dcgm-exporter is straightforward.

**Scrape interval 5s, not the usual 15s**
vLLM logs its own summary every 10 s, and Stage 9 ramps concurrency hunting for the exact point an SLO breaks. A 15 s scrape would smooth over the transition being looked for. The cost is more stored samples, which is irrelevant at this scale.

**Datasource and dashboard are provisioned from files, not configured in the UI**
A dashboard that requires someone to remember six configuration clicks is a dashboard that stops existing the moment the Grafana volume is deleted. Provisioning also means the dashboard is reviewable in the repo alongside the code it measures. The datasource is marked `editable: false` because a UI edit would be silently reverted on the next restart, which is more confusing than disallowing it.

**The dashboard leads with KV cache usage and prefix cache hit rate**
Boundary 3's failure mode is a monitoring layer that drops precisely these two while forwarding everything else. Both have already earned their place: KV cache at 18.5% under load (Stage 2) is why Stage 9 hunts for a thermal ceiling rather than a memory one, and prefix hit rate reaching 88% (Stage 6) is why chat TTFT stays flat as context grows. Putting them in the first row rather than buried below throughput is the design expressing the point.

**A preemptions panel, added before it is needed (Stage 8)**
`vllm:num_preemptions_total` is Boundary 4's signal — non-zero means the engine ran out of KV cache and evicted running sequences. Stage 10 forces this deliberately, and having the panel already in place means the evidence is captured when it happens rather than reconstructed afterwards.

---

## Made during Stage 10 (deliberate breakage)

**`/health` is liveness, `/ready` is readiness — split after breaking it**
Originally `/health` returned 503 whenever vLLM was unreachable, and the container healthcheck treats non-200 as failure. Killing vLLM in Stage 10(c) therefore marked the *gateway* unhealthy while the gateway was entirely fine and returning honest 502s.

That matters because of what a failing liveness probe *means*: "restart this container". Restarting a healthy gateway helps nothing and drops every in-flight request that was about to succeed once the engine returned. **Liveness answers "is this process alive?", not "is everything it depends on alive?"**

`/health` now always returns 200 while the process runs, with upstream status as information. `/ready` returns 503 when the gateway cannot serve — the probe a load balancer should route on, and Stage 12's `readinessProbe`. Failing readiness removes a pod from service without killing it, so it rejoins automatically, which is exactly the recovery behaviour observed.

---

## Made during Stage 11 (measured optimization)

**The measurement protocol is assembled from this project's own failures, not from best practice**
Every rule in `benchmarks/optimization_results.md` exists because ignoring it earlier produced numbers that looked like results and were not: restart between pairs (Stage 8's 16% cache-warming swing), discarded warm-up (Stage 2's 900% thermal TTFT shift), `--ignore-eos` (Stages 2 and 9's varying output volume), n ≥ 40 (Stage 10c's p95-is-the-maximum), quote the input/output ratio (Stage 10a's 2.3x workload-shape difference), verify P0 (Stage 2's silent 11.6x loss).

**Experiment 1 added to the plan: `--max-num-seqs`**
`PROJECT_PLAN.md` did not list it, but Stage 9 produced direct evidence that it was the binding constraint, and running the planned experiments while ignoring our own findings would have been strange. Result: raising 8 → 24 gave **+34.4% peak throughput** and **−83.6% TTFT p50 at concurrency 24**, at the cost of **24x worse ITL p99**.

**The product keeps `--max-num-seqs 8` despite that gain.** Under the stated SLO (p95 TTFT < 1.5 s) the change is worth exactly nothing: the SLO already fails at concurrency 8, and the cap only binds above it. Concurrency 4 remains the highest passing level at ~159 tok/s either way. **An optimization is only meaningful relative to a stated objective** — "+34% throughput" would have been a true and thoroughly misleading headline. Stating the SLO before measuring is what made the difference visible.

**Session drift is ~24%, and only consecutive pairs are comparable**
Experiment 2's control arm measured 194.2 tok/s where Experiment 1's identical baseline measured 256.1 — nothing changed but roughly 35 minutes of sustained load. This is larger than most of the effects being measured, including Experiment 1's own +34%. **Every comparison in this project must be between arms run back to back**, and every cross-session number in the Stage 13 report must be qualified.

**A `prompt` column was added to the ramp output before Experiment 2, not after**
The shared-prefix arm builds prompts from a fixed preamble while the unique arm generates to a word count, so comparable length could not be assumed. It turned out the shared prompts were **15% longer** — the winning arm did more nominal work, making the result conservative. Measuring the thing you are assuming is cheap; discovering afterwards that the assumption was wrong is not.

**Experiment 3 uses a Compose override file rather than more environment variables**
`--max-num-seqs` was parameterised because only its *value* changed. Swapping to an unquantized model changes the *shape* of the command — `--quantization` must be absent rather than different, and the 3B's pinned revision hash does not apply to another repository. Compose replaces the whole `command` list on override, which is the semantics required.

**Recorded deviation:** the 1.5B model is **not revision-pinned**. Acceptable for a one-off comparison recorded as a single measurement; not acceptable for anything shipped. If it were ever adopted, Boundary 2 requires pinning it first.

---

## Made during Finalization Phase 0 (clean slate)

**The 1.5B-AWQ is revision-pinned before it is measured, not after**
`FINALIZATION_PLAN.md` puts the pin in Phase 5.1, after Phase 2 has already recorded the model's canonical numbers. That ordering is wrong for the same reason Stage 1 discovered: **adding `--revision` changes the vLLM config, and the `torch.compile` cache is keyed on the config.** So the pin is not a no-op that can be applied later — it produces a different cache key and a fresh ~40 s compile. Measuring first and pinning afterwards would mean the canonical figures describe a configuration that was never shipped, and would drop a recompile into a measurement. Pinned to `3ecffa0ceb27851800f45519bab9c457a04405e1`, read from the downloaded snapshot rather than the Hub, closing the recorded Stage 11 deviation. Boundary 2 now holds on the finalized model.

**Phase 2 (1.5B) runs before Phase 1 (3B), and the cold card is the reason**
Phase 0.7 exists to produce one scarce resource: a card cold enough that the canonical numbers sit outside the ~24% session-drift band. Phase 1 as written spends it on ~25 minutes of 3B ramps — a model this project is not shipping — and hands Phase 2 a hot card for the model it is. The 3B's numbers already exist, and the comparison that mattered (Experiment 3c) was measured back to back and remains valid regardless. **Spend the cold card on the model being shipped.** If the 3B is re-measured at all it is afterwards, labelled hot-card.

**The context window arithmetic now rests on verified architecture, not recollection**
`config.json` read directly from the snapshot: `num_hidden_layers 28`, `num_key_value_heads 2`, `num_attention_heads 12`, `hidden_size 1536`, `max_position_embeddings 32768`, `rope_scaling null`. Independently: head_dim = 1536/12 = 128, and KV/token = 2 x 28 x 2 x 128 x 2 bytes = **28,672 bytes = 28.0 KiB**, matching the value Stage 11 back-solved out of a vLLM error message. Consequences: 69,616 tokens x 28,672 = **1.859 GiB** of cache (matches), and one full 32,768-token sequence costs **896 MiB against 1,859 MiB available** — so the model's native maximum context fits, and Phase 2.5's claim is now derived rather than recalled. `rope_scaling: null` also means **32,768 is a hard ceiling**, not a YaRN-extendable one: this checkpoint ships no context-extension config, so anything above 32k would require enabling YaRN explicitly and is out of scope.

**Clearing the `torch.compile` cache is a hygiene measure, not a disk measure**
`inference-product_vllm-cache` held every config compiled across Stage 11 and totalled **17.32 MB** — roughly 1.4 MB per config, against a predicted 0.3-1.5 GB. Compiled graphs are metadata and serialized artifacts, not weights. The volume was still worth deleting, because a dozen stale config keys are clutter and the recompile is cheap, but the justification is "no dead keys" rather than "reclaimed space". **The compile cache's entire value is the ~40 s it saves, and its entire cost is measurement contamination if that 40 s lands in the wrong place.**

**`--gpu-memory-utilization 0.78` is reopened, not settled — the constraint was partly self-inflicted**
`FINALIZATION_PLAN.md` Phase 3 lists this dial as *"Cannot move. This is a Windows constraint, not a choice."* Phase 0 produced evidence against that. Stage 1 calibrated 0.78 against **3.21 of 4.0 GiB free**, and attributed the shortfall entirely to the WDDM compositor reservation. But `nvidia-smi` with the stack down showed **290 MiB held by ordinary desktop applications** — Edge WebView2, WhatsApp, Phone Link — which are part of what made free memory 3.21 GiB. With those closed the card reports **0 MiB in use**.

The dial is a fraction of *total* memory, so closing apps does not enlarge the KV cache by itself; what it enlarges is the headroom against vLLM's startup free-memory check, which is the thing that actually forced 0.78 down. Weights are fixed at 1.1018 GiB, so **every additional MiB of budget becomes KV cache** at ~28 KiB/token.

This is not a config tweak. The shared prefix cache is the binding constraint on v2's entire capacity question, so a 15-25% larger pool moves the knee in the headline curve. **Decision: do not change the dial yet.** Phase 2 runs at the shipping 0.78 and we read the free-memory figure out of its startup log, which vLLM prints anyway. One variable at a time — the canonical measurement first, the dial as a separate arm afterwards.

**Recorded as part of the measurement environment: desktop GPU applications are closed during measurement.** This is a reproducibility fact, not a tuning trick, and it must appear beside any number it affects. It is also not a posture a real deployment could assume.

**The cooldown target is dropped — it was never load-bearing, and it is unreachable**
`FINALIZATION_PLAN.md` Phase 0.7 called for cooling to ≤50 °C before any canonical measurement. Two things killed it.

**It is unreachable.** With every GPU process killed — 0 MiB VRAM in use, 0% utilization — the card still sat at **74 °C, P0, 1057 MHz core, 5501 MHz memory, 16.53 W**. Stage 2's *Prefer maximum performance* fix pins the clocks permanently, including at idle, so there is no low-power state to fall into and the card dissipates ~16.5 W continuously doing nothing. The 50 °C figure was chosen in advance rather than measured, which is precisely the error this project keeps writing rules about.

**It was also unnecessary.** The protocol already runs a **discarded warm-up before every measured run**, which drives the card to thermal steady state (~87 °C, ~712 MHz) before anything is recorded. Starting at 50 °C versus 74 °C changes how quickly steady state is reached, not where it settles — and the measured run happens afterwards either way. The drift that genuinely matters is the ~24% Stage 11 measured across ~35 minutes of *sustained load*, and its mitigation is the existing rule that compared arms run back to back. A cold start was never the control for it.

**Replacement rule:** record the idle temperature at the start of any measurement block as environment metadata, keep the discarded warm-up, and compare only arms measured consecutively. No waiting.

**Related correction to Stage 1's account of free VRAM.** Stage 1 attributed the 3.21-of-4.0 GiB shortfall entirely to the WDDM compositor reservation. `nvidia-smi` with the stack down shows **290 MiB held by ordinary desktop applications** rendering with hardware acceleration (Electron/Chromium: the IDE, WhatsApp, Edge WebView2, Start Menu). These are `C+G` graphics clients, unrelated to the NVIDIA power-management setting — that setting governs P-states, not which GPU renders an application. The compositor is real; it was not the whole story.

## Made during Finalization Phase 2

**The warm-up rule is tightened: it must reach FULL heat soak, verified, not merely "run something first"**
The protocol has said "discarded warm-up, then measure without pausing" since Stage 2. Phase 2 showed that is not specific enough. Two runs of an identical command differed by **42% throughput and 56% on ITL**, because one began heat-soaked and the other from 58 C after an interruption. The existing warm-up (levels 4, min-requests 16 — roughly 25 s) does not reach steady state: the watch_gpu trace shows the card needs **~85 seconds** under load to transition from SwPowerCap to sustained SwThermalSlowdown.

Worse, an under-soaked ramp is not merely noisy, it is **systematically biased**: levels are measured in ascending order while the card heats, so low concurrency is measured cool and high concurrency hot. ITL rose +38% across the cool-start ramp against +19% across the soaked one — the difference is temperature masquerading as contention.

**Revised rule: warm up until watch_gpu shows sustained 0x20 and a plateaued temperature, then measure. Any interruption between warm-up and measurement invalidates the warm-up.** Assuming thermal state instead of reading it is what produced this, and the telemetry that catches it costs nothing.

**Decode is core-clock-gated on this card, and the v1 scaling law gains a second term**
clocks.mem held at 5501 MHz across both runs while ITL swung from 10.1 to 15.8 ms, tracking clocks.sm to within 1%. A purely bandwidth-bound decode cannot behave that way. The correct model is decode_time = max(bandwidth_time, issue_time), and on this card **issue time dominates** — the SMs cannot generate memory requests fast enough at thermally-limited clocks to saturate the 192 GB/s bus. Decode proportional to weight bytes still holds across models at comparable thermal state; it is now understood as a special case rather than the whole law. FINALIZATION_PLAN.md's appendix predicted this and called it untestable here; it was tested accidentally and confirmed.

**Consequence: the 712 MHz sustained-clock figure is model-dependent and must not be reused.** It was measured while serving the 3B. The lighter 1.5B sustains ~1,100-1,200 MHz, so every derived quantity resting on 712 MHz — notably the "~97 GB/s effective bandwidth, 50.5% of rating" calculation — is stale and needs re-deriving against the model actually being measured.
