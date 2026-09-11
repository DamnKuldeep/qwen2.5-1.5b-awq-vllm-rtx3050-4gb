# Product Spec — v1

## What this is

A self-hosted LLM inference product: an authenticated, budget-limited API in front of a properly-tuned vLLM server, with two separate dashboards for two separate audiences, deployed via Docker, benchmarked and load-tested on real hardware, and hardened — provably, not just described — against the four self-hosting failure boundaries.

**This is v1.** It deliberately does not include retrieval-augmented generation, guardrails, multi-model routing, or a real user-account system. Those are good future directions and explicitly out of scope for this build — see `CLAUDE.md`'s Scope section.

## How it behaves

**For a developer calling it:**
- `POST /v1/chat/completions` — OpenAI-compatible request/response shape, so any existing OpenAI client library works against it by changing only the base URL.
- Requires `Authorization: Bearer <api_key>` on every request.
- Each key has a token budget. Once exhausted, the API returns `429`.
- Supports `response_format: {"type": "json_object"}` for structured output — and this is contract-tested (Boundary 1 from the reading), so it's *proven*, not assumed, that the gateway doesn't silently drop it.

**For a person using it directly:**
- A minimal chat page in a browser, using an API key, talking to the same endpoint above.

**For the person operating it — two dashboards, answering two different questions:**
- **Technical dashboard** (Grafana, `localhost:3000`) — GPU health, KV cache %, prefix cache hit rate, TTFT/ITL/throughput. Answers: *is the system healthy?*
- **Product usage dashboard** (served by the gateway itself, `localhost:8080/dashboard`) — requests per key, tokens consumed per key, remaining budget, request history. Answers: *how is the product being used?*

These are deliberately two different views. Conflating "is the infrastructure okay" with "who's using how much of my product" is a real-world mistake worth avoiding on purpose here.

## What's underneath

See `DECISIONS.md` for the reasoning behind each of these — nothing below is arbitrary:

- **Model:** Qwen2.5-3B-Instruct-AWQ, served by vLLM with the Marlin kernel and prefix caching enabled from the start.
- **Gateway:** FastAPI.
- **Usage/budget storage:** SQLite.
- **Observability:** Prometheus scraping vLLM's own `/metrics` directly (not through a wrapper), visualized in Grafana.
- **Deployment:** Docker Compose for the full local stack; an optional Kubernetes stage reuses your GPU-enabled Minikube cluster to demonstrate self-healing concretely.
- **Load testing:** a small custom asyncio script — deliberately not a new framework, so every part of what's being measured stays visible and explainable.

## Definition of done for v1

- All four self-hosting boundaries are **demonstrated**, not just described:
  1. A contract test that actually catches a regression when structured output silently breaks (Boundary 1).
  2. Pinned versions (vLLM image tag, model revision) with the reasoning written down (Boundary 2).
  3. A dashboard reading the engine's real metrics, including KV-cache-% and prefix-hit-rate specifically (Boundary 3).
  4. A request-state test proving the gateway survives an engine-side preemption/eviction without corrupting its own tracking (Boundary 4).
- A written benchmark report with real numbers from this exact machine: baseline TTFT/ITL/throughput, a measured concurrency ceiling against a stated SLO, and a before/after table for at least one optimization.
- The whole stack comes up with one command and is reachable at documented URLs.
