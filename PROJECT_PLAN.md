# Project Plan — Stage by Stage

Follow in order. Each stage ends only when the report-back confirms it works. Update `PROGRESS_LOG.md` at the end of every stage. This file is the skeleton — the actual teaching, file-by-file reasoning, and troubleshooting happens live, per `CLAUDE.md`.

---

### Stage 0 — Confirm the environment is still ready
No new files. Re-verify (by asking the user to run and report, not by running anything): `docker run --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi` still works, and Docker Desktop is up. This project assumes the Docker+GPU passthrough already proven working in the earlier course — this stage just confirms nothing regressed since then.

### Stage 1 — Stand up vLLM, tuned from the start
Create: `deployment/docker/run_vllm.sh` (or a documented `docker run` command) with the model, `--quantization awq_marlin`, a deliberate `--max-model-len`, `--max-num-seqs`, and `--enable-prefix-caching` — decisions already made in `DECISIONS.md`. Explain each flag as it's written. User runs it, confirms Marlin was selected in the logs and the server reports a nonzero KV-cache block count.

### Stage 2 — Baseline benchmark
Use `vllm bench latency` and `vllm bench throughput` (built into the vLLM image — no new install). Capture real TTFT/ITL/throughput numbers into `benchmarks/baseline.md`. This is the number every later "improvement" gets compared against — don't skip it to save time.

### Stage 3 — Gateway skeleton: auth + proxy
Create `gateway/main.py` (FastAPI), `gateway/requirements.txt`. Explain what FastAPI is and why async matters here. Just auth + pass-through to vLLM at this stage — no budget tracking yet, kept small on purpose. User runs it locally with `uvicorn`, confirms a request with a valid key succeeds and an invalid key gets `401`.

### Stage 4 — Usage tracking, budget enforcement, product usage dashboard
Add SQLite-backed request logging and per-key budgets to the gateway. Add the `/dashboard` route (Jinja2 template) showing usage per key. Explain SQLite's role and why this dashboard is a different thing from Grafana (which doesn't exist yet — that's Stage 8). User confirms budget exhaustion returns `429`, and `/dashboard` shows real logged requests.

### Stage 5 — Contract test (Boundary 1)
Create `gateway/tests/test_structured_output.py`. Explain what a contract test is and why checking the response body (not just status 200) is the whole point. User runs `pytest`, sees it pass. Later, in Stage 10, this same test gets deliberately broken and re-passed as proof it actually catches regressions.

### Stage 6 — Minimal chat UI
Create `ui/index.html` (plain HTML/JS, no framework) that calls the gateway. User opens it in a browser, sends a message, sees a response.

### Stage 7 — Containerize and bring the stack up together
Create `gateway/Dockerfile`, `deployment/docker/docker-compose.yml` wiring together vLLM + gateway (+ Prometheus/Grafana placeholders for Stage 8). User runs `docker compose up`, confirms every service reports healthy.

### Stage 8 — Observability: Prometheus + Grafana
Create `observability/prometheus/prometheus.yml` (scraping vLLM's `/metrics` directly — Boundary 3, made concrete) and Grafana dashboard config with panels for KV-cache-%, prefix-hit-rate, requests running/waiting, TTFT/ITL, throughput, GPU stats. User opens `localhost:3000`, confirms real numbers moving as they send traffic.

### Stage 9 — Load testing and the concurrency ceiling
Create `load_testing/ramp.py`. User runs it at increasing concurrency while watching the Grafana dashboard live, and reports back the concurrency level where a stated SLO (e.g. p95 TTFT < 1.5s) breaks, and whether the dashboard shows it breaking on KV-cache-% (memory ceiling) or GPU compute (throughput ceiling).

### Stage 10 — Deliberately break it, on purpose, three ways
No new components — deliberate misuse of what already exists. (a) Force KV-cache pressure and observe an eviction (Boundary 4) — confirm the gateway's own state survives it. (b) Temporarily break the `response_format` forwarding and confirm the Stage 5 test actually fails, then fix it. (c) `docker compose kill` or `kubectl delete pod` (if Stage 12 has happened) the vLLM service mid-load-test and observe how the gateway behaves. Each gets a short write-up in `PROGRESS_LOG.md`.

### Stage 11 — Optimization pass, measured
Re-run the Stage 2 benchmark after each: (a) unquantized 1.5B vs. the AWQ 3B — which actually wins on this card; (b) a shared-system-prompt workload vs. no shared prefix — prove prefix caching is doing something, using the Grafana panel from Stage 8; (c) optionally, n-gram speculative decoding. Every result goes in `benchmarks/optimization_results.md` as a before/after table.

### Stage 12 — Optional: Kubernetes deployment
Create `deployment/k8s/` manifests (Deployment + Service for both vLLM and the gateway) on the GPU-enabled Minikube cluster from the earlier course. Repeat the Module 4 self-healing lab for real: kill a pod mid-load-test, time the recovery.

### Stage 13 — Final report
Assemble `docs/FINAL_REPORT.md`: architecture diagram, Stage 2 baseline, Stage 11 before/after table, Stage 9's concurrency ceiling, Stage 10's three incident write-ups, and a short "what I'd need for real scale" closing section.
