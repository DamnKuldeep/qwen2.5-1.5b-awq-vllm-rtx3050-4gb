# gateway/

The FastAPI service in front of vLLM: bearer-token auth, per-key token budgets backed by SQLite, a streaming proxy that forwards request bodies as untyped dicts so no field can be silently dropped, admission control with per-tenant fairness, a sliding-window context policy, and the product usage dashboard.

| File | What it is |
| --- | --- |
| `main.py` | Routes, auth, budget enforcement, the streaming tee, `/metrics`, `/ready` vs `/health` |
| `admission.py` | Bounded concurrency, weighted by prompt size; dynamic per-key share; Prometheus exposition |
| `context.py` | What happens when a conversation outgrows the context window, and why that shape |
| `db.py` | SQLite ledger — keys, budgets, request history |
| `seed_db.py` | Demo keys and the simulator's per-user key pool |
| `tests/` | 22 contract tests against a stub upstream, 1 integration test against the live stack |
| `test_gateway.ps1`, `test_budget.ps1` | Behavioural checks against a running stack |

Run locally without Docker: `python -m gateway.seed_db` then `uvicorn gateway.main:app --port 8080`.
