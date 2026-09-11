# load_testing/

| File | What it is |
| --- | --- |
| `chat_sim.py` | Multi-user chat simulator: Poisson arrivals, log-normal think time, conversations that accumulate context, per-user reporting. The instrument the capacity model is built on |
| `ramp.py` | The v1 concurrency ramp — holds N requests in flight with unique prompts. Measures the engine, not a chat workload |
| `failure_matrix.py` | The worst cases reachable over HTTP, each with a stated expectation |
| `kill_test.py` | Engine crash and gateway restart under load |
| `run_matrix.ps1` | The whole measurement matrix, one command, seeded |
| `analyze.py` | Turns `results/*.json` into the tables in the docs |
| `results/` | Every measurement behind the numbers in `docs/`, with the engine config embedded in each file |
