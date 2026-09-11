# deployment/docker/

| File | What it is |
| --- | --- |
| `docker-compose.yml` | The stack: vLLM, gateway, Prometheus, Grafana. Every engine flag carries its derivation in a comment |
| `docker-compose.1_5b-awq.yml` | **The shipped override** — Qwen2.5-1.5B-Instruct-AWQ, pinned revision, 32k context, `--max-num-seqs 32`. Always pass both files, or set `COMPOSE_FILE` |
| `docker-compose.1_5b.yml` | The unquantized 1.5B arm from Stage 11 Experiment 3. Cannot start at 4k context on this card; kept as the record of that |
| `run_vllm.ps1` / `run_vllm.sh` | Pre-Compose launch scripts from Stage 1 (3B model). Superseded; kept for the flag history |
| `smoke_test.ps1` | One real generation against the engine |

The base file alone runs the v1 3B model. The two-file form is what every number in the docs was measured on.
