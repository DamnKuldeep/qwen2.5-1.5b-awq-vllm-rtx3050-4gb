# benchmarks/

| File | What it is |
| --- | --- |
| `baseline.md` | Stage 2 baseline for the 3B model, and the thermal protocol that measuring it forced |
| `optimization_results.md` | Stage 11 experiments, and **the measurement protocol** — ten rules, each traceable to the failure that produced it |
| `watch_gpu.ps1` | One-second sampling of clocks, power, temperature and throttle reasons. The only instrument that saw the 11.6x power-state loss |
| `run_bench_serve.ps1` | `vllm bench serve` wrapper from Stage 2 |

Every number here is from one specific RTX 3050 laptop. Read `optimization_results.md` before comparing anything to anything.
