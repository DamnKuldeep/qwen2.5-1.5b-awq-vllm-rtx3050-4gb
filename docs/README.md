# Documentation map

Four kinds of document, and they answer different questions. Start with
whichever matches yours.

## If you want the result

| Document | Answers |
| --- | --- |
| **[../README.md](../README.md)** | What is this, how many users, how do I run it. The pitch, with the headline numbers and the diagrams |
| **[RESULTS.md](RESULTS.md)** | *Every scenario, every number, in one place*, generated from the result files and never typed. SLO attainment from the user's side (retries included) across the capacity sweep, traffic shapes, conversation lengths and an abusive client; the ablations behind every shipped setting; failure injection; live Grafana captures |
| **[CAPACITY_MODEL.md](CAPACITY_MODEL.md)** | *How many people can talk to this at once, and what decides the answer.* The derivation, the measured curve, the levers, and exactly where each number stops being true |
| **[FAILURE_MATRIX.md](FAILURE_MATRIX.md)** | *What happens when it breaks.* Fourteen failure cases, each with the behaviour expected **before** testing next to the behaviour measured after |

## If you want the reasoning

| Document | Answers |
| --- | --- |
| **[FINAL_REPORT.md](FINAL_REPORT.md)** | The full account, v1 through v2. Contains **the wrong-predictions table** — twenty-four predictions that were written down and then contradicted by measurement. The most useful thing here for anyone doing similar work |
| **[../DECISIONS.md](../DECISIONS.md)** | Every architectural choice with the reasoning that produced it, in the order the choices were made |
| **[CONCEPTS_EXPLAINED.md](CONCEPTS_EXPLAINED.md)** | Every tool, flag and concept, explained the first time it appeared. Written for someone meeting `awq_marlin` or WAL mode for the first time |
| **[RETROSPECTIVE_STAGES_1_2.md](RETROSPECTIVE_STAGES_1_2.md)** | A narrative of how the engine was stood up and the baseline measured — what was predicted, what happened, what changed |

## If you want the method

| Document | Answers |
| --- | --- |
| **[../benchmarks/run_evidence_suite.ps1](../benchmarks/run_evidence_suite.ps1)** | The one command behind RESULTS.md: heat soak, capacity sweep with a control, traffic shapes, conversation lengths, fairness, dashboard capture, failure injection, kill tests. `ablate_long_prefill.ps1` and `ablate_gateway.ps1` produce the tuning arms |
| **[../benchmarks/optimization_results.md](../benchmarks/optimization_results.md)** | **The measurement protocol** — ten rules, each traceable to the specific failure that produced it. Read this before comparing any number here to any number elsewhere |
| **[../benchmarks/baseline.md](../benchmarks/baseline.md)** | The v1 baseline, and the thermal protocol that measuring it forced into existence |
| **[../PROGRESS_LOG.md](../PROGRESS_LOG.md)** | The raw chronological record. Append-only: where something later turned out wrong, a correction was added rather than the original edited. Long, and deliberately not summarised |

## Plans, kept for the reasoning rather than the conclusions

Both were written *before* the work and both contain predictions that
measurement contradicted. Each carries a banner saying so.

| Document | Was |
| --- | --- |
| **[../PRODUCT_SPEC.md](../PRODUCT_SPEC.md)** | The v1 spec: what the product should do and what was deliberately out of scope |
| **[../PROJECT_PLAN.md](../PROJECT_PLAN.md)** | The v1 stage-by-stage build plan (Stages 0–13) |
| **[V2_SCOPE.md](V2_SCOPE.md)** | The v2 plan. Predicted a prefix-cache knee that never appeared |
| **[FINALIZATION_PLAN.md](FINALIZATION_PLAN.md)** | The v1 close-out plan. Two of its instructions were wrong and are corrected in place |
| **[reading_reference/](reading_reference/)** | Background study notes from before the project started, synthesised from 14 sources |

## Images — all generated, none hand-drawn

| Files | Made by | From |
| --- | --- | --- |
| `img/*-light.svg`, `img/*-dark.svg` | [`img/_gen_diagrams.py`](img/_gen_diagrams.py) | Constants, and `load_testing/results/*.json` for the charts |
| `img/session-*.svg` | [`../benchmarks/plot_prometheus.py`](../benchmarks/plot_prometheus.py) | Prometheus range queries over a whole test session |
| `img/grafana-*.png` | [`../benchmarks/capture_dashboards.py`](../benchmarks/capture_dashboards.py) | Grafana's own renderer, over the absolute window the evidence suite ran in |

Regenerate rather than edit. Every chart reads the same data the dashboards do,
so none of them can drift from the measurements they draw:

```bash
python docs/img/_gen_diagrams.py
python benchmarks/build_results_page.py
```

Each SVG exists twice, once per theme, because GitHub strips `<style>` and media
queries from SVG — `<picture><source media="(prefers-color-scheme: dark)">` is the
only theme mechanism it honours.
