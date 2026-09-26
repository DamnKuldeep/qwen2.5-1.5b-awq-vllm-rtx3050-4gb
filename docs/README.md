# Documentation map

Four kinds of document, and they answer different questions. Start with
whichever matches yours.

## If you want the result

| Document | Answers |
| --- | --- |
| **[../README.md](../README.md)** | What is this, how many users, how do I run it. The pitch, with the headline numbers and the diagrams |
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
| **[../benchmarks/optimization_results.md](../benchmarks/optimization_results.md)** | **The measurement protocol** — ten rules, each traceable to the specific failure that produced it. Read this before comparing any number here to any number elsewhere |
| **[../benchmarks/baseline.md](../benchmarks/baseline.md)** | The v1 baseline, and the thermal protocol that measuring it forced into existence |
| **[../PROGRESS_LOG.md](../PROGRESS_LOG.md)** | The raw chronological record. Append-only: where something later turned out wrong, a correction was added rather than the original edited. Long, and deliberately not summarised |

## Plans, kept for the reasoning rather than the conclusions

Both were written *before* the work and both contain predictions that
measurement contradicted. Each carries a banner saying so.

| Document | Was |
| --- | --- |
| **[V2_SCOPE.md](V2_SCOPE.md)** | The v2 plan. Predicted a prefix-cache knee that never appeared |
| **[FINALIZATION_PLAN.md](FINALIZATION_PLAN.md)** | The v1 close-out plan. Two of its instructions were wrong and are corrected in place |
| **[reading_reference/](reading_reference/)** | Background study notes from before the project started, synthesised from 14 sources |

## Diagrams

`img/` holds the SVGs used in the README, in light and dark variants.
They are generated — edit **[img/_gen_diagrams.py](img/_gen_diagrams.py)** and
re-run it, rather than editing the SVGs by hand:

```bash
python docs/img/_gen_diagrams.py
```

The degradation chart reads its numbers directly from
`load_testing/results/*.json`, so it cannot drift from the measurements it draws.
