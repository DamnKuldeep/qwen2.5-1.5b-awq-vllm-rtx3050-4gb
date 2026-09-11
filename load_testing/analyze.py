"""
Turn chat_sim result files into the tables that go in the docs.

    python load_testing/analyze.py                       # all results
    python load_testing/analyze.py --glob "capacity_*"   # just the sweep

Two views, because they answer different questions:

  * The AGGREGATE p95 is what a dashboard shows.
  * The PER-USER p95 distribution is what users experience. A service can have
    a healthy aggregate p95 while a minority of users are consistently unlucky,
    and averaging across requests hides exactly those people. The
    "users over SLO" column is the honest headline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join("load_testing/results", pattern + ".json"))):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d["_name"] = os.path.basename(path)[:-5]
        out.append(d)
    return out


def capacity_table(runs: list[dict]) -> str:
    rows = ["| users | turns ok | shed 503 | shed % | out tok/s | mean prompt | "
            "TTFT p50 | TTFT p95 | worst user p95 | users over SLO | prefix hit % | preempt |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |"]
    for d in runs:
        c, t = d["config"], d["totals"]
        a, pu, pc = d["ttft_ms_all_requests"], d["ttft_ms_per_user_p95"], d["prefix_cache"]
        rows.append(
            f"| {c['users']} | {t['turns_ok']} | {t['turns_shed_503']} | {t['shed_rate_pct']} | "
            f"{t['output_tok_per_s']} | {t['mean_prompt_tokens']:.0f} | {a['p50']} | {a['p95']} | "
            f"{pu['worst_user']} | {pu['users_breaching_slo']}/{pu['users_total']} | "
            f"{pc['hit_rate_pct']} | {pc['preemptions_delta']:.0f} |"
        )
    return "\n".join(rows)


def scenario_table(runs: list[dict]) -> str:
    rows = ["| scenario | users | turns ok | shed 503 | shed % | TTFT p95 | worst user p95 | "
            "users over SLO | prefix hit % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |"]
    for d in runs:
        c, t = d["config"], d["totals"]
        a, pu, pc = d["ttft_ms_all_requests"], d["ttft_ms_per_user_p95"], d["prefix_cache"]
        label = c.get("label", d["_name"])
        rows.append(
            f"| {label} | {c['users']} | {t['turns_ok']} | {t['turns_shed_503']} | "
            f"{t['shed_rate_pct']} | {a['p95']} | {pu['worst_user']} | "
            f"{pu['users_breaching_slo']}/{pu['users_total']} | {pc['hit_rate_pct']} |"
        )
    return "\n".join(rows)


def ascii_curve(runs: list[dict]) -> str:
    """A plain-text plot of p95 TTFT and cache hit rate against user count.

    ASCII rather than an image because it renders in a terminal, in a diff, and
    in any markdown viewer, and because the shape is the whole point - a knee
    is visible at this resolution and needs no axis labels to be read.
    """
    pts = [(d["config"]["users"], d["ttft_ms_all_requests"]["p95"] or 0,
            d["prefix_cache"]["hit_rate_pct"] or 0, d["totals"]["shed_rate_pct"])
           for d in runs if d["config"].get("pattern") == "steady"]
    if not pts:
        return "(no steady-state runs)"
    pts.sort()
    max_ttft = max(p[1] for p in pts) or 1
    width = 46
    lines = ["users |  p95 TTFT (ms)                                 | hit% | shed%"]
    lines.append("------+------------------------------------------------+------+------")
    for users, p95, hit, shed in pts:
        bar = "#" * max(1, int(width * p95 / max_ttft))
        lines.append(f"{users:5d} | {bar:<46} | {hit:4.0f} | {shed:5.1f}   {p95:.0f} ms")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*")
    args = ap.parse_args()
    runs = load(args.glob)
    if not runs:
        print("no results found")
        return 1

    cap = [d for d in runs if d["config"].get("pattern") == "steady"]
    other = [d for d in runs if d["config"].get("pattern") != "steady"]

    if cap:
        print("\n## Capacity sweep\n")
        print(capacity_table(cap))
        print("\n```")
        print(ascii_curve(cap))
        print("```")
    if other:
        print("\n## Traffic shapes\n")
        print(scenario_table(other))

    print("\n## Engine config recorded in these runs\n")
    print("```")
    print(json.dumps(runs[0].get("engine", {}), indent=2))
    print(json.dumps({k: v for k, v in runs[0]["admission"].items()
                      if k.startswith("max_")}, indent=2))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
