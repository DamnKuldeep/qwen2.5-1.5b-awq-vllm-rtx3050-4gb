"""
Plot real Prometheus series as SVG, for docs/img/.

    python benchmarks/plot_prometheus.py --start <epoch> --end <epoch> --out docs/img/session

Grafana shows these series live and the committed PNGs prove it. This exists
for a different job: one chart of a WHOLE test session, so the shape of a
whole session of deliberately varied load is visible at once - which no dashboard crop
shows, because a dashboard is built for the last five minutes. The evidence
suite calls this with the absolute window it ran in.

Vector output, two themes, and the numbers come from the same Prometheus the
dashboards read, so a chart here cannot disagree with a panel there.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROM = "http://localhost:9090"
FONT = "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

THEMES = {
    "light": dict(bg="#ffffff", panel="#f6f8fa", line="#d0d7de", ink="#1f2328",
                  dim="#59636e", faint="#8c959f", accent="#0969da", good="#1a7f37",
                  warn="#9a6700", bad="#cf222e", gpu="#8250df"),
    "dark": dict(bg="#0d1117", panel="#161b22", line="#30363d", ink="#e6edf3",
                 dim="#9198a1", faint="#6e7681", accent="#4493f8", good="#3fb950",
                 warn="#d29922", bad="#f85149", gpu="#ab7df8"),
}


def query_range(expr: str, start: float, end: float, step: int):
    q = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{PROM}/api/v1/query_range?{q}", timeout=60) as r:
        d = json.load(r)
    if d.get("status") != "success" or not d["data"]["result"]:
        return []
    return [(float(t), float(v) if v not in ("NaN", "+Inf") else 0.0)
            for t, v in d["data"]["result"][0]["values"]]


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, fill, size=12, weight=400, anchor="start", font=FONT):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{font}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>')


def nice_max(v: float) -> tuple[float, list[float]]:
    """A round axis maximum and its ticks, so labels read 0/5/10 rather than 0/3.5/7."""
    if v <= 0:
        return 1.0, [0.5, 1.0]
    for step in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if v / step <= 4:
            top = step * math.ceil(v / step)
            return top, [step * i for i in range(1, int(round(top / step)) + 1)]
    return v, [v / 2, v]


SHORT = {"capacity_10users": "10u", "capacity_20users": "20u", "capacity_30users": "30u",
         "capacity_40users": "40u", "capacity_60users": "60u", "capacity_10users_control": "ctl",
         "shape_burst": "burst", "shape_herd": "herd", "shape_ramp": "ramp",
         "shape_diurnal": "diurnal", "conv_300tok": "0.3k", "conv_2000tok": "2k",
         "conv_8000tok": "8k", "adversarial": "abuse"}


def phases(start: float, end: float) -> list[tuple[float, float, str]]:
    """Where each evidence run sat in time, from the result files themselves.

    Each file is written the moment its run ends and records its own duration,
    so (mtime - duration, mtime) is the run. No separate timeline to keep in
    sync, and a run that did not happen simply has no band.
    """
    out = []
    for p in (ROOT / "load_testing" / "results").glob("evidence_*.json"):
        key = p.stem.removeprefix("evidence_")
        if key not in SHORT:
            continue
        t1 = p.stat().st_mtime
        try:
            dur = float(json.loads(p.read_text(encoding="utf-8"))["config"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if start <= t1 - dur and t1 <= end + 60:
            out.append((t1 - dur, t1, SHORT[key]))
    return sorted(out)


def build(series: dict, start: float, end: float, limit: float, t: dict) -> str:
    W, L, R = 1100, 78, 900
    panels = [
        ("Concurrency: peak in flight and queued, against the admission limit",
         [("in flight", "inflight", t["accent"]), ("queued", "queued", t["warn"])], None, True),
        ("Refused, as a share of offered requests  (%)",
         [("refused", "shedpct", t["bad"])], 100.0, False),
        ("Engine output  (tokens/s)",
         [("output tok/s", "tokps", t["good"])], None, False),
        ("Prefix-cache hit rate  (%)",
         [("hit rate", "cachehit", t["gpu"])], 100.0, False),
    ]
    band_top, ph, gap = 104, 96, 14
    bottom = band_top + len(panels) * (ph + gap) - gap
    H = bottom + 80
    span = max(end - start, 1)
    minutes = round(span / 60)

    def px(ts):
        return L + (ts - start) / span * (R - L)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img"><rect width="{W}" height="{H}" fill="{t["bg"]}"/>']
    s.append(txt(28, 34, f"The {minutes}-minute evidence session, as Prometheus recorded it",
                 t["ink"], 17, 650))
    s.append(txt(28, 55, "Capacity sweep, traffic shapes, conversation lengths, then an abusive "
                         "client. Bands mark each run; every line is a real scraped series.",
                 t["dim"], 12.5))

    # Phase bands behind every panel, labelled once along the top.
    for a, b, label in phases(start, end):
        x0, x1 = px(a), px(b)
        s.append(f'<rect x="{x0:.1f}" y="{band_top-22}" width="{max(x1-x0,1):.1f}" '
                 f'height="{bottom-band_top+22}" fill="{t["line"]}" opacity="0.35"/>')
        s.append(txt((x0 + x1) / 2, band_top - 8, label, t["dim"], 10, 500, "middle", MONO))

    for n, (title, keys, fixed, has_limit) in enumerate(panels):
        top = band_top + n * (ph + gap)
        bot = top + ph
        s.append(f'<rect x="{L}" y="{top}" width="{R-L}" height="{ph}" rx="5" '
                 f'fill="none" stroke="{t["line"]}"/>')
        s.append(txt(L + 8, top + 15, title, t["ink"], 11.5, 620))
        if fixed:
            vmax, ticks = nice_max(fixed)
        else:
            peak = max((v for _, k, _ in keys for _, v in series.get(k, [])), default=0.0)
            if has_limit:
                peak = max(peak, limit)
            vmax, ticks = nice_max(peak * 1.08)
        inner = ph - 22

        def py(v, bot=bot, vmax=vmax, inner=inner):
            return bot - min(v, vmax) / vmax * inner

        for tk in ticks:
            y = py(tk)
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{t["line"]}" '
                     f'stroke-width="1" stroke-dasharray="2 4"/>')
            s.append(txt(L - 8, y + 4, f"{tk:g}", t["faint"], 10, 400, "end", MONO))
        if has_limit:
            y = py(limit)
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{t["bad"]}" '
                     f'stroke-width="1.4" stroke-dasharray="6 4"/>')
        for _, k, colour in keys:
            pts = series.get(k, [])
            if len(pts) < 2:
                continue
            poly = " ".join(f"{px(ts):.1f},{py(v):.1f}" for ts, v in pts)
            s.append(f'<polyline points="{poly}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.6" stroke-linejoin="round"/>')
        items = [(label, colour) for label, _, colour in keys]
        if has_limit:
            items.append((f"limit ({limit:g})", t["bad"]))
        for i, (label, colour) in enumerate(items):
            s.append(f'<rect x="{R+16}" y="{top+8+i*18}" width="10" height="10" rx="2" fill="{colour}"/>')
            s.append(txt(R + 32, top + 17 + i * 18, label, t["dim"], 11.5))

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        ts = start + frac * span
        s.append(txt(px(ts), bottom + 18, time.strftime("%H:%M", time.gmtime(ts)),
                     t["faint"], 10.5, 400, "middle", MONO))
    s.append(txt(R, bottom + 34, "UTC", t["faint"], 10, 400, "end", MONO))

    y0 = H - 16
    notes = [("in flight never crosses the limit", t["accent"]),
             ("overload becomes refusals, not latency", t["bad"]),
             ("cache dips only when many conversations open cold", t["gpu"])]
    x = 28
    for note, colour in notes:
        s.append(f'<rect x="{x}" y="{y0-9}" width="9" height="9" rx="2" fill="{colour}"/>')
        s.append(txt(x + 15, y0, note, t["dim"], 11.5))
        x += 300
    s.append("</svg>")
    return "".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=75)
    ap.add_argument("--start", type=int, help="absolute start, epoch seconds")
    ap.add_argument("--end", type=int, help="absolute end, epoch seconds")
    ap.add_argument("--out", default="docs/img/session")
    args = ap.parse_args()

    if args.start and args.end:
        start, end = float(args.start), float(args.end)
    else:
        end = time.time()
        start = end - args.minutes * 60
    step = max(15, int((end - start) / 700))

    w = f"{step}s"
    # Rate windows are short (30 s against a 5 s scrape) so each run's band
    # shows that run: a 1-minute window smeared the heat soak's refusals into
    # the first 10-user band, which refused nothing.
    series = {
        # Peaks within each step, not samples: a gauge sampled every 15 s can
        # miss the limit being reached, and the claim here is that it is never
        # EXCEEDED - which only a max can support.
        "inflight": query_range(f"max_over_time(gateway_inflight[{w}])", start, end, step),
        "queued":   query_range(f"max_over_time(gateway_queue_depth[{w}])", start, end, step),
        "shedpct":  query_range(
            "100 * sum(rate(gateway_rejected_total[30s])) / clamp_min("
            "sum(rate(gateway_rejected_total[30s])) + sum(rate(gateway_admitted_total[30s])), 1e-9)",
            start, end, step),
        "tokps":    query_range("rate(vllm:generation_tokens_total[30s])", start, end, step),
        "cachehit": query_range(
            "100 * rate(vllm:prefix_cache_hits_total[1m]) / "
            "clamp_min(rate(vllm:prefix_cache_queries_total[1m]), 1)", start, end, step),
    }
    lim = query_range("max(gateway_max_inflight)", start, end, step)
    limit = max((v for _, v in lim), default=6.0)
    got = {k: len(v) for k, v in series.items()}
    print("  samples:", got)
    if not any(got.values()):
        print("  no data - is Prometheus up and has load run?")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        p = out.with_name(f"{out.name}-{theme}.svg")
        p.write_text(build(series, start, end, limit, t), encoding="utf-8")
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
