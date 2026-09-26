"""
Generates the repository's SVG diagrams.

    python docs/img/_gen_diagrams.py

WHY GENERATED RATHER THAN HAND-DRAWN, AND WHY NOT MERMAID
----------------------------------------------------------
Mermaid renders natively on GitHub, which is why most repositories use it - and
why most repository diagrams look identical. It also gives you no control over
spacing, no way to align a label with a measured number, and no way to put a
real axis on a chart.

These are plain SVG instead. Two consequences worth knowing:

  * GitHub strips <style> blocks and CSS media queries from inline SVG, so
    "one file that adapts to dark mode" does not work. Each diagram is emitted
    TWICE, once per theme, and the README selects between them with
    <picture><source media="(prefers-color-scheme: dark)">. That is the only
    mechanism GitHub honours.
  * Every colour, size and coordinate is a Python constant here, so a change to
    the palette is one edit rather than forty.

The chart diagrams take their numbers from load_testing/results/*.json where
possible, so a diagram cannot silently disagree with the measurements it draws.
"""

import json
import pathlib

OUT = pathlib.Path(__file__).parent
RESULTS = OUT.parent.parent / "load_testing" / "results"

THEMES = {
    "light": dict(
        bg="#ffffff", panel="#f6f8fa", panel2="#eef1f4", line="#d0d7de",
        ink="#1f2328", dim="#59636e", faint="#8c959f",
        accent="#0969da", good="#1a7f37", warn="#9a6700", bad="#cf222e",
        gpu="#8250df", shadow="#00000012",
    ),
    "dark": dict(
        bg="#0d1117", panel="#161b22", panel2="#1c2128", line="#30363d",
        ink="#e6edf3", dim="#9198a1", faint="#6e7681",
        accent="#4493f8", good="#3fb950", warn="#d29922", bad="#f85149",
        gpu="#ab7df8", shadow="#00000040",
    ),
}

FONT = "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace"


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, fill, size=13, weight=400, anchor="start", font=FONT, opacity=1.0):
    o = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-family="{font}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{o}>{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke=None, rx=8, sw=1, dash=None):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}{da}/>'


def arrow(x1, y1, x2, y2, color, mid=None, dash=None, width=1.6):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
         f'stroke-width="{width}" marker-end="url(#ah)"{da}/>')
    if mid:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 7
        s += text(mx, my, mid, color, 11, 500, "middle")
    return s


def svg_open(w, h, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{t["dim"]}"/></marker></defs>'
            f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>')


# ---------------------------------------------------------------------------
# 1. Request lifecycle - what happens to one request, and where it can stop
# ---------------------------------------------------------------------------

def diagram_lifecycle(t):
    W, H = 900, 470
    s = [svg_open(W, H, t)]
    s.append(text(28, 34, "The path of one request", t["ink"], 17, 650))
    s.append(text(28, 55, "Five gates before the GPU. Each can refuse, and each refusal is cheap.",
                 t["dim"], 12.5))

    gates = [
        ("Auth",      "401",  "invalid key"),
        ("Budget",    "429",  "tokens spent"),
        ("Size",      "413",  "cannot fit window"),
        ("Output",    "clamp", "max_tokens bounded"),
        ("Admission", "503",  "at capacity"),
    ]
    x0, y, bw, bh = 28, 96, 152, 58
    gap = (W - 56 - bw) / (len(gates) - 1)

    for i, (name, code, why) in enumerate(gates):
        x = x0 + i * gap
        is_clamp = code == "clamp"
        col = t["warn"] if is_clamp else t["bad"]
        s.append(rect(x, y, bw, bh, t["panel"], t["line"]))
        s.append(text(x + bw / 2, y + 24, name, t["ink"], 13.5, 600, "middle"))
        s.append(text(x + bw / 2, y + 42, why, t["dim"], 10.5, 400, "middle"))
        # refusal drops downward
        s.append(f'<line x1="{x + bw/2}" y1="{y + bh}" x2="{x + bw/2}" y2="{y + bh + 26}" '
                 f'stroke="{col}" stroke-width="1.4" stroke-dasharray="3 3"/>')
        s.append(rect(x + bw / 2 - 28, y + bh + 26, 56, 21, t["bg"], col, 10))
        s.append(text(x + bw / 2, y + bh + 41, code, col, 11, 650, "middle", MONO))
        if i < len(gates) - 1:
            s.append(arrow(x + bw + 4, y + bh / 2, x + gap - 4, y + bh / 2, t["dim"]))

    # engine
    ey = 236
    s.append(rect(28, ey, W - 56, 96, t["panel2"], t["line"]))
    s.append(text(48, ey + 26, "vLLM  ·  Qwen2.5-1.5B-Instruct-AWQ", t["gpu"], 13.5, 650))
    s.append(text(48, ey + 47, "continuous batching  ·  paged KV, 16-token blocks  ·  prefix cache",
                  t["dim"], 11.5))
    s.append(text(48, ey + 70, "at most 6 requests decoding at once", t["ink"], 12, 600))
    s.append(text(48 + 214, ey + 70, "— the gateway's limit, not the engine's",
                  t["faint"], 11.5))

    # pool bar showing KV usage
    bx, bw2 = 560, 300
    s.append(text(bx, ey + 26, "KV pool  69,760 tokens", t["dim"], 11.5))
    s.append(rect(bx, ey + 36, bw2, 14, t["panel"], t["line"], 7))
    s.append(rect(bx, ey + 36, bw2 * 0.17, 14, t["good"], None, 7))
    s.append(text(bx, ey + 66, "peak measured: 17%", t["good"], 11.5, 600))
    s.append(text(bx + 120, ey + 66, "0 preemptions", t["faint"], 11.5))

    s.append(arrow(W / 2, y + bh + 54, W / 2, ey - 6, t["dim"], None, None, 1.8))
    s.append(text(W / 2 + 10, y + bh + 74, "admitted", t["good"], 11.5, 600))

    # stream back
    sy = 372
    s.append(rect(28, sy, W - 56, 58, t["panel"], t["line"]))
    s.append(text(48, sy + 24, "streamed back token by token", t["ink"], 13, 600))
    s.append(text(48, sy + 43,
                  "slot held for the WHOLE stream · usage recorded in a finally block, "
                  "so a disconnect still bills", t["dim"], 11.5))
    s.append(arrow(W / 2, ey + 96 + 4, W / 2, sy - 6, t["dim"], None, None, 1.8))
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------------------
# 2. The degradation chart - the repository's headline evidence
#    Numbers read from the result files, so the chart cannot drift from them.
# ---------------------------------------------------------------------------

def load_curve():
    """(users, p95_ms, shed_pct, over_slo) for the shipped queue-timeout config."""
    pts = []
    for name, users in [("baseline_10users_qt06", 10), ("qt06_capacity_20users", 20),
                        ("qt06_capacity_30users", 30), ("qt06_capacity_40users", 40),
                        ("qt06_capacity_60users", 60)]:
        f = RESULTS / f"{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        pts.append((users,
                    d["ttft_ms_all_requests"]["p95"],
                    d["totals"]["shed_rate_pct"],
                    d["ttft_ms_per_user_p95"]["users_breaching_slo"],
                    d["ttft_ms_per_user_p95"]["users_total"]))
    return pts


def diagram_degradation(t):
    pts = load_curve()
    W, H = 900, 480
    L, R, TOP, BOT = 74, 700, 92, 372          # plot box
    SLO = 1500
    YMAX = 1700
    s = [svg_open(W, H, t)]

    s.append(text(28, 34, "Offered load rises 6x. Latency does not.", t["ink"], 17, 650))
    s.append(text(28, 55,
                  "p95 time-to-first-token, as the client sees it (queue wait included). "
                  "Excess load is refused, not queued.", t["dim"], 12.5))

    def px(u):   return L + (u - 10) / 50 * (R - L)
    def py(ms):  return BOT - (ms / YMAX) * (BOT - TOP)

    # y gridlines
    for ms in (0, 500, 1000, 1500):
        y = py(ms)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{t["line"]}" '
                 f'stroke-width="1"{" stroke-dasharray=\'4 4\'" if ms else ""}/>')
        s.append(text(L - 12, y + 4, f"{ms:,}", t["faint"], 11, 400, "end", MONO))
    s.append(text(L - 12, py(YMAX) + 4, "ms", t["faint"], 11, 400, "end"))

    # SLO band
    s.append(rect(L, TOP, R - L, py(SLO) - TOP, t["bad"], None, 0))
    s.append(f'<rect x="{L}" y="{TOP}" width="{R-L}" height="{py(SLO)-TOP:.1f}" '
             f'fill="{t["bad"]}" opacity="0.06"/>')
    s.append(f'<line x1="{L}" y1="{py(SLO):.1f}" x2="{R}" y2="{py(SLO):.1f}" '
             f'stroke="{t["bad"]}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    s.append(text(R - 6, py(SLO) - 9, "SLO  p95 < 1,500 ms", t["bad"], 11.5, 600, "end"))

    # the measured line
    poly = " ".join(f"{px(u):.1f},{py(p):.1f}" for u, p, *_ in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{t["good"]}" stroke-width="2.6" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    area = f"{L},{BOT} " + poly + f" {px(pts[-1][0]):.1f},{BOT}"
    s.append(f'<polygon points="{area}" fill="{t["good"]}" opacity="0.09"/>')

    for u, p, shed, over, total in pts:
        x, y = px(u), py(p)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{t["bg"]}" '
                 f'stroke="{t["good"]}" stroke-width="2.4"/>')
        s.append(text(x, y - 15, f"{p:.0f}", t["good"], 12, 650, "middle", MONO))
        s.append(text(x, BOT + 22, str(u), t["ink"], 12.5, 600, "middle", MONO))
        s.append(text(x, BOT + 40, f"{shed:.0f}% shed", t["faint"], 10.5, 400, "middle"))
        s.append(text(x, BOT + 56, f"{over}/{total} over", t["good"] if over == 0 else t["bad"],
                      10.5, 600, "middle"))
    s.append(text((L + R) / 2, BOT + 82, "simulated concurrent users", t["dim"], 12, 500, "middle"))

    # axes
    s.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}" stroke="{t["line"]}" stroke-width="1.4"/>')
    s.append(f'<line x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}" stroke="{t["line"]}" stroke-width="1.4"/>')

    # side panel: the tuning result
    PX = R + 32
    s.append(rect(PX, TOP, W - PX - 28, 200, t["panel"], t["line"]))
    s.append(text(PX + 16, TOP + 26, "Queue timeout decides", t["ink"], 12.5, 650))
    s.append(text(PX + 16, TOP + 44, "where the flat line sits", t["ink"], 12.5, 650))
    rows = [("2.0 s", "2,070 ms", "49/58 over", t["bad"]),
            ("0.6 s", "778 ms", "0/58 over", t["good"])]
    yy = TOP + 74
    for label, val, note, col in rows:
        s.append(text(PX + 16, yy, label, t["dim"], 11.5, 500, "start", MONO))
        s.append(text(PX + 62, yy, val, col, 12.5, 650, "start", MONO))
        s.append(text(PX + 16, yy + 17, note, col, 11, 500))
        yy += 48
    s.append(text(PX + 16, TOP + 178, "worst TTFT ≈ queue", t["faint"], 10.5))
    s.append(text(PX + 16, TOP + 192, "timeout + engine TTFT", t["faint"], 10.5))

    # footnote
    s.append(text(28, H - 22,
                  "chat_sim.py · Poisson arrivals · log-normal think time (median 12 s) · "
                  "6-turn conversations · seed 42 · heat-soaked card",
                  t["faint"], 10.5))
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------------------
# 3. The three numbers people call "concurrency"
# ---------------------------------------------------------------------------

def diagram_concurrency(t):
    W, H = 900, 330
    s = [svg_open(W, H, t)]
    s.append(text(28, 34, 'Three numbers are called "concurrency". Only one is a limit.',
                  t["ink"], 17, 650))
    s.append(text(28, 55, "Confusing them is the fastest way to a wrong capacity model.",
                  t["dim"], 12.5))

    cards = [
        ("6", "requests on the GPU at once", t["good"],
         ["THE limit. Set by the gateway.", "Measured: p95 TTFT holds at 6,",
          "fails by 36% at 12.", "", "Cap fixed; what fits is dynamic —",
          "a 20k prompt costs 4 of the 6."]),
        ("32", "--max-num-seqs", t["accent"],
         ["The engine's own batch ceiling.", "Deliberately above 6 so it can",
          "never be the binding limit.", "", "Raising it 8 → 32 bought +87%",
          "peak throughput and 0% SLO gain."]),
        ("2.13x", 'vLLM\'s "maximum concurrency"', t["faint"],
         ["Not a limit at all.", "How many FULL-WINDOW sequences",
          "the 69,760-token pool holds.", "", "Blocks are allocated 16 at a time,",
          "so ~95 real chat sequences fit."]),
    ]
    cw, gap2 = 272, 20
    for i, (big, sub, col, lines) in enumerate(cards):
        x = 28 + i * (cw + gap2)
        s.append(rect(x, 84, cw, 216, t["panel"], t["line"]))
        s.append(f'<rect x="{x}" y="84" width="4" height="216" rx="2" fill="{col}"/>')
        s.append(text(x + 22, 128, big, col, 34, 700, "start", MONO))
        s.append(text(x + 22, 150, sub, t["ink"], 12, 600))
        yy = 178
        for ln in lines:
            if ln:
                s.append(text(x + 22, yy, ln, t["dim"], 11.3))
            yy += 17
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------------------

DIAGRAMS = {
    "request-lifecycle": diagram_lifecycle,
    "degradation": diagram_degradation,
    "concurrency": diagram_concurrency,
}

if __name__ == "__main__":
    for name, fn in DIAGRAMS.items():
        for theme, t in THEMES.items():
            path = OUT / f"{name}-{theme}.svg"
            path.write_text(fn(t), encoding="utf-8")
            print(f"  wrote {path.relative_to(OUT.parent.parent)}  ({path.stat().st_size:,} bytes)")
