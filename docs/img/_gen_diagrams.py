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
# 0. Architecture - the components and what talks to what
# ---------------------------------------------------------------------------

def diagram_architecture(t):
    W, H = 900, 480
    s = [svg_open(W, H, t)]
    s.append(text(28, 34, "One GPU, one engine, one gateway that decides what reaches it",
                  t["ink"], 17, 650))
    s.append(text(28, 55, "Every request crosses the gateway; the engine never sees load it cannot serve.",
                  t["dim"], 12.5))

    def module(x, y, w, title, sub, col):
        out = [rect(x, y, w, 40, t["bg"], t["line"], 6),
               f'<rect x="{x}" y="{y}" width="3" height="40" rx="1.5" fill="{col}"/>',
               text(x + 12, y + 17, title, t["ink"], 12, 620),
               text(x + 12, y + 32, sub, t["dim"], 10.5)]
        return "".join(out)

    top, bot = 84, 318

    # clients
    s.append(rect(28, top, 140, bot - top, t["panel"], t["line"]))
    s.append(text(44, top + 26, "Clients", t["ink"], 13.5, 650))
    for i, (a, b) in enumerate([("Chat UI", "/chat, streaming"),
                                ("OpenAI SDK", "any client, base_url"),
                                ("chat_sim.py", "retrying load"),
                                ("failure tests", "14 injected cases")]):
        y = top + 50 + i * 44
        s.append(text(44, y, a, t["ink"], 12, 600))
        s.append(text(44, y + 15, b, t["faint"], 10.5))

    # gateway
    gx, gw = 200, 300
    s.append(rect(gx, top, gw, bot - top, t["panel"], t["line"]))
    s.append(text(gx + 16, top + 26, "Gateway", t["ink"], 13.5, 650))
    s.append(text(gx + gw - 16, top + 26, "FastAPI :8080", t["faint"], 11, 400, "end", MONO))
    mods = [("Auth and budgets", "API keys · token budgets · 429", t["warn"]),
            ("Context policy", "trim oldest turns · exact-count 413", t["warn"]),
            ("Admission control", "10 slots · uncached-token cost · fair share", t["bad"]),
            ("Streaming proxy", "billed in finally · 300 s stream cap", t["accent"])]
    for i, (a, b, col) in enumerate(mods):
        s.append(module(gx + 14, top + 44 + i * 47, gw - 28, a, b, col))

    # engine
    ex, ew = 580, 292
    s.append(rect(ex, top, ew, bot - top, t["panel2"], t["line"]))
    s.append(text(ex + 16, top + 26, "vLLM 0.11", t["gpu"], 13.5, 650))
    s.append(text(ex + ew - 16, top + 26, "V1 engine :8000", t["faint"], 11, 400, "end", MONO))
    emods = [("Qwen2.5-1.5B-Instruct AWQ", "Marlin kernels · 1.10 GiB weights"),
             ("Scheduler", "chunked prefill 2,048 · long cap 512"),
             ("Paged KV cache", "69,760-token pool · prefix caching"),
             ("RTX 3050 Laptop", "4 GiB · 35 W · 32k context")]
    for i, (a, b) in enumerate(emods):
        s.append(module(ex + 14, top + 44 + i * 47, ew - 28, a, b, t["gpu"]))

    # request path
    mid = top + 120
    s.append(arrow(168 + 4, mid, gx - 4, mid, t["dim"], "HTTP", None, 1.8))
    s.append(arrow(gx + gw + 4, mid - 12, ex - 4, mid - 12, t["good"], "admitted", None, 1.8))
    s.append(arrow(gx + gw + 4, mid + 34, ex - 4, mid + 34, t["faint"], "/tokenize", "4 3", 1.4))

    # bottom row
    by, bh = 356, 72
    boxes = [
        (gx, 160, "SQLite ledger", "WAL · usage and budgets", "breaker fails closed", t["warn"]),
        (390, 180, "Prometheus", ":9090 · scrapes both", "10 predictive alerts", t["accent"]),
        (600, 272, "Grafana", ":3000 · two dashboards", "engine + admission view", t["accent"]),
    ]
    for x, w, a, b, c, col in boxes:
        s.append(rect(x, by, w, bh, t["panel"], t["line"]))
        s.append(f'<rect x="{x}" y="{by}" width="{w}" height="3" rx="1.5" fill="{col}"/>')
        s.append(text(x + 14, by + 25, a, t["ink"], 12.5, 620))
        s.append(text(x + 14, by + 43, b, t["dim"], 10.5))
        s.append(text(x + 14, by + 59, c, t["faint"], 10.5))

    s.append(arrow(gx + 80, bot + 4, gx + 80, by - 4, t["dim"]))
    px = 480
    s.append(arrow(px, by - 4, gx + gw - 60, bot + 4, t["faint"], None, "4 3", 1.3))
    s.append(arrow(px + 40, by - 4, ex + 60, bot + 4, t["faint"], None, "4 3", 1.3))
    s.append(text(px + 20, by - 14, "scrape /metrics", t["faint"], 10.5, 500, "middle"))
    s.append(arrow(570 + 4, by + bh / 2, 600 - 4, by + bh / 2, t["dim"]))

    s.append(text(28, H - 18, "docker compose: vllm + gateway, with Prometheus and Grafana "
                  "behind the observability profile", t["faint"], 10.5))
    s.append("</svg>")
    return "".join(s)


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
    s.append(text(48, ey + 70, "at most 10 requests decoding at once", t["ink"], 12, 600))
    s.append(text(48 + 222, ey + 70, "— the gateway's limit, not the engine's",
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
    """Per-level results from the heat-soaked, control-validated evidence sweep.

    Returns dicts with users, attainment, first-try share, users within SLO,
    admitted p95 and user-perceived p95.

    WHY SLO ATTAINMENT AND NOT p95 OF ADMITTED REQUESTS. Latency measured only
    on requests the gateway admitted is the engine's view: a gateway that
    refused everyone would score perfectly. Clients retry a 503, so every
    refusal costs its user at least the Retry-After wait. Attainment counts a
    message as good only if its first token arrived within the SLO of its FIRST
    send, retries included - the user's view.
    """
    pts = []
    for users in (10, 20, 25, 30, 40, 60):
        f = RESULTS / f"evidence_capacity_{users}users.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        m, pp = d["messages"], d["per_user_perceived"]
        pts.append(dict(
            users=users,
            attain=m["slo_attainment_pct"],
            first=100 * m["served_first_try"] / max(1, m["sent"]),
            within=pp["users_within_slo"], total=pp["users_total"],
            admitted_p95=d["ttft_ms_all_requests"]["p95"],
            user_p95=d["ttft_ms_user_perceived"]["p95"],
        ))
    return pts


def diagram_degradation(t):
    pts = load_curve()
    W, H = 900, 500
    L, R, TOP, BOT = 74, 640, 92, 372
    s = [svg_open(W, H, t)]
    if not pts:
        s.append(text(28, 34, "No evidence runs found", t["ink"], 17, 650))
        s.append("</svg>")
        return "".join(s)

    ok90 = [p["users"] for p in pts if p["attain"] >= 90]
    head = (f"Up to {max(ok90)} users, 9 in 10 messages start within 1.5 s, retries included."
            if ok90 else "Share of messages that started within 1.5 s, retries included.")
    s.append(text(28, 34, head, t["ink"], 17, 650))
    s.append(text(28, 55,
                  "Measured from each message's FIRST send: a refused message is retried, "
                  "and the wait counts against it.", t["dim"], 12.5))

    # Evenly spaced levels, not a proportional axis: 20/25/30 would otherwise
    # sit on top of each other exactly where the knee is.
    idx = {p["users"]: i for i, p in enumerate(pts)}

    def px(u):  return L + 20 + idx[u] / max(1, len(pts) - 1) * (R - L - 40)
    def py(v):  return BOT - (v / 100) * (BOT - TOP)

    for v in (0, 25, 50, 75, 90, 100):
        y = py(v)
        dash = ' stroke-dasharray="4 4"' if v else ""
        col = t["good"] if v == 90 else t["line"]
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{col}" '
                 f'stroke-width="{1.4 if v == 90 else 1}"{dash}/>')
        s.append(text(L - 12, y + 4, f"{v}%", t["faint"], 11, 400, "end", MONO))
    s.append(text(R - 6, py(90) - 8, "90% attainment target", t["good"], 11, 600, "end"))

    first = " ".join(f"{px(p['users']):.1f},{py(p['first']):.1f}" for p in pts)
    att = " ".join(f"{px(p['users']):.1f},{py(p['attain']):.1f}" for p in pts)
    s.append(f'<polyline points="{first}" fill="none" stroke="{t["accent"]}" '
             f'stroke-width="1.8" stroke-dasharray="5 4"/>')
    s.append(f'<polyline points="{att}" fill="none" stroke="{t["good"]}" stroke-width="2.8" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')

    for p in pts:
        x = px(p["users"])
        s.append(f'<circle cx="{x:.1f}" cy="{py(p["first"]):.1f}" r="3.5" fill="{t["accent"]}"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{py(p["attain"]):.1f}" r="5" fill="{t["bg"]}" '
                 f'stroke="{t["good"]}" stroke-width="2.4"/>')
        # Label on whichever side of the point is away from the dashed line.
        below = p["attain"] < p["first"] - 1
        s.append(text(x, py(p["attain"]) + (20 if below else -12), f"{p['attain']:.0f}%",
                      t["good"], 11.5, 650, "middle", MONO))
        s.append(text(x, BOT + 24, str(p["users"]), t["ink"], 12.5, 600, "middle", MONO))
        s.append(text(x, BOT + 48, f"{p['within']}/{p['total']}",
                      t["good"] if p["within"] == p["total"] else t["dim"], 11.5, 600,
                      "middle", MONO))

    # Row captions, read like a table: the percentages above count MESSAGES,
    # the fractions in the second row count PEOPLE. Unlabelled, "35%" next to
    # "0/40" read as a contradiction.
    cap_x = R + 28
    s.append(text(cap_x, BOT + 24, "← simulated chat users", t["dim"], 11.5, 500))
    s.append(text(cap_x, BOT + 48, "← users with every message on time", t["dim"], 11.5, 500))
    ymid = (TOP + BOT) / 2
    s.append(f'<text x="22" y="{ymid:.0f}" fill="{t["dim"]}" font-family="{FONT}" font-size="11.5" '
             f'font-weight="500" text-anchor="middle" transform="rotate(-90 22 {ymid:.0f})">'
             f'share of messages</text>')
    s.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}" stroke="{t["line"]}" stroke-width="1.4"/>')
    s.append(f'<line x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}" stroke="{t["line"]}" stroke-width="1.4"/>')

    PX = R + 28
    s.append(rect(PX, TOP, W - PX - 24, 92, t["panel"], t["line"]))
    s.append(f'<line x1="{PX+14}" y1="{TOP+22}" x2="{PX+40}" y2="{TOP+22}" '
             f'stroke="{t["good"]}" stroke-width="2.8"/>')
    s.append(text(PX + 48, TOP + 26, "within 1.5 s of first send", t["ink"], 11.5, 600))
    s.append(f'<line x1="{PX+14}" y1="{TOP+46}" x2="{PX+40}" y2="{TOP+46}" '
             f'stroke="{t["accent"]}" stroke-width="1.8" stroke-dasharray="5 4"/>')
    s.append(text(PX + 48, TOP + 50, "served on the first try", t["ink"], 11.5, 600))
    s.append(text(PX + 14, TOP + 76, "% of all messages sent, not of users", t["faint"], 10.5))

    # The lesson: the engine's view and the user's view of the same run.
    # The first level below target is where the two views part company most
    # usefully: the engine still looks healthy, the users do not.
    mid = next((p for p in pts if p["attain"] < 90), pts[-1])
    s.append(rect(PX, TOP + 106, W - PX - 24, 174, t["panel"], t["line"]))
    s.append(text(PX + 14, TOP + 130, f"Same run, {mid['users']} users:", t["ink"], 12, 650))
    s.append(text(PX + 14, TOP + 156, "admitted requests", t["dim"], 11))
    s.append(text(PX + 14, TOP + 174, f"p95 {mid['admitted_p95']:,.0f} ms", t["good"], 13, 650,
                  "start", MONO))
    s.append(text(PX + 14, TOP + 200, "what users waited", t["dim"], 11))
    s.append(text(PX + 14, TOP + 218, f"p95 {mid['user_p95']:,.0f} ms", t["bad"], 13, 650,
                  "start", MONO))
    s.append(text(PX + 14, TOP + 244, "Refusals look free from", t["faint"], 10.5))
    s.append(text(PX + 14, TOP + 260, "the engine. They are not.", t["faint"], 10.5))

    s.append(text(28, H - 22,
                  "chat_sim.py - Poisson arrivals - log-normal think time - 6-message conversations - "
                  "retries honour Retry-After - seed 42 - heat-soaked, control-validated",
                  t["faint"], 10.5))
    s.append("</svg>")
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
        ("10", "requests on the GPU at once", t["good"],
         ["THE limit. Set by the gateway.", "Measured on chat, which hits the",
          "prefix cache; cold prompts held 6.", "", "Cap fixed; what fits is dynamic —",
          "a cold 20k prompt costs 8 of 10."]),
        ("32", "--max-num-seqs", t["accent"],
         ["The engine's own batch ceiling.", "Deliberately above 10 so it can",
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
    "architecture": diagram_architecture,
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
