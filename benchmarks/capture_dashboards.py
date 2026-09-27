"""
Capture the Grafana dashboards as PNGs, for docs/img/.

    python benchmarks/capture_dashboards.py --label under-load --minutes 15

Requires the renderer, which lives behind its own compose profile because it is
~400 MB of headless Chromium that nobody serving traffic should have to pull:

    docker compose --profile observability --profile capture up -d

TWO THINGS THAT BROKE, BOTH WORTH KNOWING
------------------------------------------
1. The renderer reported Etc/Unknown as its timezone, which Grafana's bundled
   moment-timezone has no data for. It threw on every render and returned the
   "renderer unavailable" placeholder - a 13 KB PNG that looks like a failure
   only if you check the size. Fixed with TZ/BROWSER_TZ on the renderer and
   `"timezone": "utc"` in the dashboards.

2. The /render/d-solo/ route (single panel) crashes in Grafana 11.5 with
   "Cannot read properties of undefined (reading 'keys')" - the dashboardScene
   feature toggles are not compatible with solo rendering. The full-dashboard
   route /render/d/ works. So these are whole-dashboard captures.

A placeholder is ~13 KB and a real render is >60 KB, which is what the size
check below is for: a silent placeholder in the docs would be worse than no
image at all.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import urllib.error
import urllib.request

GRAFANA = "http://localhost:3000"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"
MIN_REAL_BYTES = 60_000

# Heights are set so every panel fits: a capture that cuts off mid-panel reads as
# broken even when the data is fine. The engine dashboard has more rows.
DASHBOARDS = {
    "grafana-capacity": ("gateway-capacity", 1400, 1720),
    "grafana-engine": ("vllm-engine", 1400, 2080),
}


def render(uid: str, width: int, height: int, window: tuple[str, str], theme: str) -> bytes:
    frm, to = window
    url = (f"{GRAFANA}/render/d/{uid}/x?width={width}&height={height}"
           f"&from={frm}&to={to}&theme={theme}&kiosk")
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="suffix, e.g. 'under-load'")
    ap.add_argument("--minutes", type=int, default=15, help="relative window: the last N minutes")
    ap.add_argument("--start", type=int, help="absolute window start, epoch seconds")
    ap.add_argument("--end", type=int, help="absolute window end, epoch seconds")
    ap.add_argument("--themes", default="light,dark")
    args = ap.parse_args()

    # An ABSOLUTE window is what makes a capture reproducible. "The last 20
    # minutes" silently dropped the whole capacity sweep from the first
    # capture, because the sweep had finished more than 20 minutes earlier.
    if args.start and args.end:
        window = (str(args.start * 1000), str(args.end * 1000))
    else:
        window = (f"now-{args.minutes}m", "now")

    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, (uid, w, h) in DASHBOARDS.items():
        for theme in args.themes.split(","):
            stem = f"{name}-{args.label}-{theme}" if args.label else f"{name}-{theme}"
            path = OUT / f"{stem}.png"
            try:
                data = render(uid, w, h, window, theme)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  FAILED {stem}: {exc}")
                failures += 1
                continue
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                print(f"  FAILED {stem}: not a PNG")
                failures += 1
                continue
            pw, ph = struct.unpack(">II", data[16:24])
            if len(data) < MIN_REAL_BYTES:
                print(f"  PLACEHOLDER {stem}: {len(data):,} bytes - renderer not working")
                failures += 1
                continue
            path.write_bytes(data)
            print(f"  wrote {path.name:44} {len(data):>8,} bytes  {pw}x{ph}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
