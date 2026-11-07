#!/usr/bin/env python3
"""Concurrent load test for POST /leads/classify.

  ./loadtest.py                              # levels 1,2,4,8 against localhost:8000
  ./loadtest.py -c 1,4,16 -n 24 --container leadguard --json out.json

For each concurrency level it keeps C requests in flight until N have completed, using
leads from samples.jsonl round-robin, and reports throughput, latency percentiles, errors
and (with --container) peak CPU/RAM sampled from `docker stats`. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent


def leads() -> list[dict]:
    rows = [json.loads(x) for x in (HERE / "samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [{"email": "lead@example.com", "phone": "0901234567", "message": r["message"]} for r in rows]


def call(url: str, body: dict, timeout: float) -> tuple[float, int, str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
            ok = set(out) == {"label", "confidence", "reason"} and out["label"] in ("spam", "valid_language")
            return time.perf_counter() - t0, r.status, "" if ok else f"bad body {out}"
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, e.code, e.read().decode()[:120]
    except Exception as e:  # noqa: BLE001
        return time.perf_counter() - t0, 0, f"{type(e).__name__}: {e}"


class StatsSampler(threading.Thread):
    """Samples `docker stats` for peak CPU% and memory while a level runs."""

    def __init__(self, container: str | None):
        super().__init__(daemon=True)
        self.container, self.stop, self.cpu, self.mem = container, threading.Event(), [], []

    def run(self) -> None:
        while self.container and not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}};{{.MemUsage}}", self.container],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                cpu, mem = out.split(";")
                self.cpu.append(float(cpu.rstrip("%")))
                used = mem.split("/")[0].strip()
                num = float(used.rstrip("GiBMKk"))
                self.mem.append(num * 1024 if "GiB" in used else num)
            except Exception:  # noqa: BLE001
                pass


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def run_level(url: str, conc: int, n: int, timeout: float, container: str | None) -> dict:
    pool_leads = leads()
    bodies = [pool_leads[i % len(pool_leads)] for i in range(n)]
    sampler = StatsSampler(container)
    sampler.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        results = list(ex.map(lambda b: call(url, b, timeout), bodies))
    wall = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join(timeout=12)

    lat = [r[0] for r in results if r[1] == 200 and not r[2]]
    errs = [r for r in results if r[1] != 200 or r[2]]
    return {
        "concurrency": conc,
        "requests": n,
        "ok": len(lat),
        "errors": len(errs),
        "error_samples": sorted({f"{e[1]} {e[2]}" for e in errs})[:3],
        "wall_s": round(wall, 1),
        "throughput_rps": round(len(lat) / wall, 3),
        "leads_per_min": round(len(lat) / wall * 60, 1),
        "p50_s": round(statistics.median(lat), 2) if lat else None,
        "p95_s": round(pct(lat, 0.95), 2) if lat else None,
        "max_s": round(max(lat), 2) if lat else None,
        "peak_cpu_pct": max(sampler.cpu) if sampler.cpu else None,
        "peak_mem_mib": round(max(sampler.mem)) if sampler.mem else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/leads/classify")
    ap.add_argument("-c", "--concurrency", default="1,2,4,8")
    ap.add_argument("-n", "--requests", type=int, default=0, help="per level (default max(8, 2*C))")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--container", help="docker container name for CPU/RAM sampling")
    ap.add_argument("--json", type=Path, help="write results as JSON")
    args = ap.parse_args()

    call(args.url, leads()[0], args.timeout)  # warm-up (prompt cache, first-request budget probe)

    rows = []
    print(f"{'conc':>4} {'ok':>4} {'err':>4} {'wall_s':>7} {'req/s':>6} {'leads/min':>9} "
          f"{'p50_s':>6} {'p95_s':>6} {'max_s':>6} {'cpu%':>6} {'mem_MiB':>7}")
    for c in [int(x) for x in args.concurrency.split(",")]:
        n = args.requests or max(8, 2 * c)
        r = run_level(args.url, c, n, args.timeout, args.container)
        rows.append(r)
        print(f"{c:>4} {r['ok']:>4} {r['errors']:>4} {r['wall_s']:>7} {r['throughput_rps']:>6} "
              f"{r['leads_per_min']:>9} {r['p50_s']!s:>6} {r['p95_s']!s:>6} {r['max_s']!s:>6} "
              f"{r['peak_cpu_pct']!s:>6} {r['peak_mem_mib']!s:>7}", flush=True)
        for e in r["error_samples"]:
            print(f"      error: {e}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
