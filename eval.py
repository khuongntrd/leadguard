#!/usr/bin/env python3
"""Measure accuracy + latency of classify.py against labeled messages.

  ./eval.py                                   # samples.jsonl (hand-written, no PII)
  ./eval.py --leads leads.csv -n 200
      # balanced sample of real leads; Stage == "Spam" is spam, everything else
      # (Unread/Archived/Contacted) is valid_language. Those labels are noisy:
      # some genuine enquiries were marked Spam by hand, so read the misses.

Writes per-message results to eval-results.jsonl (gitignored; may hold PII).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify import classify

HERE = Path(__file__).parent


def load_samples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_leads(path: Path, n: int, seed: int) -> list[dict]:
    csv.field_size_limit(10**8)
    spam, valid = [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            msg = (row.get("Message") or "").strip()
            if not msg:
                continue
            (spam if row.get("Stage") == "Spam" else valid).append({"message": msg})
    rng = random.Random(seed)
    k = n // 2
    picked = [dict(r, expected="spam") for r in rng.sample(spam, min(k, len(spam)))]
    picked += [dict(r, expected="valid_language") for r in rng.sample(valid, min(n - k, len(valid)))]
    rng.shuffle(picked)
    return picked


def run_one(item: dict) -> dict:
    try:
        r = classify(item["message"])
        return {**item, "label": r["label"], "confidence": r["confidence"], "reason": r["reason"],
                "latency_s": r["_meta"]["latency_s"]}
    except Exception as e:  # noqa: BLE001 - POC: record and keep going
        return {**item, "label": None, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=HERE / "samples.jsonl")
    ap.add_argument("--leads", type=Path, help="CSV export with Message and Stage columns")
    ap.add_argument("-n", type=int, default=100, help="lead sample size (balanced)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-j", "--jobs", type=int, default=2, help="concurrent requests (match serve.sh PARALLEL)")
    ap.add_argument("--out", type=Path, default=HERE / "eval-results.jsonl")
    args = ap.parse_args()

    items = load_leads(args.leads, args.n, args.seed) if args.leads else load_samples(args.samples)
    print(f"evaluating {len(items)} messages ...", file=sys.stderr)

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_one, items))

    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["label"]]
    errors = len(results) - len(ok)
    tp = sum(r["expected"] == "spam" and r["label"] == "spam" for r in ok)
    fp = sum(r["expected"] != "spam" and r["label"] == "spam" for r in ok)
    fn = sum(r["expected"] == "spam" and r["label"] != "spam" for r in ok)
    tn = len(ok) - tp - fp - fn
    acc = (tp + tn) / len(ok) if ok else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    lat = sorted(r["latency_s"] for r in ok)

    print(f"accuracy   {acc:.3f}  ({tp + tn}/{len(ok)}, errors {errors})")
    print(f"spam       precision {prec:.3f}  recall {rec:.3f}")
    print(f"confusion  TP {tp}  FP {fp}  FN {fn}  TN {tn}   (positive = spam)")
    if lat:
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"latency    median {statistics.median(lat):.2f}s  p95 {p95:.2f}s  (per request, {args.jobs} concurrent)")
    print("\nmisses:")
    for r in ok:
        if r["label"] != r["expected"]:
            print(f"  [{r['expected']} -> {r['label']} {r['confidence']:.2f}] {r['message'][:90]!r}  // {r['reason']}")
    print(f"\nfull results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
