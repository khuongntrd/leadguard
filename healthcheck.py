#!/usr/bin/env python3
"""Probe llama-server and the API; exit 0 only if both answer /health with 200.

Used by Docker HEALTHCHECK and the entrypoint watchdog. Prints one JSON line.
"""

import json
import sys
import urllib.request

TARGETS = {
    "llama_server": "http://127.0.0.1:8080/health",
    "api": "http://127.0.0.1:8000/live",
}


def probe(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return "ok" if r.status == 200 else f"http {r.status}"
    except Exception as e:  # noqa: BLE001
        return f"down: {type(e).__name__}"


status = {name: probe(url) for name, url in TARGETS.items()}
print(json.dumps(status))
sys.exit(0 if all(v == "ok" for v in status.values()) else 1)
