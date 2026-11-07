#!/usr/bin/env bash
# Runs llama-server (internal, 127.0.0.1:8080) and the API (0.0.0.0:8000) in one container.
# Watchdog: if either process exits, or either /health fails WATCHDOG_FAILS times in a row
# (hung, not just dead), stop both and exit 1 so Docker/k8s restarts the container.
set -uo pipefail

INTERVAL="${WATCHDOG_INTERVAL:-10}"
MAX_FAILS="${WATCHDOG_FAILS:-3}"

llama-server -m /models/ggml-model-i2_s.gguf \
  --host 127.0.0.1 --port 8080 \
  -t "$THREADS" -c "$CTX" -np "$PARALLEL" -ngl 0 --temp 0 --override-kv tokenizer.ggml.pre=str:llama-bpe &
LLAMA=$!

uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 &
API=$!

shutdown() {
  kill -TERM "$LLAMA" "$API" 2>/dev/null
  wait 2>/dev/null
  exit "${1:-0}"
}
trap 'shutdown 0' TERM INT

# Model load takes a few seconds; don't count failures until both are up once.
for _ in $(seq 1 180); do
  python /app/healthcheck.py >/dev/null && break
  kill -0 "$LLAMA" 2>/dev/null || { echo "watchdog: llama-server exited during startup" >&2; shutdown 1; }
  kill -0 "$API" 2>/dev/null || { echo "watchdog: api exited during startup" >&2; shutdown 1; }
  sleep 1
done
echo "watchdog: ready $(python /app/healthcheck.py)"

fails=0
while true; do
  sleep "$INTERVAL" &
  wait $!
  kill -0 "$LLAMA" 2>/dev/null || { echo "watchdog: llama-server exited" >&2; shutdown 1; }
  kill -0 "$API" 2>/dev/null || { echo "watchdog: api exited" >&2; shutdown 1; }
  if status=$(python /app/healthcheck.py); then
    fails=0
  else
    fails=$((fails + 1))
    echo "watchdog: unhealthy ($fails/$MAX_FAILS) $status" >&2
    [ "$fails" -ge "$MAX_FAILS" ] && shutdown 1
  fi
done
