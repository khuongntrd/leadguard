#!/usr/bin/env bash
# Starts bitnet.cpp llama-server with BitNet b1.58 2B-4T on 127.0.0.1:8080 (CPU only).
# Env: PORT (8080), THREADS (perf cores), PARALLEL (2 slots), CTX (4096 total).
set -euo pipefail
cd "$(dirname "$0")"
THREADS="${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)}"
exec vendor/BitNet/build/bin/llama-server \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  -t "$THREADS" -c "${CTX:-4096}" -np "${PARALLEL:-2}" \
  -ngl 0 --temp 0 --override-kv tokenizer.ggml.pre=str:llama-bpe
