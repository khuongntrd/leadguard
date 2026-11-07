#!/usr/bin/env bash
# Builds bitnet.cpp (llama-server with BitNet kernels) natively and downloads the
# pre-quantized BitNet b1.58 2B-4T GGUF. Mirrors vendor/BitNet/setup_env.py for
# `-md models/BitNet-b1.58-2B-4T -q i2_s` but skips its torch/transformers pip
# install, which is only needed to convert HF weights (we download the GGUF).
#
# BitNet is pinned to 765741d (2025-05-21): newer main produces garbage with the
# published GGUF (see README). Same pin as the Dockerfile.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
BITNET="$ROOT/vendor/BitNet"
BITNET_REF=765741d80bda4d3069e367bf0062134891232cce
MODEL_DIR="$ROOT/models/BitNet-b1.58-2B-4T"

if [ ! -d "$BITNET/.git" ]; then
  git init -q "$BITNET"
  git -C "$BITNET" remote add origin https://github.com/microsoft/BitNet.git
fi
if [ "$(git -C "$BITNET" rev-parse HEAD 2>/dev/null)" != "$BITNET_REF" ]; then
  git -C "$BITNET" fetch -q --depth 1 origin "$BITNET_REF"
  git -C "$BITNET" checkout -q -f FETCH_HEAD
  git -C "$BITNET" submodule update -q --init --recursive --depth 1
  rm -rf "$BITNET/build"
fi

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q numpy fastapi==0.115.12 uvicorn==0.34.2 langdetect==1.0.9

ARCH="$(uname -m)"
cd "$BITNET"
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  "$ROOT/.venv/bin/python" utils/codegen_tl1.py --model bitnet_b1_58-3B \
    --BM 160,320,320 --BK 64,128,64 --bm 32,64,32
  EXTRA="-DBITNET_ARM_TL1=OFF"
else
  "$ROOT/.venv/bin/python" utils/codegen_tl2.py --model bitnet_b1_58-3B \
    --BM 160,320,320 --BK 96,96,96 --bm 32,32,32
  EXTRA="-DBITNET_X86_TL2=OFF"
fi

# CMAKE_POLICY_VERSION_MINIMUM: the pinned llama.cpp declares cmake < 3.5, which CMake 4 rejects.
cmake -B build $EXTRA -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release -j --target llama-server

mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL_DIR/ggml-model-i2_s.gguf" ]; then
  huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf \
    --revision 29f884c2ae --local-dir "$MODEL_DIR"
fi

echo "OK: $BITNET/build/bin/llama-server + $MODEL_DIR/ggml-model-i2_s.gguf"
