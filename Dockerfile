# BitNet b1.58 2B-4T lead spam classifier: llama-server (bitnet.cpp) + FastAPI in one container.
#   docker build -t leadguard .
#   docker run -d -p 8000:8000 --cpus 4 --memory 2g --restart unless-stopped leadguard
#
# BITNET_REF is pinned to 765741d (2025-05-21). Newer BitNet main (>= 2025-11 kernel rewrite,
# 2026-07 llama.cpp bump) produces garbage with the published April-2025 GGUF.
# Portable build (GGML_NATIVE=OFF): amd64 needs AVX2, arm64 needs dotprod
# (Graviton2+, Ampere Altra, Apple M*). Ubuntu 24.04 for clang 18 (bitnet.cpp needs >= 18).

ARG BITNET_REF=765741d80bda4d3069e367bf0062134891232cce

FROM ubuntu:24.04 AS build
ARG BITNET_REF
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git clang cmake make python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q BitNet && cd BitNet \
    && git remote add origin https://github.com/microsoft/BitNet.git \
    && git fetch -q --depth 1 origin "$BITNET_REF" && git checkout -q FETCH_HEAD \
    && git submodule update -q --init --recursive --depth 1
WORKDIR /src/BitNet
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      python3 utils/codegen_tl1.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 64,128,64 --bm 32,64,32; \
      FLAGS="-DBITNET_ARM_TL1=OFF"; CFLAGS_ARCH="-march=armv8.2-a+dotprod"; \
    else \
      python3 utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32; \
      FLAGS="-DBITNET_X86_TL2=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"; CFLAGS_ARCH=""; \
    fi \
    && cmake -B build $FLAGS -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
         -DCMAKE_C_FLAGS="$CFLAGS_ARCH" -DCMAKE_CXX_FLAGS="$CFLAGS_ARCH" \
         -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server

FROM ubuntu:24.04 AS model
ARG MODEL_URL=https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/29f884c2ae/ggml-model-i2_s.gguf
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 3 -o /model.gguf "$MODEL_URL" \
    && echo "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162  /model.gguf" | sha256sum -c -

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.2 langdetect==1.0.9 \
    && useradd -r -u 10001 app
ENV PATH=/opt/venv/bin:$PATH
COPY --from=build /src/BitNet/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=model /model.gguf /models/ggml-model-i2_s.gguf
WORKDIR /app
COPY classify.py rules.py api.py healthcheck.py entrypoint.sh ./
ENV THREADS=4 PARALLEL=2 CTX=4096 BITNET_URL=http://127.0.0.1:8080
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=8s --start-period=90s --retries=3 CMD ["python", "/app/healthcheck.py"]
ENTRYPOINT ["/app/entrypoint.sh"]
