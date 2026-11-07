# leadguard

Classifies a lead/contact message as `valid_language` or `spam` with
[BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) running on CPU
through [bitnet.cpp](https://github.com/microsoft/BitNet). The goal is to see whether a ~1.2 GB
ternary model is good and fast enough to pre-filter contact-form lead spam without a GPU.

```
classify.py / eval.py ──HTTP──► llama-server (bitnet.cpp, i2_s kernels) ──► ggml-model-i2_s.gguf
                       /v1/chat/completions + response_format json_schema
```

## Setup (once)

Needs clang, cmake, python3 and `huggingface-cli`.

```bash
./setup.sh       # clones bitnet.cpp into vendor/, builds llama-server, downloads the GGUF (~1.2 GB)
```

`setup.sh` follows the upstream `setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s` steps
but skips its torch/transformers install, which is only needed to convert HF weights.

## Run

```bash
./serve.sh                                    # 127.0.0.1:8080; THREADS / PARALLEL / CTX env overrides
./classify.py "Congratulations, you won an iPhone, pay 500k shipping"
echo "Cho em xin bảng giá căn 2PN" | ./classify.py --raw
./eval.py                                     # samples.jsonl: 12 hand-written vi/en cases
./eval.py --leads leads.csv -n 200                # your CSV export: Message + Stage columns
./loadtest.py -c 1,2,4,8 --container leadguard   # against the Docker API
```

Output:

```json
{ "label": "spam", "confidence": 0.95, "reason": "..." }
```

## Docker API

One container, one exposed port. llama-server is an internal process on `127.0.0.1:8080`.

```bash
docker build -t leadguard .            # builds bitnet.cpp + bakes the GGUF in (2.45 GB image)
docker run -d --name leadguard -p 8000:8000 --cpus 4 --memory 2g --restart unless-stopped leadguard

curl -s localhost:8000/leads/classify -H 'content-type: application/json' -d '{
  "email": "promo@cheap-seo.example", "phone": "+18005550100",
  "message": "We can rank your site #1 on Google in 7 days, guaranteed!"}'
# {"label":"spam","confidence":1.0,"reason":"The message is an unsolicited offer ..."}
```

The body is exactly `{label, confidence, reason}`. Diagnostics go in headers:
`X-Truncated`, `X-Message-Tokens`, `X-Latency-Ms`. Errors: 422 bad input, 503 model down,
502 bad model output. Env vars: `THREADS` (4), `PARALLEL` (2 slots), `CTX` (4096 total,
2048 per slot). amd64 needs AVX2, arm64 needs dotprod.

**Health:** `GET /health` → `{"status","api","llama_server"}`, 200 only when both are up.
`GET /live` checks the API process only. The entrypoint runs a watchdog: if either process
exits, or `healthcheck.py` fails 3× in a row (every 10 s, so a hang is caught too), it stops
both and exits 1, and `--restart` / k8s brings the container back. Docker `HEALTHCHECK` uses
the same script.

**Context cut:** on first request the API reads the per-slot `n_ctx` from llama-server `/props`
and measures the prompt template with `/tokenize`. Message budget =
`n_ctx − template − 120 output − 48 safety` (~1.8k tokens at the defaults). A longer lead
is cut **by tokens**: first 75% + `[...truncated...]` + last 25% (links/CTAs usually sit at
the end), and `X-Truncated: true` is set.

**Version pin:** BitNet is pinned to `765741d` (2025-05-21). Current `main` (Nov-2025 I2_S
kernel rewrite, Jul-2026 llama.cpp bump) loads the published GGUF but outputs garbage
(`????`, `@@@@`) at ~2 tok/s, and doesn't compile on macOS. The server runs with
`--override-kv tokenizer.ggml.pre=str:llama-bpe` because the GGUF lacks pre-tokenizer metadata
(LLaMA-3 tokenizer); that raised the sample eval from 9/12 to 10/12.

## Releases → GHCR

Publishing a GitHub release runs `.github/workflows/release.yml`. It builds linux/amd64 and
linux/arm64 on native runners and pushes one multi-arch image to
`ghcr.io/khuongntrd/leadguard`:

| Release tag | Image tags |
|---|---|
| `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` |
| `v0.4.0` | `0.4.0`, `0.4`, `latest` (no `0` major tag while in 0.x) |
| `v1.3.0-rc.1` (pre-release) | `1.3.0-rc.1` only |

Tags that are not `vX.Y.Z[-pre]` fail the run before building.

```bash
gh release create v0.1.0 --generate-notes          # stable
gh release create v0.2.0-rc.1 --prerelease         # pre-release
docker run -d -p 8000:8000 --cpus 4 --memory 2g ghcr.io/khuongntrd/leadguard:0.1.0
```

The first build takes ~20–30 min per arch (bitnet.cpp compile + 1.2 GB model). Later builds
reuse the GitHub Actions layer cache. GHCR packages start **private** even for a public repo,
so after the first release set the package to public under
*Package settings → Change visibility* so `docker pull` works without logging in.

## Accuracy (2026-11-07)

The CSV `Stage` labels are not usable as ground truth: the "not Spam" side is mostly untriaged
`Unread` leads, and ~90% of those are bot spam. I hand-reviewed two balanced samples
(`data/reviewed-leads-seed{42,123}.jsonl`, gitignored, contains customer text). Seed 123 was
not looked at while tuning, so it is the honest number. Each sample holds only 2–6 real leads,
so the false-positive rate is not well measured yet.

| Pipeline | seed 42 (100) | holdout seed 123 (60) | spam precision | spam recall |
|---|---|---|---|---|
| spec prompt only (`PROMPT_VERSION=spec USE_RULES=0`) | 82% | 88% | 98.7% / 100% | 82% / 88% |
| **rules + v2 prompt (default)** | **94%** | **95%** | 98.9% / 100% | 94.7% / 94.8% |

- **`rules.py`** (runs first, ~0 ms) catches random-character posts, non-Latin scripts and a
  bot that posts "I wanted to know your price" in dozens of languages. Guards keep accent-less
  Vietnamese, CamelCase names and Title Case project names ("Masteri Centre Point 2PN") out.
  CJK/Korean go to the model.
- **v2 prompt** = the spec rules plus domain context and full-JSON few-shot examples, with
  `reason` generated before `label`. A JSON skeleton with placeholders made the model copy
  "one short sentence" in 42% of answers, so fields are described in prose instead.
- Still missed: the English "Hi, I wanted to know your price." bot, Vietnamese dating-spam
  `rb.gy` links, Galician/Portuguese price posts.

## Benchmark (measured 2026-11-07, Docker Desktop on Apple M-series, arm64)

`./loadtest.py -c 1,2,4,8 --container leadguard` sends the 12 sample leads round-robin,
keeping C requests in flight. Zero errors in every run.

**Current build** (rules + v2 + prefix cache), 4 CPU, `THREADS=4 PARALLEL=2`:

| Concurrency | Leads/min | p50 | p95 |
|---|---|---|---|
| 1 | 23.9 | 2.4 s | 3.7 s |
| 2 | 19.9 | 3.9 s | 13.2 s |
| 4 | 32.2 | 6.8 s | 8.6 s |
| 8 | 34.9 | 13.2 s | 14.6 s |

Peak memory ~1 GiB. The API calls `/completion` with `cache_prompt` and the exact BitNet chat
format, so each slot keeps the fixed system + instructions + examples prefix in its KV
cache. Per lead only ~10–30 new tokens are processed (the first request on a slot pays the
full ~470 tokens, about 5 s). Rule hits skip the model entirely.

**Earlier build** (spec prompt via `/v1/chat/completions`, no prefix cache):

| Config | Concurrency | Leads/min | p50 | p95 | Peak RAM |
|---|---|---|---|---|---|
| 4 CPU, `THREADS=4 PARALLEL=2` (default) | 1 | 12.6 | 5.1 s | 6.2 s | 505 MiB |
| | 2 | 12.2 | 10.0 s | 13.5 s | 507 MiB |
| | 4 | 12.0 | 19.8 s | 25.5 s | 507 MiB |
| | 8 | 13.4 | 32.8 s | 37.5 s | 508 MiB |
| 4 CPU, `THREADS=4 PARALLEL=1` | 1 | 13.6 | 3.6 s | 6.6 s | 287 MiB |
| | 4 | 11.5 | 18.5 s | 23.9 s | 288 MiB |
| 8 CPU, `THREADS=8 PARALLEL=4` | 1 | 7.3 | 5.1 s | 18.9 s | 744 MiB |
| | 4 | 6.9 | 33.7 s | 62.0 s | 768 MiB |

A 6.4k-token message cut to 1.8k tokens took 51 s.

**What it means:**
- **It handles concurrent requests, but it doesn't speed up with them.** One request
  already saturates 4 cores, so throughput is flat at ~12–13 leads/min (~0.2 req/s) and
  extra requests queue: latency ≈ concurrency × ~4–5 s.
- **More slots don't add throughput on CPU.** `PARALLEL=2` vs 1 is the same leads/min; it
  only lets two requests interleave and adds KV memory.
- **More threads made it slower here** (8 threads: 7 leads/min). Apple chips mix P and E cores
  and llama.cpp threads wait for the slowest, so this is host-specific; on uniform server cores
  (EPYC/Graviton) 8 threads should help. Re-measure on the target node.
- **Scale with replicas, not slots or threads:** each 4-CPU replica handles ~12 leads/min
  with the earlier build and ~30 leads/min with the current one (≈ 45k leads/day). For bursts, put a queue in front instead of holding HTTP requests 30 s+.
- **Prompt processing dominated** (~80% of the time), which is why the prefix cache gave a
  ~2.6× gain. Long messages still cost ~10–30 ms/token.
- Requests abandoned by the client keep running in llama-server until they finish.

## Sizing

| | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU (AVX2 / ARM dotprod) | 4 vCPU, `THREADS=4` |
| RAM | 1 GiB | 1.5 GiB request, 2 GiB limit |

Measured process memory was 290–510 MiB (1–2 slots). The 1.2 GB weights are mmapped, so
they sit in page cache rather than process RSS. Leave room for them, or cold pages get
re-read from disk under memory pressure.

## How it works

- `rules.py` first; if no rule fires, the v2 prompt (or `PROMPT_VERSION=spec` for the spec
  prompt verbatim) with the lead in the `"""` fence (`"""` in the message is replaced so it
  can't close the fence). Email and phone go in the fence too.
- `json_schema` makes llama-server compile the schema to a grammar, so the
  model can only emit `label` ∈ {`valid_language`, `spam`}, a number `confidence` and a
  `reason`. No retry/repair parsing needed.
- `temperature: 0` for repeatable results.

## Caveats

- **`confidence` is self-reported** by a 2B model and is not calibrated. Don't threshold on
  it without checking `eval.py` output first; token logprobs would be a better signal.
- **Lead labels are noisy** in both directions: `Stage == "Spam"` includes genuine enquiries,
  and `Unread` is mostly spam. Use the reviewed sets, not raw `Stage`.
- **Borderline leads can flip between a cold and a cached prefix** (slightly different
  numerics), even at temperature 0. Seen once in the smoke test: a valid Vietnamese
  price request was labelled spam on the very first request after a restart.
- **The language rule blocks real leads in other languages** (not Vietnamese/English/CJK).
  Fine for current sites; revisit if a site targets other foreign buyers.
- BitNet 2B-4T was trained mostly on English; Vietnamese quality is the main thing to measure.
- The leads CSV and `eval-results.jsonl` hold customer PII — keep them local.
