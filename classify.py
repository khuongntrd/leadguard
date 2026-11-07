#!/usr/bin/env python3
"""Classify a lead (email / phone / message) as valid_language or spam with BitNet b1.58 2B-4T.

Talks to a local bitnet.cpp llama-server over its /completion endpoint (prefix-cached).
Output is constrained by a JSON schema (grammar), so the model can only emit the
three fields with the two allowed labels. Messages longer than the model's
per-slot context are cut by tokens (head + tail kept) before sending.

Usage:
  ./classify.py "Your account has been suspended, verify at http://x.co"
  ./classify.py --email a@b.com --phone 0901234567 "Cho em xin bảng giá"
  echo "Tôi muốn xem căn 2PN" | ./classify.py --raw
Stdlib only, no pip install needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

import rules

SERVER = os.environ.get("BITNET_URL", "http://127.0.0.1:8080")
MAX_OUTPUT_TOKENS = 120
SAFETY_TOKENS = 48       # chat-template wrapper tokens not counted by /tokenize, plus slack
HEAD_SHARE = 0.75        # when cutting, keep 75% from the start, 25% from the end (links/CTAs)
PRE_CUT_CHARS = 32_000   # cheap char cap before tokenizing huge payloads
CUT_MARKER = "\n[...truncated...]\n"

SPEC_PROMPT = """Classify the user message.

Return JSON only:
{{
  "label": "valid_language" | "spam",
  "confidence": 0.0-1.0,
  "reason": "one short sentence"
}}

Spam includes scams, phishing, unsolicited ads, deceptive offers,
and manipulative requests. Do not classify ordinary questions,
support requests, or informal language as spam.

Message:
\"\"\"
{message}
\"\"\""""

# v2: the spec rules plus domain context and few-shot examples. A 2B model given only the
# spec prompt labelled ~78% of real leads as spam (50% accuracy on 100 balanced leads).
V2_SYSTEM = (
    "You screen messages sent through the contact form of Vietnamese real-estate websites. "
    "Most messages come from real home buyers, renters or investors. They are often very short "
    "and informal, in Vietnamese (sometimes without accents, with abbreviations like e, a, ko, PN, "
    "can ho) or English. A real lead is valid even if it only asks for the price, a call back, "
    "more information, or a viewing."
)

V2_PROMPT = """Classify the user message.

Answer with one JSON object with these fields, in this order:
- reason: one short sentence that explains your decision for this message
- label: "valid_language" or "spam"
- confidence: a number from 0.0 to 1.0

Spam includes scams, phishing, unsolicited ads, deceptive offers,
and manipulative requests. Do not classify ordinary questions,
support requests, or informal language as spam.

Spam on these websites is usually: someone selling to the website owner (SEO, marketing,
loan offers, crypto, betting, affiliate programs), suspicious links, random characters that
are not words, or a bot that posts the same one-line "I wanted to know your price" in many
languages. Real leads write in Vietnamese or English about a property: price, payment,
bank loan support, location, handover, a viewing or a call back. A message that is not in
Vietnamese or English is almost always that bot: label it spam.

Message: "cho e xin bang gia can 2PN"
{{"reason": "A buyer asks for the price list of a two-bedroom unit.", "label": "valid_language", "confidence": 0.95}}

Message: "Hello, I am interested in [Masteri Centre Point]"
{{"reason": "A visitor shows interest in a named project through the contact form.", "label": "valid_language", "confidence": 0.9}}

Message: "Can ho nay co ho tro vay ngan hang khong a?"
{{"reason": "A buyer asks whether a bank loan is available for the apartment.", "label": "valid_language", "confidence": 0.95}}

Message: "Goi lai cho toi sau 6h nhe"
{{"reason": "The lead asks to be called back after 6 pm.", "label": "valid_language", "confidence": 0.95}}

Message: "We can boost your website to Google page 1, reply for a free audit"
{{"reason": "An unsolicited SEO service offer to the site owner.", "label": "spam", "confidence": 0.95}}

Message: "Hæ, ég vildi vita verð þitt."
{{"reason": "A generic price question in Icelandic, a known multilingual bot template.", "label": "spam", "confidence": 0.85}}

Message: "kKEUPBJlj"
{{"reason": "Random characters, not a real message.", "label": "spam", "confidence": 0.9}}

Message:
\"\"\"
{message}
\"\"\""""

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")
if PROMPT_VERSION == "spec":
    SYSTEM, PROMPT, FIELD_ORDER = None, SPEC_PROMPT, ["label", "confidence", "reason"]
else:
    SYSTEM, PROMPT, FIELD_ORDER = V2_SYSTEM, V2_PROMPT, ["reason", "label", "confidence"]

_FIELDS = {
    "label": {"type": "string", "enum": ["valid_language", "spam"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "reason": {"type": "string", "maxLength": 200},
}
# Property order is generation order: v2 makes the model explain before it decides.
SCHEMA = {
    "type": "object",
    "properties": {k: _FIELDS[k] for k in FIELD_ORDER},
    "required": FIELD_ORDER,
    "additionalProperties": False,
}


def render_prompt(text: str) -> str:
    """BitNet chat format, byte-identical to llama-server's built-in "bitnet" template
    (BOS is added by the server). Built here so /completion can reuse the KV cache for the
    fixed system + instructions prefix: only the lead's own tokens are processed per request.
    """
    system = f"System: {SYSTEM}" if SYSTEM else ""
    return f"{system}User: {PROMPT.format(message=text)}<|eot_id|>Assistant: "


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{SERVER}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _get(path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=timeout) as resp:
        return json.load(resp)


def _tokenize(text: str) -> list[int]:
    return _post("/tokenize", {"content": text, "add_special": False})["tokens"]


def _detokenize(tokens: list[int]) -> str:
    return _post("/detokenize", {"tokens": tokens})["content"] if tokens else ""


_budget_lock = threading.Lock()
_budget: int | None = None


def message_token_budget() -> int:
    """Tokens left for the lead text: per-slot n_ctx minus prompt, output and safety."""
    global _budget
    with _budget_lock:
        if _budget is None:
            props = _get("/props")
            n_ctx = int(props["default_generation_settings"]["n_ctx"])  # per slot (-c / -np)
            overhead = len(_tokenize(render_prompt("")))
            _budget = n_ctx - overhead - MAX_OUTPUT_TOKENS - SAFETY_TOKENS
            if _budget < 64:
                raise RuntimeError(f"context too small: n_ctx={n_ctx}, overhead={overhead}")
        return _budget


def fit_to_context(text: str) -> tuple[str, int, bool]:
    """Return (text, token_count, truncated). Cuts the middle when over budget."""
    text = text[:PRE_CUT_CHARS]
    budget = message_token_budget()
    tokens = _tokenize(text)
    if len(tokens) <= budget:
        return text, len(tokens), False
    keep = budget - len(_tokenize(CUT_MARKER))
    head = int(keep * HEAD_SHARE)
    tail = keep - head
    cut = _detokenize(tokens[:head]) + CUT_MARKER + _detokenize(tokens[-tail:])
    return cut, budget, True


def lead_text(message: str, email: str | None = None, phone: str | None = None) -> str:
    parts = []
    if email:
        parts.append(f"Email: {email.strip()}")
    if phone:
        parts.append(f"Phone: {phone.strip()}")
    parts.append(f"Message: {message.strip()}" if parts else message.strip())
    # Keep user text from closing the """ fence and steering the prompt.
    return "\n".join(parts).replace('"""', "”””")


USE_RULES = os.environ.get("USE_RULES", "1") != "0"


def classify_lead(message: str, email: str | None = None, phone: str | None = None) -> dict:
    if USE_RULES and (hit := rules.check(message)):
        return {"label": "spam", "confidence": 0.95, "reason": hit["reason"],
                "_meta": {"latency_s": 0.0, "message_tokens": 0, "truncated": False,
                          "rule": hit["rule"], "usage": None, "raw": None}}
    text, n_tokens, truncated = fit_to_context(lead_text(message, email, phone))
    payload = {
        "prompt": render_prompt(text),
        "temperature": 0,
        "n_predict": MAX_OUTPUT_TOKENS,
        "json_schema": SCHEMA,
        "cache_prompt": True,  # reuse the shared prefix already in the slot's KV cache
    }
    t0 = time.perf_counter()
    data = _post("/completion", payload)
    elapsed = time.perf_counter() - t0

    raw = data["content"]
    result = json.loads(raw)
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    result["_meta"] = {
        "latency_s": round(elapsed, 3),
        "message_tokens": n_tokens,
        "truncated": truncated,
        "rule": None,
        "usage": {k: data.get("timings", {}).get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")},
        "raw": raw,
    }
    return result


def classify(message: str) -> dict:
    return classify_lead(message)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="?", help="message text (default: stdin)")
    ap.add_argument("--email")
    ap.add_argument("--phone")
    ap.add_argument("--raw", action="store_true", help="include raw output, latency, token usage")
    args = ap.parse_args()

    message = args.message if args.message is not None else sys.stdin.read()
    if not message.strip():
        ap.error("empty message")

    result = classify_lead(message, args.email, args.phone)
    meta = result.pop("_meta")
    if args.raw:
        result["_meta"] = meta
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
