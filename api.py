"""HTTP API: POST /leads/classify {email?, phone?, message} -> {label, confidence, reason}.

The body is exactly the spec format; diagnostics go in response headers:
X-Truncated, X-Message-Tokens, X-Latency-Ms.

Run: uvicorn api:app --host 0.0.0.0 --port 8000   (needs llama-server on BITNET_URL)
"""

from __future__ import annotations

import json
import urllib.error
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import classify

app = FastAPI(title="BitNet lead spam classifier", version="0.1.0")


class Lead(BaseModel):
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=32)
    message: str = Field(min_length=1, max_length=100_000)


class Verdict(BaseModel):
    label: Literal["valid_language", "spam"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@app.post("/leads/classify", response_model=Verdict)
def classify_lead(lead: Lead, response: Response) -> Verdict:
    # Sync def: FastAPI runs it in its threadpool; llama-server queues across its -np slots.
    if not lead.message.strip():
        raise HTTPException(422, "message is blank")
    try:
        r = classify.classify_lead(lead.message, lead.email, lead.phone)
        verdict = Verdict(label=r["label"], confidence=round(r["confidence"], 2), reason=r["reason"].strip())
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(503, f"model server unavailable: {e}") from e
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        raise HTTPException(502, f"bad model output: {e}") from e
    meta = r["_meta"]
    response.headers["X-Truncated"] = str(meta["truncated"]).lower()
    response.headers["X-Message-Tokens"] = str(meta["message_tokens"])
    response.headers["X-Latency-Ms"] = str(int(meta["latency_s"] * 1000))
    return verdict


@app.get("/live")
def live() -> dict:
    """API process only (used by the watchdog; avoids a loop through llama-server)."""
    return {"status": "ok"}


@app.get("/health")
def health() -> JSONResponse:
    """Both components: 200 only when the API and llama-server are up."""
    try:
        classify._get("/health", timeout=3)
        llama = "ok"
    except Exception as e:  # noqa: BLE001
        llama = f"down: {type(e).__name__}"
    ok = llama == "ok"
    body = {"status": "ok" if ok else "degraded", "api": "ok", "llama_server": llama}
    return JSONResponse(body, status_code=200 if ok else 503)
