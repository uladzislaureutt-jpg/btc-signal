from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Header, HTTPException

from .runner import run_once


app = FastAPI(title="BTC Signal Core", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(x_cron_secret: str | None = Header(default=None)) -> dict:
    expected = os.getenv("CRON_SECRET")
    if not expected or not x_cron_secret or not hmac.compare_digest(expected, x_cron_secret):
        raise HTTPException(status_code=401, detail="Invalid cron secret")
    result = run_once(dry_run=False)
    return {
        "decision": result.decision.to_dict(),
        "duplicate": result.duplicate,
        "notified": result.notified,
    }

