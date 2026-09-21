import os

from fastapi.testclient import TestClient

from btc_signal.api import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_run_requires_secret():
    os.environ["CRON_SECRET"] = "expected"
    response = TestClient(app).post("/run", headers={"X-Cron-Secret": "wrong"})
    assert response.status_code == 401

