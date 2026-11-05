from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Protocol

import requests

from .strategy import Decision


class Store(Protocol):
    def seen(self, dedup_key: str) -> bool: ...
    def cooldown_active(self, candle_time: str, hours: int) -> bool: ...
    def same_side_episode_active(self, side: str, candle_time: str, hours: int) -> bool: ...
    def recent_signal_side(self, candle_time: str, hours: int) -> str | None: ...
    def most_recent_notified_signal(self, candle_time: str, hours: int) -> dict[str, str] | None: ...
    def save(self, decision: Decision, notified: bool) -> None: ...


class SQLiteStore:
    def __init__(self, path: str | Path = "data/btc_signal.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            create table if not exists signals (
                id integer primary key autoincrement,
                dedup_key text not null unique,
                candle_time text not null,
                state text not null,
                side text,
                score integer not null,
                price real not null,
                payload text not null,
                notified integer not null default 0,
                created_at text not null default current_timestamp
            )
            """
        )
        self.connection.commit()

    def seen(self, dedup_key: str) -> bool:
        row = self.connection.execute(
            "select 1 from signals where dedup_key = ?", (dedup_key,)
        ).fetchone()
        return row is not None

    def cooldown_active(self, candle_time: str, hours: int) -> bool:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        row = self.connection.execute(
            """
            select 1 from signals
            where state in ('signal', 'strong_signal') and candle_time >= ?
            limit 1
            """,
            (threshold.isoformat(),),
        ).fetchone()
        return row is not None

    def same_side_episode_active(self, side: str, candle_time: str, hours: int) -> bool:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        row = self.connection.execute(
            """
            select side from signals
            where state in ('signal', 'strong_signal') and candle_time >= ?
            order by candle_time desc
            limit 1
            """,
            (threshold.isoformat(),),
        ).fetchone()
        return bool(row and row[0] == side)

    def recent_signal_side(self, candle_time: str, hours: int) -> str | None:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        row = self.connection.execute(
            """
            select side from signals
            where state in ('signal', 'strong_signal') and candle_time >= ?
            order by candle_time desc
            limit 1
            """,
            (threshold.isoformat(),),
        ).fetchone()
        return row[0] if row and row[0] else None

    def most_recent_notified_signal(self, candle_time: str, hours: int) -> dict[str, str] | None:
        end = datetime.fromisoformat(candle_time).astimezone(timezone.utc)
        threshold = end - timedelta(hours=hours)
        row = self.connection.execute(
            """
            select side, candle_time from signals
            where state in ('signal', 'strong_signal') and notified = 1
              and candle_time >= ? and candle_time < ?
            order by candle_time desc limit 1
            """,
            (threshold.isoformat(), end.isoformat()),
        ).fetchone()
        return {"side": row[0], "candle_time": row[1]} if row else None

    def save(self, decision: Decision, notified: bool) -> None:
        self.connection.execute(
            """
            insert or ignore into signals
            (dedup_key, candle_time, state, side, score, price, payload, notified)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.dedup_key,
                decision.timestamp,
                decision.state,
                decision.side,
                decision.score,
                decision.price,
                json.dumps(decision.to_dict(), ensure_ascii=False),
                int(notified),
            ),
        )
        self.connection.commit()


class SupabaseStore:
    def __init__(self, url: str, secret_key: str, timeout: int = 20):
        self.endpoint = f"{url.rstrip('/')}/rest/v1/btc_signals"
        self.timeout = timeout
        self.headers = {
            "apikey": secret_key,
            "Content-Type": "application/json",
        }
        # New sb_secret_* keys are API keys, not JWTs. Legacy service-role keys
        # still use the Bearer header for compatibility.
        if not secret_key.startswith("sb_secret_"):
            self.headers["Authorization"] = f"Bearer {secret_key}"

    def seen(self, dedup_key: str) -> bool:
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params={"dedup_key": f"eq.{dedup_key}", "select": "id", "limit": 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return bool(response.json())

    def cooldown_active(self, candle_time: str, hours: int) -> bool:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params={
                "state": "in.(signal,strong_signal)",
                "candle_time": f"gte.{threshold.isoformat()}",
                "select": "id",
                "limit": 1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return bool(response.json())

    def same_side_episode_active(self, side: str, candle_time: str, hours: int) -> bool:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params={
                "state": "in.(signal,strong_signal)",
                "candle_time": f"gte.{threshold.isoformat()}",
                "select": "side,candle_time",
                "order": "candle_time.desc",
                "limit": 1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json()
        return bool(rows and rows[0].get("side") == side)

    def recent_signal_side(self, candle_time: str, hours: int) -> str | None:
        threshold = datetime.fromisoformat(candle_time).astimezone(timezone.utc) - timedelta(hours=hours)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params={
                "state": "in.(signal,strong_signal)",
                "candle_time": f"gte.{threshold.isoformat()}",
                "select": "side,candle_time",
                "order": "candle_time.desc",
                "limit": 1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json()
        return rows[0].get("side") if rows and rows[0].get("side") else None

    def most_recent_notified_signal(self, candle_time: str, hours: int) -> dict[str, str] | None:
        end = datetime.fromisoformat(candle_time).astimezone(timezone.utc)
        threshold = end - timedelta(hours=hours)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params={
                "state": "in.(signal,strong_signal)",
                "notified": "eq.true",
                "candle_time": f"gte.{threshold.isoformat()}",
                "and": f"(candle_time.lt.{end.isoformat()})",
                "select": "side,candle_time",
                "order": "candle_time.desc",
                "limit": 1,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json()
        return {"side": rows[0]["side"], "candle_time": rows[0]["candle_time"]} if rows else None

    def save(self, decision: Decision, notified: bool) -> None:
        payload = decision.to_dict()
        record = {
            "dedup_key": decision.dedup_key,
            "candle_time": decision.timestamp,
            "state": decision.state,
            "side": decision.side,
            "score": decision.score,
            "price": decision.price,
            "entry_low": decision.entry_low,
            "entry_high": decision.entry_high,
            "stop": decision.stop,
            "target_1": decision.target_1,
            "target_2": decision.target_2,
            "rr": decision.rr,
            "provider": decision.provider,
            "strategy_version": decision.strategy_version,
            "payload": payload,
            "notified": notified,
        }
        response = requests.post(
            self.endpoint,
            headers={**self.headers, "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json=record,
            timeout=self.timeout,
        )
        response.raise_for_status()


def build_store() -> Store:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        return SupabaseStore(url, key)
    return SQLiteStore()
