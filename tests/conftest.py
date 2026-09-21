from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    with (ROOT / "config" / "strategy.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candles(periods: int = 500, freq: str = "1h", trend: float = 2.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    base = 50_000 + np.arange(periods) * trend + np.sin(np.arange(periods) / 4) * 300
    frame = pd.DataFrame(index=index)
    frame["open"] = base
    frame["high"] = base + 250
    frame["low"] = base - 250
    frame["close"] = base + 30
    frame["volume"] = 100 + np.cos(np.arange(periods) / 5) * 10
    delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    frame["close_time"] = index + delta - pd.Timedelta(milliseconds=1)
    return frame

