from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative))
    return result.fillna(100.0).where(avg_gain.ne(0), 0.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def enrich(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    params = config["indicators"]
    enriched = frame.copy()
    enriched["ema_fast"] = ema(enriched["close"], int(params["ema_fast"]))
    enriched["ema_slow"] = ema(enriched["close"], int(params["ema_slow"]))
    enriched["rsi"] = rsi(enriched["close"], int(params["rsi_period"]))
    enriched["atr"] = atr(enriched, int(params["atr_period"]))
    enriched["volume_median"] = enriched["volume"].rolling(
        int(params["volume_median_period"]), min_periods=5
    ).median()
    return enriched

