from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, ema
from .strategy import Decision


def _neutral(
    *,
    timestamp: str,
    price: float,
    provider: str | None,
    vetoes: list[str],
    factors: dict[str, dict[str, object]],
) -> Decision:
    return Decision(
        timestamp=timestamp,
        state="neutral",
        side=None,
        score=0,
        price=price,
        entry_low=None,
        entry_high=None,
        stop=None,
        target_1=None,
        target_2=None,
        rr=None,
        regime="no_breakout",
        level=None,
        factors=factors,
        vetoes=vetoes,
        provider=provider,
        strategy_version="0.3.1",
    )


def evaluate_trend_signal(
    frame_4h: pd.DataFrame,
    config: dict,
    provider: str | None,
    current_price: float,
    *,
    now: pd.Timestamp | None = None,
) -> Decision:
    """Create an operational signal from the frozen 0.3.0 trend rules.

    The breakout is evaluated on the latest completed 4H candle. The live
    ticker is the proposed market entry. TP1/TP2 are price landmarks; the
    tested exit is the ATR trailing stop recorded in ``factors.exit_plan``.
    """
    raw = config["trend_bidirectional"]
    breakout_bars = int(raw["breakout_bars"])
    ema_period = int(raw["ema_period"])
    atr_period = int(raw["atr_period"])
    stop_atr = float(raw["initial_stop_atr"])
    trail_atr = float(raw["trailing_atr"])
    activation_atr = float(raw["trailing_activation_atr"])
    cost_bps = float(raw["round_trip_cost_bps"])
    min_risk_to_cost = float(raw["min_risk_to_cost"])
    target_1_r = float(raw.get("reference_target_1_r", 2.0))
    target_2_r = float(raw.get("reference_target_2_r", 3.0))

    required = max(ema_period, breakout_bars, atr_period) + 2
    if len(frame_4h) < required:
        timestamp = pd.Timestamp.now(tz="UTC").isoformat()
        return _neutral(
            timestamp=timestamp,
            price=current_price,
            provider=provider,
            vetoes=["insufficient_4h_history"],
            factors={"history": {"value": len(frame_4h), "required": required}},
        )

    frame = frame_4h.copy().sort_index()
    frame["ema"] = ema(frame["close"], ema_period)
    frame["atr"] = atr(frame, atr_period)
    frame["prior_high"] = frame["high"].shift(1).rolling(breakout_bars).max()
    frame["prior_low"] = frame["low"].shift(1).rolling(breakout_bars).min()
    latest = frame.iloc[-1]
    open_time = pd.Timestamp(frame.index[-1])
    close_time = latest.get("close_time")
    signal_time = (
        pd.Timestamp(close_time)
        if close_time is not None and not pd.isna(close_time)
        else open_time + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1)
    )
    if signal_time.tzinfo is None:
        signal_time = signal_time.tz_localize("UTC")
    else:
        signal_time = signal_time.tz_convert("UTC")

    values = {
        "close": float(latest["close"]),
        "ema": float(latest["ema"]),
        "atr": float(latest["atr"]),
        "prior_high": float(latest["prior_high"]),
        "prior_low": float(latest["prior_low"]),
    }
    base_factors: dict[str, dict[str, object]] = {
        "breakout": {
            "close": values["close"],
            "prior_high": values["prior_high"],
            "prior_low": values["prior_low"],
            "bars": breakout_bars,
        },
        "trend": {"close": values["close"], "ema": values["ema"], "period": ema_period},
        "volatility": {"atr": values["atr"], "period": atr_period},
    }
    if not all(np.isfinite(value) for value in values.values()) or current_price <= 0:
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["indicators_not_ready"],
            factors=base_factors,
        )

    long_signal = values["close"] > values["prior_high"] and values["close"] > values["ema"]
    short_signal = values["close"] < values["prior_low"] and values["close"] < values["ema"]
    if not long_signal and not short_signal:
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["no_4h_trend_breakout"],
            factors=base_factors,
        )

    max_age = int(config["signal"].get("max_signal_age_minutes", 75))
    current_time = now or pd.Timestamp.now(tz="UTC")
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    if current_time - signal_time > pd.Timedelta(minutes=max_age):
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["stale_breakout"],
            factors=base_factors,
        )

    side = "long" if long_signal else "short"
    direction = 1.0 if side == "long" else -1.0
    risk = stop_atr * values["atr"]
    stop = current_price - direction * risk
    risk_pct = risk / current_price
    cost_pct = cost_bps / 10_000
    risk_to_cost = risk_pct / cost_pct if cost_pct > 0 else float("inf")
    if risk <= 0 or stop <= 0 or risk_to_cost < min_risk_to_cost:
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["risk_to_cost_below_minimum"],
            factors={
                **base_factors,
                "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
            },
        )

    target_1 = current_price + direction * target_1_r * risk
    target_2 = current_price + direction * target_2_r * risk
    activation = current_price + direction * activation_atr * values["atr"]
    factors = {
        **base_factors,
        "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
        "exit_plan": {
            "mode": "atr_trailing",
            "activation_price": activation,
            "activation_atr": activation_atr,
            "trailing_distance_atr": trail_atr,
            "target_1_is_reference": True,
            "target_2_is_reference": True,
        },
    }
    channel = values["prior_high"] if side == "long" else values["prior_low"]
    return Decision(
        timestamp=signal_time.isoformat(),
        state="signal",
        side=side,
        score=100,
        price=current_price,
        entry_low=current_price,
        entry_high=current_price,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        rr=target_2_r,
        regime="above_ema200" if side == "long" else "below_ema200",
        level={"type": "donchian_breakout", "bars": breakout_bars, "price": channel},
        factors=factors,
        vetoes=[],
        provider=provider,
        strategy_version="0.3.1",
    )
