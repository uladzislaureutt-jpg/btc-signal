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
        strategy_version="0.3.4",
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
    early = config.get("early_short", {})
    early_long = config.get("early_long", early)
    early_short_enabled = bool(early.get("enabled", True))
    early_long_enabled = bool(early_long.get("enabled", True))
    early_short_min_breakout_atr = float(early.get("min_breakout_atr", 0.25))
    early_close_bars = int(early.get("close_breakout_bars", 6))
    early_min_close_breakout_atr = float(early.get("min_close_breakout_atr", 0.10))
    early_momentum_bars = int(early.get("momentum_bars", 3))
    early_min_momentum_atr = float(early.get("min_momentum_atr", 0.75))
    early_min_lower_closes = int(early.get("min_lower_closes", 2))
    early_long_min_breakout_atr = float(early_long.get("min_breakout_atr", 0.25))
    early_long_close_bars = int(early_long.get("close_breakout_bars", 6))
    early_long_min_close_breakout_atr = float(early_long.get("min_close_breakout_atr", 0.10))
    early_long_momentum_bars = int(early_long.get("momentum_bars", 3))
    early_long_min_momentum_atr = float(early_long.get("min_momentum_atr", 0.75))
    early_long_min_higher_closes = int(early_long.get("min_higher_closes", 2))

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
    frame["prior_close_low"] = frame["close"].shift(1).rolling(early_close_bars).min()
    frame["prior_close_high"] = frame["close"].shift(1).rolling(early_long_close_bars).max()
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
        "prior_close_low": float(latest["prior_close_low"]),
        "prior_close_high": float(latest["prior_close_high"]),
    }
    base_factors: dict[str, dict[str, object]] = {
        "breakout": {
            "close": values["close"],
            "prior_high": values["prior_high"],
            "prior_low": values["prior_low"],
            "prior_close_low": values["prior_close_low"],
            "prior_close_high": values["prior_close_high"],
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

    downside_breakout_atr = (values["prior_low"] - values["close"]) / values["atr"]
    close_breakout_atr = (values["prior_close_low"] - values["close"]) / values["atr"]
    momentum_reference = float(frame["close"].iloc[-1 - early_momentum_bars])
    momentum_atr = (momentum_reference - values["close"]) / values["atr"]
    recent_closes = frame["close"].iloc[-1 - early_momentum_bars :].astype(float)
    lower_closes = int((recent_closes.diff().dropna() < 0).sum())
    wick_breakdown = (
        values["close"] < values["prior_low"]
        and downside_breakout_atr >= early_short_min_breakout_atr
    )
    close_breakdown = (
        values["close"] < values["prior_close_low"]
        and close_breakout_atr >= early_min_close_breakout_atr
    )
    momentum_breakdown = (
        momentum_atr >= early_min_momentum_atr
        and lower_closes >= early_min_lower_closes
    )
    early_short_signal = (
        early_short_enabled
        and not short_signal
        and (wick_breakdown or close_breakdown or momentum_breakdown)
    )
    base_factors["early_short"] = {
        "enabled": early_short_enabled,
        "wick_breakout_atr": downside_breakout_atr,
        "minimum_wick_breakout_atr": early_short_min_breakout_atr,
        "close_breakout_atr": close_breakout_atr,
        "minimum_close_breakout_atr": early_min_close_breakout_atr,
        "close_breakout_bars": early_close_bars,
        "momentum_atr": momentum_atr,
        "minimum_momentum_atr": early_min_momentum_atr,
        "momentum_bars": early_momentum_bars,
        "lower_closes": lower_closes,
        "minimum_lower_closes": early_min_lower_closes,
        "wick_breakdown": wick_breakdown,
        "close_breakdown": close_breakdown,
        "momentum_breakdown": momentum_breakdown,
        "above_ema200": values["close"] >= values["ema"],
    }
    upside_breakout_atr = (values["close"] - values["prior_high"]) / values["atr"]
    close_up_breakout_atr = (values["close"] - values["prior_close_high"]) / values["atr"]
    long_momentum_reference = float(frame["close"].iloc[-1 - early_long_momentum_bars])
    long_momentum_atr = (values["close"] - long_momentum_reference) / values["atr"]
    recent_long_closes = frame["close"].iloc[-1 - early_long_momentum_bars :].astype(float)
    higher_closes = int((recent_long_closes.diff().dropna() > 0).sum())
    wick_up_breakout = (
        values["close"] > values["prior_high"]
        and upside_breakout_atr >= early_long_min_breakout_atr
    )
    close_up_breakout = (
        values["close"] > values["prior_close_high"]
        and close_up_breakout_atr >= early_long_min_close_breakout_atr
    )
    momentum_up_breakout = (
        long_momentum_atr >= early_long_min_momentum_atr
        and higher_closes >= early_long_min_higher_closes
    )
    early_long_signal = (
        early_long_enabled
        and values["close"] >= values["ema"]
        and not long_signal
        and (wick_up_breakout or close_up_breakout or momentum_up_breakout)
    )
    base_factors["early_long"] = {
        "enabled": early_long_enabled,
        "wick_breakout_atr": upside_breakout_atr,
        "minimum_wick_breakout_atr": early_long_min_breakout_atr,
        "close_breakout_atr": close_up_breakout_atr,
        "minimum_close_breakout_atr": early_long_min_close_breakout_atr,
        "close_breakout_bars": early_long_close_bars,
        "momentum_atr": long_momentum_atr,
        "minimum_momentum_atr": early_long_min_momentum_atr,
        "momentum_bars": early_long_momentum_bars,
        "higher_closes": higher_closes,
        "minimum_higher_closes": early_long_min_higher_closes,
        "wick_breakout": wick_up_breakout,
        "close_breakout": close_up_breakout,
        "momentum_breakout": momentum_up_breakout,
        "above_ema200": values["close"] >= values["ema"],
    }
    if not long_signal and not short_signal and not early_short_signal and not early_long_signal:
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

    side = "long" if (long_signal or early_long_signal) else "short"
    if early_long_signal:
        if close_up_breakout:
            signal_type = "early_close_breakout_long"
        elif momentum_up_breakout:
            signal_type = "early_momentum_breakout_long"
        else:
            signal_type = "early_wick_breakout_long"
    elif early_short_signal:
        if close_breakdown:
            signal_type = "early_close_breakdown"
        elif momentum_breakdown:
            signal_type = "early_momentum_breakdown"
        else:
            signal_type = "early_wick_breakdown"
    else:
        signal_type = "trend_breakout"
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
    if early_long_signal and close_up_breakout:
        channel = values["prior_close_high"]
    elif early_long_signal and momentum_up_breakout:
        channel = long_momentum_reference
    elif side == "long":
        channel = values["prior_high"]
    elif early_short_signal and close_breakdown:
        channel = values["prior_close_low"]
    elif early_short_signal and momentum_breakdown:
        channel = momentum_reference
    else:
        channel = values["prior_low"]
    level_type = "early_long" if early_long_signal else ("early_short" if early_short_signal else "donchian_breakout")
    factors["signal_type"] = {"value": signal_type}
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
        regime=(
            "early_long_above_ema200" if early_long_signal
            else (
                ("early_short_above_ema200" if values["close"] >= values["ema"] else "early_short_below_ema200")
                if early_short_signal
                else ("above_ema200" if side == "long" else "below_ema200")
            )
        ),
        level={"type": level_type, "bars": breakout_bars, "price": channel},
        factors=factors,
        vetoes=[],
        provider=provider,
        strategy_version="0.3.4",
    )
