from __future__ import annotations

import numpy as np
import pandas as pd

from . import __version__
from .indicators import atr, ema
from .strategy import Decision


def _fib_context(frame: pd.DataFrame, config: dict, price: float) -> dict[str, object]:
    raw = config.get("structure", {})
    lookback = int(raw.get("fib_lookback_bars", 180))
    ratios = [float(x) for x in raw.get("fib_ratios", [0.236, 0.382, 0.5, 0.618, 0.786, 1.0])]
    sample = frame.tail(lookback)
    low_idx = sample["low"].idxmin()
    high_idx = sample["high"].idxmax()
    low = float(sample.loc[low_idx, "low"])
    high = float(sample.loc[high_idx, "high"])
    span = high - low
    if span <= 0:
        return {"direction": "flat", "low": low, "high": high, "levels": []}
    direction = "up" if low_idx < high_idx else "down"
    if direction == "up":
        levels = [{"ratio": r, "price": high - span * r} for r in ratios]
        progress = (high - price) / span
    else:
        levels = [{"ratio": r, "price": low + span * r} for r in ratios]
        progress = (price - low) / span
    return {
        "direction": direction,
        "low": low,
        "high": high,
        "progress": progress,
        "levels": sorted(levels, key=lambda x: x["price"]),
    }


def assess_early_entry_stage(
    *, side: str, price: float, close: float, ema_value: float,
    atr_value: float, prior_20bar_extreme: float, prior_swing_extreme: float,
    maximum_ema_distance_atr: float, minimum_extreme_room_atr: float,
) -> tuple[dict[str, object], str | None]:
    """Causal EARLY entry quality gate, independently testable on live snapshots."""
    distance = abs(close - ema_value) / atr_value
    if side == "short":
        candidates = [x for x in (prior_20bar_extreme, prior_swing_extreme) if x < price]
        nearest = max(candidates) if candidates else None
        room = (price - nearest) / atr_value if nearest is not None else None
    else:
        candidates = [x for x in (prior_20bar_extreme, prior_swing_extreme) if x > price]
        nearest = min(candidates) if candidates else None
        room = (nearest - price) / atr_value if nearest is not None else None
    stage = {
        "ema_distance_atr": distance,
        "maximum_ema_distance_atr": maximum_ema_distance_atr,
        "prior_20bar_extreme": prior_20bar_extreme,
        "prior_swing_extreme": prior_swing_extreme,
        "nearest_extreme": nearest,
        "extreme_room_atr": room,
        "minimum_extreme_room_atr": minimum_extreme_room_atr,
    }
    if distance > maximum_ema_distance_atr:
        return stage, "early_trend_overextended"
    if room is not None and room < minimum_extreme_room_atr:
        return stage, "early_entry_near_prior_extreme"
    return stage, None



def assess_trend_breakout_quality(
    *, side: str, close: float, live_price: float, ema_value: float,
    atr_value: float, channel_level: float,
    extended_ema_distance_atr: float, extended_min_breakout_atr: float,
) -> tuple[dict[str, float], list[str]]:
    """Reject a reclaimed Donchian break or a late, negligible new extreme.

    Production-only entry validation: historical research logic is unchanged.
    """
    direction = 1.0 if side == "long" else -1.0
    close_depth_atr = direction * (close - channel_level) / atr_value
    live_depth_atr = direction * (live_price - channel_level) / atr_value
    ema_distance_atr = direction * (close - ema_value) / atr_value
    factors = {
        "channel_level": channel_level,
        "close_breakout_atr": close_depth_atr,
        "live_breakout_atr": live_depth_atr,
        "ema_distance_atr": ema_distance_atr,
        "extended_ema_distance_atr": extended_ema_distance_atr,
        "extended_min_breakout_atr": extended_min_breakout_atr,
    }
    vetoes: list[str] = []
    if live_depth_atr <= 0:
        vetoes.append("breakout_reclaimed_before_entry")
    if ema_distance_atr >= extended_ema_distance_atr and close_depth_atr < extended_min_breakout_atr:
        vetoes.append("overextended_shallow_breakout")
    return factors, vetoes

def assess_exceptional_entry(
    frame: pd.DataFrame, config: dict, *, side: str, price: float,
    close: float, ema_value: float, atr_value: float,
    prior_20bar_extreme: float, prior_swing_extreme: float,
    momentum_atr: float, directional_closes: int,
    close_breakout_atr: float, previous_ema: float,
) -> dict[str, object]:
    """Evaluate one-candle shock or fresh base breakout using only closed bars.

    These are separate operational entry routes, not modifications to the
    frozen 20-bar research engine. Protect both paths from late entries near
    old swing extremes and large price rebounds after candle close.
    """
    crash_cfg = config.get("crash_impulse", {})
    base_cfg = config.get("base_breakout", {})
    previous = frame.iloc[-2]
    latest = frame.iloc[-1]
    prev_close = float(previous["close"])
    candle_open = float(latest["open"])
    candle_high = float(latest["high"])
    candle_low = float(latest["low"])
    candle_range = candle_high - candle_low
    if side == "short":
        one_bar_change_atr = (prev_close - close) / atr_value
        body_atr = (candle_open - close) / atr_value
        close_location = (close - candle_low) / candle_range if candle_range > 0 else 0.5
        previous_extension_atr = (previous_ema - prev_close) / atr_value
        extrema_ahead = [x for x in (prior_20bar_extreme, prior_swing_extreme) if x < price]
        nearest_extreme = max(extrema_ahead) if extrema_ahead else None
        extreme_room_atr = (price - nearest_extreme) / atr_value if nearest_extreme is not None else None
        penetration_atr = max(0.0, (prior_20bar_extreme - price) / atr_value)
        ema_separation_atr = (ema_value - close) / atr_value
    else:
        one_bar_change_atr = (close - prev_close) / atr_value
        body_atr = (close - candle_open) / atr_value
        close_location = (candle_high - close) / candle_range if candle_range > 0 else 0.5
        previous_extension_atr = (prev_close - previous_ema) / atr_value
        extrema_ahead = [x for x in (prior_20bar_extreme, prior_swing_extreme) if x > price]
        nearest_extreme = min(extrema_ahead) if extrema_ahead else None
        extreme_room_atr = (nearest_extreme - price) / atr_value if nearest_extreme is not None else None
        penetration_atr = max(0.0, (price - prior_20bar_extreme) / atr_value)
        ema_separation_atr = (close - ema_value) / atr_value

    def safe_structure(cfg: dict) -> bool:
        minimum_room = float(cfg.get("min_extreme_room_atr", 1.0))
        maximum_penetration = float(cfg.get("max_extreme_penetration_atr", 0.75))
        max_entry_deviation = float(cfg.get("max_entry_deviation_atr", 0.60))
        return (
            (extreme_room_atr is None or extreme_room_atr >= minimum_room)
            and penetration_atr <= maximum_penetration
            and abs(price - close) / atr_value <= max_entry_deviation
        )

    width_bars = max(2, int(base_cfg.get("consolidation_bars", 6)))
    prior_closes = frame["close"].iloc[-1 - width_bars:-1]
    close_range_atr = (
        (float(prior_closes.max()) - float(prior_closes.min())) / atr_value
        if len(prior_closes) == width_bars else float("inf")
    )
    crash = (
        bool(crash_cfg.get("enabled", False))
        and one_bar_change_atr >= float(crash_cfg.get("min_single_close_move_atr", 2.0))
        and body_atr >= float(crash_cfg.get("min_candle_body_atr", 1.25))
        and close_location <= float(crash_cfg.get("max_close_location", 0.30))
        and previous_extension_atr <= float(crash_cfg.get("max_previous_ema_distance_atr", 3.0))
        and ema_separation_atr >= float(crash_cfg.get("min_ema_separation_atr", 0.25))
        and safe_structure(crash_cfg)
    )
    base_setup = (
        bool(base_cfg.get("enabled", False))
        and close_range_atr <= float(base_cfg.get("max_close_range_atr", 1.25))
        and close_breakout_atr >= float(base_cfg.get("min_close_breakout_atr", 0.80))
        and momentum_atr >= float(base_cfg.get("min_momentum_atr", 1.25))
        and directional_closes >= int(base_cfg.get("min_directional_closes", 2))
        and ema_separation_atr >= float(base_cfg.get("min_ema_separation_atr", 0.25))
    )
    return {
        "crash": bool(crash),
        "base": bool(base_setup and safe_structure(base_cfg)),
        "base_setup": bool(base_setup),
        "single_close_move_atr": one_bar_change_atr,
        "body_atr": body_atr,
        "close_location": close_location,
        "previous_ema_extension_atr": previous_extension_atr,
        "base_close_range_atr": close_range_atr,
        "close_breakout_atr": close_breakout_atr,
        "momentum_atr": momentum_atr,
        "directional_closes": directional_closes,
        "ema_separation_atr": ema_separation_atr,
        "nearest_extreme": nearest_extreme,
        "extreme_room_atr": extreme_room_atr,
        "prior_extreme_penetration_atr": penetration_atr,
        "entry_deviation_atr": abs(price - close) / atr_value,
    }


def episode_reset_status(
    frame_4h: pd.DataFrame,
    previous_candle_time: str,
    current_candle_time: str,
    side: str,
    atr_value: float,
    *,
    minimum_pullback_atr: float = 1.25,
    minimum_counter_closes: int = 2,
) -> dict[str, object]:
    """Causal reset after the most recent notified entry, using closed 4H candles only.

    Measures the pullback route to a same-direction episode reset.
    """
    frame = frame_4h.sort_index()
    if "close_time" in frame.columns:
        times = pd.to_datetime(frame["close_time"], utc=True)
    else:
        times = pd.to_datetime(frame.index, utc=True) + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1)
    start = pd.Timestamp(previous_candle_time)
    end = pd.Timestamp(current_candle_time)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    closes = frame.loc[(times >= start) & (times <= end), "close"].to_numpy(dtype=float)
    result = {
        "confirmed": False,
        "previous_candle_time": previous_candle_time,
        "observed_bars": int(len(closes)),
        "maximum_pullback_atr": 0.0,
        "minimum_pullback_atr": minimum_pullback_atr,
        "minimum_counter_closes": minimum_counter_closes,
    }
    if len(closes) < minimum_counter_closes + 2 or not np.isfinite(atr_value) or atr_value <= 0 or side not in ("long", "short"):
        return result
    extreme = float(closes[0])
    counter_count = 0
    for previous, close in zip(closes[:-1], closes[1:]):
        if side == "short":
            extreme = min(extreme, float(previous))
            counter_count = counter_count + 1 if close > previous else 0
            pullback = max(0.0, (close - extreme) / atr_value)
        else:
            extreme = max(extreme, float(previous))
            counter_count = counter_count + 1 if close < previous else 0
            pullback = max(0.0, (extreme - close) / atr_value)
        result["maximum_pullback_atr"] = max(float(result["maximum_pullback_atr"]), pullback)
        if counter_count >= minimum_counter_closes and pullback >= minimum_pullback_atr:
            result["confirmed"] = True
    return result


def _countertrend_watch(frame: pd.DataFrame, config: dict, factors: dict) -> dict | None:
    settings = config.get("countertrend_watch", {})
    if not settings.get("enabled", False):
        return None

    latest = frame.iloc[-1]
    side = "long" if latest["close"] < latest["ema"] else "short"
    direction = 1 if side == "long" else -1
    recent_changes = frame["close"].tail(5).diff().dropna() * direction
    # Alert on the first three directional closes, once per local swing.
    if recent_changes.iloc[0] > 0 or not (recent_changes.iloc[1:] > 0).all():
        return None
    setup = factors[f"early_{side}"]
    if (
        setup["momentum_atr"] < float(settings["min_momentum_atr"])
        or setup["close_breakout_atr"] < float(settings["min_close_breakout_atr"])
    ):
        return None

    extreme_column, opposing_column = ("low", "high") if side == "long" else ("high", "low")
    recent = frame.iloc[-4:-1]
    pivot = (recent[extreme_column] * direction).idxmin()
    pivot_price = float(frame.loc[pivot, extreme_column])
    if direction * (latest[extreme_column] - pivot_price) <= 0:
        return None
    before_pivot = frame.loc[:pivot].iloc[-7:-1]
    if not frame.tail(10).index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=4)).all():
        return None
    prior_extreme = float((before_pivot[opposing_column] * direction).max())
    prior_swing_atr = (prior_extreme - direction * pivot_price) / latest["atr"]
    recovery_atr = direction * (latest["close"] - pivot_price) / latest["atr"]
    if (
        prior_swing_atr < float(settings["min_prior_swing_atr"])
        or recovery_atr < float(settings["min_recovery_atr"])
    ):
        return None
    return {
        "side": side,
        "pivot_price": pivot_price,
        "prior_swing_atr": prior_swing_atr,
        "recovery_atr": recovery_atr,
    }


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
        strategy_version=__version__,
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
    structure_cfg = config.get("structure", {})
    late_after = float(structure_cfg.get("late_entry_after_retracement", 0.50))
    late_room_atr = float(structure_cfg.get("late_entry_max_room_atr", 1.50))
    minimum_tp1_r = float(structure_cfg.get("minimum_tp1_r", 1.0))
    minimum_tp2_r = float(structure_cfg.get("minimum_tp2_r", 1.5))
    early_max_ema_distance_atr = float(structure_cfg.get("early_max_ema_distance_atr", 3.0))
    early_min_extreme_room_atr = float(structure_cfg.get("early_min_extreme_room_atr", 1.0))
    swing_extreme_lookback_bars = int(structure_cfg.get("swing_extreme_lookback_bars", 180))
    fast = config.get("fast_trend", {})
    fast_enabled = bool(fast.get("enabled", True))
    fast_min_close_breakout_atr = float(fast.get("min_close_breakout_atr", 0.50))
    fast_min_momentum_atr = float(fast.get("min_momentum_atr", 1.50))
    fast_min_directional_closes = int(fast.get("min_directional_closes", 3))
    fast_min_ema_sep_atr = float(fast.get("min_ema_separation_atr", 0.25))
    fast_reversal_min_ema_sep_atr = float(fast.get("reversal_min_ema_separation_atr", 0.50))
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
    early_short_ema_confirm_bars = int(early.get("ema_confirm_bars", 2))
    early_short_min_ema_sep_atr = float(early.get("min_ema_separation_atr", 0.25))
    early_short_reversal_min_ema_sep_atr = float(early.get("reversal_min_ema_separation_atr", 0.50))
    early_long_min_breakout_atr = float(early_long.get("min_breakout_atr", 0.25))
    early_long_close_bars = int(early_long.get("close_breakout_bars", 6))
    early_long_min_close_breakout_atr = float(early_long.get("min_close_breakout_atr", 0.10))
    early_long_momentum_bars = int(early_long.get("momentum_bars", 3))
    early_long_min_momentum_atr = float(early_long.get("min_momentum_atr", 0.75))
    early_long_min_higher_closes = int(early_long.get("min_higher_closes", 2))
    early_long_ema_confirm_bars = int(early_long.get("ema_confirm_bars", 2))
    early_long_min_ema_sep_atr = float(early_long.get("min_ema_separation_atr", 0.25))
    early_long_reversal_min_ema_sep_atr = float(early_long.get("reversal_min_ema_separation_atr", 0.50))

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
    recent_short_regime = frame.tail(early_short_ema_confirm_bars)
    short_ema_confirmed = bool((recent_short_regime["close"] < recent_short_regime["ema"]).all())
    short_ema_separation_atr = (values["ema"] - values["close"]) / values["atr"]
    short_ema_separation_ok = short_ema_separation_atr >= early_short_min_ema_sep_atr
    short_reversal_confirmed = (
        short_ema_confirmed
        and short_ema_separation_atr >= early_short_reversal_min_ema_sep_atr
        and momentum_breakdown
    )
    fast_short_signal = (
        fast_enabled
        and not short_signal
        and close_breakout_atr >= fast_min_close_breakout_atr
        and momentum_atr >= fast_min_momentum_atr
        and lower_closes >= fast_min_directional_closes
        and values["close"] < values["ema"]
        and short_ema_separation_atr >= fast_min_ema_sep_atr
    )
    early_short_signal = (
        early_short_enabled
        and not short_signal
        and not fast_short_signal
        and close_breakdown
        and momentum_breakdown
        and short_ema_confirmed
        and short_ema_separation_ok
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
        "ema_confirmed": short_ema_confirmed,
        "ema_separation_atr": short_ema_separation_atr,
        "minimum_ema_separation_atr": early_short_min_ema_sep_atr,
        "reversal_confirmed": short_reversal_confirmed,
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
    recent_long_regime = frame.tail(early_long_ema_confirm_bars)
    long_ema_confirmed = bool((recent_long_regime["close"] > recent_long_regime["ema"]).all())
    long_ema_separation_atr = (values["close"] - values["ema"]) / values["atr"]
    long_ema_separation_ok = long_ema_separation_atr >= early_long_min_ema_sep_atr
    long_reversal_confirmed = (
        long_ema_confirmed
        and long_ema_separation_atr >= early_long_reversal_min_ema_sep_atr
        and momentum_up_breakout
    )
    fast_long_signal = (
        fast_enabled
        and not long_signal
        and close_up_breakout_atr >= fast_min_close_breakout_atr
        and long_momentum_atr >= fast_min_momentum_atr
        and higher_closes >= fast_min_directional_closes
        and values["close"] > values["ema"]
        and long_ema_separation_atr >= fast_min_ema_sep_atr
    )
    early_long_signal = (
        early_long_enabled
        and not long_signal
        and not fast_long_signal
        and close_up_breakout
        and momentum_up_breakout
        and long_ema_confirmed
        and long_ema_separation_ok
    )
    fast_side = "long" if fast_long_signal else ("short" if fast_short_signal else None)
    fast_reversal_confirmed = (
        (fast_long_signal and long_ema_separation_atr >= fast_reversal_min_ema_sep_atr)
        or (fast_short_signal and short_ema_separation_atr >= fast_reversal_min_ema_sep_atr)
    )
    base_factors["fast_trend"] = {
        "enabled": fast_enabled,
        "side": fast_side,
        "confirmed": bool(fast_long_signal or fast_short_signal),
        "minimum_close_breakout_atr": fast_min_close_breakout_atr,
        "minimum_momentum_atr": fast_min_momentum_atr,
        "minimum_directional_closes": fast_min_directional_closes,
        "minimum_ema_separation_atr": fast_min_ema_sep_atr,
        "reversal_minimum_ema_separation_atr": fast_reversal_min_ema_sep_atr,
        "reversal_confirmed": bool(fast_reversal_confirmed),
    }
    exception_short = assess_exceptional_entry(
        frame, config, side="short", price=current_price, close=values["close"],
        ema_value=values["ema"], atr_value=values["atr"],
        prior_20bar_extreme=values["prior_low"],
        prior_swing_extreme=float(frame["low"].iloc[-181:-1].min()),
        momentum_atr=momentum_atr, directional_closes=lower_closes,
        close_breakout_atr=close_breakout_atr,
        previous_ema=float(frame["ema"].iloc[-2]),
    )
    exception_long = assess_exceptional_entry(
        frame, config, side="long", price=current_price, close=values["close"],
        ema_value=values["ema"], atr_value=values["atr"],
        prior_20bar_extreme=values["prior_high"],
        prior_swing_extreme=float(frame["high"].iloc[-181:-1].max()),
        momentum_atr=long_momentum_atr, directional_closes=higher_closes,
        close_breakout_atr=close_up_breakout_atr,
        previous_ema=float(frame["ema"].iloc[-2]),
    )
    crash_short_signal = not short_signal and not fast_short_signal and bool(exception_short["crash"])
    crash_long_signal = not long_signal and not fast_long_signal and bool(exception_long["crash"])
    base_short_signal = (
        not short_signal and not fast_short_signal and not crash_short_signal
        and bool(exception_short["base"]) and short_ema_confirmed
    )
    base_long_signal = (
        not long_signal and not fast_long_signal and not crash_long_signal
        and bool(exception_long["base"]) and long_ema_confirmed
    )
    base_factors["exceptional_short"] = exception_short
    base_factors["exceptional_long"] = exception_long
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
        "ema_confirmed": long_ema_confirmed,
        "ema_separation_atr": long_ema_separation_atr,
        "minimum_ema_separation_atr": early_long_min_ema_sep_atr,
        "reversal_confirmed": long_reversal_confirmed,
        "above_ema200": values["close"] >= values["ema"],
    }
    has_entry = any((long_signal, short_signal, fast_short_signal, fast_long_signal, crash_short_signal, crash_long_signal, base_short_signal, base_long_signal, early_short_signal, early_long_signal))
    watch = None if has_entry else _countertrend_watch(frame, config, base_factors)
    if not has_entry and watch is None:
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
            vetoes=["stale_breakout" if has_entry else "stale_countertrend_watch"],
            factors=base_factors,
        )

    if watch is not None:
        decision = _neutral(
            timestamp=signal_time.isoformat(), price=current_price, provider=provider,
            vetoes=[], factors={**base_factors, "countertrend_watch": watch},
        )
        decision.state = "watch"
        decision.regime = "countertrend_watch"
        return decision

    side = "long" if (long_signal or fast_long_signal or crash_long_signal or base_long_signal or early_long_signal) else "short"
    if fast_long_signal:
        signal_type = "fast_trend_long"
    elif fast_short_signal:
        signal_type = "fast_trend_short"
    elif crash_long_signal:
        signal_type = "crash_impulse_long"
    elif crash_short_signal:
        signal_type = "crash_impulse_short"
    elif base_long_signal:
        signal_type = "base_breakout_long"
    elif base_short_signal:
        signal_type = "base_breakout_short"
    elif early_long_signal:
        signal_type = "confirmed_early_long"
    elif early_short_signal:
        signal_type = "confirmed_early_short"
    else:
        signal_type = "trend_breakout"
    direction = 1.0 if side == "long" else -1.0

    if signal_type == "trend_breakout":
        quality, quality_vetoes = assess_trend_breakout_quality(
            side=side, close=values["close"], live_price=current_price,
            ema_value=values["ema"], atr_value=values["atr"],
            channel_level=values["prior_high"] if side == "long" else values["prior_low"],
            extended_ema_distance_atr=float(structure_cfg.get("extended_breakout_ema_distance_atr", 6.0)),
            extended_min_breakout_atr=float(structure_cfg.get("extended_breakout_min_depth_atr", 0.20)),
        )
        base_factors["breakout_quality"] = quality
        base_factors["signal_type"] = {"value": signal_type}
        if quality_vetoes:
            return _neutral(
                timestamp=signal_time.isoformat(), price=current_price, provider=provider,
                vetoes=quality_vetoes, factors=base_factors,
            )

    base_cfg = config.get("base_breakout", {})
    maximum_penetration = structure_cfg.get("episode_base_max_penetration_atr")
    if signal_type == "trend_breakout" and base_cfg.get("enabled") and maximum_penetration is not None:
        setup = exception_long if side == "long" else exception_short
        room = setup["extreme_room_atr"]
        width_bars = max(2, int(base_cfg["consolidation_bars"]))
        window = frame.tail(width_bars + 1)
        if (
            len(window) == width_bars + 1
            and window.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=4)).all()
            and setup["base_setup"]
            and (long_ema_confirmed if side == "long" else short_ema_confirmed)
            and setup["close_location"] <= float(config.get("crash_impulse", {}).get("max_close_location", 0.30))
            and (room is None or room >= float(base_cfg.get("min_extreme_room_atr", 1.0)))
            and 0 < setup["prior_extreme_penetration_atr"] <= float(maximum_penetration)
            and setup["entry_deviation_atr"] <= float(base_cfg.get("max_entry_deviation_atr", 0.60))
        ):
            base_factors["episode_base"] = {
                "first_close_time": (signal_time - pd.Timedelta(hours=4 * width_bars)).isoformat(),
            }

    # EARLY is an entry ahead of a confirmed 20-bar breakout, not permission
    # to chase an extended trend. FAST and the frozen Donchian core remain
    # unchanged to preserve detection of genuine large initial impulses.
    if (early_long_signal or early_short_signal) and not (base_long_signal or base_short_signal or crash_long_signal or crash_short_signal):
        previous_swings = frame.iloc[-1 - swing_extreme_lookback_bars : -1]
        prior_swing_extreme = float(
            previous_swings["low"].min() if side == "short"
            else previous_swings["high"].max()
        )
        stage, stage_veto = assess_early_entry_stage(
            side=side, price=current_price, close=values["close"], ema_value=values["ema"],
            atr_value=values["atr"],
            prior_20bar_extreme=values["prior_low"] if side == "short" else values["prior_high"],
            prior_swing_extreme=prior_swing_extreme,
            maximum_ema_distance_atr=early_max_ema_distance_atr,
            minimum_extreme_room_atr=early_min_extreme_room_atr,
        )
        stage["lookback_bars"] = swing_extreme_lookback_bars
        base_factors["trend_stage"] = stage
        if stage_veto:
            return _neutral(
                timestamp=signal_time.isoformat(), price=current_price, provider=provider,
                vetoes=[stage_veto], factors=base_factors,
            )

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

    raw_target_1 = current_price + direction * target_1_r * risk
    raw_target_2 = current_price + direction * target_2_r * risk
    fib = _fib_context(frame, config, current_price)
    fib_levels = fib.get("levels", [])
    directional_levels = (
        sorted([x for x in fib_levels if float(x["price"]) < current_price], key=lambda x: float(x["price"]), reverse=True)
        if side == "short"
        else sorted([x for x in fib_levels if float(x["price"]) > current_price], key=lambda x: float(x["price"]))
    )
    nearest_structure = directional_levels[0] if directional_levels else None
    second_structure = directional_levels[1] if len(directional_levels) > 1 else None
    room_atr = (
        abs(current_price - float(nearest_structure["price"])) / values["atr"]
        if nearest_structure else float("inf")
    )
    progress = float(fib.get("progress", 0.0))
    matching_retracement = (
        (side == "short" and fib.get("direction") == "up")
        or (side == "long" and fib.get("direction") == "down")
    )
    fast_signal = bool(fast_long_signal or fast_short_signal or crash_long_signal or crash_short_signal or base_long_signal or base_short_signal)
    if not fast_signal and matching_retracement and progress >= late_after and room_atr < late_room_atr:
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["late_entry_near_structure"],
            factors={
                **base_factors,
                "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
                "fibonacci": {
                    **fib,
                    "nearest_level": nearest_structure,
                    "room_atr": room_atr,
                    "late_entry_after": late_after,
                    "minimum_room_atr": late_room_atr,
                },
            },
        )
    def reward_r(level: dict[str, float]) -> float:
        return abs(float(level["price"]) - current_price) / risk

    eligible_tp1 = [level for level in directional_levels if reward_r(level) >= minimum_tp1_r]
    eligible_tp2 = [level for level in directional_levels if reward_r(level) >= minimum_tp2_r]
    target_1_structure = eligible_tp1[0] if eligible_tp1 else None
    target_2_structure = None
    if eligible_tp2:
        for level in eligible_tp2:
            if target_1_structure is None or float(level["price"]) != float(target_1_structure["price"]):
                target_2_structure = level
                break

    if not fast_signal and directional_levels and target_1_structure is None:
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["insufficient_structural_reward"],
            factors={
                **base_factors,
                "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
                "fibonacci": {
                    **fib,
                    "nearest_level": nearest_structure,
                    "room_atr": room_atr,
                    "minimum_tp1_r": minimum_tp1_r,
                    "minimum_tp2_r": minimum_tp2_r,
                },
            },
        )

    if target_1_structure is None:
        target_1 = raw_target_1
        target_2 = raw_target_2
    elif side == "short":
        target_1 = max(raw_target_1, float(target_1_structure["price"]))
        target_2 = (
            max(raw_target_2, float(target_2_structure["price"]))
            if target_2_structure
            else raw_target_2
        )
    else:
        target_1 = min(raw_target_1, float(target_1_structure["price"]))
        target_2 = (
            min(raw_target_2, float(target_2_structure["price"]))
            if target_2_structure
            else raw_target_2
        )

    target_1_r_actual = abs(target_1 - current_price) / risk
    target_2_r_actual = abs(target_2 - current_price) / risk
    if not fast_signal and (target_1_r_actual < minimum_tp1_r or target_2_r_actual < minimum_tp2_r):
        return _neutral(
            timestamp=signal_time.isoformat(),
            price=current_price,
            provider=provider,
            vetoes=["insufficient_structural_reward"],
            factors={
                **base_factors,
                "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
                "fibonacci": {
                    **fib,
                    "nearest_level": nearest_structure,
                    "target_1_level": target_1_structure,
                    "target_2_level": target_2_structure,
                    "target_1_r": target_1_r_actual,
                    "target_2_r": target_2_r_actual,
                    "minimum_tp1_r": minimum_tp1_r,
                    "minimum_tp2_r": minimum_tp2_r,
                },
            },
        )
    activation = current_price + direction * activation_atr * values["atr"]
    factors = {
        **base_factors,
        "cost_filter": {"risk_to_cost": risk_to_cost, "minimum": min_risk_to_cost},
        "fibonacci": {
            **fib,
            "nearest_level": nearest_structure,
            "second_level": second_structure,
            "target_1_level": target_1_structure,
            "target_2_level": target_2_structure,
            "room_atr": room_atr,
            "raw_target_1": raw_target_1,
            "raw_target_2": raw_target_2,
            "target_1_structured": target_1,
            "target_2_structured": target_2,
            "target_1_r": target_1_r_actual,
            "target_2_r": target_2_r_actual,
            "minimum_tp1_r": minimum_tp1_r,
            "minimum_tp2_r": minimum_tp2_r,
        },
        "exit_plan": {
            "mode": "atr_trailing",
            "activation_price": activation,
            "activation_atr": activation_atr,
            "trailing_distance_atr": trail_atr,
            "target_1_is_reference": True,
            "target_2_is_reference": True,
        },
    }
    if fast_long_signal or crash_long_signal or base_long_signal:
        channel = values["prior_close_high"]
    elif fast_short_signal or crash_short_signal or base_short_signal:
        channel = values["prior_close_low"]
    elif early_long_signal and close_up_breakout:
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
    level_type = (
        "crash_impulse" if crash_short_signal or crash_long_signal
        else ("base_breakout" if base_short_signal or base_long_signal
        else ("fast_trend" if fast_long_signal or fast_short_signal
        else ("early_long" if early_long_signal else ("early_short" if early_short_signal else "donchian_breakout"))))
    )
    factors["signal_type"] = {"value": signal_type}
    return Decision(
        timestamp=signal_time.isoformat(),
        state="signal",
        side=side,
        score=(100 if signal_type == "trend_breakout" else (95 if fast_signal and not (base_long_signal or base_short_signal) else (92 if base_long_signal or base_short_signal else 90))),
        price=current_price,
        entry_low=current_price,
        entry_high=current_price,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        rr=target_2_r,
        regime=(
            ("crash_long_above_ema200" if side == "long" else "crash_short_below_ema200")
            if crash_long_signal or crash_short_signal
            else (("base_long_above_ema200" if side == "long" else "base_short_below_ema200")
            if base_long_signal or base_short_signal
            else (("fast_long_above_ema200" if side == "long" else "fast_short_below_ema200")
            if fast_long_signal or fast_short_signal
            else (
                "early_long_above_ema200" if early_long_signal
                else (
                    ("early_short_above_ema200" if values["close"] >= values["ema"] else "early_short_below_ema200")
                    if early_short_signal
                    else ("above_ema200" if side == "long" else "below_ema200")
                )
            )))
        ),
        level={"type": level_type, "bars": breakout_bars, "price": channel},
        factors=factors,
        vetoes=[],
        provider=provider,
        strategy_version=__version__,
    )
