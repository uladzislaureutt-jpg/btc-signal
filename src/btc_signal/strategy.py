from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .indicators import enrich
from .levels import LevelZone, build_levels, nearest_zone


Side = Literal["long", "short"]


@dataclass
class Decision:
    timestamp: str
    state: str
    side: Side | None
    score: int
    price: float
    entry_low: float | None
    entry_high: float | None
    stop: float | None
    target_1: float | None
    target_2: float | None
    rr: float | None
    regime: str
    level: dict | None
    factors: dict[str, dict[str, object]]
    vetoes: list[str]
    provider: str | None = None
    strategy_version: str = "0.1.6"

    @property
    def dedup_key(self) -> str:
        # A deployment within the same candle must not resend an existing alert.
        dedup_version = "0.4.3" if self.strategy_version in {"0.4.4", "0.4.5"} else self.strategy_version
        if dedup_version in {"0.3.1", "0.3.2", "0.3.3", "0.3.4", "0.3.5", "0.3.6", "0.4.0", "0.4.1", "0.4.2", "0.4.3"}:
            identity = [
                dedup_version,
                self.timestamp,
                self.state,
                self.side,
                self.level,
            ]
        else:
            identity = [
                self.timestamp,
                self.state,
                self.side,
                round(self.price, 2),
                self.level,
            ]
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            return value

        data = asdict(self)
        data["dedup_key"] = self.dedup_key
        return json_safe(data)


@dataclass(frozen=True)
class EvaluationTrace:
    """Diagnostics emitted beside a decision without affecting signal selection."""

    timestamp: str
    price: float
    atr: float
    regime: str
    structure_4h: str
    level_count: int
    closest_distance_atr: dict[str, float | None]
    candidates: tuple[Decision, ...]


def _regime(frame_1d: pd.DataFrame, config: dict) -> str:
    slope_bars = int(config["indicators"]["ema_slope_bars"])
    if len(frame_1d) < slope_bars + 2:
        return "no_trade"
    latest = frame_1d.iloc[-1]
    previous = frame_1d.iloc[-1 - slope_bars]
    values = [latest["close"], latest["ema_fast"], latest["ema_slow"], previous["ema_fast"]]
    if not all(np.isfinite(value) for value in values):
        return "no_trade"
    bullish = latest["close"] > latest["ema_fast"] > latest["ema_slow"] and latest["ema_fast"] > previous["ema_fast"]
    bearish = latest["close"] < latest["ema_fast"] < latest["ema_slow"] and latest["ema_fast"] < previous["ema_fast"]
    if bullish:
        return "long_allowed"
    if bearish:
        return "short_allowed"
    return "both_allowed"


def _structure(frame: pd.DataFrame) -> str:
    if len(frame) < 7:
        return "neutral"
    recent = frame.tail(7)
    first = recent.iloc[:3]
    last = recent.iloc[-3:]
    if last["high"].max() > first["high"].max() and last["low"].min() > first["low"].min():
        return "bullish"
    if last["high"].max() < first["high"].max() and last["low"].min() < first["low"].min():
        return "bearish"
    return "neutral"


def _factor(value: object, passed: bool, points: int, maximum: int) -> dict[str, object]:
    return {"value": value, "passed": passed, "points": points if passed else 0, "max": maximum}


def _risk_geometry(side: Side, price: float, stop: float, atr: float) -> tuple[float, bool]:
    minimum_risk = atr * 0.1 if np.isfinite(atr) and atr > 0 else max(abs(price) * 1e-6, 1e-8)
    raw_risk = price - stop if side == "long" else stop - price
    valid = bool(np.isfinite(raw_risk) and raw_risk >= minimum_risk)
    return (float(raw_risk) if valid else float(minimum_risk), valid)


def _zone_distance(zone: LevelZone, price: float) -> float:
    if zone.low <= price <= zone.high:
        return 0.0
    return min(abs(price - zone.low), abs(price - zone.high))


def _closest_distance_atr(
    levels: list[LevelZone], price: float, side: Side, atr: float
) -> float | None:
    candidates = [zone for zone in levels if zone.side == side]
    if not candidates or not np.isfinite(atr) or atr <= 0:
        return None
    closest = min(candidates, key=lambda zone: abs(zone.midpoint - price))
    return float(_zone_distance(closest, price) / atr)


def _candidate(
    side: Side,
    regime: str,
    structure_4h: str,
    trigger: pd.DataFrame,
    zone: LevelZone,
    levels: list[LevelZone],
    config: dict,
) -> Decision:
    latest = trigger.iloc[-1]
    previous = trigger.iloc[-2]
    weights = config["weights"]
    price = float(latest["close"])
    atr_1h = float(latest["atr"])
    if side == "long":
        price_action = latest["low"] <= zone.high and latest["close"] > zone.midpoint
        momentum = latest["rsi"] > previous["rsi"] and latest["rsi"] < 70
        aligned_regime = regime in ("long_allowed", "both_allowed")
        aligned_structure = structure_4h in ("bullish", "neutral")
        stop = zone.low - atr_1h * float(config["risk"]["stop_atr"])
        entry_low, entry_high = zone.midpoint, max(zone.high, price)
        risk, stop_geometry_valid = _risk_geometry(side, price, stop, atr_1h)
        opposing = sorted(item.midpoint for item in levels if item.side == "resistance" and item.midpoint > price)
        target_1 = opposing[0] if opposing else price + risk * float(config["risk"]["target_rr"])
        target_2 = opposing[1] if len(opposing) > 1 else price + risk * float(config["risk"]["target_rr"])
    else:
        price_action = latest["high"] >= zone.low and latest["close"] < zone.midpoint
        momentum = latest["rsi"] < previous["rsi"] and latest["rsi"] > 30
        aligned_regime = regime in ("short_allowed", "both_allowed")
        aligned_structure = structure_4h in ("bearish", "neutral")
        stop = zone.high + atr_1h * float(config["risk"]["stop_atr"])
        entry_low, entry_high = min(zone.low, price), zone.midpoint
        risk, stop_geometry_valid = _risk_geometry(side, price, stop, atr_1h)
        opposing = sorted(
            (item.midpoint for item in levels if item.side == "support" and item.midpoint < price),
            reverse=True,
        )
        target_1 = opposing[0] if opposing else price - risk * float(config["risk"]["target_rr"])
        target_2 = opposing[1] if len(opposing) > 1 else price - risk * float(config["risk"]["target_rr"])
    volume_ratio = float(latest["volume"] / latest["volume_median"]) if latest["volume_median"] else 0.0
    volume_ok = volume_ratio < float(config["signal"]["max_rejection_volume_ratio"])
    volatility_ratio = atr_1h / price if price else 0.0
    volatility_ok = 0.002 <= volatility_ratio <= 0.035
    level_points = round(int(weights["level_quality"]) * min(zone.quality, 10) / 10)
    factors = {
        "level_quality": {
            "value": zone.quality,
            "passed": zone.quality >= int(config["levels"]["min_quality"]),
            "points": level_points,
            "max": int(weights["level_quality"]),
        },
        "regime": _factor(regime, aligned_regime, int(weights["regime"]), int(weights["regime"])),
        "structure_4h": _factor(structure_4h, aligned_structure, int(weights["structure_4h"]), int(weights["structure_4h"])),
        "price_action_1h": _factor(bool(price_action), bool(price_action), int(weights["price_action_1h"]), int(weights["price_action_1h"])),
        "volume": _factor(round(volume_ratio, 3), volume_ok, int(weights["volume"]), int(weights["volume"])),
        "momentum": _factor(round(float(latest["rsi"]), 2), bool(momentum), int(weights["momentum"]), int(weights["momentum"])),
        "volatility": _factor(round(volatility_ratio, 5), volatility_ok, int(weights["volatility"]), int(weights["volatility"])),
    }
    score = sum(int(item["points"]) for item in factors.values())
    rr = abs(target_1 - price) / risk if stop_geometry_valid else 0.0
    trading_cost = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    estimated_cost_r = trading_cost * price / risk if stop_geometry_valid else float("inf")
    vetoes: list[str] = []
    if zone.quality < int(config["levels"]["min_quality"]):
        vetoes.append("weak_level")
    if zone.quality > int(config["levels"]["max_signal_quality"]):
        vetoes.append("saturated_level")
    if not aligned_regime:
        vetoes.append("daily_regime_conflict")
    if bool(config["signal"]["require_price_action"]) and not price_action:
        vetoes.append("missing_price_action_confirmation")
    if not volume_ok:
        vetoes.append("breakout_volume_risk")
    if not stop_geometry_valid:
        vetoes.append("invalid_stop_geometry")
    elif rr < float(config["risk"]["min_rr"]):
        vetoes.append("rr_below_minimum")
    elif estimated_cost_r > float(config["risk"]["max_cost_r"]):
        vetoes.append("cost_above_limit")
    distance = _zone_distance(zone, price)
    if distance > atr_1h * float(config["levels"]["max_distance_atr"]):
        vetoes.append("late_entry")
    if vetoes:
        state = "neutral"
    elif score >= int(config["signal"]["strong_score"]):
        state = "strong_signal"
    elif score >= int(config["signal"]["min_score"]):
        state = "signal"
    elif score >= int(config["signal"]["watch_score"]):
        state = "watch"
    else:
        state = "neutral"
    return Decision(
        timestamp=trigger.index[-1].isoformat(),
        state=state,
        side=side,
        score=score,
        price=price,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        stop=float(stop),
        target_1=float(target_1),
        target_2=float(target_2),
        rr=float(rr),
        regime=regime,
        level=zone.to_dict(),
        factors=factors,
        vetoes=vetoes,
    )


def evaluate_with_trace(
    frame_1d: pd.DataFrame,
    frame_4h: pd.DataFrame,
    frame_1h: pd.DataFrame,
    config: dict,
    provider: str | None = None,
) -> tuple[Decision, EvaluationTrace]:
    day = enrich(frame_1d, config)
    setup = enrich(frame_4h, config)
    trigger = enrich(frame_1h, config)
    return evaluate_enriched_with_trace(day, setup, trigger, config, provider)


def evaluate_enriched_with_trace(
    day: pd.DataFrame,
    setup: pd.DataFrame,
    trigger: pd.DataFrame,
    config: dict,
    provider: str | None = None,
) -> tuple[Decision, EvaluationTrace]:
    """Evaluate already-enriched causal frames.

    Backtests call this path after calculating rolling indicators once for the
    complete history. All indicators used by ``enrich`` are backward-looking,
    so slicing the precomputed frames at an evaluation timestamp is equivalent
    to recalculating them for every historical hour.
    """
    if min(len(day), len(setup), len(trigger)) < 20:
        raise ValueError("Not enough closed candles")
    regime = _regime(day, config)
    structure_4h = _structure(setup)
    levels = build_levels(setup, config)
    latest = trigger.iloc[-1]
    price = float(latest["close"])
    atr = float(latest["atr"])
    max_distance = atr * float(config["levels"]["max_distance_atr"])
    support = nearest_zone(levels, price, "support", max_distance)
    resistance = nearest_zone(levels, price, "resistance", max_distance)
    candidates: list[Decision] = []
    if support:
        candidates.append(_candidate("long", regime, structure_4h, trigger, support, levels, config))
    if resistance:
        candidates.append(_candidate("short", regime, structure_4h, trigger, resistance, levels, config))
    if candidates:
        decision = max(candidates, key=lambda item: item.score)
        decision.provider = provider
    else:
        decision = Decision(
            timestamp=trigger.index[-1].isoformat(),
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
            regime=regime,
            level=None,
            factors={},
            vetoes=["no_nearby_4h_level"],
            provider=provider,
        )
    trace = EvaluationTrace(
        timestamp=trigger.index[-1].isoformat(),
        price=price,
        atr=atr,
        regime=regime,
        structure_4h=structure_4h,
        level_count=len(levels),
        closest_distance_atr={
            "long": _closest_distance_atr(levels, price, "support", atr),
            "short": _closest_distance_atr(levels, price, "resistance", atr),
        },
        candidates=tuple(candidates),
    )
    return decision, trace


def evaluate(
    frame_1d: pd.DataFrame,
    frame_4h: pd.DataFrame,
    frame_1h: pd.DataFrame,
    config: dict,
    provider: str | None = None,
) -> Decision:
    decision, _ = evaluate_with_trace(frame_1d, frame_4h, frame_1h, config, provider)
    return decision
