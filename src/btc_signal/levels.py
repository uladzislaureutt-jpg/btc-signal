from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LevelZone:
    side: str
    low: float
    high: float
    midpoint: float
    touches: int
    quality: int
    sources: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data


def _pivots(frame: pd.DataFrame, window: int) -> list[tuple[float, str, str]]:
    points: list[tuple[float, str, str]] = []
    highs = frame["high"]
    lows = frame["low"]
    for i in range(window, len(frame) - window):
        if highs.iloc[i] >= highs.iloc[i - window : i + window + 1].max():
            points.append((float(highs.iloc[i]), "resistance", "swing"))
        if lows.iloc[i] <= lows.iloc[i - window : i + window + 1].min():
            points.append((float(lows.iloc[i]), "support", "swing"))
    return points


def _reference_points(frame: pd.DataFrame) -> list[tuple[float, str, str]]:
    points: list[tuple[float, str, str]] = []
    if len(frame) >= 7:
        previous_day = frame.iloc[-7:-1]
        points.extend(
            [
                (float(previous_day["high"].max()), "resistance", "previous_day_high"),
                (float(previous_day["low"].min()), "support", "previous_day_low"),
            ]
        )
    if len(frame) >= 42:
        previous_week = frame.iloc[-42:-6]
        points.extend(
            [
                (float(previous_week["high"].max()), "resistance", "previous_week_high"),
                (float(previous_week["low"].min()), "support", "previous_week_low"),
            ]
        )
    return points


def build_levels(frame_4h: pd.DataFrame, config: dict) -> list[LevelZone]:
    params = config["levels"]
    lookback = int(params["lookback_bars"])
    sample = frame_4h.tail(lookback)
    if sample.empty or not np.isfinite(sample["atr"].iloc[-1]):
        return []
    current_atr = float(sample["atr"].iloc[-1])
    cluster_distance = current_atr * float(params["cluster_atr"])
    half_width = current_atr * float(params["atr_zone"])
    points = _pivots(sample, int(params["swing_window"])) + _reference_points(sample)
    zones: list[LevelZone] = []
    for side in ("support", "resistance"):
        side_points = sorted((p for p in points if p[1] == side), key=lambda p: p[0])
        clusters: list[list[tuple[float, str, str]]] = []
        for point in side_points:
            if not clusters or abs(point[0] - np.mean([p[0] for p in clusters[-1]])) > cluster_distance:
                clusters.append([point])
            else:
                clusters[-1].append(point)
        for cluster in clusters:
            midpoint = float(np.mean([p[0] for p in cluster]))
            sources = tuple(sorted({p[2] for p in cluster}))
            touches = sum(1 for p in cluster if p[2] == "swing")
            reference_bonus = sum(1 for source in sources if source != "swing")
            reaction = sample.loc[
                (sample["low"] <= midpoint + half_width)
                & (sample["high"] >= midpoint - half_width)
            ]
            reaction_bonus = 2 if len(reaction) and (
                (reaction["high"] - reaction["low"]).median() >= current_atr
            ) else 0
            quality = min(10, touches * 2 + reference_bonus + reaction_bonus)
            if touches >= int(params["min_touches"]) or reference_bonus:
                zones.append(
                    LevelZone(
                        side=side,
                        low=midpoint - half_width,
                        high=midpoint + half_width,
                        midpoint=midpoint,
                        touches=touches,
                        quality=quality,
                        sources=sources,
                    )
                )
    return sorted(zones, key=lambda zone: (zone.side, zone.midpoint))


def nearest_zone(
    levels: list[LevelZone], price: float, side: str, max_distance: float
) -> LevelZone | None:
    candidates = [
        zone
        for zone in levels
        if zone.side == side
        and (
            zone.low - max_distance <= price <= zone.high + max_distance
        )
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda zone: abs(zone.midpoint - price))

