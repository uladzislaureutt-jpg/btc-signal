from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .strategy import Decision, EvaluationTrace, evaluate_with_trace


BACKTEST_WARMUP_HOURS = 24 * 210


@dataclass
class Trade:
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    score: int
    regime: str
    rr: float
    risk_pct: float
    gross_r: float
    cost_r: float
    level_quality: int
    price_action_ok: bool
    volume_ok: bool
    momentum_ok: bool
    structure_ok: bool
    result: str
    r_multiple: float


def audit_market_data(frame: pd.DataFrame, interval: str = "1h") -> dict:
    """Report structural market-data defects without hiding exchange outages."""
    required = {"open", "high", "low", "close", "volume", "close_time"}
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        errors.append("missing_columns")
        return {
            "status": "error",
            "errors": errors,
            "warnings": warnings,
            "missing_columns": missing_columns,
        }
    if not isinstance(frame.index, pd.DatetimeIndex):
        errors.append("index_not_datetime")
        return {
            "status": "error",
            "errors": errors,
            "warnings": warnings,
            "rows": len(frame),
            "timezone": None,
        }
    elif frame.index.tz is None:
        errors.append("index_not_timezone_aware")
    elif str(frame.index.tz) != "UTC":
        errors.append("index_not_utc")
    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        errors.append("duplicate_open_times")
    if not frame.index.is_monotonic_increasing:
        errors.append("open_times_not_monotonic")
    numeric = frame[["open", "high", "low", "close", "volume"]]
    nan_cells = int(frame[list(required)].isna().sum().sum())
    non_finite_cells = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    invalid_ohlc = int(
        (
            (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame["high"])
        ).sum()
    )
    negative_volume = int((frame["volume"] < 0).sum())
    nonpositive_price = int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if nan_cells:
        errors.append("missing_values")
    if non_finite_cells:
        errors.append("non_finite_numeric_values")
    if invalid_ohlc:
        errors.append("invalid_ohlc_geometry")
    if negative_volume:
        errors.append("negative_volume")
    if nonpositive_price:
        errors.append("nonpositive_price")

    expected = pd.Timedelta(interval)
    deltas = frame.index.to_series().diff().dropna() if isinstance(frame.index, pd.DatetimeIndex) else pd.Series(dtype="timedelta64[ns]")
    gaps = deltas[deltas > expected]
    overlaps = deltas[deltas < expected]
    gap_details = []
    if isinstance(frame.index, pd.DatetimeIndex):
        for position in range(1, len(frame.index)):
            delta = frame.index[position] - frame.index[position - 1]
            if delta > expected:
                gap_details.append({
                    "after": frame.index[position - 1].isoformat(),
                    "before": frame.index[position].isoformat(),
                    "duration_hours": float(delta / pd.Timedelta(hours=1)),
                    "estimated_missing_bars": max(int(round(delta / expected)) - 1, 0),
                })
    if len(gaps):
        warnings.append("market_data_gaps")
    if len(overlaps):
        errors.append("sub_interval_or_overlapping_bars")
    close_times = pd.to_datetime(frame["close_time"], utc=True)
    durations = close_times.array - frame.index.array
    expected_duration = expected - pd.Timedelta(milliseconds=1)
    irregular_close_duration = int((durations != expected_duration).sum())
    if irregular_close_duration:
        warnings.append("irregular_candle_duration")
    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "rows": len(frame),
        "duplicate_open_times": duplicates,
        "nan_cells": nan_cells,
        "non_finite_numeric_cells": non_finite_cells,
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "nonpositive_price_rows": nonpositive_price,
        "gap_count": int(len(gaps)),
        "estimated_missing_bars": int(sum(item["estimated_missing_bars"] for item in gap_details)),
        "gaps": gap_details[:20],
        "sub_interval_or_overlapping_bars": int(len(overlaps)),
        "irregular_candle_duration_rows": irregular_close_duration,
        "expected_interval": interval,
        "timezone": str(frame.index.tz) if isinstance(frame.index, pd.DatetimeIndex) else None,
    }


def required_history_hours(evaluation_days: int, max_holding_hours: int) -> int:
    if evaluation_days <= 0:
        raise ValueError("evaluation_days must be positive")
    if max_holding_hours < 0:
        raise ValueError("max_holding_hours must be non-negative")
    return BACKTEST_WARMUP_HOURS + evaluation_days * 24 + max_holding_hours


def resample(frame_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregated = frame_1h.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "close"])
    if frame_1h.empty:
        return aggregated
    duration = pd.Timedelta(rule)
    base_interval = pd.Timedelta(hours=1)
    expected_bars = int(duration / base_interval)
    counts = frame_1h["open"].resample(rule, label="left", closed="left").count()
    complete_groups = counts[counts == expected_bars].index
    cutoff = frame_1h["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)
    return aggregated.loc[
        aggregated.index.isin(complete_groups) & (aggregated.index + duration <= cutoff)
    ]


def _execution_rejection_reason(decision: Decision, future: pd.DataFrame, config: dict) -> str | None:
    if decision.side is None or decision.stop is None or decision.target_1 is None:
        return "missing_trade_geometry"
    if future.empty:
        return "no_next_candle"
    entry = float(future.iloc[0]["open"])
    risk = entry - decision.stop if decision.side == "long" else decision.stop - entry
    target_valid = decision.target_1 > entry if decision.side == "long" else decision.target_1 < entry
    if risk <= 0:
        return "invalid_stop_at_next_open"
    if not target_valid:
        return "target_crossed_before_entry"
    actual_rr = abs(decision.target_1 - entry) / risk
    if actual_rr < float(config["risk"]["min_rr"]):
        return "rr_below_minimum_at_next_open"
    costs = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    if costs * entry / risk > float(config["risk"]["max_cost_r"]):
        return "cost_above_limit_at_next_open"
    return None


def _bar_close_time(timestamp: pd.Timestamp, candle: pd.Series) -> pd.Timestamp:
    value = candle.get("close_time")
    return pd.Timestamp(value) if value is not None and not pd.isna(value) else timestamp + pd.Timedelta(hours=1)


def _simulate(
    decision: Decision,
    future: pd.DataFrame,
    config: dict,
    signal_time: str | None = None,
) -> Trade | None:
    if _execution_rejection_reason(decision, future, config) is not None:
        return None
    entry = float(future.iloc[0]["open"])
    entry_time = future.index[0]
    risk = entry - decision.stop if decision.side == "long" else decision.stop - entry
    actual_rr = abs(float(decision.target_1) - entry) / risk
    max_hours = int(config["risk"]["max_holding_hours"])
    last_position = min(len(future), max_hours) - 1
    last_candle = future.iloc[last_position]
    exit_price = float(last_candle["close"])
    exit_time = _bar_close_time(future.index[last_position], last_candle)
    result = "timeout"
    for bar_number, (timestamp, candle) in enumerate(future.head(max_hours).iterrows()):
        candle_open = float(candle["open"])
        if decision.side == "long":
            if bar_number and candle_open <= decision.stop:
                exit_price, exit_time, result = candle_open, _bar_close_time(timestamp, candle), "gap_stop"
                break
            if bar_number and candle_open >= decision.target_1:
                exit_price, exit_time, result = candle_open, _bar_close_time(timestamp, candle), "gap_target"
                break
            stop_hit = candle["low"] <= decision.stop
            target_hit = candle["high"] >= decision.target_1
        else:
            if bar_number and candle_open >= decision.stop:
                exit_price, exit_time, result = candle_open, _bar_close_time(timestamp, candle), "gap_stop"
                break
            if bar_number and candle_open <= decision.target_1:
                exit_price, exit_time, result = candle_open, _bar_close_time(timestamp, candle), "gap_target"
                break
            stop_hit = candle["high"] >= decision.stop
            target_hit = candle["low"] <= decision.target_1
        if stop_hit:
            exit_price, exit_time, result = decision.stop, _bar_close_time(timestamp, candle), "stop"
            break
        if target_hit:
            exit_price, exit_time, result = decision.target_1, _bar_close_time(timestamp, candle), "target"
            break
    raw_r = (exit_price - entry) / risk if decision.side == "long" else (entry - exit_price) / risk
    costs = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    cost_r = costs * ((entry + float(exit_price)) / 2) / risk
    net_r = raw_r - cost_r
    factors = decision.factors
    return Trade(
        side=decision.side,
        signal_time=signal_time or decision.timestamp,
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        entry=entry,
        exit=float(exit_price),
        stop=float(decision.stop),
        target=float(decision.target_1),
        score=decision.score,
        regime=decision.regime,
        rr=float(actual_rr),
        risk_pct=float(risk / entry) if entry else 0.0,
        gross_r=float(raw_r),
        cost_r=float(cost_r),
        level_quality=int(factors.get("level_quality", {}).get("value", 0)),
        price_action_ok=bool(factors.get("price_action_1h", {}).get("passed", False)),
        volume_ok=bool(factors.get("volume", {}).get("passed", False)),
        momentum_ok=bool(factors.get("momentum", {}).get("passed", False)),
        structure_ok=bool(factors.get("structure_4h", {}).get("passed", False)),
        result=result,
        r_multiple=float(net_r),
    )


def _distance_summary(values: list[float], missing: int, threshold: float) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "observations": int(len(array)),
        "missing": int(missing),
        "within_configured_limit": int((array <= threshold).sum()) if len(array) else 0,
        "median": float(np.median(array)) if len(array) else None,
        "p75": float(np.quantile(array, 0.75)) if len(array) else None,
        "p90": float(np.quantile(array, 0.90)) if len(array) else None,
        "p95": float(np.quantile(array, 0.95)) if len(array) else None,
    }


def _counterfactual_summary(trades: list[Trade]) -> dict:
    values = np.asarray([trade.r_multiple for trade in trades], dtype=float)
    return {
        "trades": int(len(values)),
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "expectancy_r": float(values.mean()) if len(values) else 0.0,
        "total_r": float(values.sum()) if len(values) else 0.0,
    }


def _trade_breakdown(trades: list[Trade], attribute: str) -> dict:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        groups[str(getattr(trade, attribute))].append(trade)
    return {name: _counterfactual_summary(items) for name, items in sorted(groups.items())}


def _record_trace(
    trace: EvaluationTrace,
    future: pd.DataFrame,
    config: dict,
    counters: dict,
) -> None:
    counters["regimes"][trace.regime] += 1
    counters["structures"][trace.structure_4h] += 1
    counters["level_counts"].append(trace.level_count)
    threshold = float(config["levels"]["max_distance_atr"])
    for side, distance in trace.closest_distance_atr.items():
        if distance is None:
            counters["distance_missing"][side] += 1
        else:
            counters["distances"][side].append(distance)
    for candidate in trace.candidates:
        side = str(candidate.side)
        counters["candidate_sides"][side] += 1
        counters["candidate_states"][candidate.state] += 1
        counters["scores"][str(candidate.score)] += 1
        for veto in candidate.vetoes:
            counters["vetoes"][veto] += 1
        if candidate.vetoes:
            combination = "+".join(sorted(candidate.vetoes))
            counters["veto_combinations"][combination] += 1
        if len(candidate.vetoes) == 1:
            counters["sole_vetoes"][candidate.vetoes[0]] += 1
        blockers = list(candidate.vetoes)
        if not blockers and candidate.score < int(config["signal"]["min_score"]):
            blockers = ["score_below_signal"]
        if len(blockers) == 1:
            trade = _simulate(candidate, future, config)
            if trade is not None:
                counters["counterfactuals"][blockers[0]].append(trade)
    counters["candidate_instances"] += len(trace.candidates)
    counters["distance_threshold"] = threshold


def _new_diagnostics() -> dict:
    return {
        "evaluated_hours": 0,
        "evaluation_errors": 0,
        "candidate_instances": 0,
        "no_nearby_4h_level_hours": 0,
        "actual_signal_hours": 0,
        "blocked_hours": 0,
        "blocked_signal_hours": 0,
        "blocked_reasons": Counter(),
        "selected_sides": Counter(),
        "decision_states": Counter(),
        "candidate_sides": Counter(),
        "candidate_states": Counter(),
        "vetoes": Counter(),
        "sole_vetoes": Counter(),
        "veto_combinations": Counter(),
        "scores": Counter(),
        "regimes": Counter(),
        "structures": Counter(),
        "level_counts": [],
        "distances": defaultdict(list),
        "distance_missing": Counter(),
        "distance_threshold": 0.0,
        "counterfactuals": defaultdict(list),
        "execution_rejections": Counter(),
    }


def _finalize_diagnostics(raw: dict) -> dict:
    threshold = float(raw["distance_threshold"])
    level_counts = np.asarray(raw["level_counts"], dtype=float)
    return {
        "method": "decision_funnel_v1",
        "note": "Single-blocker counterfactuals are isolated and may overlap; they are not portfolio results.",
        "evaluated_hours": raw["evaluated_hours"],
        "evaluation_errors": raw["evaluation_errors"],
        "candidate_instances": raw["candidate_instances"],
        "no_nearby_4h_level_hours": raw["no_nearby_4h_level_hours"],
        "eligible_signal_hours_before_portfolio_blocks": raw["actual_signal_hours"],
        "blocked_hours": raw["blocked_hours"],
        "blocked_signal_hours": raw["blocked_signal_hours"],
        "blocked_reasons": dict(raw["blocked_reasons"].most_common()),
        "selected_sides": dict(raw["selected_sides"].most_common()),
        "decision_states": dict(raw["decision_states"].most_common()),
        "candidate_sides": dict(raw["candidate_sides"].most_common()),
        "candidate_states": dict(raw["candidate_states"].most_common()),
        "veto_counts": dict(raw["vetoes"].most_common()),
        "sole_veto_counts": dict(raw["sole_vetoes"].most_common()),
        "veto_combinations": dict(raw["veto_combinations"].most_common(25)),
        "score_histogram": dict(sorted(raw["scores"].items(), key=lambda item: int(item[0]))),
        "regimes": dict(raw["regimes"].most_common()),
        "structures_4h": dict(raw["structures"].most_common()),
        "level_count": {
            "median": float(np.median(level_counts)) if len(level_counts) else None,
            "p90": float(np.quantile(level_counts, 0.90)) if len(level_counts) else None,
        },
        "closest_zone_distance_atr": {
            side: _distance_summary(
                raw["distances"][side], raw["distance_missing"][side], threshold
            )
            for side in ("long", "short")
        },
        "single_blocker_counterfactual": {
            reason: _counterfactual_summary(trades)
            for reason, trades in sorted(raw["counterfactuals"].items())
        },
        "execution_rejections": dict(raw["execution_rejections"].most_common()),
    }


def run_backtest(frame_1h: pd.DataFrame, config: dict) -> tuple[list[Trade], dict]:
    data_quality = audit_market_data(frame_1h)
    if data_quality["status"] == "error":
        raise ValueError(f"Invalid market data: {', '.join(data_quality['errors'])}")
    trades: list[Trade] = []
    diagnostics = _new_diagnostics()
    warmup_hours = BACKTEST_WARMUP_HOURS
    position_until: pd.Timestamp | None = None
    cooldown_until: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=int(config["signal"]["cooldown_hours"]))
    max_holding = int(config["risk"]["max_holding_hours"])
    for i in range(warmup_hours, len(frame_1h) - max_holding):
        history_1h = frame_1h.iloc[: i + 1]
        timestamp = history_1h.index[-1]
        evaluation_time = pd.Timestamp(history_1h["close_time"].iloc[-1])
        history_4h = resample(history_1h, "4h")
        history_1d = resample(history_1h, "1D")
        try:
            decision, trace = evaluate_with_trace(
                history_1d, history_4h, history_1h, config, "historical"
            )
        except ValueError:
            diagnostics["evaluation_errors"] += 1
            continue
        diagnostics["evaluated_hours"] += 1
        diagnostics["decision_states"][decision.state] += 1
        if decision.side:
            diagnostics["selected_sides"][decision.side] += 1
        if "no_nearby_4h_level" in decision.vetoes:
            diagnostics["no_nearby_4h_level_hours"] += 1
        if decision.state in ("signal", "strong_signal"):
            diagnostics["actual_signal_hours"] += 1
        future = frame_1h.iloc[i + 1 :]
        _record_trace(trace, future, config, diagnostics)

        position_blocked = position_until is not None and evaluation_time < position_until
        cooldown_blocked = cooldown_until is not None and evaluation_time < cooldown_until
        if position_blocked or cooldown_blocked:
            diagnostics["blocked_hours"] += 1
            reason = "position_and_cooldown" if position_blocked and cooldown_blocked else (
                "open_position" if position_blocked else "cooldown"
            )
            diagnostics["blocked_reasons"][reason] += 1
            if decision.state in ("signal", "strong_signal"):
                diagnostics["blocked_signal_hours"] += 1
            continue
        if decision.state not in ("signal", "strong_signal"):
            continue
        rejection = _execution_rejection_reason(decision, future, config)
        if rejection:
            diagnostics["execution_rejections"][rejection] += 1
            continue
        trade = _simulate(decision, future, config, signal_time=evaluation_time.isoformat())
        if trade:
            trades.append(trade)
            position_until = pd.Timestamp(trade.exit_time)
            cooldown_until = pd.Timestamp(trade.entry_time) + cooldown
    values = np.array([trade.r_multiple for trade in trades], dtype=float)
    wins = values[values > 0]
    losses = values[values <= 0]
    equity = values.cumsum() if len(values) else np.array([])
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([])
    drawdowns = equity - peaks if len(equity) else np.array([])
    summary = {
        "trades": len(trades),
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "expectancy_r": float(values.mean()) if len(values) else 0.0,
        "total_r": float(values.sum()) if len(values) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else None,
        "max_drawdown_r": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "total_cost_r": float(sum(trade.cost_r for trade in trades)),
        "average_cost_r": float(np.mean([trade.cost_r for trade in trades])) if trades else 0.0,
        "average_risk_pct": float(np.mean([trade.risk_pct for trade in trades])) if trades else 0.0,
        "execution_model": "signal_on_close_entry_at_next_1h_open",
        "intrabar_collision_rule": "stop_first",
        "funding_model": "not_applicable_spot",
        "higher_timeframe_partial_bar_policy": "exclude_groups_missing_any_1h_bar",
        "data_quality": data_quality,
        "by_side": _trade_breakdown(trades, "side"),
        "by_regime": _trade_breakdown(trades, "regime"),
        "diagnostics": _finalize_diagnostics(diagnostics),
    }
    return trades, summary


def trades_as_dicts(trades: list[Trade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
