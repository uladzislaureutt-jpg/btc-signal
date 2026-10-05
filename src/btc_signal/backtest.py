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
    cutoff = frame_1h["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)
    duration = pd.Timedelta(rule)
    return aggregated.loc[aggregated.index + duration <= cutoff]


def _simulate(decision: Decision, future: pd.DataFrame, config: dict) -> Trade | None:
    if decision.side is None or decision.stop is None or decision.target_1 is None:
        return None
    entry = decision.price
    risk = entry - decision.stop if decision.side == "long" else decision.stop - entry
    target_valid = decision.target_1 > entry if decision.side == "long" else decision.target_1 < entry
    if risk <= 0 or not target_valid:
        return None
    max_hours = int(config["risk"]["max_holding_hours"])
    exit_price = float(future.iloc[min(len(future), max_hours) - 1]["close"])
    exit_time = future.index[min(len(future), max_hours) - 1]
    result = "timeout"
    for timestamp, candle in future.head(max_hours).iterrows():
        if decision.side == "long":
            stop_hit = candle["low"] <= decision.stop
            target_hit = candle["high"] >= decision.target_1
        else:
            stop_hit = candle["high"] >= decision.stop
            target_hit = candle["low"] <= decision.target_1
        if stop_hit:
            exit_price, exit_time, result = decision.stop, timestamp, "stop"
            break
        if target_hit:
            exit_price, exit_time, result = decision.target_1, timestamp, "target"
            break
    raw_r = (exit_price - entry) / risk if decision.side == "long" else (entry - exit_price) / risk
    costs = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    cost_r = costs * entry / risk
    net_r = raw_r - cost_r
    factors = decision.factors
    return Trade(
        side=decision.side,
        entry_time=decision.timestamp,
        exit_time=exit_time.isoformat(),
        entry=entry,
        exit=float(exit_price),
        stop=float(decision.stop),
        target=float(decision.target_1),
        score=decision.score,
        regime=decision.regime,
        rr=float(decision.rr or 0.0),
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
    }


def run_backtest(frame_1h: pd.DataFrame, config: dict) -> tuple[list[Trade], dict]:
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

        position_blocked = position_until is not None and timestamp < position_until
        cooldown_blocked = cooldown_until is not None and timestamp < cooldown_until
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
        trade = _simulate(decision, future, config)
        if trade:
            trades.append(trade)
            position_until = pd.Timestamp(trade.exit_time)
            cooldown_until = timestamp + cooldown
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
        "diagnostics": _finalize_diagnostics(diagnostics),
    }
    return trades, summary


def trades_as_dicts(trades: list[Trade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
