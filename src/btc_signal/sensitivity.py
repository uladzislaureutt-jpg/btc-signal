from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import (
    BACKTEST_WARMUP_HOURS,
    Trade,
    _execution_rejection_reason,
    _simulate,
    audit_market_data,
    prepare_backtest_frames,
)
from .strategy import Decision, evaluate_enriched_with_trace
from .validation import (
    fixed_walk_forward,
    moving_block_bootstrap,
    outcome_metrics,
    time_breakdown,
    trade_breakdown,
)


DEFAULT_GRID = {
    "min_score": (70, 75, 80),
    "min_rr": (1.6, 1.8, 2.0),
    "max_distance_atr": (0.8, 1.0, 1.2),
    "max_cost_r": (0.5, 0.6, 0.7),
}

_VARIABLE_VETOES = {"rr_below_minimum", "cost_above_limit", "late_entry"}


@dataclass(frozen=True)
class TapeCandidate:
    decision: Decision
    distance_atr: float


@dataclass(frozen=True)
class TapePoint:
    frame_index: int
    evaluation_time: pd.Timestamp
    candidates: tuple[TapeCandidate, ...]


def _set_parameter(config: dict, name: str, value: float | int) -> None:
    paths = {
        "min_score": ("signal", "min_score"),
        "min_rr": ("risk", "min_rr"),
        "max_distance_atr": ("levels", "max_distance_atr"),
        "max_cost_r": ("risk", "max_cost_r"),
    }
    section, key = paths[name]
    config[section][key] = value


def _get_parameter(config: dict, name: str) -> float | int:
    paths = {
        "min_score": ("signal", "min_score"),
        "min_rr": ("risk", "min_rr"),
        "max_distance_atr": ("levels", "max_distance_atr"),
        "max_cost_r": ("risk", "max_cost_r"),
    }
    section, key = paths[name]
    return config[section][key]


def _envelope_config(config: dict, grid: dict[str, Iterable[float | int]]) -> dict:
    envelope = deepcopy(config)
    _set_parameter(envelope, "min_score", min(grid["min_score"]))
    _set_parameter(envelope, "min_rr", min(grid["min_rr"]))
    _set_parameter(envelope, "max_distance_atr", max(grid["max_distance_atr"]))
    _set_parameter(envelope, "max_cost_r", max(grid["max_cost_r"]))
    return envelope


def build_decision_tape(
    frame_1h: pd.DataFrame,
    config: dict,
    grid: dict[str, Iterable[float | int]] = DEFAULT_GRID,
) -> tuple[list[TapePoint], dict]:
    """Evaluate the widest parameter envelope once for deterministic replay."""
    data_quality = audit_market_data(frame_1h)
    if data_quality["status"] == "error":
        raise ValueError(f"Invalid market data: {', '.join(data_quality['errors'])}")
    envelope = _envelope_config(config, grid)
    enriched_1h, enriched_4h, enriched_1d = prepare_backtest_frames(frame_1h, envelope)
    max_holding = int(envelope["risk"]["max_holding_hours"])
    points: list[TapePoint] = []
    evaluation_errors = 0
    candidate_instances = 0
    for i in range(BACKTEST_WARMUP_HOURS, len(frame_1h) - max_holding):
        history_1h = enriched_1h.iloc[: i + 1]
        evaluation_time = pd.Timestamp(history_1h["close_time"].iloc[-1])
        history_4h = enriched_4h.loc[enriched_4h["close_time"] <= evaluation_time]
        history_1d = enriched_1d.loc[enriched_1d["close_time"] <= evaluation_time]
        try:
            _, trace = evaluate_enriched_with_trace(
                history_1d, history_4h, history_1h, envelope, "historical"
            )
        except ValueError:
            evaluation_errors += 1
            continue
        candidates = tuple(
            TapeCandidate(
                decision=candidate,
                distance_atr=float(trace.closest_distance_atr[candidate.side]),
            )
            for candidate in trace.candidates
            if candidate.side is not None
            and trace.closest_distance_atr.get(candidate.side) is not None
        )
        candidate_instances += len(candidates)
        points.append(TapePoint(i, evaluation_time, candidates))
    return points, {
        "evaluated_hours": len(points),
        "evaluation_errors": evaluation_errors,
        "candidate_instances": candidate_instances,
        "envelope": {
            name: _get_parameter(envelope, name) for name in DEFAULT_GRID
        },
        "data_quality": data_quality,
    }


def _variant_decision(candidate: TapeCandidate, config: dict) -> Decision:
    decision = candidate.decision
    vetoes = [item for item in decision.vetoes if item not in _VARIABLE_VETOES]
    if decision.rr is None or decision.rr < float(config["risk"]["min_rr"]):
        vetoes.append("rr_below_minimum")
    risk = (
        decision.price - float(decision.stop)
        if decision.side == "long"
        else float(decision.stop) - decision.price
    )
    costs = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    estimated_cost_r = costs * decision.price / risk if risk > 0 else float("inf")
    if estimated_cost_r > float(config["risk"]["max_cost_r"]):
        vetoes.append("cost_above_limit")
    if candidate.distance_atr > float(config["levels"]["max_distance_atr"]):
        vetoes.append("late_entry")
    if vetoes:
        state = "neutral"
    elif decision.score >= int(config["signal"]["strong_score"]):
        state = "strong_signal"
    elif decision.score >= int(config["signal"]["min_score"]):
        state = "signal"
    elif decision.score >= int(config["signal"]["watch_score"]):
        state = "watch"
    else:
        state = "neutral"
    return replace(decision, state=state, vetoes=vetoes)


def replay_decision_tape(
    points: Iterable[TapePoint], frame_1h: pd.DataFrame, config: dict
) -> list[Trade]:
    """Replay one parameter set with the production position/cooldown rules."""
    trades: list[Trade] = []
    position_until: pd.Timestamp | None = None
    cooldown_until: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=int(config["signal"]["cooldown_hours"]))
    distance_limit = float(config["levels"]["max_distance_atr"])
    for point in points:
        available = [
            candidate
            for candidate in point.candidates
            if candidate.distance_atr <= distance_limit + 1e-12
        ]
        if not available:
            continue
        selected = max(available, key=lambda item: item.decision.score)
        decision = _variant_decision(selected, config)
        position_blocked = position_until is not None and point.evaluation_time < position_until
        cooldown_blocked = cooldown_until is not None and point.evaluation_time < cooldown_until
        if position_blocked or cooldown_blocked:
            continue
        if decision.state not in ("signal", "strong_signal"):
            continue
        future = frame_1h.iloc[point.frame_index + 1 :]
        if _execution_rejection_reason(decision, future, config):
            continue
        trade = _simulate(decision, future, config, signal_time=point.evaluation_time.isoformat())
        if trade is not None:
            trades.append(trade)
            position_until = pd.Timestamp(trade.exit_time)
            cooldown_until = pd.Timestamp(trade.entry_time) + cooldown
    return trades


def _scenario_summary(
    trades: list[Trade], evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp
) -> dict:
    return {
        **outcome_metrics(trade.r_multiple for trade in trades),
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "total_cost_r": float(sum(trade.cost_r for trade in trades)),
        "average_cost_r": float(np.mean([trade.cost_r for trade in trades])) if trades else 0.0,
        "average_risk_pct": float(np.mean([trade.risk_pct for trade in trades])) if trades else 0.0,
        "by_side": trade_breakdown(trades, "side"),
        "by_year": time_breakdown(trades, "year"),
        "bootstrap": moving_block_bootstrap(trades),
        "walk_forward": fixed_walk_forward(
            trades, evaluation_start, evaluation_end, train_days=180, test_days=90
        ),
    }


def _scenario_configs(config: dict, grid: dict[str, Iterable[float | int]]) -> list[tuple[str, str | None, float | int | None, dict]]:
    scenarios: list[tuple[str, str | None, float | int | None, dict]] = [
        ("baseline", None, None, deepcopy(config))
    ]
    for name, values in grid.items():
        baseline_value = _get_parameter(config, name)
        for value in values:
            if value == baseline_value:
                continue
            variant = deepcopy(config)
            _set_parameter(variant, name, value)
            scenarios.append((f"{name}={value}", name, value, variant))
    return scenarios


def run_sensitivity(
    frame_1h: pd.DataFrame,
    config: dict,
    grid: dict[str, Iterable[float | int]] = DEFAULT_GRID,
) -> dict:
    points, tape_summary = build_decision_tape(frame_1h, config, grid)
    max_holding = int(config["risk"]["max_holding_hours"])
    evaluation_start = frame_1h.index[BACKTEST_WARMUP_HOURS]
    evaluation_end = pd.Timestamp(frame_1h["close_time"].iloc[len(frame_1h) - max_holding - 1])
    scenarios = []
    for label, parameter, value, scenario_config in _scenario_configs(config, grid):
        trades = replay_decision_tape(points, frame_1h, scenario_config)
        scenarios.append({
            "label": label,
            "parameter": parameter,
            "value": value,
            "config": {
                name: _get_parameter(scenario_config, name) for name in DEFAULT_GRID
            },
            **_scenario_summary(trades, evaluation_start, evaluation_end),
        })
    positive = sum(item["expectancy_r"] > 0 for item in scenarios)
    return {
        "method": "one_at_a_time_causal_decision_tape_replay",
        "note": "Each scenario changes one parameter from baseline; results are robustness diagnostics, not tuned production settings.",
        "grid": {name: list(values) for name, values in grid.items()},
        "scenario_count": len(scenarios),
        "positive_expectancy_scenarios": positive,
        "tape": tape_summary,
        "scenarios": scenarios,
    }


def verify_baseline(expected: dict, sensitivity: dict, tolerance: float = 1e-9) -> dict:
    baseline = next(item for item in sensitivity["scenarios"] if item["label"] == "baseline")
    keys = ("trades", "total_r", "gross_total_r", "total_cost_r", "profit_factor", "max_drawdown_r")
    differences: dict[str, dict[str, float | int]] = {}
    for key in keys:
        left = expected[key]
        right = baseline[key]
        if key == "trades":
            matches = left == right
        else:
            matches = bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
        if not matches:
            differences[key] = {"backtest": left, "replay": right}
    return {
        "status": "passed" if not differences else "failed",
        "tolerance": tolerance,
        "checked_metrics": list(keys),
        "differences": differences,
    }
