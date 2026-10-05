from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def outcome_metrics(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    wins = array[array > 0]
    losses = array[array <= 0]
    equity = np.concatenate(([0.0], array.cumsum()))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks
    return {
        "trades": int(len(array)),
        "win_rate": float((array > 0).mean()) if len(array) else 0.0,
        "expectancy_r": float(array.mean()) if len(array) else 0.0,
        "total_r": float(array.sum()) if len(array) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum()
        else None,
        "max_drawdown_r": float(drawdowns.min()) if len(drawdowns) else 0.0,
    }


def trade_metrics(trades: Sequence[object]) -> dict:
    return outcome_metrics(float(getattr(trade, "r_multiple")) for trade in trades)


def trade_breakdown(trades: Sequence[object], attribute: str) -> dict:
    groups: dict[str, list[object]] = defaultdict(list)
    for trade in trades:
        groups[str(getattr(trade, attribute))].append(trade)
    return {name: trade_metrics(items) for name, items in sorted(groups.items())}


def time_breakdown(trades: Sequence[object], period: str) -> dict:
    groups: dict[str, list[object]] = defaultdict(list)
    for trade in trades:
        timestamp = pd.Timestamp(getattr(trade, "entry_time"))
        if period == "year":
            key = str(timestamp.year)
        elif period == "quarter":
            key = f"{timestamp.year}-Q{timestamp.quarter}"
        else:
            raise ValueError(f"Unsupported period: {period}")
        groups[key].append(trade)
    return {name: trade_metrics(items) for name, items in sorted(groups.items())}


def moving_block_bootstrap(
    trades: Sequence[object],
    *,
    samples: int = 5000,
    seed: int = 20261005,
    block_size: int | None = None,
) -> dict:
    values = np.asarray([float(getattr(trade, "r_multiple")) for trade in trades], dtype=float)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not len(values):
        return {
            "method": "moving_block_bootstrap",
            "samples": samples,
            "seed": seed,
            "block_size": 0,
            "trades": 0,
            "expectancy_r_ci95": [0.0, 0.0],
            "total_r_ci95": [0.0, 0.0],
            "max_drawdown_r_ci95": [0.0, 0.0],
            "probability_expectancy_positive": 0.0,
        }
    size = block_size or max(2, min(10, round(math.sqrt(len(values)))))
    size = min(size, len(values))
    rng = np.random.default_rng(seed)
    block_count = math.ceil(len(values) / size)
    expectancy = np.empty(samples)
    totals = np.empty(samples)
    drawdowns = np.empty(samples)
    offsets = np.arange(size)
    for sample in range(samples):
        starts = rng.integers(0, len(values), size=block_count)
        indices = ((starts[:, None] + offsets) % len(values)).reshape(-1)[: len(values)]
        selected = values[indices]
        metrics = outcome_metrics(selected)
        expectancy[sample] = metrics["expectancy_r"]
        totals[sample] = metrics["total_r"]
        drawdowns[sample] = metrics["max_drawdown_r"]
    return {
        "method": "moving_block_bootstrap",
        "samples": samples,
        "seed": seed,
        "block_size": size,
        "trades": int(len(values)),
        "expectancy_r_ci95": [float(x) for x in np.quantile(expectancy, [0.025, 0.975])],
        "total_r_ci95": [float(x) for x in np.quantile(totals, [0.025, 0.975])],
        "max_drawdown_r_ci95": [float(x) for x in np.quantile(drawdowns, [0.025, 0.975])],
        "probability_expectancy_positive": float((expectancy > 0).mean()),
    }


def fixed_walk_forward(
    trades: Sequence[object],
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    *,
    train_days: int = 180,
    test_days: int = 90,
) -> dict:
    if train_days <= 0 or test_days <= 0:
        raise ValueError("walk-forward windows must be positive")
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    dated = [(pd.Timestamp(getattr(trade, "entry_time")), trade) for trade in trades]
    windows = []
    test_start = start + pd.Timedelta(days=train_days)
    while test_start + pd.Timedelta(days=test_days) <= end:
        train_start = test_start - pd.Timedelta(days=train_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        train = [trade for timestamp, trade in dated if train_start <= timestamp < test_start]
        test = [trade for timestamp, trade in dated if test_start <= timestamp < test_end]
        windows.append({
            "train_start": train_start.isoformat(),
            "train_end": test_start.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train": trade_metrics(train),
            "test": trade_metrics(test),
        })
        test_start += pd.Timedelta(days=test_days)
    positive_tests = sum(1 for window in windows if window["test"]["total_r"] > 0)
    nonempty_tests = sum(1 for window in windows if window["test"]["trades"] > 0)
    return {
        "method": "fixed_parameter_rolling_walk_forward",
        "note": "No parameters are selected on train windows; tuning walk-forward follows after sensitivity analysis.",
        "train_days": train_days,
        "test_days": test_days,
        "windows": windows,
        "window_count": len(windows),
        "nonempty_test_windows": nonempty_tests,
        "positive_test_windows": positive_tests,
    }
