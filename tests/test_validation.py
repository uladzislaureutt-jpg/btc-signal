from types import SimpleNamespace

import pandas as pd

from btc_signal.validation import (
    fixed_walk_forward,
    moving_block_bootstrap,
    outcome_metrics,
    time_breakdown,
)


def _trade(entry_time: str, result: float):
    return SimpleNamespace(entry_time=entry_time, r_multiple=result)


def test_drawdown_includes_initial_zero_equity():
    metrics = outcome_metrics([-1.0, 2.0])
    assert metrics["total_r"] == 1.0
    assert metrics["max_drawdown_r"] == -1.0


def test_bootstrap_is_reproducible():
    trades = [_trade(f"2025-01-{day:02d}T00:00:00+00:00", value) for day, value in enumerate([1.0, -1.0, 2.0, -0.5], start=1)]
    first = moving_block_bootstrap(trades, samples=100, seed=7, block_size=2)
    second = moving_block_bootstrap(trades, samples=100, seed=7, block_size=2)
    assert first == second
    assert first["trades"] == 4
    assert len(first["expectancy_r_ci95"]) == 2


def test_time_breakdown_uses_entry_year_and_quarter():
    trades = [
        _trade("2025-12-31T23:00:00+00:00", 1.0),
        _trade("2026-01-01T00:00:00+00:00", -1.0),
    ]
    years = time_breakdown(trades, "year")
    quarters = time_breakdown(trades, "quarter")
    assert years["2025"]["total_r"] == 1.0
    assert years["2026"]["total_r"] == -1.0
    assert set(quarters) == {"2025-Q4", "2026-Q1"}


def test_fixed_walk_forward_keeps_train_and_test_separate():
    trades = [
        _trade("2025-02-01T00:00:00+00:00", 1.0),
        _trade("2025-03-15T00:00:00+00:00", -1.0),
        _trade("2025-04-15T00:00:00+00:00", 2.0),
    ]
    result = fixed_walk_forward(
        trades,
        pd.Timestamp("2025-01-01T00:00:00+00:00"),
        pd.Timestamp("2025-07-01T00:00:00+00:00"),
        train_days=90,
        test_days=30,
    )
    assert result["window_count"] == 3
    assert result["windows"][0]["train"]["trades"] == 2
    assert result["windows"][0]["test"]["trades"] == 1
