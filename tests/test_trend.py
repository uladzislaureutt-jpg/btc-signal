from dataclasses import replace

import pandas as pd

from btc_signal.trend import TrendParameters, run_trend_backtest


def _four_hour_groups(groups: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for group_open, high, low, close in groups:
        for hour in range(4):
            rows.append((group_open if hour == 0 else close, high, low, close))
    index = pd.date_range("2020-01-01", periods=len(rows), freq="1h", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    frame["close_time"] = index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return frame


def _params() -> TrendParameters:
    return TrendParameters(
        breakout_bars=2,
        ema_period=2,
        ema_slope_bars=1,
        atr_period=2,
        initial_stop_atr=2.0,
        trailing_atr=1.0,
        trailing_activation_atr=1.0,
        min_risk_to_cost=0.0,
        max_holding_days=2,
        require_channel_reset=False,
        regime_lookback_days=1,
        regime_threshold=0.15,
        round_trip_cost_bps=0,
    )


def test_breakout_enters_at_next_1h_open():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (103, 105, 102, 104),
        (105, 106, 50, 100),
    ])
    trades, _ = run_trend_backtest(
        frame,
        evaluation_start="2020-01-01",
        evaluation_end="2020-01-03",
        params=_params(),
        detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].signal_time == "2020-01-01T11:59:59.999000+00:00"
    assert trades[0].entry_time == "2020-01-01T12:00:00+00:00"
    assert trades[0].entry == 105
    assert trades[0].result == "stop"


def test_cost_gate_is_checked_at_actual_entry():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (103, 105, 102, 104),
        (105, 106, 50, 100),
    ])
    params = replace(_params(), round_trip_cost_bps=100, min_risk_to_cost=100)
    trades, summary = run_trend_backtest(
        frame,
        evaluation_start="2020-01-01",
        evaluation_end="2020-01-03",
        params=params,
        detailed=False,
    )
    assert trades == []
    assert summary["diagnostics"]["cost_gate_rejections"] == 1


def test_trailing_stop_updates_only_after_completed_4h_bar():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (103, 105, 102, 104),
        (105, 113, 100, 112),
        (112, 113, 103, 106),
    ])
    trades, _ = run_trend_backtest(
        frame,
        evaluation_start="2020-01-01",
        evaluation_end="2020-01-03",
        params=_params(),
        detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].trail_activated is True
    assert trades[0].exit_stop > trades[0].initial_stop
    assert trades[0].exit_time == "2020-01-01T16:59:59.999000+00:00"
