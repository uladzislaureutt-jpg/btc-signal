from dataclasses import replace

import pandas as pd

from btc_signal.trend_v030 import TrendV030Parameters, run_trend_v030_backtest


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


def _funding(rows: list[tuple[str, float]] | None = None) -> pd.DataFrame:
    rows = rows or []
    index = pd.DatetimeIndex([pd.Timestamp(timestamp) for timestamp, _ in rows], tz="UTC")
    return pd.DataFrame(
        {"funding_rate": [rate for _, rate in rows], "interval_hours": 8.0},
        index=index,
    )


def _params() -> TrendV030Parameters:
    return TrendV030Parameters(
        breakout_bars=2,
        ema_period=2,
        atr_period=2,
        initial_stop_atr=2.0,
        trailing_atr=1.0,
        trailing_activation_atr=1.0,
        min_risk_to_cost=0.0,
        max_holding_days=2,
        round_trip_cost_bps=0,
        regime_lookback_days=1,
        regime_threshold=0.15,
    )


def test_long_breakout_enters_at_exact_next_1h_open():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (103, 105, 102, 104),
        (105, 106, 50, 100),
    ])
    trades, summary = run_trend_v030_backtest(
        frame, _funding(), evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=_params(), side_mode="long",
        detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].side == "long"
    assert trades[0].signal_time == "2020-01-01T11:59:59.999000+00:00"
    assert trades[0].entry_time == "2020-01-01T12:00:00+00:00"
    assert trades[0].entry == 105
    assert summary["side_mode"] == "long"


def test_short_breakout_enters_at_exact_next_1h_open():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (99, 100, 98, 99),
        (97, 98, 95, 96),
        (95, 110, 94, 100),
    ])
    trades, summary = run_trend_v030_backtest(
        frame, _funding(), evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=_params(), side_mode="short",
        detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].side == "short"
    assert trades[0].signal_time == "2020-01-01T11:59:59.999000+00:00"
    assert trades[0].entry_time == "2020-01-01T12:00:00+00:00"
    assert trades[0].entry == 95
    assert summary["side_mode"] == "short"


def test_positive_funding_is_income_for_short_and_cost_for_long():
    long_frame = _four_hour_groups([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (103, 105, 102, 104),
        (105, 106, 103, 105),
        (105, 106, 103, 105),
    ])
    short_frame = _four_hour_groups([
        (100, 101, 99, 100),
        (99, 100, 98, 99),
        (97, 98, 95, 96),
        (95, 97, 94, 95),
        (95, 97, 94, 95),
    ])
    funding = _funding([("2020-01-01T16:00:00Z", 0.001)])
    params = replace(_params(), initial_stop_atr=10.0)
    longs, _ = run_trend_v030_backtest(
        long_frame, funding, evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=params, side_mode="long", detailed=False,
    )
    shorts, _ = run_trend_v030_backtest(
        short_frame, funding, evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=params, side_mode="short", detailed=False,
    )
    assert longs[0].funding_r < 0
    assert shorts[0].funding_r > 0


def test_cost_gate_uses_actual_entry_and_side_specific_counter():
    frame = _four_hour_groups([
        (100, 101, 99, 100),
        (99, 100, 98, 99),
        (97, 98, 95, 96),
        (95, 110, 94, 100),
    ])
    params = replace(_params(), round_trip_cost_bps=100, min_risk_to_cost=100)
    trades, summary = run_trend_v030_backtest(
        frame, _funding(), evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=params, side_mode="short", detailed=False,
    )
    assert trades == []
    assert summary["diagnostics"]["cost_gate_rejections_short"] == 1
