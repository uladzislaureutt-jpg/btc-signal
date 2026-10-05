from dataclasses import replace

import pandas as pd

from btc_signal.short_reversion import ShortParameters, run_short_backtest


def _market(groups: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for group_open, high, low, close in groups:
        for hour in range(4):
            rows.append((group_open if hour == 0 else close, high, low, close))
    index = pd.date_range("2020-01-01", periods=len(rows), freq="1h", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    frame["close_time"] = index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return frame


def _funding(times: list[str] | None = None, rates: list[float] | None = None) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(times or [], utc=True))
    return pd.DataFrame({
        "funding_rate": rates or [],
        "interval_hours": [8.0] * len(index),
    }, index=index)


def _params() -> ShortParameters:
    return ShortParameters(
        resistance_lookback_bars=3,
        atr_period=2,
        rsi_period=2,
        min_rsi=60,
        rejection_close_location_max=0.5,
        initial_stop_atr=2,
        target_r=2,
        min_risk_to_cost=0,
        max_holding_days=2,
        cooldown_hours=0,
        round_trip_cost_bps=0,
        regime_lookback_days=1,
    )


def test_rejection_enters_next_hour_and_uses_stop_first():
    frame = _market([
        (100, 101, 99, 100),
        (102, 103, 101, 102),
        (104, 105, 103, 104),
        (106, 106, 102, 104),
        (104, 112, 80, 90),
    ])
    trades, _ = run_short_backtest(
        frame, _funding(), evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=_params(), detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].signal_time == "2020-01-01T15:59:59.999000+00:00"
    assert trades[0].entry_time == "2020-01-01T16:00:00+00:00"
    assert trades[0].entry == 104
    assert trades[0].result == "stop"


def test_cost_gate_uses_actual_entry_risk():
    frame = _market([
        (100, 101, 99, 100),
        (102, 103, 101, 102),
        (104, 105, 103, 104),
        (106, 106, 102, 104),
        (104, 106, 100, 102),
    ])
    params = replace(_params(), round_trip_cost_bps=100, min_risk_to_cost=100)
    trades, summary = run_short_backtest(
        frame, _funding(), evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=params, detailed=False,
    )
    assert trades == []
    assert summary["diagnostics"]["cost_gate_rejections"] == 1


def test_positive_funding_is_income_for_short():
    frame = _market([
        (100, 101, 99, 100),
        (102, 103, 101, 102),
        (104, 105, 103, 104),
        (106, 106, 102, 104),
        (104, 106, 100, 102),
        (102, 104, 99, 101),
        (101, 103, 98, 100),
    ])
    funding = _funding(["2020-01-02T00:00:00Z"], [0.001])
    trades, _ = run_short_backtest(
        frame, funding, evaluation_start="2020-01-01",
        evaluation_end="2020-01-03", params=_params(), detailed=False,
    )
    assert len(trades) == 1
    assert trades[0].funding_events == 1
    assert trades[0].funding_r > 0
