import pandas as pd

from btc_signal.backtest import _simulate, required_history_hours, resample
from btc_signal.strategy import Decision
from conftest import candles, config


def test_resample_excludes_incomplete_higher_timeframe_candle():
    frame = candles(6, freq="1h")
    result = resample(frame, "4h")
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2025-01-01 00:00:00+00:00")


def test_240_day_backtest_adds_warmup_and_holding_window():
    assert required_history_hours(240, 48) == (210 + 240) * 24 + 48


def test_simulate_rejects_invalid_short_stop_geometry():
    decision = Decision(
        timestamp="2026-01-01T00:00:00+00:00",
        state="signal",
        side="short",
        score=80,
        price=110.0,
        entry_low=100.0,
        entry_high=102.0,
        stop=105.0,
        target_1=95.0,
        target_2=90.0,
        rr=0.0,
        regime="short_allowed",
        level=None,
        factors={},
        vetoes=[],
    )
    assert _simulate(decision, candles(4), config()) is None
