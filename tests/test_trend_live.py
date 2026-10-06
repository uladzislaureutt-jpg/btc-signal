import pandas as pd

from btc_signal.trend_live import evaluate_trend_signal
from conftest import candles, config


def _breakout_frame(side: str) -> pd.DataFrame:
    trend = 15.0 if side == "long" else -15.0
    frame = candles(260, freq="4h", trend=trend)
    frame["high"] = frame["close"] + 900.0
    frame["low"] = frame["close"] - 900.0
    if side == "long":
        frame.loc[frame.index[-1], "close"] = frame["high"].iloc[-21:-1].max() + 250.0
        frame.loc[frame.index[-1], "high"] = frame["close"].iloc[-1] + 900.0
        frame.loc[frame.index[-1], "low"] = frame["close"].iloc[-1] - 900.0
    else:
        frame.loc[frame.index[-1], "close"] = frame["low"].iloc[-21:-1].min() - 250.0
        frame.loc[frame.index[-1], "high"] = frame["close"].iloc[-1] + 900.0
        frame.loc[frame.index[-1], "low"] = frame["close"].iloc[-1] - 900.0
    return frame


def test_long_signal_contains_entry_stop_targets_and_trailing_plan():
    frame = _breakout_frame("long")
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 55_000.0, now=now)
    assert decision.state == "signal"
    assert decision.side == "long"
    assert decision.entry_low == decision.entry_high == 55_000.0
    assert decision.stop < decision.entry_low < decision.target_1 < decision.target_2
    assert decision.factors["exit_plan"]["trailing_distance_atr"] == 3.0
    assert decision.strategy_version == "0.3.1"


def test_short_signal_has_reversed_price_geometry():
    frame = _breakout_frame("short")
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 45_000.0, now=now)
    assert decision.state == "signal"
    assert decision.side == "short"
    assert decision.stop > decision.entry_low > decision.target_1 > decision.target_2


def test_stale_breakout_is_not_emitted():
    frame = _breakout_frame("long")
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(hours=2)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 55_000.0, now=now)
    assert decision.state == "neutral"
    assert "stale_breakout" in decision.vetoes


def test_no_breakout_is_neutral():
    frame = candles(260, freq="4h", trend=0.0)
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 50_000.0, now=now)
    assert decision.state == "neutral"
    assert "no_4h_trend_breakout" in decision.vetoes
