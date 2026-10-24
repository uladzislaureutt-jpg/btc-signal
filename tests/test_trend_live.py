import json

import pandas as pd

from btc_signal.strategy import Decision
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
    assert decision.strategy_version == "0.4.1"


def test_short_signal_has_reversed_price_geometry():
    frame = _breakout_frame("short")
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 45_000.0, now=now)
    assert decision.state == "signal"
    assert decision.side == "short"
    assert decision.stop > decision.entry_low > decision.target_1 > decision.target_2


def test_wick_breakdown_alone_does_not_trigger_early_short():
    frame = candles(260, freq="4h", trend=30.0)
    frame["high"] = frame["close"] + 900.0
    frame["low"] = frame["close"] - 900.0
    prior_low = frame["low"].iloc[-21:-1].min()
    frame.loc[frame.index[-1], "close"] = prior_low - 600.0
    frame.loc[frame.index[-1], "high"] = frame["close"].iloc[-1] + 900.0
    frame.loc[frame.index[-1], "low"] = frame["close"].iloc[-1] - 900.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 55_000.0, now=now)
    assert decision.state == "neutral"


def test_early_short_requires_confirmed_breakdown_below_ema():
    frame = candles(260, freq="4h", trend=0.0)
    frame["high"] = frame["close"] + 300.0
    frame["low"] = frame["close"] - 300.0
    base = float(frame["close"].iloc[-4])
    stair = [base - 900.0, base - 2200.0, base - 3800.0]
    for offset, close in zip((-3, -2, -1), stair):
        idx = frame.index[offset]
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 300.0
        frame.loc[idx, "low"] = close - 300.0
    old_idx = frame.index[-10]
    frame.loc[old_idx, "low"] = stair[-1] - 1800.0
    cfg = config()
    cfg["fast_trend"]["enabled"] = False
    cfg["structure"]["late_entry_max_room_atr"] = 0.0
    cfg["structure"]["minimum_tp1_r"] = 0.0
    cfg["structure"]["minimum_tp2_r"] = 0.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", stair[-1], now=now)
    assert decision.state == "signal"
    assert decision.side == "short"
    assert decision.score == 90
    assert decision.factors["early_short"]["close_breakdown"] is True
    assert decision.factors["early_short"]["momentum_breakdown"] is True
    assert decision.factors["early_short"]["ema_confirmed"] is True
    assert decision.factors["early_short"]["ema_separation_atr"] >= 0.25


def test_small_drift_does_not_trigger_early_short():
    frame = candles(260, freq="4h", trend=25.0)
    frame["high"] = frame["close"] + 700.0
    frame["low"] = frame["close"] - 700.0
    base = float(frame["close"].iloc[-4])
    for offset, close in zip((-3, -2, -1), (base - 40.0, base - 65.0, base - 80.0)):
        idx = frame.index[offset]
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 700.0
        frame.loc[idx, "low"] = close - 700.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", base - 80.0, now=now)
    assert decision.state == "neutral"


def test_early_long_requires_confirmed_recovery_above_ema():
    frame = candles(260, freq="4h", trend=5.0)
    frame["high"] = frame["close"] + 350.0
    frame["low"] = frame["close"] - 350.0
    base = float(frame["close"].iloc[-4])
    stair = [base + 500.0, base + 1300.0, base + 2400.0]
    for offset, close in zip((-3, -2, -1), stair):
        idx = frame.index[offset]
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 350.0
        frame.loc[idx, "low"] = close - 350.0
    old_idx = frame.index[-10]
    frame.loc[old_idx, "high"] = stair[-1] + 1600.0
    cfg = config()
    cfg["fast_trend"]["enabled"] = False
    cfg["structure"]["minimum_tp1_r"] = 0.0
    cfg["structure"]["minimum_tp2_r"] = 0.0
    cfg["structure"]["late_entry_max_room_atr"] = 0.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", stair[-1], now=now)
    assert decision.state == "signal"
    assert decision.side == "long"
    assert decision.score == 90
    assert decision.factors["early_long"]["close_breakout"] is True
    assert decision.factors["early_long"]["momentum_breakout"] is True
    assert decision.factors["early_long"]["ema_confirmed"] is True
    assert decision.factors["early_long"]["ema_separation_atr"] >= 0.25


def test_near_ema_early_long_is_rejected_even_with_momentum():
    frame = candles(260, freq="4h", trend=0.0)
    frame["high"] = frame["close"] + 350.0
    frame["low"] = frame["close"] - 350.0
    # Force strong short-term momentum but keep the final close near EMA200.
    base = float(frame["close"].iloc[-4])
    for offset, close in zip((-3, -2, -1), (base - 500.0, base + 250.0, base + 700.0)):
        idx = frame.index[offset]
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 350.0
        frame.loc[idx, "low"] = close - 350.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", base + 700.0, now=now)
    if decision.state == "signal":
        assert decision.factors["signal_type"]["value"] == "trend_breakout"
    else:
        assert decision.state == "neutral"


def test_live_cost_filter_accepts_ratio_above_four():
    frame = _breakout_frame("long")
    cfg = config()
    cfg["trend_bidirectional"]["min_risk_to_cost"] = 4.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", 80_000.0, now=now)
    assert decision.state == "signal"
    assert decision.factors["cost_filter"]["minimum"] == 4.0



def test_structural_fibonacci_caps_short_targets_and_blocks_late_entry():
    frame = candles(260, freq="4h", trend=0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + 30_000.0
    # Create an ordered upswing inside the 30-day lookback: low first, high later.
    low_idx = frame.index[-160]
    high_idx = frame.index[-40]
    frame.loc[low_idx, "low"] = 75_000.0
    frame.loc[low_idx, "close"] = 75_500.0
    frame.loc[high_idx, "high"] = 87_500.0
    frame.loc[high_idx, "close"] = 87_000.0
    # Keep recent volatility near the production episode (~900 ATR).
    for idx in frame.index[-14:]:
        close = float(frame.loc[idx, "close"])
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    # Current short is beyond the 50% retracement and close to the 0.618 level.
    for offset, close in zip((-4, -3, -2, -1), (83_700.0, 82_800.0, 81_800.0, 80_900.0)):
        idx = frame.index[offset]
        frame.loc[idx, "open"] = close + 150.0
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    cfg = config()
    cfg["fast_trend"]["enabled"] = False
    cfg["trend_bidirectional"]["min_risk_to_cost"] = 0.0
    cfg["trend_bidirectional"]["ema_period"] = 2
    cfg["early_short"]["min_close_breakout_atr"] = 0.0
    cfg["early_short"]["close_breakout_bars"] = 1
    cfg["early_short"]["min_momentum_atr"] = 0.0
    cfg["early_short"]["min_lower_closes"] = 1
    cfg["early_short"]["ema_confirm_bars"] = 1
    cfg["early_short"]["min_ema_separation_atr"] = -10.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", 80_900.0, now=now)
    assert decision.state == "neutral"
    assert "late_entry_near_structure" in decision.vetoes
    assert decision.factors["fibonacci"]["progress"] >= 0.5
    assert decision.factors["fibonacci"]["room_atr"] < 1.5



def test_structural_target_skips_too_close_level_and_preserves_minimum_reward():
    frame = candles(260, freq="4h", trend=0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + 30_000.0
    low_idx = frame.index[-160]
    high_idx = frame.index[-40]
    frame.loc[low_idx, "low"] = 75_000.0
    frame.loc[low_idx, "close"] = 75_500.0
    frame.loc[high_idx, "high"] = 87_500.0
    frame.loc[high_idx, "close"] = 87_000.0
    for idx in frame.index[-14:]:
        close = float(frame.loc[idx, "close"])
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    for offset, close in zip((-4, -3, -2, -1), (80_700.0, 81_300.0, 82_000.0, 82_380.0)):
        idx = frame.index[offset]
        frame.loc[idx, "open"] = close - 120.0
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    cfg = config()
    cfg["trend_bidirectional"]["min_risk_to_cost"] = 0.0
    cfg["structure"]["late_entry_max_room_atr"] = 0.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", 82_382.0, now=now)
    assert decision.state == "signal"
    assert decision.side == "long"
    risk = decision.entry_low - decision.stop
    assert (decision.target_1 - decision.entry_low) / risk >= 1.0
    assert (decision.target_2 - decision.entry_low) / risk >= 1.5
    assert decision.factors["fibonacci"]["target_1_r"] >= 1.0
    assert decision.factors["fibonacci"]["target_2_r"] >= 1.5


def test_signal_is_rejected_when_no_structural_level_offers_one_r():
    frame = candles(260, freq="4h", trend=0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + 30_000.0
    high_idx = frame.index[-160]
    low_idx = frame.index[-40]
    frame.loc[high_idx, "high"] = 83_000.0
    frame.loc[high_idx, "close"] = 82_900.0
    frame.loc[low_idx, "low"] = 75_000.0
    frame.loc[low_idx, "close"] = 75_500.0
    for idx in frame.index[-14:]:
        close = float(frame.loc[idx, "close"])
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    for offset, close in zip((-4, -3, -2, -1), (80_700.0, 81_300.0, 82_000.0, 82_380.0)):
        idx = frame.index[offset]
        frame.loc[idx, "open"] = close - 120.0
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 500.0
        frame.loc[idx, "low"] = close - 500.0
    cfg = config()
    cfg["trend_bidirectional"]["min_risk_to_cost"] = 0.0
    cfg["structure"]["late_entry_max_room_atr"] = 0.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", 82_382.0, now=now)
    assert decision.state == "neutral"
    assert "insufficient_structural_reward" in decision.vetoes



def test_fast_short_catches_strong_impulse_before_second_ema_close():
    frame = candles(260, freq="4h", trend=0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + 33_500.0
    frame["high"] = frame["close"] + 450.0
    frame["low"] = frame["close"] - 450.0

    # A sharp 3-candle decline comparable to the 22 Oct production episode.
    closes = [84_500.0, 84_050.0, 83_800.0, 82_200.0]
    for offset, close in zip((-4, -3, -2, -1), closes):
        idx = frame.index[offset]
        frame.loc[idx, "open"] = close + 100.0
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 450.0
        frame.loc[idx, "low"] = close - 450.0
    # Keep the 20-bar wick floor below price, as in production on 22 Oct.
    frame.loc[frame.index[-10], "low"] = 80_500.0

    cfg = config()
    cfg["structure"]["late_entry_max_room_atr"] = 0.0
    cfg["structure"]["minimum_tp1_r"] = 10.0
    cfg["structure"]["minimum_tp2_r"] = 10.0
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, cfg, "test_perpetual", 82_150.0, now=now)

    assert decision.state == "signal"
    assert decision.side == "short"
    assert decision.score == 95
    assert decision.factors["signal_type"]["value"] == "fast_trend_short"
    assert decision.factors["fast_trend"]["confirmed"] is True
    assert decision.factors["early_short"]["ema_confirmed"] is False
    assert decision.strategy_version == "0.4.1"


def test_decision_payload_replaces_non_finite_values_with_null():
    decision = Decision(
        timestamp="2026-10-24T16:00:00+00:00",
        state="signal",
        side="short",
        score=95,
        price=79_700.0,
        entry_low=79_700.0,
        entry_high=79_700.0,
        stop=81_000.0,
        target_1=78_000.0,
        target_2=77_000.0,
        rr=2.0,
        regime="fast_short_below_ema200",
        level={"type": "fast_trend", "price": 80_000.0},
        factors={"fibonacci": {"room_atr": float("inf"), "other": float("nan")}},
        vetoes=[],
        provider="okx_swap",
        strategy_version="0.4.1",
    )
    payload = decision.to_dict()
    assert payload["factors"]["fibonacci"]["room_atr"] is None
    assert payload["factors"]["fibonacci"]["other"] is None
    json.dumps(payload, allow_nan=False)



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


def test_same_breakout_deduplicates_when_live_price_moves():
    frame = _breakout_frame("long")
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    first = evaluate_trend_signal(frame, config(), "test_perpetual", 55_000.0, now=now)
    second = evaluate_trend_signal(frame, config(), "test_perpetual", 55_100.0, now=now)
    assert first.state == second.state == "signal"
    assert first.dedup_key == second.dedup_key
