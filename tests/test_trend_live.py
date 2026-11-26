import json
from dataclasses import replace

import pandas as pd

from btc_signal import __version__
from btc_signal.strategy import Decision
from btc_signal.trend_live import assess_early_entry_stage, assess_exceptional_entry, assess_trend_breakout_quality, episode_reset_status, evaluate_trend_signal
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
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 55_200.0, now=now)
    assert decision.state == "signal"
    assert decision.side == "long"
    assert decision.entry_low == decision.entry_high == 55_200.0
    assert decision.stop < decision.entry_low < decision.target_1 < decision.target_2
    assert decision.factors["exit_plan"]["trailing_distance_atr"] == 3.0
    assert decision.strategy_version == __version__
    assert decision.dedup_key == replace(decision, strategy_version="0.4.3").dedup_key


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
    cfg["structure"]["early_max_ema_distance_atr"] = 999.0
    cfg["structure"]["early_min_extreme_room_atr"] = 0.0
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
    cfg["structure"]["early_max_ema_distance_atr"] = 999.0
    cfg["structure"]["early_min_extreme_room_atr"] = 0.0
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
    cfg["structure"]["early_max_ema_distance_atr"] = 999.0
    cfg["structure"]["early_min_extreme_room_atr"] = 0.0
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
    cfg["structure"]["early_max_ema_distance_atr"] = 999.0
    cfg["structure"]["early_min_extreme_room_atr"] = 0.0
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
    cfg["structure"]["early_max_ema_distance_atr"] = 999.0
    cfg["structure"]["early_min_extreme_room_atr"] = 0.0
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
    assert decision.strategy_version == __version__


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
    first = evaluate_trend_signal(frame, config(), "test_perpetual", 55_200.0, now=now)
    second = evaluate_trend_signal(frame, config(), "test_perpetual", 55_300.0, now=now)
    assert first.state == second.state == "signal"
    assert first.dedup_key == second.dedup_key


def test_production_three_short_stage_replay():
    """Frozen 0.4.1 production snapshots: no OHLC lookahead, no synthetic prices."""
    cases = [
        # candle, entry, close, ema, atr, 20-bar floor, 30-day floor, expected veto
        ("2026-10-26", 79690.2, 79560.3, 82016.5986365635, 744.4771250947526, 79168.0, 79168.0),
        ("2026-11-02", 77540.1, 77486.2, 81155.84049338721, 746.5008954905356, 76260.0, 76260.0),
        ("2026-11-04", 76609.7, 76450.2, 81149.24544870135, 816.4275250292864, 76360.0, 75690.1),
    ]
    for date, entry, close, ema_value, atr_value, prior_low, swing_low in cases:
        stage, veto = assess_early_entry_stage(
            side="short", price=entry, close=close, ema_value=ema_value,
            atr_value=atr_value, prior_20bar_extreme=prior_low,
            prior_swing_extreme=swing_low, maximum_ema_distance_atr=3.0,
            minimum_extreme_room_atr=1.0,
        )
        assert veto == "early_trend_overextended", date
        assert stage["ema_distance_atr"] > 3.0


def test_prior_support_veto_is_not_replaced_by_fibonacci():
    stage, veto = assess_early_entry_stage(
        side="short", price=76609.7, close=76450.2, ema_value=81149.2454,
        atr_value=816.4275, prior_20bar_extreme=76360.0,
        prior_swing_extreme=75690.1, maximum_ema_distance_atr=99.0,
        minimum_extreme_room_atr=1.0,
    )
    assert veto == "early_entry_near_prior_extreme"
    assert 0 < stage["extreme_room_atr"] < 0.35
    assert stage["nearest_extreme"] == 76360.0


def test_both_sides_have_symmetric_stage_guards():
    _, veto = assess_early_entry_stage(
        side="long", price=60900.0, close=60800.0, ema_value=58500.0,
        atr_value=800.0, prior_20bar_extreme=61000.0,
        prior_swing_extreme=62000.0, maximum_ema_distance_atr=3.0,
        minimum_extreme_room_atr=1.0,
    )
    assert veto == "early_entry_near_prior_extreme"


def test_episode_reset_requires_real_countertrend_pullback():
    frame = candles(260, freq="4h", trend=0.0)
    values = [80000, 79500, 79000, 79200, 79500, 79600, 79200, 78800]
    for idx, close in zip(frame.index[-8:], values):
        frame.loc[idx, "close"] = close
    start = pd.Timestamp(frame["close_time"].iloc[-8]).isoformat()
    end = pd.Timestamp(frame["close_time"].iloc[-1]).isoformat()
    confirmed = episode_reset_status(
        frame, start, end, "short", 400.0,
        minimum_pullback_atr=1.25, minimum_counter_closes=2,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["maximum_pullback_atr"] >= 1.25


def test_episode_reset_stays_locked_without_meaningful_retracement():
    frame = candles(260, freq="4h", trend=0.0)
    values = [80000, 79800, 79500, 79600, 79400, 79000, 78900, 78500]
    for idx, close in zip(frame.index[-8:], values):
        frame.loc[idx, "close"] = close
    status = episode_reset_status(
        frame, pd.Timestamp(frame["close_time"].iloc[-8]).isoformat(),
        pd.Timestamp(frame["close_time"].iloc[-1]).isoformat(),
        "short", 400.0, minimum_pullback_atr=1.25, minimum_counter_closes=2,
    )
    assert status["confirmed"] is False


def test_episode_reset_stays_locked_when_only_one_countertrend_close():
    frame = candles(260, freq="4h", trend=0.0)
    values = [80000, 79000, 79600, 78900, 78400, 78000, 77700, 77300]
    for idx, close in zip(frame.index[-8:], values):
        frame.loc[idx, "close"] = close
    status = episode_reset_status(
        frame, pd.Timestamp(frame["close_time"].iloc[-8]).isoformat(),
        pd.Timestamp(frame["close_time"].iloc[-1]).isoformat(),
        "short", 400.0, minimum_pullback_atr=1.25, minimum_counter_closes=2,
    )
    assert status["confirmed"] is False


def _exception_frame(prev_close: float, close: float, opening: float, high: float, low: float):
    frame = candles(260, freq="4h", trend=0.0)
    prev = frame.index[-2]
    current = frame.index[-1]
    frame.loc[prev, "close"] = prev_close
    frame.loc[current, "open"] = opening
    frame.loc[current, "close"] = close
    frame.loc[current, "high"] = high
    frame.loc[current, "low"] = low
    return frame


def test_new_crash_route_catches_one_bar_without_three_lower_closes():
    frame = _exception_frame(50_000, 47_500, 50_000, 50_100, 47_400)
    result = assess_exceptional_entry(
        frame, config(), side="short", price=47_500, close=47_500,
        ema_value=49_700, atr_value=800, prior_20bar_extreme=45_000,
        prior_swing_extreme=45_000, momentum_atr=3.2,
        directional_closes=1, close_breakout_atr=2.8,
        previous_ema=50_000,
    )
    assert result["crash"] is True
    assert result["single_close_move_atr"] >= 2
    assert result["base"] is False


def test_crash_at_support_or_after_big_rebound_is_rejected():
    frame = _exception_frame(50_000, 47_500, 50_000, 50_100, 47_400)
    common = dict(
        frame=frame, config=config(), side="short", close=47_500,
        ema_value=49_700, atr_value=800, momentum_atr=3.2,
        directional_closes=1, close_breakout_atr=2.8,
        previous_ema=50_000,
    )
    near_support = assess_exceptional_entry(
        price=47_500, prior_20bar_extreme=47_200,
        prior_swing_extreme=46_500, **common,
    )
    after_rebound = assess_exceptional_entry(
        price=48_400, prior_20bar_extreme=45_000,
        prior_swing_extreme=45_000, **common,
    )
    assert near_support["crash"] is False
    assert near_support["extreme_room_atr"] < 1
    assert after_rebound["crash"] is False
    assert after_rebound["entry_deviation_atr"] > 0.60


def test_crash_avoids_chasing_a_prior_overextended_trend():
    frame = _exception_frame(46_000, 43_500, 46_000, 46_100, 43_400)
    result = assess_exceptional_entry(
        frame, config(), side="short", price=43_500, close=43_500,
        ema_value=49_000, atr_value=800, prior_20bar_extreme=42_000,
        prior_swing_extreme=42_000, momentum_atr=3.2,
        directional_closes=1, close_breakout_atr=2.8,
        previous_ema=49_000,
    )
    assert result["crash"] is False
    assert result["previous_ema_extension_atr"] > 3.0


def test_8_nov_new_base_impulse_passes_even_far_below_ema():
    frame = candles(260, freq="4h", trend=0.0)
    closes = [75_940, 75_900, 75_920, 76_035, 75_893, 75_829.9, 74_818.1]
    for index, value in zip(frame.index[-7:], closes):
        frame.loc[index, "close"] = value
    frame.loc[frame.index[-1], "open"] = 75_850
    frame.loc[frame.index[-1], "high"] = 75_900
    frame.loc[frame.index[-1], "low"] = 74_700
    result = assess_exceptional_entry(
        frame, config(), side="short", price=74_606.5, close=74_818.1,
        ema_value=79_987.9, atr_value=778.967, prior_20bar_extreme=74_800.1,
        prior_swing_extreme=74_800.1, momentum_atr=1.439,
        directional_closes=2, close_breakout_atr=1.299,
        previous_ema=80_009,
    )
    assert result["base"] is True
    assert result["crash"] is False
    assert result["base_close_range_atr"] < 0.5


def test_new_base_does_not_reopen_old_false_short_signals():
    frame = candles(260, freq="4h", trend=0.0)
    for breakout, momentum in ((0.75, 1.40), (0.38, 1.47), (0.42, 1.48)):
        result = assess_exceptional_entry(
            frame, config(), side="short", price=49_800, close=49_800,
            ema_value=52_000, atr_value=800, prior_20bar_extreme=47_000,
            prior_swing_extreme=47_000, momentum_atr=momentum,
            directional_closes=2, close_breakout_atr=breakout,
            previous_ema=52_000,
        )
        assert result["base"] is False
        assert result["crash"] is False


def test_base_breakout_rejects_new_signal_with_support_under_half_atr():
    frame = candles(260, freq="4h", trend=0.0)
    for index, value in zip(frame.index[-7:], [75_940, 75_900, 75_920, 76_035, 75_893, 75_829, 73_837]):
        frame.loc[index, "close"] = value
    result = assess_exceptional_entry(
        frame, config(), side="short", price=73_783.2, close=73_837,
        ema_value=79_902, atr_value=841.83, prior_20bar_extreme=73_385,
        prior_swing_extreme=73_385, momentum_atr=2.44,
        directional_closes=2, close_breakout_atr=1.16,
        previous_ema=79_933,
    )
    assert result["base"] is False
    assert result["extreme_room_atr"] < 0.5


def test_exceptional_path_symmetric_for_long_shock():
    frame = _exception_frame(50_000, 52_500, 50_000, 52_600, 49_900)
    result = assess_exceptional_entry(
        frame, config(), side="long", price=52_500, close=52_500,
        ema_value=50_300, atr_value=800, prior_20bar_extreme=55_000,
        prior_swing_extreme=55_000, momentum_atr=3.1,
        directional_closes=1, close_breakout_atr=2.7,
        previous_ema=50_000,
    )
    assert result["crash"] is True


def test_crash_path_in_real_signal_evaluator():
    frame = candles(260, freq="4h", trend=0.0)
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 250
    frame["low"] = frame["close"] - 250
    frame.loc[frame.index[-10], "low"] = 45_000
    frame.loc[frame.index[-2], "open"] = 50_000
    frame.loc[frame.index[-2], "close"] = 50_000
    frame.loc[frame.index[-1], "open"] = 50_000
    frame.loc[frame.index[-1], "close"] = 47_500
    frame.loc[frame.index[-1], "high"] = 50_100
    frame.loc[frame.index[-1], "low"] = 47_400
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    result = evaluate_trend_signal(frame, config(), "test_perpetual", 47_500, now=now)
    assert result.state == "signal"
    assert result.side == "short"
    assert result.factors["signal_type"]["value"] == "crash_impulse_short"


def test_base_consolidation_reentry_in_full_signal_evaluator():
    frame = candles(260, freq="4h", trend=0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] += 30_000
    frame.loc[frame.index[-10], "low"] = 74_800.1
    base_closes = [75_940, 75_900, 75_920, 76_035, 75_893, 75_829.9, 74_818.1]
    for idx, close in zip(frame.index[-7:], base_closes):
        frame.loc[idx, "open"] = close + 80
        frame.loc[idx, "close"] = close
        frame.loc[idx, "high"] = close + 250
        frame.loc[idx, "low"] = close - 250
    frame.loc[frame.index[-10], "low"] = 74_800.1
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", 74_606.5, now=now)
    assert decision.state == "signal", (decision.vetoes, decision.factors.get("exceptional_short"))
    assert decision.side == "short"
    assert decision.factors["signal_type"]["value"] == "base_breakout_short"
    assert decision.factors["trend"]["ema"] - decision.factors["trend"]["close"] > 3 * decision.factors["volatility"]["atr"]


def test_reclaimed_standard_short_is_vetoed_at_execution():
    frame = _breakout_frame("short")
    prior_low = float(frame["low"].iloc[-21:-1].min())
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", prior_low + 20, now=now)
    assert decision.state == "neutral"
    assert "breakout_reclaimed_before_entry" in decision.vetoes
    assert decision.factors["breakout_quality"]["live_breakout_atr"] < 0


def test_reclaimed_standard_long_is_vetoed_at_execution():
    frame = _breakout_frame("long")
    prior_high = float(frame["high"].iloc[-21:-1].max())
    now = pd.Timestamp(frame["close_time"].iloc[-1]) + pd.Timedelta(minutes=5)
    decision = evaluate_trend_signal(frame, config(), "test_perpetual", prior_high - 20, now=now)
    assert decision.state == "neutral"
    assert "breakout_reclaimed_before_entry" in decision.vetoes


def test_november_25_2026_micro_breakdown_rejected_before_short_alert():
    # Production row 375: only information available at decision time.
    quality, vetoes = assess_trend_breakout_quality(
        side="short", close=58710.8, live_price=58770.3, ema_value=70168.37778435311,
        atr_value=1254.6915469321443, channel_level=58731.0,
        extended_ema_distance_atr=6.0, extended_min_breakout_atr=0.20,
    )
    assert set(vetoes) == {"breakout_reclaimed_before_entry", "overextended_shallow_breakout"}
    assert 0 < quality["close_breakout_atr"] < 0.02
    assert quality["ema_distance_atr"] > 9


def test_november_10_and_12_meaningful_breakdowns_still_pass_quality_gate():
    for close, live_price, ema_value, atr_value, old_low in [
        (73177.4, 73155.6, 79400.0, 950.0, 73385.0),
        (71682.7, 71893.3, 78800.0, 950.0, 72688.0),
    ]:
        quality, vetoes = assess_trend_breakout_quality(
            side="short", close=close, live_price=live_price,
            ema_value=ema_value, atr_value=atr_value, channel_level=old_low,
            extended_ema_distance_atr=6.0, extended_min_breakout_atr=0.20,
        )
        assert vetoes == []
        assert quality["close_breakout_atr"] >= 0.20
