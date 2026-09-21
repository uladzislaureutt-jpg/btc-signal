import pandas as pd

from btc_signal.levels import LevelZone
from btc_signal.strategy import _candidate, evaluate
from conftest import candles, config


def test_evaluate_never_signals_without_nearby_level():
    cfg = config()
    decision = evaluate(
        candles(300, freq="1d", trend=100.0),
        candles(500, freq="4h", trend=100.0),
        candles(1000, freq="1h", trend=100.0),
        cfg,
        "test",
    )
    if decision.level is None:
        assert decision.state == "neutral"
        assert "no_nearby_4h_level" in decision.vetoes


def test_decision_has_stable_dedup_key():
    cfg = config()
    decision = evaluate(
        candles(300, freq="1d", trend=1.0),
        candles(500, freq="4h", trend=1.0),
        candles(1000, freq="1h", trend=1.0),
        cfg,
        "test",
    )
    assert decision.dedup_key == decision.dedup_key
    assert len(decision.dedup_key) == 64




def _trigger_frame(price: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [price, price],
            "high": [price + 1, price + 1],
            "low": [price - 1, price - 1],
            "close": [price, price],
            "atr": [10.0, 10.0],
            "rsi": [50.0, 55.0],
            "volume": [110.0, 110.0],
            "volume_median": [100.0, 100.0],
        },
        index=index,
    )


def test_short_candidate_vetoes_price_above_stop():
    cfg = config()
    zone = LevelZone(
        side="resistance",
        low=98.0,
        high=102.0,
        midpoint=100.0,
        touches=2,
        quality=6,
        sources=("swing",),
    )
    decision = _candidate(
        "short",
        "short_allowed",
        "bearish",
        _trigger_frame(110.0),
        zone,
        [zone],
        cfg,
    )
    assert decision.stop is not None and decision.price >= decision.stop
    assert decision.state == "neutral"
    assert decision.rr == 0.0
    assert "invalid_stop_geometry" in decision.vetoes


def test_long_candidate_vetoes_price_below_stop():
    cfg = config()
    zone = LevelZone(
        side="support",
        low=98.0,
        high=102.0,
        midpoint=100.0,
        touches=2,
        quality=6,
        sources=("swing",),
    )
    decision = _candidate(
        "long",
        "long_allowed",
        "bullish",
        _trigger_frame(90.0),
        zone,
        [zone],
        cfg,
    )
    assert decision.stop is not None and decision.price <= decision.stop
    assert decision.state == "neutral"
    assert decision.rr == 0.0
    assert "invalid_stop_geometry" in decision.vetoes
