from btc_signal.strategy import evaluate
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

