from collections import Counter, defaultdict

from btc_signal.backtest import _finalize_diagnostics, _new_diagnostics
from btc_signal.strategy import evaluate, evaluate_with_trace
from conftest import candles, config


def test_evaluate_with_trace_preserves_public_decision():
    cfg = config()
    frames = (
        candles(300, freq="1d", trend=1.0),
        candles(500, freq="4h", trend=1.0),
        candles(1000, freq="1h", trend=1.0),
    )
    public = evaluate(*frames, cfg, "test")
    traced, trace = evaluate_with_trace(*frames, cfg, "test")
    assert public.to_dict() == traced.to_dict()
    assert trace.timestamp == public.timestamp
    assert trace.price == public.price


def test_empty_diagnostic_funnel_is_json_ready():
    raw = _new_diagnostics()
    result = _finalize_diagnostics(raw)
    assert result["method"] == "decision_funnel_v1"
    assert result["evaluated_hours"] == 0
    assert result["closest_zone_distance_atr"]["long"]["observations"] == 0
    assert result["single_blocker_counterfactual"] == {}


def test_diagnostic_counters_use_independent_collections():
    first = _new_diagnostics()
    second = _new_diagnostics()
    first["vetoes"]["example"] += 1
    first["distances"]["long"].append(0.5)
    assert isinstance(first["vetoes"], Counter)
    assert isinstance(first["distances"], defaultdict)
    assert second["vetoes"] == Counter()
    assert second["distances"]["long"] == []
