from copy import deepcopy

import pandas as pd

from btc_signal.sensitivity import (
    TapeCandidate,
    TapePoint,
    _variant_decision,
    replay_decision_tape,
    verify_baseline,
)
from btc_signal.strategy import Decision
from conftest import config


def _decision(side: str, score: int = 80) -> Decision:
    return Decision(
        timestamp="2026-01-01T00:00:00+00:00",
        state="signal",
        side=side,
        score=score,
        price=100.0,
        entry_low=99.0,
        entry_high=101.0,
        stop=90.0 if side == "long" else 110.0,
        target_1=140.0 if side == "long" else 60.0,
        target_2=150.0 if side == "long" else 50.0,
        rr=4.0,
        regime="both_allowed",
        level=None,
        factors={},
        vetoes=[],
    )


def test_variant_decision_recomputes_parameter_vetoes():
    cfg = config()
    candidate = TapeCandidate(_decision("long", score=74), distance_atr=0.7)
    assert _variant_decision(candidate, cfg).state == "watch"
    strict = deepcopy(cfg)
    strict["levels"]["max_distance_atr"] = 0.5
    changed = _variant_decision(candidate, strict)
    assert changed.state == "neutral"
    assert "late_entry" in changed.vetoes


def test_replay_filters_distance_before_selecting_highest_score():
    cfg = config()
    index = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 50.0],
            "close": [100.0, 60.0],
            "volume": [100.0, 100.0],
            "close_time": index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )
    point = TapePoint(
        frame_index=0,
        evaluation_time=frame.iloc[0]["close_time"],
        candidates=(
            TapeCandidate(_decision("long", score=90), distance_atr=1.0),
            TapeCandidate(_decision("short", score=80), distance_atr=0.5),
        ),
    )
    trades = replay_decision_tape([point], frame, cfg)
    assert len(trades) == 1
    assert trades[0].side == "short"
    assert trades[0].result == "target"


def test_baseline_verification_reports_metric_drift():
    metrics = {
        "trades": 2,
        "total_r": 1.0,
        "gross_total_r": 2.0,
        "total_cost_r": 1.0,
        "profit_factor": 2.0,
        "max_drawdown_r": -1.0,
    }
    sensitivity = {"scenarios": [{"label": "baseline", **metrics}]}
    assert verify_baseline(metrics, sensitivity)["status"] == "passed"
    changed = dict(metrics, total_r=0.5)
    assert verify_baseline(changed, sensitivity)["differences"] == {
        "total_r": {"backtest": 0.5, "replay": 1.0}
    }
