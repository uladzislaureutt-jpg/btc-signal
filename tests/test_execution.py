import pandas as pd

from btc_signal.backtest import _simulate, audit_market_data, resample
from btc_signal.strategy import Decision
from conftest import config


def _decision(side: str = "long") -> Decision:
    return Decision(
        timestamp="2026-01-01T00:00:00+00:00",
        state="signal",
        side=side,
        score=80,
        price=100.0,
        entry_low=99.0,
        entry_high=101.0,
        stop=90.0 if side == "long" else 110.0,
        target_1=140.0 if side == "long" else 60.0,
        target_2=150.0 if side == "long" else 50.0,
        rr=4.0,
        regime="long_allowed" if side == "long" else "short_allowed",
        level=None,
        factors={},
        vetoes=[],
    )


def _future(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 01:00", periods=len(rows), freq="1h", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 100.0
    frame["close_time"] = index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return frame


def test_simulate_enters_at_next_candle_open():
    cfg = config()
    cfg["risk"]["max_holding_hours"] = 2
    future = _future([(105.0, 110.0, 100.0, 106.0), (106.0, 112.0, 101.0, 108.0)])
    trade = _simulate(_decision(), future, cfg, signal_time="2026-01-01T00:59:59.999000+00:00")
    assert trade is not None
    assert trade.signal_time == "2026-01-01T00:59:59.999000+00:00"
    assert trade.entry_time == "2026-01-01T01:00:00+00:00"
    assert trade.entry == 105.0
    assert trade.rr == (140.0 - 105.0) / (105.0 - 90.0)


def test_same_candle_stop_and_target_counts_stop_first():
    cfg = config()
    future = _future([(100.0, 145.0, 85.0, 120.0)])
    trade = _simulate(_decision(), future, cfg)
    assert trade is not None
    assert trade.result == "stop"
    assert trade.exit == 90.0
    assert trade.exit_time == "2026-01-01T01:59:59.999000+00:00"


def test_later_gap_through_stop_uses_open_price():
    cfg = config()
    cfg["risk"]["max_holding_hours"] = 2
    future = _future([(100.0, 105.0, 95.0, 100.0), (85.0, 88.0, 80.0, 82.0)])
    trade = _simulate(_decision(), future, cfg)
    assert trade is not None
    assert trade.result == "gap_stop"
    assert trade.exit == 85.0


def test_market_data_audit_reports_gap_and_short_candle():
    frame = _future([(100.0, 105.0, 95.0, 100.0)] * 3)
    frame = frame.drop(frame.index[1])
    frame.loc[frame.index[-1], "close_time"] = frame.index[-1] + pd.Timedelta(minutes=56)
    audit = audit_market_data(frame)
    assert audit["status"] == "warning"
    assert audit["gap_count"] == 1
    assert audit["estimated_missing_bars"] == 1
    assert audit["irregular_candle_duration_rows"] == 1


def test_resample_drops_higher_timeframe_group_with_missing_hour():
    frame = _future([(100.0, 105.0, 95.0, 100.0)] * 8)
    frame = frame.drop(frame.index[2])
    result = resample(frame, "4h")
    assert list(result.index) == [pd.Timestamp("2026-01-01T04:00:00+00:00")]


def test_runner_does_not_repeat_short_without_structural_reset(monkeypatch):
    from btc_signal import runner
    from conftest import candles

    frame = candles(260, freq="4h", trend=0.0)
    closes = [80000, 79800, 79500, 79600, 79400, 79000, 78900, 78500]
    for candle_idx, close in zip(frame.index[-8:], closes):
        frame.loc[candle_idx, "close"] = close
    previous_candle = pd.Timestamp(frame["close_time"].iloc[-8]).isoformat()
    latest_candle = pd.Timestamp(frame["close_time"].iloc[-1]).isoformat()

    class FakeMarket:
        last_provider = "test"
        def snapshot(self, **kwargs):
            return {"4h": frame}, 78500.0

    class FakeStore:
        saved = None
        def seen(self, key):
            return False
        def cooldown_active(self, candle_time, hours):
            return False
        def most_recent_notified_signal(self, candle_time, hours):
            return {"side": "short", "candle_time": previous_candle}
        def save(self, decision, notified):
            self.saved = (decision, notified)

    store = FakeStore()
    sent = []
    def signal(*args, **kwargs):
        return Decision(
            timestamp=latest_candle, state="signal", side="short", score=90,
            price=78500.0, entry_low=78500.0, entry_high=78500.0,
            stop=80500.0, target_1=74500.0, target_2=72500.0,
            rr=2.0, regime="test", level={"price": 78600.0},
            factors={
                "volatility": {"atr": 400.0},
                "signal_type": {"value": "confirmed_early_short"},
            }, vetoes=[], provider="test", strategy_version="0.4.2",
        )

    monkeypatch.setattr(runner, "load_config", config)
    monkeypatch.setattr(runner, "build_market_data", lambda cfg: FakeMarket())
    monkeypatch.setattr(runner, "evaluate_trend_signal", signal)
    monkeypatch.setattr(runner, "build_store", lambda: store)
    monkeypatch.setattr(runner, "send", lambda dec: sent.append(dec))
    result = runner.run_once()
    assert result.decision.state == "neutral"
    assert "same_direction_without_structural_reset" in result.decision.vetoes
    assert sent == []
    assert store.saved[1] is False


def test_runner_allows_new_short_after_confirmed_pullback(monkeypatch):
    from btc_signal import runner
    from conftest import candles

    frame = candles(260, freq="4h", trend=0.0)
    closes = [80000, 79500, 79000, 79200, 79500, 79600, 79200, 78800]
    for candle_idx, close in zip(frame.index[-8:], closes):
        frame.loc[candle_idx, "close"] = close
    previous_candle = pd.Timestamp(frame["close_time"].iloc[-8]).isoformat()
    latest_candle = pd.Timestamp(frame["close_time"].iloc[-1]).isoformat()

    class FakeMarket:
        last_provider = "test"
        def snapshot(self, **kwargs):
            return {"4h": frame}, 78800.0
    class FakeStore:
        def seen(self, key):
            return False
        def cooldown_active(self, candle_time, hours):
            return False
        def most_recent_notified_signal(self, candle_time, hours):
            return {"side": "short", "candle_time": previous_candle}
        def save(self, decision, notified):
            self.saved = (decision, notified)
    store = FakeStore()
    sent = []

    monkeypatch.setattr(runner, "load_config", config)
    monkeypatch.setattr(runner, "build_market_data", lambda cfg: FakeMarket())
    monkeypatch.setattr(runner, "build_store", lambda: store)
    monkeypatch.setattr(runner, "send", lambda dec: sent.append(dec))
    monkeypatch.setattr(
        runner, "evaluate_trend_signal",
        lambda *args: Decision(
            timestamp=latest_candle, state="signal", side="short", score=90,
            price=78800.0, entry_low=78800.0, entry_high=78800.0,
            stop=80800.0, target_1=74800.0, target_2=72800.0,
            rr=2.0, regime="test", level={"price": 78800.0},
            factors={
                "volatility": {"atr": 400.0},
                "signal_type": {"value": "confirmed_early_short"},
            }, vetoes=[], provider="test", strategy_version="0.4.2",
        ),
    )
    result = runner.run_once()
    assert result.decision.state == "signal"
    assert result.notified is True
    assert result.decision.factors["episode_reset"]["confirmed"] is True
    assert len(sent) == 1
    assert store.saved[1] is True
