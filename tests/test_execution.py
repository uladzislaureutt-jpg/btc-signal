import pandas as pd
import pytest

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
        def most_recent_candle_before(self, candle_time):
            return None
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
        def most_recent_candle_before(self, candle_time):
            return None
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


def test_gap_between_candle_slots_is_logged_without_replaying_old_signal(monkeypatch, capsys):
    from btc_signal import runner
    from conftest import candles
    frame = candles(260, freq="4h", trend=0.0)
    current = pd.Timestamp(frame["close_time"].iloc[-1])
    previous = current - pd.Timedelta(hours=8)
    class Market:
        last_provider = "test"
        def snapshot(self, **kwargs):
            return {"4h": frame}, 50_000.0
    class Store:
        def seen(self, key):
            return False
        def most_recent_candle_before(self, candle_time):
            return previous.isoformat()
        def save(self, decision, notified):
            self.saved = (decision, notified)
    store = Store()
    def neutral(*args, **kwargs):
        return Decision(
            timestamp=current.isoformat(), state="neutral", side=None, score=0,
            price=50_000.0, entry_low=None, entry_high=None, stop=None,
            target_1=None, target_2=None, rr=None, regime="no_breakout",
            level=None, factors={}, vetoes=["no_4h_trend_breakout"],
            strategy_version="0.4.3",
        )
    monkeypatch.setattr(runner, "load_config", config)
    monkeypatch.setattr(runner, "build_market_data", lambda cfg: Market())
    monkeypatch.setattr(runner, "evaluate_trend_signal", neutral)
    monkeypatch.setattr(runner, "build_store", lambda: store)
    monkeypatch.setattr(runner, "send", lambda d: (_ for _ in ()).throw(AssertionError("no stale alert")))
    result = runner.run_once()
    assert result.notified is False
    assert result.decision.factors["coverage"]["missed_4h_bars"] == 1
    assert "::warning::" in capsys.readouterr().out
    assert store.saved[1] is False


@pytest.mark.parametrize("previous_in_base", [False, True])
def test_november_selloff_and_rebound_through_live_runner(monkeypatch, tmp_path, previous_in_base):
    from btc_signal import runner
    from btc_signal.storage import SQLiteStore
    from btc_signal.telegram import format_decision
    from btc_signal.trend_live import evaluate_trend_signal
    from conftest import candles

    # OKX 4H OHLC from 19 Nov 2026 12:00 UTC through 21 Nov 20:00 UTC, by close time.
    episode = [
        (63714.2, 64054.0, 63530.0, 63918.7),
        (63918.7, 64048.5, 62748.9, 63075.1),
        (63075.0, 63395.7, 62188.0, 62341.0),
        (62340.9, 62437.7, 61388.0, 61949.1),
        (61949.1, 62621.5, 61256.6, 62123.7),
        (62123.6, 62126.5, 61350.0, 61818.1),
        (61818.1, 62317.1, 61520.0, 61782.1),
        (61782.2, 62155.0, 59360.0, 59444.4),
        (59444.4, 59796.3, 57695.3, 58522.5),
        (58522.5, 59374.0, 58166.0, 58919.5),
        (58919.4, 59209.9, 56334.2, 56700.1),
        (56700.0, 58636.2, 55100.0, 57959.6),
        (57959.6, 58627.9, 57341.3, 58177.2),
        (58177.1, 60970.0, 58115.0, 59908.9),
        (59908.9, 60448.2, 59750.0, 60331.2),
    ]
    frame = candles(260, freq="4h", trend=0.0)
    frame["close"] += 15_000
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 600
    frame["low"] = frame["close"] - 600
    frame.iloc[-len(episode):, :4] = episode
    breakout_bar = -8

    store = SQLiteStore(tmp_path / "live.db")
    previous = _decision("short")
    previous.timestamp = frame["close_time"].iloc[breakout_bar - (3 if previous_in_base else 30)].isoformat()
    store.save(previous, True)
    messages = []

    class Market:
        last_provider = "okx_swap"

        def snapshot(self, **kwargs):
            return {"4h": snapshot}, price

    monkeypatch.setattr(runner, "build_market_data", lambda _: Market())
    monkeypatch.setattr(runner, "build_store", lambda: store)
    monkeypatch.setattr(runner, "load_config", config)
    monkeypatch.setattr(runner, "send", lambda decision: messages.append(format_decision(decision)))
    monkeypatch.setattr(
        runner, "evaluate_trend_signal",
        lambda frame, cfg, provider, ticker: evaluate_trend_signal(frame, cfg, provider, ticker, now=now),
    )

    def run(bar, ticker, **kwargs):
        nonlocal snapshot, price, now
        snapshot = frame.iloc[:len(frame) + bar + 1]
        price = ticker
        now = snapshot["close_time"].iloc[-1] + pd.Timedelta(minutes=7)
        return runner.run_once(**kwargs)

    snapshot = price = now = None
    result = run(breakout_bar, 59_361.2)
    assert result.notified is not previous_in_base
    if previous_in_base:
        assert result.decision.vetoes == ["same_direction_without_structural_reset"]
    else:
        assert result.decision.factors["episode_reset"]["method"] == "consolidation_breakout"
        assert not run(-7, 58_510.1).notified
        assert not run(-5, 55_713.8).notified

    assert not run(-3, 58_155.2).notified
    watch = run(-2, 59_886.3)
    assert watch.decision.state == "watch"
    assert watch.notified
    assert watch.decision.entry_low is watch.decision.stop is watch.decision.target_1 is None
    assert "Отскок после падения" in messages[-1]
    assert "Стоп-лосс" not in messages[-1]
    assert run(-2, 59_950.0).duplicate
    assert not run(-1, 60_381.6).notified
