from btc_signal.storage import SupabaseStore


def test_new_supabase_secret_is_not_sent_as_bearer_jwt():
    store = SupabaseStore("https://example.supabase.co", "sb_secret_example")
    assert store.headers["apikey"] == "sb_secret_example"
    assert "Authorization" not in store.headers


def test_legacy_service_role_keeps_bearer_header():
    store = SupabaseStore("https://example.supabase.co", "legacy.jwt.value")
    assert store.headers["Authorization"] == "Bearer legacy.jwt.value"


def test_sqlite_structural_episode_reads_last_notified_only(tmp_path):
    from btc_signal.storage import SQLiteStore
    from btc_signal.strategy import Decision

    store = SQLiteStore(tmp_path / "episodes.sqlite")
    def saved(time, side, notified):
        return Decision(
            timestamp=time, state="signal", side=side, score=90,
            price=80000.0, entry_low=80000.0, entry_high=80000.0,
            stop=82000.0 if side == "short" else 78000.0,
            target_1=76000.0 if side == "short" else 84000.0,
            target_2=74000.0 if side == "short" else 86000.0,
            rr=2.0, regime="test", level={"price": 80000.0},
            factors={}, vetoes=[], strategy_version="0.4.2",
        )
    store.save(saved("2026-11-02T15:59:59.999000+00:00", "short", True), True)
    store.save(saved("2026-11-03T15:59:59.999000+00:00", "long", False), False)
    latest = store.most_recent_notified_signal("2026-11-05T03:59:59.999000+00:00", 720)
    assert latest == {
        "side": "short",
        "candle_time": "2026-11-02T15:59:59.999000+00:00",
    }
    assert store.most_recent_notified_signal("2026-12-20T03:59:59.999000+00:00", 720) is None


def test_supabase_episode_query_filters_notified_and_no_future(monkeypatch):
    from btc_signal.storage import SupabaseStore

    captured = {}
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return [{"side": "short", "candle_time": "2026-11-04T03:59:59.999+00:00"}]

    def fake_get(url, *, headers, params, timeout):
        captured.update(params)
        return Response()

    monkeypatch.setattr("btc_signal.storage.requests.get", fake_get)
    result = SupabaseStore("https://example.supabase.co", "sb_secret_example").most_recent_notified_signal(
        "2026-11-05T03:59:59.999+00:00", 720
    )
    assert result["side"] == "short"
    assert captured["notified"] == "eq.true"
    assert captured["order"] == "candle_time.desc"
    assert "candle_time.lt." in captured["and"]


def test_sqlite_coverage_returns_previous_record_only(tmp_path):
    from btc_signal.storage import SQLiteStore
    from btc_signal.strategy import Decision

    store = SQLiteStore(tmp_path / "coverage.sqlite")
    def neutral(t):
        return Decision(
            timestamp=t, state="neutral", side=None, score=0, price=50_000,
            entry_low=None, entry_high=None, stop=None, target_1=None,
            target_2=None, rr=None, regime="test", level=None,
            factors={}, vetoes=[], strategy_version="0.4.3",
        )
    first = "2026-11-05T11:59:59.999+00:00"
    second = "2026-11-05T19:59:59.999+00:00"
    store.save(neutral(first), False)
    store.save(neutral(second), False)
    assert store.most_recent_candle_before(second) == first
    assert store.most_recent_candle_before(first) is None


def test_supabase_coverage_query_uses_strictly_earlier_candle(monkeypatch):
    from btc_signal.storage import SupabaseStore
    captured = {}
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return [{"candle_time": "2026-11-05T11:59:59.999+00:00"}]
    def fake_get(url, *, headers, params, timeout):
        captured.update(params)
        return Response()
    monkeypatch.setattr("btc_signal.storage.requests.get", fake_get)
    prior = SupabaseStore("https://example.supabase.co", "sb_secret_example").most_recent_candle_before(
        "2026-11-05T19:59:59.999+00:00"
    )
    assert prior == "2026-11-05T11:59:59.999+00:00"
    assert captured["candle_time"].startswith("lt.")
    assert captured["order"] == "candle_time.desc"
