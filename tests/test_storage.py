from btc_signal.storage import SupabaseStore


def test_new_supabase_secret_is_not_sent_as_bearer_jwt():
    store = SupabaseStore("https://example.supabase.co", "sb_secret_example")
    assert store.headers["apikey"] == "sb_secret_example"
    assert "Authorization" not in store.headers


def test_legacy_service_role_keeps_bearer_header():
    store = SupabaseStore("https://example.supabase.co", "legacy.jwt.value")
    assert store.headers["Authorization"] == "Bearer legacy.jwt.value"
