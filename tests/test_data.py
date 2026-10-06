import pandas as pd

from btc_signal.data import MarketDataError, ResilientMarketData, closed_candles
from conftest import candles


def test_unclosed_candle_is_removed():
    frame = candles(3)
    now = frame["close_time"].iloc[-2] + pd.Timedelta(milliseconds=1)
    result = closed_candles(frame, now=now)
    assert len(result) == 2


def test_provider_chain_falls_back():
    frame = candles(3)

    class Broken:
        name = "broken"

        def klines(self, **kwargs):
            raise MarketDataError("down")

    class Working:
        name = "working"

        def klines(self, **kwargs):
            return frame

    client = ResilientMarketData([Broken(), Working()])
    result = client.klines(symbol="BTCUSDT", interval="1h", limit=3)
    assert len(result) == 3
    assert client.last_provider == "working"


def test_bundle_does_not_mix_providers():
    frame = candles(3)

    class Partial:
        name = "partial"

        def klines(self, symbol, interval, limit):
            if interval == "4h":
                raise MarketDataError("missing 4h")
            return frame

    class Complete:
        name = "complete"

        def klines(self, **kwargs):
            return frame

    client = ResilientMarketData([Partial(), Complete()])
    result = client.bundle("BTCUSDT", {"1h": 3, "4h": 3, "1d": 3})
    assert set(result) == {"1h", "4h", "1d"}
    assert client.last_provider == "complete"


def test_snapshot_keeps_candles_and_ticker_on_one_provider():
    frame = candles(3)

    class Broken:
        name = "broken"

        def klines(self, **kwargs):
            return frame

        def ticker(self, symbol):
            raise MarketDataError("ticker down")

    class Complete:
        name = "complete"

        def klines(self, **kwargs):
            return frame

        def ticker(self, symbol):
            return 50_123.0

    client = ResilientMarketData([Broken(), Complete()])
    frames, price = client.snapshot("BTCUSDT", {"4h": 3})
    assert set(frames) == {"4h"}
    assert price == 50_123.0
    assert client.last_provider == "complete"
