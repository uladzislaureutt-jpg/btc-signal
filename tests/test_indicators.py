from btc_signal.indicators import atr, ema, rsi

from conftest import candles


def test_indicators_return_finite_latest_values():
    frame = candles(250)
    assert ema(frame["close"], 50).iloc[-1] > 0
    assert 0 <= rsi(frame["close"], 14).iloc[-1] <= 100
    assert atr(frame, 14).iloc[-1] > 0

