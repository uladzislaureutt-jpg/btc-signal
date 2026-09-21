import pandas as pd

from btc_signal.backtest import resample
from conftest import candles


def test_resample_excludes_incomplete_higher_timeframe_candle():
    frame = candles(6, freq="1h")
    result = resample(frame, "4h")
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2025-01-01 00:00:00+00:00")
