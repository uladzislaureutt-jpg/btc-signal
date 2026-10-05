import io
import zipfile

import pandas as pd

from btc_signal.historical_data import month_range, parse_kline_zip


def _zip_csv(row: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("BTCUSDT-1h-2020-01.csv", row)
    return output.getvalue()


def test_month_range_is_inclusive():
    assert month_range("2020-11", "2021-02") == [
        "2020-11", "2020-12", "2021-01", "2021-02"
    ]


def test_parse_binance_millisecond_archive():
    row = "1577836800000,100,105,95,102,10,1577840399999,0,1,0,0,0\n"
    frame = parse_kline_zip(_zip_csv(row))
    assert frame.loc[0, "open_time"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame.loc[0, "close_time"] == pd.Timestamp("2020-01-01T00:59:59.999Z")
    assert frame.loc[0, "close"] == 102


def test_parse_binance_microsecond_archive():
    row = "1735689600000000,100,105,95,102,10,1735693199999999,0,1,0,0,0\n"
    frame = parse_kline_zip(_zip_csv(row))
    assert frame.loc[0, "open_time"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert frame.loc[0, "close_time"] == pd.Timestamp("2025-01-01T00:59:59.999999Z")
