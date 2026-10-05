import io
import zipfile

import pandas as pd

from btc_signal.perpetual_data import parse_funding_zip, parse_perpetual_kline_zip


def _zip(name: str, content: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, content)
    return output.getvalue()


def test_parse_perpetual_kline_archive():
    payload = _zip(
        "BTCUSDT-1h-2020-01.csv",
        "1577836800000,100,105,95,102,10,1577840399999,0,1,0,0,0\n",
    )
    frame = parse_perpetual_kline_zip(payload)
    assert frame.loc[0, "open_time"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame.loc[0, "close"] == 102


def test_parse_funding_archive_with_official_headers():
    payload = _zip(
        "BTCUSDT-fundingRate-2020-01.csv",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1577836800000,8,0.0001\n",
    )
    frame = parse_funding_zip(payload)
    assert frame.loc[0, "funding_time"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame.loc[0, "funding_rate"] == 0.0001
    assert frame.loc[0, "interval_hours"] == 8
