from btc_signal.indicators import enrich
from btc_signal.levels import build_levels
from conftest import candles, config


def test_levels_are_zones_with_bounded_quality():
    cfg = config()
    frame = enrich(candles(350, freq="4h", trend=0.0), cfg)
    zones = build_levels(frame, cfg)
    assert zones
    assert all(zone.low < zone.high for zone in zones)
    assert all(0 <= zone.quality <= 10 for zone in zones)

