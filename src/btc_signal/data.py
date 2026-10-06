from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import requests


COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


class MarketDataError(RuntimeError):
    pass


class MarketProvider(Protocol):
    name: str

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame: ...

    def ticker(self, symbol: str) -> float: ...


def _frame(rows: list[list[object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        raise MarketDataError("Exchange returned no candles")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return (
        frame.drop_duplicates("open_time")
        .sort_values("open_time")
        .set_index("open_time")
    )


@dataclass
class BinanceProvider:
    timeout: int = 20
    base_url: str = "https://api.binance.com"
    name: str = "binance"
    klines_path: str = "/api/v3/klines"
    ticker_path: str = "/api/v3/ticker/price"

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        remaining = limit
        cursor = start_ms
        rows: list[list[object]] = []
        while remaining > 0:
            batch = min(remaining, 1000)
            params: dict[str, object] = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch,
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            response = requests.get(
                f"{self.base_url}{self.klines_path}", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            converted = [
                [row[0], row[1], row[2], row[3], row[4], row[5], row[6]]
                for row in payload
            ]
            rows.extend(converted)
            remaining -= len(converted)
            next_cursor = int(payload[-1][6]) + 1
            if len(converted) < batch or (end_ms is not None and next_cursor > end_ms):
                break
            cursor = next_cursor
            time.sleep(0.05)
        return _frame(rows)

    def ticker(self, symbol: str) -> float:
        response = requests.get(
            f"{self.base_url}{self.ticker_path}",
            params={"symbol": symbol},
            timeout=self.timeout,
        )
        response.raise_for_status()
        price = float(response.json()["price"])
        if price <= 0:
            raise MarketDataError("Exchange returned an invalid ticker price")
        return price


BYBIT_INTERVALS = {"1h": "60", "4h": "240", "1d": "D"}


@dataclass
class BybitProvider:
    timeout: int = 20
    base_url: str = "https://api.bybit.com"
    name: str = "bybit"
    category: str = "spot"

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        if interval not in BYBIT_INTERVALS:
            raise MarketDataError(f"Unsupported Bybit interval: {interval}")
        remaining = limit
        cursor_end = end_ms
        rows: list[list[object]] = []
        duration_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
        while remaining > 0:
            batch = min(remaining, 1000)
            params: dict[str, object] = {
                "category": self.category,
                "symbol": symbol,
                "interval": BYBIT_INTERVALS[interval],
                "limit": batch,
            }
            if start_ms is not None:
                params["start"] = start_ms
            if cursor_end is not None:
                params["end"] = cursor_end
            response = requests.get(
                f"{self.base_url}/v5/market/kline", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") != 0:
                raise MarketDataError(payload.get("retMsg", "Bybit API error"))
            batch_rows = payload.get("result", {}).get("list", [])
            if not batch_rows:
                break
            converted = [
                [
                    int(row[0]),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    int(row[0]) + duration_ms - 1,
                ]
                for row in batch_rows
            ]
            rows.extend(converted)
            remaining -= len(converted)
            oldest = min(int(row[0]) for row in batch_rows)
            if len(converted) < batch or start_ms is not None and oldest <= start_ms:
                break
            cursor_end = oldest - 1
            time.sleep(0.05)
        return _frame(rows)

    def ticker(self, symbol: str) -> float:
        response = requests.get(
            f"{self.base_url}/v5/market/tickers",
            params={"category": self.category, "symbol": symbol},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") != 0:
            raise MarketDataError(payload.get("retMsg", "Bybit API error"))
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            raise MarketDataError("Bybit returned no ticker")
        price = float(rows[0]["lastPrice"])
        if price <= 0:
            raise MarketDataError("Exchange returned an invalid ticker price")
        return price


class ResilientMarketData:
    def __init__(self, providers: list[MarketProvider]):
        if not providers:
            raise ValueError("At least one market-data provider is required")
        self.providers = providers
        self.last_provider: str | None = None

    def klines(self, **kwargs: object) -> pd.DataFrame:
        errors: list[str] = []
        for provider in self.providers:
            try:
                result = provider.klines(**kwargs)
                self.last_provider = provider.name
                return closed_candles(result)
            except (requests.RequestException, MarketDataError, ValueError) as exc:
                errors.append(f"{provider.name}: {exc}")
        raise MarketDataError("; ".join(errors))

    def bundle(self, symbol: str, intervals: dict[str, int]) -> dict[str, pd.DataFrame]:
        """Fetch every timeframe from one provider to avoid mixed-exchange signals."""
        errors: list[str] = []
        for provider in self.providers:
            frames: dict[str, pd.DataFrame] = {}
            try:
                for interval, limit in intervals.items():
                    frames[interval] = closed_candles(
                        provider.klines(symbol=symbol, interval=interval, limit=limit)
                    )
                self.last_provider = provider.name
                return frames
            except (requests.RequestException, MarketDataError, ValueError) as exc:
                errors.append(f"{provider.name}: {exc}")
        raise MarketDataError("; ".join(errors))

    def snapshot(
        self, symbol: str, intervals: dict[str, int]
    ) -> tuple[dict[str, pd.DataFrame], float]:
        """Fetch candles and the live price from one derivatives venue."""
        errors: list[str] = []
        for provider in self.providers:
            frames: dict[str, pd.DataFrame] = {}
            try:
                for interval, limit in intervals.items():
                    frames[interval] = closed_candles(
                        provider.klines(symbol=symbol, interval=interval, limit=limit)
                    )
                price = provider.ticker(symbol)
                self.last_provider = provider.name
                return frames, price
            except (requests.RequestException, MarketDataError, ValueError, KeyError) as exc:
                errors.append(f"{provider.name}: {exc}")
        raise MarketDataError("; ".join(errors))


def closed_candles(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    current = now or pd.Timestamp.now(tz="UTC")
    return frame.loc[frame["close_time"] <= current].copy()


def build_market_data(config: dict) -> ResilientMarketData:
    timeout = int(config["market"]["request_timeout_seconds"])
    providers = {
        "binance": BinanceProvider(timeout=timeout),
        "bybit": BybitProvider(timeout=timeout),
        "binance_us": BinanceProvider(
            timeout=timeout,
            base_url="https://api.binance.us",
            name="binance_us",
        ),
        "binance_data": BinanceProvider(
            timeout=timeout,
            base_url="https://data-api.binance.vision",
            name="binance_spot_proxy",
        ),
        "binance_usdm": BinanceProvider(
            timeout=timeout,
            base_url="https://fapi.binance.com",
            name="binance_usdm",
            klines_path="/fapi/v1/klines",
            ticker_path="/fapi/v1/ticker/price",
        ),
        "bybit_linear": BybitProvider(
            timeout=timeout,
            name="bybit_linear",
            category="linear",
        ),
    }
    selected = config["market"].get("sources")
    if not selected:
        selected = [config["market"]["primary"], config["market"]["fallback"]]
    return ResilientMarketData([providers[name] for name in selected])
