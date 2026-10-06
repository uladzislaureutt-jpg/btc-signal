from __future__ import annotations

import hashlib
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .historical_data import KLINE_COLUMNS, month_range


@dataclass(frozen=True)
class PerpetualArchiveRecord:
    dataset: str
    partition: str
    frequency: str
    url: str
    checksum_url: str
    expected_sha256: str
    actual_sha256: str
    bytes: int
    rows: int


def _request_bytes(url: str, *, timeout: int, retries: int) -> bytes:
    import requests

    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {url} after {retries} attempts") from error


def _checksum(payload: bytes) -> str:
    fields = payload.decode("utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("Invalid Binance checksum response")
    return fields[0].lower()


def _single_csv(payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in archive, found {len(members)}")
        return archive.read(members[0])


def _timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    unit = "us" if float(numeric.abs().max()) >= 100_000_000_000_000 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def parse_perpetual_kline_zip(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(_single_csv(payload)), header=None, names=KLINE_COLUMNS)
    if frame.empty:
        raise ValueError("Perpetual kline archive contains no rows")
    if str(frame.iloc[0]["open_time"]).strip().lower() in {"open_time", "opentime"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = _timestamp(frame["open_time"])
    frame["close_time"] = _timestamp(frame["close_time"])
    return frame[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def parse_funding_zip(payload: bytes) -> pd.DataFrame:
    raw = _single_csv(payload)
    frame = pd.read_csv(io.BytesIO(raw))
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    time_column = (
        normalized.get("calc_time")
        or normalized.get("fundingtime")
        or normalized.get("funding_time")
    )
    rate_column = (
        normalized.get("last_funding_rate")
        or normalized.get("fundingrate")
        or normalized.get("funding_rate")
    )
    interval_column = normalized.get("funding_interval_hours")
    if time_column is None or rate_column is None:
        frame = pd.read_csv(
            io.BytesIO(raw), header=None,
            names=["calc_time", "funding_interval_hours", "last_funding_rate"],
        )
        time_column = "calc_time"
        interval_column = "funding_interval_hours"
        rate_column = "last_funding_rate"
    result = pd.DataFrame({
        "funding_time": _timestamp(frame[time_column]),
        "funding_rate": pd.to_numeric(frame[rate_column], errors="raise"),
        "interval_hours": (
            pd.to_numeric(frame[interval_column], errors="coerce").fillna(8.0)
            if interval_column is not None else 8.0
        ),
    })
    if result.empty:
        raise ValueError("Funding archive contains no rows")
    return result


def _url(
    symbol: str, partition: str, dataset: str,
    interval: str = "1h", frequency: str = "monthly",
) -> str:
    if dataset == "klines":
        filename = f"{symbol}-{interval}-{partition}.zip"
        return f"https://data.binance.vision/data/futures/um/{frequency}/klines/{symbol}/{interval}/{filename}"
    if dataset == "fundingRate":
        filename = f"{symbol}-fundingRate-{partition}.zip"
        return f"https://data.binance.vision/data/futures/um/{frequency}/fundingRate/{symbol}/{filename}"
    raise ValueError(f"Unsupported perpetual dataset: {dataset}")


def _download(
    *, symbol: str, partition: str, frequency: str, dataset: str, interval: str,
    cache_dir: Path, timeout: int, retries: int,
) -> tuple[pd.DataFrame, PerpetualArchiveRecord]:
    url = _url(symbol, partition, dataset, interval, frequency)
    checksum_url = f"{url}.CHECKSUM"
    expected = _checksum(_request_bytes(checksum_url, timeout=timeout, retries=retries))
    cache_path = cache_dir / dataset / Path(url).name
    payload = cache_path.read_bytes() if cache_path.exists() else b""
    if hashlib.sha256(payload).hexdigest() != expected:
        payload = _request_bytes(url, timeout=timeout, retries=retries)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {url}: expected {expected}, got {actual}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or hashlib.sha256(cache_path.read_bytes()).hexdigest() != actual:
        cache_path.write_bytes(payload)
    frame = (
        parse_perpetual_kline_zip(payload)
        if dataset == "klines" else parse_funding_zip(payload)
    )
    return frame, PerpetualArchiveRecord(
        dataset=dataset, partition=partition, frequency=frequency,
        url=url, checksum_url=checksum_url,
        expected_sha256=expected, actual_sha256=actual,
        bytes=len(payload), rows=len(frame),
    )


def download_perpetual_history(
    *, symbol: str = "BTCUSDT", interval: str = "1h",
    start_month: str = "2020-01", end_month: str = "2024-01",
    cache_dir: str | Path = ".cache/binance-perpetual",
    workers: int = 8, timeout: int = 60, retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    months = month_range(start_month, end_month)
    if start_month < "2020-01":
        raise ValueError(
            "Official Binance USD-M BTCUSDT public archives begin in 2020-01; "
            "use 2020-01 or later and allow indicators to warm up in-sample."
        )
    cache = Path(cache_dir)
    jobs = (
        [("klines", month, "monthly") for month in months]
        + [("fundingRate", month, "monthly") for month in months]
    )
    downloaded: dict[tuple[str, str, str], tuple[pd.DataFrame, PerpetualArchiveRecord]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                _download, symbol=symbol, partition=partition, frequency=frequency,
                dataset=dataset,
                interval=interval, cache_dir=cache, timeout=timeout, retries=retries,
            ): (dataset, partition, frequency)
            for dataset, partition, frequency in jobs
        }
        for future in as_completed(futures):
            downloaded[futures[future]] = future.result()

    klines = pd.concat(
        [downloaded[job][0] for job in jobs if job[0] == "klines"], ignore_index=True
    ).sort_values("open_time")
    if klines["open_time"].duplicated().any():
        raise ValueError("Duplicate perpetual kline timestamps")
    klines = klines.set_index("open_time")
    funding = pd.concat(
        [downloaded[job][0] for job in jobs if job[0] == "fundingRate"], ignore_index=True
    ).sort_values("funding_time")
    if funding["funding_time"].duplicated().any():
        duplicates = funding[funding["funding_time"].duplicated(keep=False)]
        if duplicates.groupby("funding_time")["funding_rate"].nunique().gt(1).any():
            raise ValueError("Conflicting duplicate funding timestamps")
        funding = funding.drop_duplicates("funding_time", keep="last")
    funding = funding.set_index("funding_time")
    manifest = [
        asdict(downloaded[job][1]) for job in jobs
    ]
    return klines, funding, manifest
