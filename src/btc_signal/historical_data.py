from __future__ import annotations

import hashlib
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ARCHIVE_ROOT = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class ArchiveRecord:
    month: str
    url: str
    checksum_url: str
    expected_sha256: str
    actual_sha256: str
    bytes: int
    rows: int


def month_range(start: str, end: str) -> list[str]:
    """Return inclusive YYYY-MM values, rejecting reversed ranges."""
    first = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    if last < first:
        raise ValueError("archive end month must not precede start month")
    return [str(period) for period in pd.period_range(first, last, freq="M")]


def _archive_url(symbol: str, interval: str, month: str) -> str:
    filename = f"{symbol}-{interval}-{month}.zip"
    return f"{ARCHIVE_ROOT}/{symbol}/{interval}/{filename}"


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


def _parse_checksum(payload: bytes) -> str:
    fields = payload.decode("utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("Invalid Binance checksum response")
    checksum = fields[0].lower()
    if any(character not in "0123456789abcdef" for character in checksum):
        raise ValueError("Invalid SHA-256 checksum")
    return checksum


def parse_kline_zip(payload: bytes) -> pd.DataFrame:
    """Parse a single official Binance monthly kline ZIP."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in archive, found {len(members)}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)
    if frame.empty:
        raise ValueError("Binance archive contains no rows")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    open_raw = pd.to_numeric(frame["open_time"], errors="raise")
    close_raw = pd.to_numeric(frame["close_time"], errors="raise")
    unit = "us" if float(open_raw.abs().max()) >= 100_000_000_000_000 else "ms"
    frame["open_time"] = pd.to_datetime(open_raw, unit=unit, utc=True)
    frame["close_time"] = pd.to_datetime(close_raw, unit=unit, utc=True)
    return frame[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def _download_month(
    symbol: str,
    interval: str,
    month: str,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[pd.DataFrame, ArchiveRecord]:
    url = _archive_url(symbol, interval, month)
    checksum_url = f"{url}.CHECKSUM"
    expected = _parse_checksum(_request_bytes(checksum_url, timeout=timeout, retries=retries))
    cache_path = cache_dir / Path(url).name
    if cache_path.exists():
        payload = cache_path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            payload = _request_bytes(url, timeout=timeout, retries=retries)
    else:
        payload = _request_bytes(url, timeout=timeout, retries=retries)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {url}: expected {expected}, got {actual}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or hashlib.sha256(cache_path.read_bytes()).hexdigest() != actual:
        cache_path.write_bytes(payload)
    frame = parse_kline_zip(payload)
    return frame, ArchiveRecord(
        month=month,
        url=url,
        checksum_url=checksum_url,
        expected_sha256=expected,
        actual_sha256=actual,
        bytes=len(payload),
        rows=len(frame),
    )


def download_monthly_klines(
    *,
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    cache_dir: str | Path = ".cache/binance",
    workers: int = 8,
    timeout: int = 60,
    retries: int = 3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Download, checksum, merge, and validate monthly Binance archives."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    months = month_range(start_month, end_month)
    cache = Path(cache_dir)
    downloaded: dict[str, tuple[pd.DataFrame, ArchiveRecord]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(months))) as executor:
        futures = {
            executor.submit(
                _download_month, symbol, interval, month, cache, timeout, retries
            ): month
            for month in months
        }
        for future in as_completed(futures):
            downloaded[futures[future]] = future.result()

    frames = [downloaded[month][0] for month in months]
    frame = pd.concat(frames, ignore_index=True).sort_values("open_time")
    duplicate_mask = frame["open_time"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask]
        value_columns = ["open", "high", "low", "close", "volume", "close_time"]
        conflicts = duplicates.groupby("open_time")[value_columns].nunique().gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError("Conflicting duplicate Binance candles")
        frame = frame.drop_duplicates("open_time", keep="first")
    frame = frame.set_index("open_time").sort_index()
    manifest = [asdict(downloaded[month][1]) for month in months]
    return frame, manifest
