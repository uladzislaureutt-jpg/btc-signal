from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import BACKTEST_WARMUP_HOURS, required_history_hours, run_backtest
from .config import load_config
from .data import build_market_data
from .runner import run_once


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: str, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _backtest(
    days: int | None,
    output: str,
    market_output: str | None = None,
    diagnostics_output: str = "reports/backtest_diagnostics.json",
    manifest_output: str = "reports/backtest_manifest.json",
) -> None:
    config = load_config()
    requested_days = days or int(config["history"]["backtest_days"])
    max_holding_hours = int(config["risk"]["max_holding_hours"])
    total_hours = required_history_hours(requested_days, max_holding_hours)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=total_hours)
    market = build_market_data(config)
    frame = market.klines(
        symbol=config["market"]["symbol"],
        interval="1h",
        limit=total_hours,
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
    )
    if market_output:
        market_output_path = Path(market_output)
        market_output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(market_output_path, index_label="open_time")
    else:
        market_output_path = Path("reports/backtest_market_1h.csv")
        market_output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(market_output_path, index_label="open_time")
    trades, summary = run_backtest(frame, config)
    diagnostics = summary.pop("diagnostics")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(trades[0]).keys()) if trades else ["no_trades"])
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))
    diagnostics_path = _write_json(diagnostics_output, diagnostics)
    evaluation_end_index = len(frame) - max_holding_hours - 1
    manifest = {
        "schema_version": 2,
        "strategy_version": __version__,
        "commit_sha": os.getenv("GITHUB_SHA"),
        "provider": market.last_provider,
        "symbol": config["market"]["symbol"],
        "requested_days": requested_days,
        "warmup_hours": BACKTEST_WARMUP_HOURS,
        "max_holding_hours": max_holding_hours,
        "history_candles": len(frame),
        "history_start": frame.index[0].isoformat() if len(frame) else None,
        "history_end": frame.index[-1].isoformat() if len(frame) else None,
        "evaluation_start": frame.index[BACKTEST_WARMUP_HOURS].isoformat()
        if len(frame) > BACKTEST_WARMUP_HOURS else None,
        "evaluation_end": frame.index[evaluation_end_index].isoformat()
        if evaluation_end_index >= BACKTEST_WARMUP_HOURS else None,
        "market_data_sha256": _sha256_file(market_output_path),
        "trades_sha256": _sha256_file(output_path),
        "config_sha256": _sha256_json(config),
        "diagnostics_file": diagnostics_path.name,
        "execution_model": summary["execution_model"],
        "intrabar_collision_rule": summary["intrabar_collision_rule"],
        "funding_model": summary["funding_model"],
        "higher_timeframe_partial_bar_policy": summary["higher_timeframe_partial_bar_policy"],
        "data_quality_status": summary["data_quality"]["status"],
        "data_quality_warnings": summary["data_quality"]["warnings"],
        "bootstrap_method": summary["bootstrap"]["method"],
        "bootstrap_samples": summary["bootstrap"]["samples"],
        "bootstrap_seed": summary["bootstrap"]["seed"],
        "walk_forward_method": summary["walk_forward"]["method"],
        "walk_forward_train_days": summary["walk_forward"]["train_days"],
        "walk_forward_test_days": summary["walk_forward"]["test_days"],
    }
    manifest_path = _write_json(manifest_output, manifest)
    print(json.dumps({
        "provider": market.last_provider,
        "requested_days": requested_days,
        "history_candles": len(frame),
        "strategy_version": __version__,
        "manifest_file": manifest_path.name,
        "diagnostics_file": diagnostics_path.name,
        **summary,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Signal Core 0.1.5")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="Evaluate the newest closed candles")
    live.add_argument("--dry-run", action="store_true", help="Do not store or notify")
    live.add_argument("--no-watch", action="store_true", help="Do not notify WATCH states")
    backtest = commands.add_parser("backtest", help="Run a historical shadow test")
    backtest.add_argument("--days", type=int, default=None)
    backtest.add_argument("--output", default="reports/backtest_trades.csv")
    backtest.add_argument("--market-output", default=None)
    backtest.add_argument("--diagnostics-output", default="reports/backtest_diagnostics.json")
    backtest.add_argument("--manifest-output", default="reports/backtest_manifest.json")
    args = parser.parse_args()
    if args.command == "live":
        result = run_once(dry_run=args.dry_run, notify_watch=not args.no_watch)
        print(json.dumps(result.decision.to_dict(), ensure_ascii=False, indent=2))
        print(json.dumps({"duplicate": result.duplicate, "notified": result.notified}))
    else:
        _backtest(
            args.days,
            args.output,
            args.market_output,
            args.diagnostics_output,
            args.manifest_output,
        )


if __name__ == "__main__":
    main()
