from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import required_history_hours, run_backtest
from .config import load_config
from .data import build_market_data
from .runner import run_once


def _backtest(days: int | None, output: str, market_output: str | None = None) -> None:
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
    trades, summary = run_backtest(frame, config)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(trades[0]).keys()) if trades else ["no_trades"])
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))
    print(
        json.dumps(
            {
                "provider": market.last_provider,
                "requested_days": requested_days,
                "history_candles": len(frame),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Signal Core 0.1.2")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="Evaluate the newest closed candles")
    live.add_argument("--dry-run", action="store_true", help="Do not store or notify")
    live.add_argument("--no-watch", action="store_true", help="Do not notify WATCH states")
    backtest = commands.add_parser("backtest", help="Run a historical shadow test")
    backtest.add_argument("--days", type=int, default=None)
    backtest.add_argument("--output", default="reports/backtest_trades.csv")
    backtest.add_argument("--market-output", default=None)
    args = parser.parse_args()
    if args.command == "live":
        result = run_once(dry_run=args.dry_run, notify_watch=not args.no_watch)
        print(json.dumps(result.decision.to_dict(), ensure_ascii=False, indent=2))
        print(json.dumps({"duplicate": result.duplicate, "notified": result.notified}))
    else:
        _backtest(args.days, args.output, args.market_output)


if __name__ == "__main__":
    main()
