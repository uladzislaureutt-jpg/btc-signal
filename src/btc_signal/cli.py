from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import BACKTEST_WARMUP_HOURS, required_history_hours, run_backtest
from .config import load_config
from .data import build_market_data
from .historical_data import download_monthly_klines
from .perpetual_data import download_perpetual_history
from .runner import run_once
from .sensitivity import run_sensitivity, verify_baseline
from .short_reversion import (
    parameters_from_config as short_parameters_from_config,
    run_short_backtest,
    run_short_sensitivity,
    trades_as_dicts as short_trades_as_dicts,
)
from .trend import (
    parameters_from_config,
    run_trend_backtest,
    run_trend_sensitivity,
    trades_as_dicts as trend_trades_as_dicts,
)
from .trend_v030 import (
    evaluate_success_criteria as evaluate_v030_success,
    parameters_from_config as trend_v030_parameters_from_config,
    run_trend_v030_backtest,
    run_trend_v030_sensitivity,
    trades_as_dicts as trend_v030_trades_as_dicts,
)


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


def _sensitivity(
    market_input: str,
    output: str,
    baseline_summary: str | None = None,
) -> None:
    config = load_config()
    frame = pd.read_csv(market_input)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, format="mixed")
    frame = frame.set_index("open_time")
    result = run_sensitivity(frame, config)
    if baseline_summary:
        expected = json.loads(Path(baseline_summary).read_text(encoding="utf-8"))
        result["baseline_verification"] = verify_baseline(expected, result)
    output_path = _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("baseline_verification", {}).get("status") == "failed":
        raise RuntimeError(f"Sensitivity replay differs from baseline; inspect {output_path}")


def _trend_backtest(
    evaluation_start: str,
    evaluation_end: str,
    archive_start: str,
    archive_end: str,
    reports_dir: str,
    cache_dir: str,
) -> None:
    config = load_config()
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, sources = download_monthly_klines(
        symbol=config["market"]["symbol"],
        interval="1h",
        start_month=archive_start,
        end_month=archive_end,
        cache_dir=cache_dir,
    )
    market_path = output_dir / f"btc_1h_{archive_start}_{archive_end}.csv"
    frame.to_csv(market_path, index_label="open_time")
    params = parameters_from_config(config)
    trades, summary = run_trend_backtest(
        frame,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        params=params,
    )
    trades_path = output_dir / "trend_trades_2020_2023.csv"
    trade_rows = trend_trades_as_dicts(trades)
    with trades_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(trade_rows[0]) if trade_rows else ["no_trades"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trade_rows)
    summary_path = _write_json(str(output_dir / "trend_summary.json"), summary)
    sensitivity = run_trend_sensitivity(
        frame,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        base=params,
    )
    replay = sensitivity["primary_replay"]
    sensitivity["primary_verification"] = {
        "status": "passed" if (
            replay["trades"] == summary["trades"]
            and abs(replay["total_r"] - summary["total_r"]) < 1e-12
            and abs(replay["gross_total_r"] - summary["gross_total_r"]) < 1e-12
        ) else "failed",
        "expected_trades": summary["trades"],
        "replayed_trades": replay["trades"],
        "total_r_delta": replay["total_r"] - summary["total_r"],
        "gross_total_r_delta": replay["gross_total_r"] - summary["gross_total_r"],
    }
    sensitivity_path = _write_json(
        str(output_dir / "trend_sensitivity.json"), sensitivity
    )
    manifest = {
        "schema_version": 1,
        "research_release": __version__,
        "commit_sha": os.getenv("GITHUB_SHA"),
        "instrument": "Binance spot BTCUSDT",
        "source": "official Binance public monthly kline archives",
        "archive_start": archive_start,
        "archive_end": archive_end,
        "evaluation_start": pd.Timestamp(evaluation_start).isoformat(),
        "evaluation_end_exclusive": pd.Timestamp(evaluation_end).isoformat(),
        "parameter_selection_policy": "frozen_before_holdout; sensitivity is descriptive and must not select a replacement",
        "primary_parameters": summary["parameters"],
        "source_archives": sources,
        "market_data_sha256": _sha256_file(market_path),
        "trades_sha256": _sha256_file(trades_path),
        "summary_sha256": _sha256_file(summary_path),
        "sensitivity_sha256": _sha256_file(sensitivity_path),
        "config_sha256": _sha256_json(config),
    }
    manifest_path = _write_json(str(output_dir / "trend_manifest.json"), manifest)
    print(json.dumps({
        "research_release": __version__,
        "manifest_file": manifest_path.name,
        "summary_file": summary_path.name,
        "sensitivity_file": sensitivity_path.name,
        "source_archive_count": len(sources),
        "holdout_policy": manifest["parameter_selection_policy"],
        **summary,
        "sensitivity_overview": {
            "scenario_count": sensitivity["scenario_count"],
            "positive_scenarios": sensitivity["positive_scenarios"],
            "positive_share": sensitivity["positive_share"],
            "by_cost_bps": sensitivity["by_cost_bps"],
            "primary_verification": sensitivity["primary_verification"],
        },
    }, ensure_ascii=False, indent=2))


def _short_backtest(
    evaluation_start: str,
    evaluation_end: str,
    archive_start: str,
    archive_end: str,
    reports_dir: str,
    cache_dir: str,
) -> None:
    config = load_config()
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, funding, sources = download_perpetual_history(
        symbol=config["market"]["symbol"], interval="1h",
        start_month=archive_start, end_month=archive_end, cache_dir=cache_dir,
    )
    market_path = output_dir / f"btcusdt_perpetual_1h_{archive_start}_{archive_end}.csv"
    funding_path = output_dir / f"btcusdt_funding_{archive_start}_{archive_end}.csv"
    frame.to_csv(market_path, index_label="open_time")
    funding.to_csv(funding_path, index_label="funding_time")
    params = short_parameters_from_config(config)
    trades, summary = run_short_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=params,
    )
    trades_path = output_dir / "short_trades_2020_2023.csv"
    rows = short_trades_as_dicts(trades)
    with trades_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["no_trades"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path = _write_json(str(output_dir / "short_summary.json"), summary)
    sensitivity = run_short_sensitivity(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, base=params,
    )
    replay = sensitivity["primary_replay"]
    sensitivity["primary_verification"] = {
        "status": "passed" if (
            replay["trades"] == summary["trades"]
            and abs(replay["total_r"] - summary["total_r"]) < 1e-12
            and abs(replay["gross_total_r"] - summary["gross_total_r"]) < 1e-12
            and abs(replay["total_funding_r"] - summary["total_funding_r"]) < 1e-12
        ) else "failed",
        "expected_trades": summary["trades"],
        "replayed_trades": replay["trades"],
        "total_r_delta": replay["total_r"] - summary["total_r"],
        "funding_r_delta": replay["total_funding_r"] - summary["total_funding_r"],
    }
    sensitivity_path = _write_json(str(output_dir / "short_sensitivity.json"), sensitivity)
    manifest = {
        "schema_version": 1,
        "research_release": __version__,
        "commit_sha": os.getenv("GITHUB_SHA"),
        "instrument": "Binance USD-M BTCUSDT perpetual",
        "source": "official Binance public monthly futures archives",
        "archive_start": archive_start,
        "archive_end": archive_end,
        "evaluation_start": pd.Timestamp(evaluation_start).isoformat(),
        "evaluation_end_exclusive": pd.Timestamp(evaluation_end).isoformat(),
        "parameter_selection_policy": "frozen_before_holdout; sensitivity is descriptive and must not select a replacement",
        "primary_parameters": summary["parameters"],
        "funding_model": summary["funding_model"],
        "source_archives": sources,
        "market_data_sha256": _sha256_file(market_path),
        "funding_data_sha256": _sha256_file(funding_path),
        "trades_sha256": _sha256_file(trades_path),
        "summary_sha256": _sha256_file(summary_path),
        "sensitivity_sha256": _sha256_file(sensitivity_path),
        "config_sha256": _sha256_json(config),
    }
    manifest_path = _write_json(str(output_dir / "short_manifest.json"), manifest)
    print(json.dumps({
        "research_release": __version__,
        "manifest_file": manifest_path.name,
        "source_archive_count": len(sources),
        "holdout_policy": manifest["parameter_selection_policy"],
        **summary,
        "sensitivity_overview": {
            "scenario_count": sensitivity["scenario_count"],
            "positive_scenarios": sensitivity["positive_scenarios"],
            "positive_share": sensitivity["positive_share"],
            "by_cost_bps": sensitivity["by_cost_bps"],
            "primary_verification": sensitivity["primary_verification"],
        },
    }, ensure_ascii=False, indent=2))
    if sensitivity["primary_verification"]["status"] != "passed":
        raise RuntimeError("Short sensitivity replay differs from primary backtest")


def _trend_v030_backtest(
    evaluation_start: str,
    evaluation_end: str,
    archive_start: str,
    archive_end: str,
    reports_dir: str,
    cache_dir: str,
) -> None:
    config = load_config()
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, funding, sources = download_perpetual_history(
        symbol=config["market"]["symbol"], interval="1h",
        start_month=archive_start, end_month=archive_end, cache_dir=cache_dir,
    )
    market_path = output_dir / f"btcusdt_perpetual_1h_{archive_start}_{archive_end}.csv"
    funding_path = output_dir / f"btcusdt_funding_{archive_start}_{archive_end}.csv"
    frame.to_csv(market_path, index_label="open_time")
    funding.to_csv(funding_path, index_label="funding_time")

    params = trend_v030_parameters_from_config(config)
    primary_trades, primary = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=params, side_mode="both",
    )
    long_trades, long_only = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=params, side_mode="long",
    )
    short_trades, short_only = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=params, side_mode="short",
    )
    stress_params = replace(params, round_trip_cost_bps=45)
    _, stress = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=stress_params, side_mode="both",
    )
    _, stress_long = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=stress_params, side_mode="long",
    )
    _, stress_short = run_trend_v030_backtest(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, params=stress_params, side_mode="short",
    )

    trade_files = {}
    for name, trades in (
        ("combined", primary_trades), ("long", long_trades), ("short", short_trades)
    ):
        path = output_dir / f"trend_v030_{name}_trades.csv"
        rows = trend_v030_trades_as_dicts(trades)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["no_trades"])
            writer.writeheader()
            writer.writerows(rows)
        trade_files[name] = path

    success = evaluate_v030_success(primary, long_only, short_only, stress, stress_short)
    summary = {
        "research_release": __version__,
        "frozen_policy": "0.3.0 parameters and success gates fixed before full-period execution",
        "period_note": "2026 is year-to-date through the last complete archived month",
        "primary_combined": primary,
        "standalone_legs": {"long": long_only, "short": short_only},
        "stress_45bps": {"combined": stress, "long": stress_long, "short": stress_short},
        "success_criteria": success,
    }
    summary_path = _write_json(str(output_dir / "trend_v030_summary.json"), summary)

    sensitivity = run_trend_v030_sensitivity(
        frame, funding, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, base=params,
    )
    replay = sensitivity["primary_replay"]
    sensitivity["primary_verification"] = {
        "status": "passed" if (
            replay["trades"] == primary["trades"]
            and abs(replay["total_r"] - primary["total_r"]) < 1e-12
            and abs(replay["gross_total_r"] - primary["gross_total_r"]) < 1e-12
            and abs(replay["total_funding_r"] - primary["total_funding_r"]) < 1e-12
        ) else "failed",
        "expected_trades": primary["trades"],
        "replayed_trades": replay["trades"],
        "total_r_delta": replay["total_r"] - primary["total_r"],
        "funding_r_delta": replay["total_funding_r"] - primary["total_funding_r"],
    }
    sensitivity_path = _write_json(
        str(output_dir / "trend_v030_sensitivity.json"), sensitivity
    )
    manifest = {
        "schema_version": 1,
        "research_release": __version__,
        "commit_sha": os.getenv("GITHUB_SHA"),
        "instrument": "Binance USD-M BTCUSDT perpetual",
        "source": "official Binance public monthly futures archives",
        "archive_start": archive_start,
        "archive_end": archive_end,
        "evaluation_start": pd.Timestamp(evaluation_start).isoformat(),
        "evaluation_end_exclusive": pd.Timestamp(evaluation_end).isoformat(),
        "parameter_selection_policy": "frozen before 0.3.0 full-period run; sensitivity is descriptive only",
        "primary_parameters": primary["parameters"],
        "success_criteria": success,
        "funding_model": primary["funding_model"],
        "source_archives": sources,
        "market_data_sha256": _sha256_file(market_path),
        "funding_data_sha256": _sha256_file(funding_path),
        "combined_trades_sha256": _sha256_file(trade_files["combined"]),
        "long_trades_sha256": _sha256_file(trade_files["long"]),
        "short_trades_sha256": _sha256_file(trade_files["short"]),
        "summary_sha256": _sha256_file(summary_path),
        "sensitivity_sha256": _sha256_file(sensitivity_path),
        "config_sha256": _sha256_json(config),
    }
    manifest_path = _write_json(str(output_dir / "trend_v030_manifest.json"), manifest)
    print(json.dumps({
        "research_release": __version__,
        "manifest_file": manifest_path.name,
        "summary_file": summary_path.name,
        "sensitivity_file": sensitivity_path.name,
        "source_archive_count": len(sources),
        "primary_combined": primary,
        "standalone_legs": {"long": long_only, "short": short_only},
        "stress_45bps": {"combined": stress, "long": stress_long, "short": stress_short},
        "success_criteria": success,
        "sensitivity_overview": {
            "scenario_count": sensitivity["scenario_count"],
            "positive_scenarios": sensitivity["positive_scenarios"],
            "positive_share": sensitivity["positive_share"],
            "by_cost_bps": sensitivity["by_cost_bps"],
            "primary_verification": sensitivity["primary_verification"],
        },
    }, ensure_ascii=False, indent=2))
    if sensitivity["primary_verification"]["status"] != "passed":
        raise RuntimeError("0.3.0 sensitivity replay differs from primary backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Signal Core 0.3.1")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="Generate the current 4H trend signal")
    live.add_argument("--dry-run", action="store_true", help="Do not store or notify")
    live.add_argument("--no-watch", action="store_true", help="Do not notify WATCH states")
    backtest = commands.add_parser("backtest", help="Run a historical shadow test")
    backtest.add_argument("--days", type=int, default=None)
    backtest.add_argument("--output", default="reports/backtest_trades.csv")
    backtest.add_argument("--market-output", default=None)
    backtest.add_argument("--diagnostics-output", default="reports/backtest_diagnostics.json")
    backtest.add_argument("--manifest-output", default="reports/backtest_manifest.json")
    sensitivity = commands.add_parser(
        "sensitivity", help="Run one-at-a-time parameter robustness analysis"
    )
    sensitivity.add_argument(
        "--market-input", default="reports/backtest_market_1h.csv"
    )
    sensitivity.add_argument(
        "--output", default="reports/backtest_sensitivity.json"
    )
    sensitivity.add_argument("--baseline-summary", default=None)
    trend = commands.add_parser(
        "trend-backtest", help="Run the frozen 2020-2023 4H trend-long holdout"
    )
    trend.add_argument("--evaluation-start", default="2020-01-01")
    trend.add_argument("--evaluation-end", default="2024-01-01")
    trend.add_argument("--archive-start", default="2019-06")
    trend.add_argument("--archive-end", default="2024-03")
    trend.add_argument("--reports-dir", default="reports/trend_2020_2023")
    trend.add_argument("--cache-dir", default=".cache/binance")
    short = commands.add_parser(
        "short-backtest", help="Run the frozen 2020-2023 perpetual short-reversion holdout"
    )
    short.add_argument("--evaluation-start", default="2020-01-01")
    short.add_argument("--evaluation-end", default="2024-01-01")
    short.add_argument("--archive-start", default="2020-01")
    short.add_argument("--archive-end", default="2024-01")
    short.add_argument("--reports-dir", default="reports/short_2020_2023")
    short.add_argument("--cache-dir", default=".cache/binance-perpetual")
    trend_v030 = commands.add_parser(
        "trend-v030-backtest",
        help="Run frozen 0.3.0 BTC perpetual long+short trend evaluation",
    )
    trend_v030.add_argument("--evaluation-start", default="2020-01-01")
    trend_v030.add_argument("--evaluation-end", default="2026-10-01")
    trend_v030.add_argument("--archive-start", default="2020-01")
    trend_v030.add_argument("--archive-end", default="2026-09")
    trend_v030.add_argument("--reports-dir", default="reports/trend_v030_2020_2026ytd")
    trend_v030.add_argument("--cache-dir", default=".cache/binance-perpetual")
    args = parser.parse_args()
    if args.command == "live":
        result = run_once(dry_run=args.dry_run, notify_watch=not args.no_watch)
        print(json.dumps(result.decision.to_dict(), ensure_ascii=False, indent=2))
        print(json.dumps({"duplicate": result.duplicate, "notified": result.notified}))
    elif args.command == "backtest":
        _backtest(
            args.days,
            args.output,
            args.market_output,
            args.diagnostics_output,
            args.manifest_output,
        )
    elif args.command == "sensitivity":
        _sensitivity(args.market_input, args.output, args.baseline_summary)
    elif args.command == "trend-backtest":
        _trend_backtest(
            args.evaluation_start,
            args.evaluation_end,
            args.archive_start,
            args.archive_end,
            args.reports_dir,
            args.cache_dir,
        )
    elif args.command == "short-backtest":
        _short_backtest(
            args.evaluation_start, args.evaluation_end,
            args.archive_start, args.archive_end,
            args.reports_dir, args.cache_dir,
        )
    else:
        _trend_v030_backtest(
            args.evaluation_start, args.evaluation_end,
            args.archive_start, args.archive_end,
            args.reports_dir, args.cache_dir,
        )


if __name__ == "__main__":
    main()
