from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from .backtest import audit_market_data, resample
from .indicators import atr, ema
from .validation import (
    fixed_walk_forward,
    moving_block_bootstrap,
    outcome_metrics,
    time_breakdown,
    trade_breakdown,
)


@dataclass(frozen=True)
class TrendV030Parameters:
    breakout_bars: int = 20
    ema_period: int = 200
    atr_period: int = 14
    initial_stop_atr: float = 2.0
    trailing_atr: float = 3.0
    trailing_activation_atr: float = 1.0
    min_risk_to_cost: float = 8.0
    max_holding_days: int = 60
    round_trip_cost_bps: int = 30
    regime_lookback_days: int = 180
    regime_threshold: float = 0.15


@dataclass(frozen=True)
class TrendV030Trade:
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    signal_atr: float
    channel_level: float
    initial_stop: float
    exit_stop: float
    risk_pct: float
    risk_to_cost: float
    gross_r: float
    trading_cost_r: float
    funding_r: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    holding_hours: float
    funding_events: int
    regime: str
    trail_activated: bool
    result: str


def parameters_from_config(config: dict) -> TrendV030Parameters:
    raw = config.get("trend_bidirectional", {})
    return TrendV030Parameters(
        breakout_bars=int(raw.get("breakout_bars", 20)),
        ema_period=int(raw.get("ema_period", 200)),
        atr_period=int(raw.get("atr_period", 14)),
        initial_stop_atr=float(raw.get("initial_stop_atr", 2.0)),
        trailing_atr=float(raw.get("trailing_atr", 3.0)),
        trailing_activation_atr=float(raw.get("trailing_activation_atr", 1.0)),
        min_risk_to_cost=float(raw.get("min_risk_to_cost", 8.0)),
        max_holding_days=int(raw.get("max_holding_days", 60)),
        round_trip_cost_bps=int(raw.get("round_trip_cost_bps", 30)),
        regime_lookback_days=int(raw.get("regime_lookback_days", 180)),
        regime_threshold=float(raw.get("regime_threshold", 0.15)),
    )


def _prepare(frame_1h: pd.DataFrame, params: TrendV030Parameters) -> pd.DataFrame:
    frame = resample(frame_1h, "4h").copy()
    frame["ema"] = ema(frame["close"], params.ema_period)
    frame["atr"] = atr(frame, params.atr_period)
    frame["prior_high"] = (
        frame["high"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
    )
    frame["prior_low"] = (
        frame["low"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).min()
    )
    regime_bars = params.regime_lookback_days * 6
    frame["regime_return"] = frame["close"] / frame["close"].shift(regime_bars) - 1.0
    return frame


def _regime(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > threshold:
        return "bull"
    if value < -threshold:
        return "bear"
    return "sideways"


def _close_time(timestamp: pd.Timestamp, candle: pd.Series) -> pd.Timestamp:
    value = candle.get("close_time")
    if value is None or pd.isna(value):
        return timestamp + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return pd.Timestamp(value)


def _funding_audit(funding: pd.DataFrame) -> dict:
    if not isinstance(funding.index, pd.DatetimeIndex):
        return {"status": "error", "errors": ["index_not_datetime"]}
    deltas = funding.index.to_series().diff().dropna()
    gaps = deltas[deltas > pd.Timedelta(hours=12)]
    return {
        "status": "warning" if len(gaps) else "ok",
        "rows": len(funding),
        "start": funding.index.min().isoformat() if len(funding) else None,
        "end": funding.index.max().isoformat() if len(funding) else None,
        "duplicate_times": int(funding.index.duplicated().sum()),
        "gap_count_over_12h": int(len(gaps)),
        "min_rate": float(funding["funding_rate"].min()) if len(funding) else None,
        "max_rate": float(funding["funding_rate"].max()) if len(funding) else None,
    }


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def run_trend_v030_backtest(
    frame_1h: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    params: TrendV030Parameters,
    side_mode: str = "both",
    detailed: bool = True,
) -> tuple[list[TrendV030Trade], dict]:
    """Causal two-sided 4H breakout simulation with 1H execution and funding.

    Signals use only completed 4H candles. Entries occur at the exact next 1H
    open. ``evaluation_end`` is exclusive for new entries.
    """
    if side_mode not in {"both", "long", "short"}:
        raise ValueError("side_mode must be one of: both, long, short")
    start, end = _utc(evaluation_start), _utc(evaluation_end)
    if end <= start:
        raise ValueError("evaluation_end must be after evaluation_start")
    data_quality = audit_market_data(frame_1h)
    if data_quality["status"] == "error":
        raise ValueError(f"Invalid perpetual data: {', '.join(data_quality['errors'])}")
    funding_quality = _funding_audit(funding)
    if funding_quality["status"] == "error":
        raise ValueError("Invalid funding data")

    four_hour = _prepare(frame_1h, params)
    events = {pd.Timestamp(row.close_time): row for row in four_hour.itertuples()}
    funding_rates = funding["funding_rate"].astype(float).to_dict()
    cost_pct = params.round_trip_cost_bps / 10_000
    max_holding = pd.Timedelta(days=params.max_holding_days)
    trades: list[TrendV030Trade] = []
    pending: dict | None = None
    position: dict | None = None
    diagnostics = {
        "eligible_long_breakouts": 0,
        "eligible_short_breakouts": 0,
        "cost_gate_rejections_long": 0,
        "cost_gate_rejections_short": 0,
        "signals_ignored_while_position_open": 0,
        "signals_missing_exact_next_1h_bar": 0,
    }

    def finish(exit_price: float, exit_time: pd.Timestamp, result: str) -> None:
        nonlocal position
        assert position is not None
        side = str(position["side"])
        entry = float(position["entry"])
        risk = float(position["risk"])
        gross_r = ((exit_price - entry) if side == "long" else (entry - exit_price)) / risk
        trading_cost_r = cost_pct * ((entry + exit_price) / 2) / risk
        funding_r = float(position["funding_r"])
        trades.append(TrendV030Trade(
            side=side,
            signal_time=str(position["signal_time"]),
            entry_time=position["entry_time"].isoformat(),
            exit_time=exit_time.isoformat(),
            entry=entry,
            exit=float(exit_price),
            signal_atr=float(position["signal_atr"]),
            channel_level=float(position["channel_level"]),
            initial_stop=float(position["initial_stop"]),
            exit_stop=float(position["stop"]),
            risk_pct=float(position["risk_pct"]),
            risk_to_cost=float(position["risk_to_cost"]),
            gross_r=float(gross_r),
            trading_cost_r=float(trading_cost_r),
            funding_r=funding_r,
            r_multiple=float(gross_r - trading_cost_r + funding_r),
            mfe_r=float(position["mfe"]),
            mae_r=float(position["mae"]),
            holding_hours=float((exit_time - position["entry_time"]) / pd.Timedelta(hours=1)),
            funding_events=int(position["funding_events"]),
            regime=str(position["regime"]),
            trail_activated=bool(position["trail_activated"]),
            result=result,
        ))
        position = None

    for timestamp, candle in frame_1h.iterrows():
        timestamp = pd.Timestamp(timestamp)
        close_time = _close_time(timestamp, candle)

        if pending is not None:
            if timestamp != pending["expected_entry_time"]:
                diagnostics["signals_missing_exact_next_1h_bar"] += 1
            elif position is not None:
                diagnostics["signals_ignored_while_position_open"] += 1
            elif start <= timestamp < end:
                side = str(pending["side"])
                entry = float(candle["open"])
                risk = params.initial_stop_atr * float(pending["atr"])
                stop = entry - risk if side == "long" else entry + risk
                risk_pct = risk / entry if entry > 0 else 0.0
                ratio = risk_pct / cost_pct if cost_pct > 0 else float("inf")
                if risk > 0 and stop > 0 and ratio >= params.min_risk_to_cost:
                    position = {
                        "side": side,
                        "signal_time": pending["signal_time"],
                        "entry_time": timestamp,
                        "entry": entry,
                        "risk": risk,
                        "signal_atr": float(pending["atr"]),
                        "channel_level": float(pending["channel_level"]),
                        "initial_stop": stop,
                        "stop": stop,
                        "risk_pct": risk_pct,
                        "risk_to_cost": ratio,
                        "regime": pending["regime"],
                        "best_close": entry,
                        "trail_activated": False,
                        "funding_r": 0.0,
                        "funding_events": 0,
                        "mfe": 0.0,
                        "mae": 0.0,
                    }
                else:
                    diagnostics[f"cost_gate_rejections_{side}"] += 1
            pending = None

        if position is not None and timestamp in funding_rates and timestamp > position["entry_time"]:
            rate = float(funding_rates[timestamp])
            direction = -1.0 if position["side"] == "long" else 1.0
            position["funding_r"] += direction * rate * float(candle["open"]) / float(position["risk"])
            position["funding_events"] += 1

        if position is not None:
            side = str(position["side"])
            entry = float(position["entry"])
            risk = float(position["risk"])
            stop = float(position["stop"])
            bar_open = float(candle["open"])
            bar_high = float(candle["high"])
            bar_low = float(candle["low"])
            favorable = (bar_high - entry) / risk if side == "long" else (entry - bar_low) / risk
            adverse = (bar_low - entry) / risk if side == "long" else (entry - bar_high) / risk
            position["mfe"] = max(float(position["mfe"]), favorable)
            position["mae"] = min(float(position["mae"]), adverse)
            exit_price: float | None = None
            result: str | None = None
            if side == "long":
                if timestamp > position["entry_time"] and bar_open <= stop:
                    exit_price, result = bar_open, "gap_stop"
                elif bar_low <= stop:
                    exit_price, result = stop, "stop"
            else:
                if timestamp > position["entry_time"] and bar_open >= stop:
                    exit_price, result = bar_open, "gap_stop"
                elif bar_high >= stop:
                    exit_price, result = stop, "stop"
            if exit_price is None and close_time >= position["entry_time"] + max_holding:
                exit_price, result = float(candle["close"]), "timeout"
            if exit_price is not None and result is not None:
                finish(exit_price, close_time, result)

        event = events.get(close_time)
        if event is None:
            continue

        if position is not None:
            completed_close = float(event.close)
            signal_atr = float(position["signal_atr"])
            if position["side"] == "long":
                position["best_close"] = max(float(position["best_close"]), completed_close)
                if completed_close >= float(position["entry"]) + params.trailing_activation_atr * signal_atr:
                    position["trail_activated"] = True
                if position["trail_activated"] and np.isfinite(float(event.atr)):
                    candidate = float(position["best_close"]) - params.trailing_atr * float(event.atr)
                    position["stop"] = max(float(position["stop"]), candidate)
            else:
                position["best_close"] = min(float(position["best_close"]), completed_close)
                if completed_close <= float(position["entry"]) - params.trailing_activation_atr * signal_atr:
                    position["trail_activated"] = True
                if position["trail_activated"] and np.isfinite(float(event.atr)):
                    candidate = float(position["best_close"]) + params.trailing_atr * float(event.atr)
                    position["stop"] = min(float(position["stop"]), candidate)

        values_ready = all(np.isfinite(value) for value in (
            float(event.prior_high), float(event.prior_low), float(event.ema), float(event.atr)
        ))
        long_signal = (
            values_ready and side_mode in {"both", "long"}
            and float(event.close) > float(event.prior_high)
            and float(event.close) > float(event.ema)
        )
        short_signal = (
            values_ready and side_mode in {"both", "short"}
            and float(event.close) < float(event.prior_low)
            and float(event.close) < float(event.ema)
        )
        if long_signal or short_signal:
            side = "long" if long_signal else "short"
            diagnostics[f"eligible_{side}_breakouts"] += 1
            if position is not None or pending is not None:
                diagnostics["signals_ignored_while_position_open"] += 1
            else:
                pending = {
                    "side": side,
                    "signal_time": close_time.isoformat(),
                    "expected_entry_time": pd.Timestamp(event.Index) + pd.Timedelta(hours=4),
                    "atr": float(event.atr),
                    "channel_level": float(event.prior_high if side == "long" else event.prior_low),
                    "regime": _regime(float(event.regime_return), params.regime_threshold),
                }

    if position is not None and len(frame_1h):
        candle = frame_1h.iloc[-1]
        finish(float(candle["close"]), _close_time(frame_1h.index[-1], candle), "end_of_data")

    summary = summarize_trend_v030(
        trades,
        frame_1h=frame_1h,
        funding=funding,
        evaluation_start=start,
        evaluation_end=end,
        params=params,
        side_mode=side_mode,
        data_quality=data_quality,
        funding_quality=funding_quality,
        detailed=detailed,
    )
    summary["diagnostics"] = diagnostics
    return trades, summary


def _strategy_equity(trades: list[TrendV030Trade], risk_fraction: float = 0.01) -> dict:
    equity = [1.0]
    for trade in trades:
        equity.append(equity[-1] * max(0.0, 1.0 + risk_fraction * trade.r_multiple))
    values = np.asarray(equity)
    drawdown = values / np.maximum.accumulate(values) - 1.0
    return {
        "risk_fraction_per_trade": risk_fraction,
        "return_pct": float((values[-1] - 1.0) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
    }


def _side_metrics(trades: list[TrendV030Trade], detailed: bool) -> dict:
    result = {
        **outcome_metrics(trade.r_multiple for trade in trades),
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "total_trading_cost_r": float(sum(trade.trading_cost_r for trade in trades)),
        "total_funding_r": float(sum(trade.funding_r for trade in trades)),
    }
    if detailed:
        result.update({
            "by_year": time_breakdown(trades, "year"),
            "by_regime": trade_breakdown(trades, "regime"),
            "bootstrap": moving_block_bootstrap(trades),
        })
    return result


def _portfolio_diagnostics(
    trades: list[TrendV030Trade], start: pd.Timestamp, end: pd.Timestamp
) -> dict:
    months = pd.period_range(start=start.tz_localize(None), end=(end - pd.Timedelta(seconds=1)).tz_localize(None), freq="M")
    monthly = pd.DataFrame(0.0, index=months, columns=["long", "short"])
    for trade in trades:
        period = pd.Timestamp(trade.exit_time).tz_convert("UTC").tz_localize(None).to_period("M")
        if period in monthly.index:
            monthly.loc[period, trade.side] += trade.r_multiple
    monthly["combined"] = monthly.sum(axis=1)
    correlation = None
    if len(monthly) > 1 and monthly["long"].std() > 0 and monthly["short"].std() > 0:
        correlation = float(monthly["long"].corr(monthly["short"]))

    rolling = {}
    for window in (6, 12):
        values = monthly["combined"].rolling(window, min_periods=window).sum()
        if values.notna().any():
            period = values.idxmin()
            rolling[f"worst_{window}_month_r"] = float(values.loc[period])
            rolling[f"worst_{window}_month_end"] = str(period)
        else:
            rolling[f"worst_{window}_month_r"] = None
            rolling[f"worst_{window}_month_end"] = None

    ordered = sorted(trades, key=lambda item: pd.Timestamp(item.exit_time))
    equity = 0.0
    peak = 0.0
    peak_time = start
    longest = pd.Timedelta(0)
    for trade in ordered:
        equity += trade.r_multiple
        timestamp = pd.Timestamp(trade.exit_time)
        if equity >= peak:
            peak = equity
            peak_time = timestamp
        else:
            longest = max(longest, timestamp - peak_time)
    if equity < peak:
        longest = max(longest, end - peak_time)

    yearly = time_breakdown(trades, "year")
    positive_totals = [max(0.0, value["total_r"]) for value in yearly.values()]
    concentration = max(positive_totals, default=0.0) / sum(positive_totals) if sum(positive_totals) else None
    return {
        "monthly_long_short_correlation": correlation,
        **rolling,
        "max_time_underwater_days": float(longest / pd.Timedelta(days=1)),
        "largest_positive_year_share": concentration,
        "monthly_returns_r": {
            str(index): {column: float(row[column]) for column in monthly.columns}
            for index, row in monthly.iterrows()
        },
    }


def summarize_trend_v030(
    trades: list[TrendV030Trade],
    *,
    frame_1h: pd.DataFrame,
    funding: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    params: TrendV030Parameters,
    side_mode: str,
    data_quality: dict,
    funding_quality: dict,
    detailed: bool,
) -> dict:
    metrics = outcome_metrics(trade.r_multiple for trade in trades)
    gross = outcome_metrics(trade.gross_r for trade in trades)
    result = {
        **metrics,
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "gross_profit_factor": gross["profit_factor"],
        "total_trading_cost_r": float(sum(trade.trading_cost_r for trade in trades)),
        "total_funding_r": float(sum(trade.funding_r for trade in trades)),
        "average_trading_cost_r": float(np.mean([trade.trading_cost_r for trade in trades])) if trades else 0.0,
        "median_risk_pct": float(np.median([trade.risk_pct for trade in trades])) if trades else 0.0,
        "median_holding_hours": float(np.median([trade.holding_hours for trade in trades])) if trades else 0.0,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "side_mode": side_mode,
        "parameters": asdict(params),
        "execution_model": "completed_4h_20_bar_breakout_next_1h_open_hourly_stop",
        "higher_timeframe_policy": "complete_4h_bars_only_trailing_effective_next_1h_bar",
        "intrabar_rule": "gap_then_stop_no_target",
        "funding_model": "archived_rate_times_1h_open_notional_signed_by_position_side",
    }
    if not detailed:
        return result
    total_holding = sum(trade.holding_hours for trade in trades)
    evaluation_hours = float((evaluation_end - evaluation_start) / pd.Timedelta(hours=1))
    result.update({
        "exposure_pct": float(100 * total_holding / evaluation_hours) if evaluation_hours else 0.0,
        "by_side": {
            side: _side_metrics([trade for trade in trades if trade.side == side], True)
            for side in ("long", "short")
        },
        "by_year": time_breakdown(trades, "year"),
        "by_quarter": time_breakdown(trades, "quarter"),
        "by_regime": trade_breakdown(trades, "regime"),
        "by_result": trade_breakdown(trades, "result"),
        "bootstrap": moving_block_bootstrap(trades),
        "walk_forward": fixed_walk_forward(
            trades, evaluation_start, evaluation_end, train_days=365, test_days=180
        ),
        "portfolio_diagnostics": _portfolio_diagnostics(trades, evaluation_start, evaluation_end),
        "strategy_equity_1pct_risk": _strategy_equity(trades),
        "data_quality": data_quality,
        "funding_quality": funding_quality,
        "top_winners": [
            asdict(trade)
            for trade in sorted(trades, key=lambda item: item.r_multiple, reverse=True)[:10]
        ],
    })
    return result


def run_trend_v030_sensitivity(
    frame_1h: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    base: TrendV030Parameters,
    breakout_values: tuple[int, ...] = (15, 20, 30),
    stop_values: tuple[float, ...] = (1.5, 2.0, 2.5),
    trail_values: tuple[float, ...] = (2.5, 3.0, 4.0),
    cost_values: tuple[int, ...] = (15, 30, 45),
) -> dict:
    scenarios = []
    for breakout_bars, stop_atr, trail_atr, cost_bps in product(
        breakout_values, stop_values, trail_values, cost_values
    ):
        params = replace(
            base,
            breakout_bars=breakout_bars,
            initial_stop_atr=stop_atr,
            trailing_atr=trail_atr,
            round_trip_cost_bps=cost_bps,
        )
        _, metrics = run_trend_v030_backtest(
            frame_1h,
            funding,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            params=params,
            side_mode="both",
            detailed=False,
        )
        scenarios.append(metrics)
    primary = next(item for item in scenarios if (
        item["parameters"]["breakout_bars"] == base.breakout_bars
        and item["parameters"]["initial_stop_atr"] == base.initial_stop_atr
        and item["parameters"]["trailing_atr"] == base.trailing_atr
        and item["parameters"]["round_trip_cost_bps"] == base.round_trip_cost_bps
    ))
    by_cost = {}
    for cost in cost_values:
        subset = [item for item in scenarios if item["parameters"]["round_trip_cost_bps"] == cost]
        by_cost[str(cost)] = {
            "scenarios": len(subset),
            "positive": sum(item["total_r"] > 0 for item in subset),
            "median_total_r": float(np.median([item["total_r"] for item in subset])),
            "min_total_r": float(min(item["total_r"] for item in subset)),
            "max_total_r": float(max(item["total_r"] for item in subset)),
        }
    return {
        "method": "frozen_full_factorial_no_selection",
        "scenario_count": len(scenarios),
        "positive_scenarios": sum(item["total_r"] > 0 for item in scenarios),
        "positive_share": sum(item["total_r"] > 0 for item in scenarios) / len(scenarios),
        "primary_parameters": asdict(base),
        "primary_replay": primary,
        "by_cost_bps": by_cost,
        "scenarios": scenarios,
    }


def evaluate_success_criteria(
    primary: dict,
    long_only: dict,
    short_only: dict,
    stress: dict,
    stress_short: dict,
) -> dict:
    years = primary.get("by_year", {})
    independent = [years.get(str(year), {"total_r": 0.0})["total_r"] for year in range(2020, 2024)]
    complete = [years.get(str(year), {"total_r": 0.0})["total_r"] for year in range(2020, 2026)]
    later_total = sum(years.get(str(year), {"total_r": 0.0})["total_r"] for year in range(2024, 2027))
    combined_gates = {
        "minimum_300_trades": primary["trades"] >= 300,
        "positive_2020_2023_total": sum(independent) > 0,
        "positive_at_least_3_of_4_years_2020_2023": sum(value > 0 for value in independent) >= 3,
        "positive_2024_2026_ytd_total": later_total > 0,
        "positive_at_least_4_of_6_complete_years_2020_2025": sum(value > 0 for value in complete) >= 4,
        "stress_45bps_total_positive": stress["total_r"] > 0,
        "stress_45bps_profit_factor_at_least_1_15": (
            stress["profit_factor"] is not None and stress["profit_factor"] >= 1.15
        ),
        "stress_45bps_bootstrap_lower_expectancy_above_zero": (
            stress["bootstrap"]["expectancy_r_ci95"][0] > 0
        ),
        "largest_positive_year_share_at_most_50pct": (
            primary["portfolio_diagnostics"]["largest_positive_year_share"] is not None
            and primary["portfolio_diagnostics"]["largest_positive_year_share"] <= 0.50
        ),
    }
    short_gates = {
        "short_total_positive_primary": short_only["total_r"] > 0,
        "short_total_positive_at_45bps": stress_short["total_r"] > 0,
        "short_bootstrap_lower_expectancy_above_zero_at_45bps": (
            stress_short["bootstrap"]["expectancy_r_ci95"][0] > 0
        ),
        "short_bear_regime_positive_primary": (
            short_only.get("by_regime", {}).get("bear", {}).get("total_r", 0.0) > 0
        ),
    }
    return {
        "policy": "predeclared_before_0.3.0_full-period_run",
        "combined": {**combined_gates, "passed": all(combined_gates.values())},
        "short_standalone": {**short_gates, "passed": all(short_gates.values())},
        "long_reference_total_r": long_only["total_r"],
        "short_reference_total_r": short_only["total_r"],
    }


def trades_as_dicts(trades: list[TrendV030Trade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
