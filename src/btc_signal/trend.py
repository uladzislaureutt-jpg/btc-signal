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
class TrendParameters:
    breakout_bars: int = 20
    ema_period: int = 200
    ema_slope_bars: int = 20
    atr_period: int = 14
    initial_stop_atr: float = 2.0
    trailing_atr: float = 3.0
    trailing_activation_atr: float = 1.0
    min_risk_to_cost: float = 8.0
    max_holding_days: int = 60
    require_channel_reset: bool = True
    regime_lookback_days: int = 180
    regime_threshold: float = 0.15
    round_trip_cost_bps: int = 30


@dataclass(frozen=True)
class TrendTrade:
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    signal_atr: float
    initial_stop: float
    exit_stop: float
    risk_pct: float
    risk_to_cost: float
    gross_r: float
    cost_r: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    holding_hours: float
    regime: str
    trail_activated: bool
    result: str


def parameters_from_config(config: dict) -> TrendParameters:
    raw = config.get("trend_long", {})
    return TrendParameters(
        breakout_bars=int(raw.get("breakout_bars", 20)),
        ema_period=int(raw.get("ema_period", 200)),
        ema_slope_bars=int(raw.get("ema_slope_bars", 20)),
        atr_period=int(raw.get("atr_period", 14)),
        initial_stop_atr=float(raw.get("initial_stop_atr", 2.0)),
        trailing_atr=float(raw.get("trailing_atr", 3.0)),
        trailing_activation_atr=float(raw.get("trailing_activation_atr", 1.0)),
        min_risk_to_cost=float(raw.get("min_risk_to_cost", 8.0)),
        max_holding_days=int(raw.get("max_holding_days", 60)),
        require_channel_reset=bool(raw.get("require_channel_reset", True)),
        regime_lookback_days=int(raw.get("regime_lookback_days", 180)),
        regime_threshold=float(raw.get("regime_threshold", 0.15)),
        round_trip_cost_bps=int(raw.get("round_trip_cost_bps", 30)),
    )


def _prepare_4h(frame_1h: pd.DataFrame, params: TrendParameters) -> pd.DataFrame:
    frame = resample(frame_1h, "4h").copy()
    frame["ema"] = ema(frame["close"], params.ema_period)
    frame["atr"] = atr(frame, params.atr_period)
    frame["prior_high"] = (
        frame["high"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
    )
    frame["ema_slope_reference"] = frame["ema"].shift(params.ema_slope_bars)
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


def run_trend_backtest(
    frame_1h: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    params: TrendParameters,
    detailed: bool = True,
) -> tuple[list[TrendTrade], dict]:
    """Causal long-only 4H breakout simulation with 1H stop execution.

    ``evaluation_end`` is exclusive for new entries. Positions opened before it
    may use the supplied post-period data to finish normally.
    """
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if end <= start:
        raise ValueError("evaluation_end must be after evaluation_start")
    data_quality = audit_market_data(frame_1h)
    if data_quality["status"] == "error":
        raise ValueError(f"Invalid market data: {', '.join(data_quality['errors'])}")

    four_hour = _prepare_4h(frame_1h, params)
    events = {pd.Timestamp(row.close_time): row for row in four_hour.itertuples()}
    cost_pct = params.round_trip_cost_bps / 10_000
    max_holding = pd.Timedelta(days=params.max_holding_days)
    trades: list[TrendTrade] = []
    pending: dict | None = None
    position: dict | None = None
    armed = not params.require_channel_reset
    rejected_cost = 0
    eligible_breakouts = 0
    ignored_while_open = 0
    missing_entry_bar = 0
    channel_resets = 0

    for timestamp, candle in frame_1h.iterrows():
        timestamp = pd.Timestamp(timestamp)
        close_time = _close_time(timestamp, candle)

        if pending is not None:
            exact_next_bar = timestamp == pending["expected_entry_time"]
            if not exact_next_bar:
                missing_entry_bar += 1
            elif position is None and start <= timestamp < end:
                entry = float(candle["open"])
                entry_atr = float(pending["atr"])
                initial_stop = entry - params.initial_stop_atr * entry_atr
                risk = entry - initial_stop
                risk_pct = risk / entry if entry > 0 else 0.0
                ratio = risk_pct / cost_pct if cost_pct > 0 else float("inf")
                if risk > 0 and ratio >= params.min_risk_to_cost:
                    position = {
                        "signal_time": pending["signal_time"],
                        "entry_time": timestamp,
                        "entry": entry,
                        "entry_atr": entry_atr,
                        "initial_stop": initial_stop,
                        "stop": initial_stop,
                        "risk": risk,
                        "risk_pct": risk_pct,
                        "risk_to_cost": ratio,
                        "regime": pending["regime"],
                        "highest_close": entry,
                        "trail_activated": False,
                        "mfe": 0.0,
                        "mae": 0.0,
                    }
                    armed = False
                else:
                    rejected_cost += 1
            elif position is not None:
                ignored_while_open += 1
            pending = None

        if position is not None:
            entry = float(position["entry"])
            risk = float(position["risk"])
            stop = float(position["stop"])
            bar_open = float(candle["open"])
            bar_high = float(candle["high"])
            bar_low = float(candle["low"])
            position["mfe"] = max(float(position["mfe"]), (bar_high - entry) / risk)
            position["mae"] = min(float(position["mae"]), (bar_low - entry) / risk)
            exit_price: float | None = None
            result: str | None = None
            if timestamp > position["entry_time"] and bar_open <= stop:
                exit_price, result = bar_open, "gap_stop"
            elif bar_low <= stop:
                exit_price, result = stop, "stop"
            elif close_time >= position["entry_time"] + max_holding:
                exit_price, result = float(candle["close"]), "timeout"
            if exit_price is not None:
                gross_r = (exit_price - entry) / risk
                cost_r = cost_pct * ((entry + exit_price) / 2) / risk
                trades.append(TrendTrade(
                    side="long",
                    signal_time=str(position["signal_time"]),
                    entry_time=position["entry_time"].isoformat(),
                    exit_time=close_time.isoformat(),
                    entry=entry,
                    exit=float(exit_price),
                    signal_atr=float(position["entry_atr"]),
                    initial_stop=float(position["initial_stop"]),
                    exit_stop=stop,
                    risk_pct=float(position["risk_pct"]),
                    risk_to_cost=float(position["risk_to_cost"]),
                    gross_r=float(gross_r),
                    cost_r=float(cost_r),
                    r_multiple=float(gross_r - cost_r),
                    mfe_r=float(position["mfe"]),
                    mae_r=float(position["mae"]),
                    holding_hours=float((close_time - position["entry_time"]) / pd.Timedelta(hours=1)),
                    regime=str(position["regime"]),
                    trail_activated=bool(position["trail_activated"]),
                    result=result,
                ))
                position = None

        event = events.get(close_time)
        if event is not None:
            prior_high = float(event.prior_high)
            values_ready = all(np.isfinite(value) for value in (
                prior_high, float(event.ema), float(event.ema_slope_reference), float(event.atr)
            ))
            inside_channel = values_ready and float(event.close) <= prior_high
            if inside_channel and not armed:
                channel_resets += 1
                armed = True

            if position is not None:
                completed_close = float(event.close)
                position["highest_close"] = max(float(position["highest_close"]), completed_close)
                if completed_close >= (
                    float(position["entry"])
                    + params.trailing_activation_atr * float(position["entry_atr"])
                ):
                    position["trail_activated"] = True
                if position["trail_activated"] and np.isfinite(float(event.atr)):
                    candidate = float(position["highest_close"]) - params.trailing_atr * float(event.atr)
                    position["stop"] = max(float(position["stop"]), candidate)

            breakout = values_ready and float(event.close) > prior_high
            trend_ok = values_ready and (
                float(event.close) > float(event.ema)
                and float(event.ema) > float(event.ema_slope_reference)
            )
            if breakout and trend_ok and armed:
                eligible_breakouts += 1
                pending = {
                    "signal_time": close_time.isoformat(),
                    "expected_entry_time": pd.Timestamp(event.Index) + pd.Timedelta(hours=4),
                    "atr": float(event.atr),
                    "regime": _regime(float(event.regime_return), params.regime_threshold),
                }

    if position is not None and len(frame_1h):
        candle = frame_1h.iloc[-1]
        timestamp = frame_1h.index[-1]
        close_time = _close_time(timestamp, candle)
        entry = float(position["entry"])
        exit_price = float(candle["close"])
        risk = float(position["risk"])
        gross_r = (exit_price - entry) / risk
        cost_r = cost_pct * ((entry + exit_price) / 2) / risk
        trades.append(TrendTrade(
            side="long",
            signal_time=str(position["signal_time"]),
            entry_time=position["entry_time"].isoformat(),
            exit_time=close_time.isoformat(),
            entry=entry,
            exit=exit_price,
            signal_atr=float(position["entry_atr"]),
            initial_stop=float(position["initial_stop"]),
            exit_stop=float(position["stop"]),
            risk_pct=float(position["risk_pct"]),
            risk_to_cost=float(position["risk_to_cost"]),
            gross_r=float(gross_r),
            cost_r=float(cost_r),
            r_multiple=float(gross_r - cost_r),
            mfe_r=float(position["mfe"]),
            mae_r=float(position["mae"]),
            holding_hours=float((close_time - position["entry_time"]) / pd.Timedelta(hours=1)),
            regime=str(position["regime"]),
            trail_activated=bool(position["trail_activated"]),
            result="end_of_data",
        ))

    summary = summarize_trend(
        trades,
        frame_1h=frame_1h,
        evaluation_start=start,
        evaluation_end=end,
        params=params,
        data_quality=data_quality,
        detailed=detailed,
    )
    summary["diagnostics"] = {
        "eligible_breakouts": eligible_breakouts,
        "cost_gate_rejections": rejected_cost,
        "signals_ignored_while_position_open": ignored_while_open,
        "signals_missing_exact_next_1h_bar": missing_entry_bar,
        "channel_resets": channel_resets,
    }
    return trades, summary


def _strategy_equity(trades: list[TrendTrade], risk_fraction: float = 0.01) -> dict:
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


def _benchmark(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    period = frame.loc[(frame.index >= start) & (frame.index < end)]
    if period.empty:
        return {"return_pct": None, "max_drawdown_pct": None}
    initial = float(period.iloc[0]["open"])
    closes = period["close"].astype(float)
    curve = closes / initial
    drawdown = curve / curve.cummax() - 1.0
    return {
        "return_pct": float((float(closes.iloc[-1]) / initial - 1.0) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
    }


def summarize_trend(
    trades: list[TrendTrade],
    *,
    frame_1h: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    params: TrendParameters,
    data_quality: dict,
    detailed: bool,
) -> dict:
    metrics = outcome_metrics(trade.r_multiple for trade in trades)
    gross = outcome_metrics(trade.gross_r for trade in trades)
    result = {
        **metrics,
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "gross_profit_factor": gross["profit_factor"],
        "total_cost_r": float(sum(trade.cost_r for trade in trades)),
        "average_cost_r": float(np.mean([trade.cost_r for trade in trades])) if trades else 0.0,
        "median_cost_r": float(np.median([trade.cost_r for trade in trades])) if trades else 0.0,
        "median_risk_pct": float(np.median([trade.risk_pct for trade in trades])) if trades else 0.0,
        "median_holding_hours": float(np.median([trade.holding_hours for trade in trades])) if trades else 0.0,
        "trailing_activation_rate": float(np.mean([trade.trail_activated for trade in trades])) if trades else 0.0,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "parameters": asdict(params),
        "execution_model": "completed_4h_signal_next_1h_open_hourly_stop",
        "higher_timeframe_policy": "complete_4h_bars_only_trailing_effective_next_1h_bar",
        "intrabar_rule": "gap_then_stop_no_target",
        "funding_model": "not_applicable_spot_long_only",
    }
    if not detailed:
        return result
    total_holding = sum(trade.holding_hours for trade in trades)
    evaluation_hours = float((evaluation_end - evaluation_start) / pd.Timedelta(hours=1))
    gross_positive = sum(max(trade.gross_r, 0.0) for trade in trades)
    largest = max((trade.gross_r for trade in trades), default=0.0)
    result.update({
        "exposure_pct": float(100 * total_holding / evaluation_hours) if evaluation_hours else 0.0,
        "largest_winner_share_of_gross_profit": float(largest / gross_positive) if gross_positive else None,
        "by_year": time_breakdown(trades, "year"),
        "by_quarter": time_breakdown(trades, "quarter"),
        "by_regime": trade_breakdown(trades, "regime"),
        "by_result": trade_breakdown(trades, "result"),
        "bootstrap": moving_block_bootstrap(trades),
        "walk_forward": fixed_walk_forward(
            trades, evaluation_start, evaluation_end, train_days=365, test_days=180
        ),
        "strategy_equity_1pct_risk": _strategy_equity(trades),
        "buy_and_hold": _benchmark(frame_1h, evaluation_start, evaluation_end),
        "data_quality": data_quality,
        "top_winners": [
            asdict(trade)
            for trade in sorted(trades, key=lambda item: item.r_multiple, reverse=True)[:10]
        ],
    })
    return result


def run_trend_sensitivity(
    frame_1h: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    base: TrendParameters,
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
        _, metrics = run_trend_backtest(
            frame_1h,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            params=params,
            detailed=False,
        )
        scenarios.append(metrics)
    positive = [item for item in scenarios if item["total_r"] > 0]
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
        "positive_scenarios": len(positive),
        "positive_share": len(positive) / len(scenarios),
        "primary_parameters": asdict(base),
        "primary_replay": primary,
        "by_cost_bps": by_cost,
        "scenarios": scenarios,
    }


def trades_as_dicts(trades: list[TrendTrade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
