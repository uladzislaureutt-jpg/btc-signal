from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from .backtest import audit_market_data, resample
from .indicators import atr, rsi
from .validation import (
    fixed_walk_forward,
    moving_block_bootstrap,
    outcome_metrics,
    time_breakdown,
    trade_breakdown,
)


@dataclass(frozen=True)
class ShortParameters:
    resistance_lookback_bars: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    min_rsi: float = 60.0
    rejection_close_location_max: float = 0.50
    initial_stop_atr: float = 2.0
    target_r: float = 2.0
    min_risk_to_cost: float = 8.0
    max_holding_days: int = 14
    cooldown_hours: int = 72
    round_trip_cost_bps: int = 30
    regime_lookback_days: int = 180
    regime_threshold: float = 0.15


@dataclass(frozen=True)
class ShortTrade:
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    signal_atr: float
    resistance: float
    stop: float
    target: float
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
    result: str


def parameters_from_config(config: dict) -> ShortParameters:
    raw = config.get("short_reversion", {})
    return ShortParameters(
        resistance_lookback_bars=int(raw.get("resistance_lookback_bars", 20)),
        atr_period=int(raw.get("atr_period", 14)),
        rsi_period=int(raw.get("rsi_period", 14)),
        min_rsi=float(raw.get("min_rsi", 60)),
        rejection_close_location_max=float(raw.get("rejection_close_location_max", 0.5)),
        initial_stop_atr=float(raw.get("initial_stop_atr", 2.0)),
        target_r=float(raw.get("target_r", 2.0)),
        min_risk_to_cost=float(raw.get("min_risk_to_cost", 8.0)),
        max_holding_days=int(raw.get("max_holding_days", 14)),
        cooldown_hours=int(raw.get("cooldown_hours", 72)),
        round_trip_cost_bps=int(raw.get("round_trip_cost_bps", 30)),
        regime_lookback_days=int(raw.get("regime_lookback_days", 180)),
        regime_threshold=float(raw.get("regime_threshold", 0.15)),
    )


def _prepare(frame_1h: pd.DataFrame, params: ShortParameters) -> pd.DataFrame:
    frame = resample(frame_1h, "4h").copy()
    frame["atr"] = atr(frame, params.atr_period)
    frame["rsi"] = rsi(frame["close"], params.rsi_period)
    frame["prior_high"] = (
        frame["high"].shift(1)
        .rolling(params.resistance_lookback_bars, min_periods=params.resistance_lookback_bars)
        .max()
    )
    regime_bars = params.regime_lookback_days * 6
    frame["regime_return"] = frame["close"] / frame["close"].shift(regime_bars) - 1.0
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_location"] = (frame["close"] - frame["low"]) / candle_range
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


def run_short_backtest(
    frame_1h: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    params: ShortParameters,
    detailed: bool = True,
) -> tuple[list[ShortTrade], dict]:
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
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
    cooldown = pd.Timedelta(hours=params.cooldown_hours)
    cooldown_until: pd.Timestamp | None = None
    pending: dict | None = None
    position: dict | None = None
    trades: list[ShortTrade] = []
    diagnostics = {
        "eligible_rejections": 0,
        "cost_gate_rejections": 0,
        "signals_during_position": 0,
        "signals_during_cooldown": 0,
        "signals_missing_exact_next_1h_bar": 0,
    }

    def finish(exit_price: float, exit_time: pd.Timestamp, result: str) -> None:
        nonlocal position, cooldown_until
        assert position is not None
        entry = float(position["entry"])
        risk = float(position["risk"])
        gross_r = (entry - exit_price) / risk
        trading_cost_r = cost_pct * ((entry + exit_price) / 2) / risk
        funding_r = float(position["funding_r"])
        trades.append(ShortTrade(
            side="short",
            signal_time=str(position["signal_time"]),
            entry_time=position["entry_time"].isoformat(),
            exit_time=exit_time.isoformat(),
            entry=entry,
            exit=float(exit_price),
            signal_atr=float(position["signal_atr"]),
            resistance=float(position["resistance"]),
            stop=float(position["stop"]),
            target=float(position["target"]),
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
            result=result,
        ))
        cooldown_until = exit_time + cooldown
        position = None

    for timestamp, candle in frame_1h.iterrows():
        timestamp = pd.Timestamp(timestamp)
        close_time = _close_time(timestamp, candle)
        if pending is not None:
            if timestamp != pending["expected_entry_time"]:
                diagnostics["signals_missing_exact_next_1h_bar"] += 1
            elif position is not None:
                diagnostics["signals_during_position"] += 1
            elif cooldown_until is not None and timestamp < cooldown_until:
                diagnostics["signals_during_cooldown"] += 1
            elif start <= timestamp < end:
                entry = float(candle["open"])
                risk = params.initial_stop_atr * float(pending["atr"])
                stop = entry + risk
                target = entry - params.target_r * risk
                risk_pct = risk / entry if entry > 0 else 0.0
                ratio = risk_pct / cost_pct if cost_pct > 0 else float("inf")
                if risk > 0 and target > 0 and ratio >= params.min_risk_to_cost:
                    position = {
                        "signal_time": pending["signal_time"],
                        "entry_time": timestamp,
                        "entry": entry,
                        "risk": risk,
                        "signal_atr": float(pending["atr"]),
                        "resistance": float(pending["resistance"]),
                        "stop": stop,
                        "target": target,
                        "risk_pct": risk_pct,
                        "risk_to_cost": ratio,
                        "regime": pending["regime"],
                        "funding_r": 0.0,
                        "funding_events": 0,
                        "mfe": 0.0,
                        "mae": 0.0,
                    }
                else:
                    diagnostics["cost_gate_rejections"] += 1
            pending = None

        if position is not None and timestamp in funding_rates and timestamp > position["entry_time"]:
            rate = float(funding_rates[timestamp])
            position["funding_r"] += rate * float(candle["open"]) / float(position["risk"])
            position["funding_events"] += 1

        if position is not None:
            entry = float(position["entry"])
            risk = float(position["risk"])
            stop = float(position["stop"])
            target = float(position["target"])
            bar_open = float(candle["open"])
            bar_high = float(candle["high"])
            bar_low = float(candle["low"])
            position["mfe"] = max(float(position["mfe"]), (entry - bar_low) / risk)
            position["mae"] = min(float(position["mae"]), (entry - bar_high) / risk)
            if timestamp > position["entry_time"] and bar_open >= stop:
                finish(bar_open, close_time, "gap_stop")
            elif timestamp > position["entry_time"] and bar_open <= target:
                finish(bar_open, close_time, "gap_target")
            elif bar_high >= stop:
                finish(stop, close_time, "stop")
            elif bar_low <= target:
                finish(target, close_time, "target")
            elif close_time >= position["entry_time"] + max_holding:
                finish(float(candle["close"]), close_time, "timeout")

        event = events.get(close_time)
        if event is not None:
            values = (
                float(event.prior_high), float(event.atr), float(event.rsi),
                float(event.close_location),
            )
            ready = all(np.isfinite(value) for value in values)
            rejection = ready and (
                float(event.high) >= float(event.prior_high)
                and float(event.close) < float(event.prior_high)
                and float(event.close) < float(event.open)
                and float(event.rsi) >= params.min_rsi
                and float(event.close_location) <= params.rejection_close_location_max
            )
            if rejection:
                diagnostics["eligible_rejections"] += 1
                pending = {
                    "signal_time": close_time.isoformat(),
                    "expected_entry_time": pd.Timestamp(event.Index) + pd.Timedelta(hours=4),
                    "atr": float(event.atr),
                    "resistance": float(event.prior_high),
                    "regime": _regime(float(event.regime_return), params.regime_threshold),
                }

    if position is not None and len(frame_1h):
        last = frame_1h.iloc[-1]
        finish(float(last["close"]), _close_time(frame_1h.index[-1], last), "end_of_data")

    summary = summarize_short(
        trades, frame_1h=frame_1h, funding=funding,
        evaluation_start=start, evaluation_end=end, params=params,
        data_quality=data_quality, funding_quality=funding_quality, detailed=detailed,
    )
    summary["diagnostics"] = diagnostics
    return trades, summary


def _strategy_equity(trades: list[ShortTrade], risk_fraction: float = 0.01) -> dict:
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


def summarize_short(
    trades: list[ShortTrade], *, frame_1h: pd.DataFrame, funding: pd.DataFrame,
    evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp,
    params: ShortParameters, data_quality: dict, funding_quality: dict,
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
        "average_trading_cost_r": float(np.mean([t.trading_cost_r for t in trades])) if trades else 0.0,
        "median_risk_pct": float(np.median([t.risk_pct for t in trades])) if trades else 0.0,
        "median_holding_hours": float(np.median([t.holding_hours for t in trades])) if trades else 0.0,
        "total_funding_events": int(sum(t.funding_events for t in trades)),
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "parameters": asdict(params),
        "execution_model": "completed_4h_rejection_next_1h_open_hourly_stop_and_target",
        "intrabar_rule": "gap_then_stop_then_target",
        "funding_model": "archived_rate_times_1h_open_notional_for_short",
    }
    if not detailed:
        return result
    evaluation_hours = float((evaluation_end - evaluation_start) / pd.Timedelta(hours=1))
    positive_gross = sum(max(t.gross_r, 0.0) for t in trades)
    result.update({
        "exposure_pct": float(100 * sum(t.holding_hours for t in trades) / evaluation_hours),
        "largest_winner_share_of_gross_profit": (
            float(max((t.gross_r for t in trades), default=0.0) / positive_gross)
            if positive_gross else None
        ),
        "by_year": time_breakdown(trades, "year"),
        "by_quarter": time_breakdown(trades, "quarter"),
        "by_regime": trade_breakdown(trades, "regime"),
        "by_result": trade_breakdown(trades, "result"),
        "bootstrap": moving_block_bootstrap(trades),
        "walk_forward": fixed_walk_forward(
            trades, evaluation_start, evaluation_end, train_days=365, test_days=180
        ),
        "strategy_equity_1pct_risk": _strategy_equity(trades),
        "data_quality": data_quality,
        "funding_quality": funding_quality,
        "top_winners": [
            asdict(trade)
            for trade in sorted(trades, key=lambda item: item.r_multiple, reverse=True)[:10]
        ],
    })
    return result


def run_short_sensitivity(
    frame_1h: pd.DataFrame, funding: pd.DataFrame, *,
    evaluation_start: str | pd.Timestamp, evaluation_end: str | pd.Timestamp,
    base: ShortParameters,
    lookback_values: tuple[int, ...] = (15, 20, 30),
    stop_values: tuple[float, ...] = (1.5, 2.0, 2.5),
    target_values: tuple[float, ...] = (1.5, 2.0, 2.5),
    cost_values: tuple[int, ...] = (15, 30, 45),
) -> dict:
    scenarios = []
    for lookback, stop_atr, target_r, cost_bps in product(
        lookback_values, stop_values, target_values, cost_values
    ):
        params = replace(
            base, resistance_lookback_bars=lookback,
            initial_stop_atr=stop_atr, target_r=target_r,
            round_trip_cost_bps=cost_bps,
        )
        _, metrics = run_short_backtest(
            frame_1h, funding, evaluation_start=evaluation_start,
            evaluation_end=evaluation_end, params=params, detailed=False,
        )
        scenarios.append(metrics)
    primary = next(item for item in scenarios if item["parameters"] == asdict(base))
    by_cost = {}
    for cost in cost_values:
        subset = [item for item in scenarios if item["parameters"]["round_trip_cost_bps"] == cost]
        totals = [item["total_r"] for item in subset]
        by_cost[str(cost)] = {
            "scenarios": len(subset),
            "positive": sum(value > 0 for value in totals),
            "median_total_r": float(np.median(totals)),
            "min_total_r": float(min(totals)),
            "max_total_r": float(max(totals)),
        }
    positive = sum(item["total_r"] > 0 for item in scenarios)
    return {
        "method": "frozen_full_factorial_no_selection",
        "scenario_count": len(scenarios),
        "positive_scenarios": positive,
        "positive_share": positive / len(scenarios),
        "primary_parameters": asdict(base),
        "primary_replay": primary,
        "by_cost_bps": by_cost,
        "scenarios": scenarios,
    }


def trades_as_dicts(trades: list[ShortTrade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
