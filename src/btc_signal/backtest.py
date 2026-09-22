from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .strategy import Decision, evaluate


BACKTEST_WARMUP_HOURS = 24 * 210


@dataclass
class Trade:
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    score: int
    regime: str
    rr: float
    risk_pct: float
    gross_r: float
    cost_r: float
    level_quality: int
    price_action_ok: bool
    volume_ok: bool
    momentum_ok: bool
    structure_ok: bool
    result: str
    r_multiple: float


def required_history_hours(evaluation_days: int, max_holding_hours: int) -> int:
    if evaluation_days <= 0:
        raise ValueError("evaluation_days must be positive")
    if max_holding_hours < 0:
        raise ValueError("max_holding_hours must be non-negative")
    return BACKTEST_WARMUP_HOURS + evaluation_days * 24 + max_holding_hours


def resample(frame_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregated = frame_1h.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "close"])
    if frame_1h.empty:
        return aggregated
    cutoff = frame_1h["close_time"].iloc[-1] + pd.Timedelta(milliseconds=1)
    duration = pd.Timedelta(rule)
    return aggregated.loc[aggregated.index + duration <= cutoff]


def _simulate(decision: Decision, future: pd.DataFrame, config: dict) -> Trade | None:
    if decision.side is None or decision.stop is None or decision.target_1 is None:
        return None
    entry = decision.price
    risk = entry - decision.stop if decision.side == "long" else decision.stop - entry
    target_valid = (
        decision.target_1 > entry
        if decision.side == "long"
        else decision.target_1 < entry
    )
    if risk <= 0 or not target_valid:
        return None
    max_hours = int(config["risk"]["max_holding_hours"])
    exit_price = float(future.iloc[min(len(future), max_hours) - 1]["close"])
    exit_time = future.index[min(len(future), max_hours) - 1]
    result = "timeout"
    for timestamp, candle in future.head(max_hours).iterrows():
        if decision.side == "long":
            stop_hit = candle["low"] <= decision.stop
            target_hit = candle["high"] >= decision.target_1
        else:
            stop_hit = candle["high"] >= decision.stop
            target_hit = candle["low"] <= decision.target_1
        # Conservative rule when both are touched inside one hourly candle.
        if stop_hit:
            exit_price, exit_time, result = decision.stop, timestamp, "stop"
            break
        if target_hit:
            exit_price, exit_time, result = decision.target_1, timestamp, "target"
            break
    raw_r = (exit_price - entry) / risk if decision.side == "long" else (entry - exit_price) / risk
    costs = (
        float(config["risk"]["fee_bps_round_trip"])
        + float(config["risk"]["slippage_bps_round_trip"])
    ) / 10_000
    cost_r = costs * entry / risk
    net_r = raw_r - cost_r
    factors = decision.factors
    return Trade(
        side=decision.side,
        entry_time=decision.timestamp,
        exit_time=exit_time.isoformat(),
        entry=entry,
        exit=float(exit_price),
        stop=float(decision.stop),
        target=float(decision.target_1),
        score=decision.score,
        regime=decision.regime,
        rr=float(decision.rr or 0.0),
        risk_pct=float(risk / entry) if entry else 0.0,
        gross_r=float(raw_r),
        cost_r=float(cost_r),
        level_quality=int(factors.get("level_quality", {}).get("value", 0)),
        price_action_ok=bool(factors.get("price_action_1h", {}).get("passed", False)),
        volume_ok=bool(factors.get("volume", {}).get("passed", False)),
        momentum_ok=bool(factors.get("momentum", {}).get("passed", False)),
        structure_ok=bool(factors.get("structure_4h", {}).get("passed", False)),
        result=result,
        r_multiple=float(net_r),
    )


def run_backtest(frame_1h: pd.DataFrame, config: dict) -> tuple[list[Trade], dict]:
    trades: list[Trade] = []
    warmup_hours = BACKTEST_WARMUP_HOURS
    blocked_until: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=int(config["signal"]["cooldown_hours"]))
    for i in range(warmup_hours, len(frame_1h) - int(config["risk"]["max_holding_hours"])):
        history_1h = frame_1h.iloc[: i + 1]
        timestamp = history_1h.index[-1]
        if blocked_until is not None and timestamp < blocked_until:
            continue
        history_4h = resample(history_1h, "4h")
        history_1d = resample(history_1h, "1D")
        try:
            decision = evaluate(history_1d, history_4h, history_1h, config, "historical")
        except ValueError:
            continue
        if decision.state not in ("signal", "strong_signal"):
            continue
        future = frame_1h.iloc[i + 1 :]
        trade = _simulate(decision, future, config)
        if trade:
            trades.append(trade)
            exit_time = pd.Timestamp(trade.exit_time)
            blocked_until = max(timestamp + cooldown, exit_time)
    values = np.array([trade.r_multiple for trade in trades], dtype=float)
    wins = values[values > 0]
    losses = values[values <= 0]
    equity = values.cumsum() if len(values) else np.array([])
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([])
    drawdowns = equity - peaks if len(equity) else np.array([])
    summary = {
        "trades": len(trades),
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "expectancy_r": float(values.mean()) if len(values) else 0.0,
        "total_r": float(values.sum()) if len(values) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else None,
        "max_drawdown_r": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "gross_total_r": float(sum(trade.gross_r for trade in trades)),
        "total_cost_r": float(sum(trade.cost_r for trade in trades)),
        "average_cost_r": float(np.mean([trade.cost_r for trade in trades])) if trades else 0.0,
        "average_risk_pct": float(np.mean([trade.risk_pct for trade in trades])) if trades else 0.0,
    }
    return trades, summary


def trades_as_dicts(trades: list[Trade]) -> list[dict]:
    return [asdict(trade) for trade in trades]
