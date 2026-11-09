from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import load_config
from .data import build_market_data
from .storage import build_store
from .strategy import Decision
from .telegram import send
from .trend_live import episode_reset_status, evaluate_trend_signal


@dataclass
class RunResult:
    decision: Decision
    duplicate: bool
    notified: bool


def run_once(dry_run: bool = False, notify_watch: bool = True) -> RunResult:
    config = load_config()
    market = build_market_data(config)
    symbol = config["market"]["symbol"]
    history = config["history"]
    frames, current_price = market.snapshot(
        symbol=symbol,
        intervals={
            "4h": int(history["live_4h_bars"]),
        },
    )
    decision = evaluate_trend_signal(
        frames["4h"], config, market.last_provider, current_price
    )
    if dry_run:
        return RunResult(decision=decision, duplicate=False, notified=False)
    store = build_store()
    prior_record = store.most_recent_candle_before(decision.timestamp)
    if prior_record:
        current_closed = datetime.fromisoformat(decision.timestamp).astimezone(timezone.utc)
        previous_closed = datetime.fromisoformat(prior_record).astimezone(timezone.utc)
        # All candle times use the same millisecond-precise 4h close convention.
        elapsed_hours = (current_closed - previous_closed).total_seconds() / 3600.0
        missed_bars = max(0, round(elapsed_hours / 4.0) - 1)
        if missed_bars:
            decision.factors["coverage"] = {
                "previous_candle_time": prior_record,
                "current_candle_time": decision.timestamp,
                "missed_4h_bars": missed_bars,
            }
            print(
                f"::warning::BTC 4h monitor missed {missed_bars} candle(s) "
                f"between {prior_record} and {decision.timestamp}; "
                "historical signals are not sent retroactively.",
                flush=True,
            )
    duplicate = store.seen(decision.dedup_key)
    if not duplicate and decision.state in ("signal", "strong_signal") and store.cooldown_active(
        decision.timestamp, int(config["signal"]["cooldown_hours"])
    ):
        decision.vetoes.append("cooldown_active")
        decision.state = "neutral"
        duplicate = store.seen(decision.dedup_key)
    structure_cfg = config.get("structure", {})
    episode_hours = int(structure_cfg.get("episode_hours", 24))
    reset_hours = int(structure_cfg.get("episode_reset_lookback_hours", 720))
    if not duplicate and decision.state in ("signal", "strong_signal") and decision.side:
        previous = store.most_recent_notified_signal(decision.timestamp, reset_hours)
        if previous and previous["side"] == decision.side:
            reset = episode_reset_status(
                frames["4h"],
                previous["candle_time"],
                decision.timestamp,
                decision.side,
                float(decision.factors["volatility"]["atr"]),
                minimum_pullback_atr=float(structure_cfg.get("episode_reset_pullback_atr", 1.25)),
                minimum_counter_closes=int(structure_cfg.get("episode_reset_counter_closes", 2)),
            )
            decision.factors["episode_reset"] = reset
            if not reset["confirmed"]:
                decision.vetoes.append("same_direction_without_structural_reset")
                decision.state = "neutral"
                duplicate = store.seen(decision.dedup_key)
        elif previous and previous["side"] != decision.side:
            last_time = datetime.fromisoformat(previous["candle_time"]).astimezone(timezone.utc)
            current_time = datetime.fromisoformat(decision.timestamp).astimezone(timezone.utc)
            hours_since_opposite = (current_time - last_time).total_seconds() / 3600.0
            if 0 <= hours_since_opposite <= episode_hours:
                signal_type = decision.factors.get("signal_type", {}).get("value")
                if signal_type in ("fast_trend_long", "fast_trend_short"):
                    reversal_confirmed = bool(
                        decision.factors.get("fast_trend", {}).get("reversal_confirmed", False)
                    )
                else:
                    factor_name = "early_long" if decision.side == "long" else "early_short"
                    reversal_confirmed = bool(
                        decision.factors.get(factor_name, {}).get("reversal_confirmed", False)
                    )
                if signal_type != "trend_breakout" and not reversal_confirmed:
                    decision.vetoes.append("directional_reset_required")
                    decision.state = "neutral"
                    duplicate = store.seen(decision.dedup_key)
    should_notify = decision.state in ("signal", "strong_signal") or (
        notify_watch and decision.state == "watch"
    )
    notified = False
    if not duplicate and should_notify:
        send(decision)
        notified = True
    if not duplicate:
        store.save(decision, notified)
    return RunResult(decision=decision, duplicate=duplicate, notified=notified)
