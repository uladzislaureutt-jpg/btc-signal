from __future__ import annotations

from dataclasses import dataclass

from .config import load_config
from .data import build_market_data
from .storage import build_store
from .strategy import Decision
from .telegram import send
from .trend_live import evaluate_trend_signal


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
    duplicate = store.seen(decision.dedup_key)
    if not duplicate and decision.state in ("signal", "strong_signal") and store.cooldown_active(
        decision.timestamp, int(config["signal"]["cooldown_hours"])
    ):
        decision.vetoes.append("cooldown_active")
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
