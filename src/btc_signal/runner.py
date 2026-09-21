from __future__ import annotations

from dataclasses import dataclass

from .config import load_config
from .data import build_market_data
from .storage import build_store
from .strategy import Decision, evaluate
from .telegram import send


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
    frames = market.bundle(
        symbol=symbol,
        intervals={
            "1h": int(history["live_1h_bars"]),
            "4h": int(history["live_4h_bars"]),
            "1d": int(history["live_1d_bars"]),
        },
    )
    decision = evaluate(frames["1d"], frames["4h"], frames["1h"], config, market.last_provider)
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
