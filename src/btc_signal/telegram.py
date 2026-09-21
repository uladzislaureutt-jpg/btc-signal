from __future__ import annotations

import html
import os

import requests

from .strategy import Decision


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}".replace(",", " ")


def format_decision(decision: Decision) -> str:
    icons = {"watch": "👀", "signal": "🟢", "strong_signal": "🔥"}
    titles = {"watch": "WATCH", "signal": "SIGNAL", "strong_signal": "STRONG SIGNAL"}
    side = (decision.side or "—").upper()
    zone = decision.level or {}
    volume = decision.factors.get("volume", {}).get("value", "—")
    rsi_value = decision.factors.get("momentum", {}).get("value", "—")
    return (
        f"{icons.get(decision.state, 'ℹ️')} <b>BTC · {side} {titles.get(decision.state, decision.state.upper())} · {decision.score}/100</b>\n\n"
        f"Цена: <b>{_money(decision.price)}</b>\n"
        f"4H-зона: {_money(zone.get('low'))}–{_money(zone.get('high'))} · качество {zone.get('quality', '—')}/10\n"
        f"1D: {html.escape(decision.regime)}\n"
        f"Volume ratio: {volume} · RSI: {rsi_value}\n\n"
        f"Entry: {_money(decision.entry_low)}–{_money(decision.entry_high)}\n"
        f"Stop: {_money(decision.stop)}\n"
        f"Target 1: {_money(decision.target_1)}\n"
        f"Target 2: {_money(decision.target_2)}\n"
        f"R/R: {decision.rr:.2f}\n\n"
        f"<i>Shadow signal · не является финансовой рекомендацией</i>"
    )


def send(decision: Decision, timeout: int = 20) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": format_decision(decision),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()

