from __future__ import annotations

import os

import requests

from .strategy import Decision


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}".replace(",", " ")


def format_decision(decision: Decision) -> str:
    side = (decision.side or "—").upper()
    breakout = decision.level or {}
    exit_plan = decision.factors.get("exit_plan", {})
    entry = decision.entry_low
    return (
        f"📈 <b>BTCUSDT PERPETUAL · {side}</b>\n\n"
        f"Вход: <b>{_money(entry)}</b>\n"
        f"Стоп-лосс: <b>{_money(decision.stop)}</b>\n"
        f"TP1 (ориентир 2R): {_money(decision.target_1)}\n"
        f"TP2 (ориентир 3R): {_money(decision.target_2)}\n\n"
        f"Пробой 4H: {_money(breakout.get('price'))}\n"
        f"Трейлинг включить от: {_money(exit_plan.get('activation_price'))}\n"
        f"Дистанция трейлинга: {exit_plan.get('trailing_distance_atr', '—')} ATR(4H)\n"
        f"Источник: {decision.provider or '—'}\n"
        f"Свеча закрыта: {decision.timestamp}\n\n"
        f"<i>Основной выход — трейлинг; TP1/TP2 являются ценовыми ориентирами. "
        f"Сигнал не исполняет ордер автоматически.</i>"
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

