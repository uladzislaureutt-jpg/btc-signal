# BTC Signal Core 0.4.2

Сервис формирует редкие торговые сигналы по `BTCUSDT perpetual`:

- направление: `LONG` или `SHORT`;
- цена входа по текущему рынку после подтверждения;
- первоначальный стоп-лосс;
- два ценовых ориентира TP1/TP2;
- момент включения и дистанция ATR-трейлинга.

Сервис не рассчитывает долю капитала и не исполняет ордера автоматически.

## Рабочая логика

Сигнал строится только по завершённой 4H-свече:

- `LONG`: закрытие выше максимума предыдущих 20 свечей и выше EMA200;
- `SHORT`: закрытие ниже минимума предыдущих 20 свечей и ниже EMA200;
- `EARLY LONG`: пробой максимума закрытий последних 6 свечей минимум на 0,25 ATR **и одновременно** рост минимум на 1 ATR за 3 свечи, минимум 2 повышающихся закрытия, 2 закрытия подряд выше EMA200 и расстояние от EMA200 не менее 0,25 ATR;
- `EARLY SHORT`: зеркально — пробой минимума закрытий последних 6 свечей минимум на 0,25 ATR **и одновременно** падение минимум на 1 ATR за 3 свечи, минимум 2 понижающихся закрытия, 2 закрытия подряд ниже EMA200 и расстояние от EMA200 не менее 0,25 ATR;
- `FAST TREND`: для сильного импульса не ждёт второго EMA-close — требуется пробой close-level минимум на 0,5 ATR, движение минимум на 1,5 ATR за 3 свечи, 3 последовательных закрытия по направлению и расстояние от EMA200 минимум 0,25 ATR;
- вход: текущая цена perpetual после подтверждённого пробоя;
- стоп: 2 ATR(4H) от входа;
- TP1/TP2: структурные ориентиры по Fibonacci доминирующего 30-дневного 4H swing; слишком близкие уровни пропускаются, TP1 должен давать не менее 1R, TP2 — не менее 1.5R;
- новый вход не выдаётся поздно в коррекции, если после прохождения 50% swing до следующего Fibonacci-уровня остаётся меньше 1,5 ATR;
- если структура не даёт минимум 1R для TP1 и 1.5R для TP2 относительно ATR-стопа, обычный/ EARLY сигнал отклоняется; FAST TREND не блокируется промежуточным Fibonacci-уровнем — структура только корректирует цели;
- повторный сигнал того же направления подавляется в течение 24 часов как один торговый эпизод;
- смена направления через EARLY-сигнал требует directional reset: 2 закрытия по другую сторону EMA200, подтверждённый momentum и расстояние от EMA200 минимум 0,5 ATR; обычный 20-bar breakout может сменить направление без этого ограничения;
- основной выход: trailing stop 3 ATR после движения на 1 ATR в прибыль;
- сигнал отменяется, если риск до стопа меньше чем в 4 раза превышает принятые round-trip издержки;
- устаревший пробой не отправляется.

TP1 и TP2 — структурные ориентиры: они учитывают Fibonacci-уровни, но не заменяют основной ATR trailing stop. Диагностика перед сохранением очищается от `NaN/Infinity`, поэтому отсутствие следующего Fibonacci-уровня не может уронить production-run. `EARLY LONG` и `EARLY SHORT` являются строго подтверждёнными оперативными overlay и не изменяет замороженный исследовательский движок 0.3.0.

Основной источник — Binance USD-M Futures; резервные perpetual-источники — Bybit Linear и OKX Swap. Последний технический fallback — официальный market-data endpoint Binance Spot: он используется только если derivatives API недоступны. Свечи и текущая цена одного расчёта всегда берутся с одной площадки, а фактический источник указан в сигнале.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install --no-deps -e .
pytest -q
btc-signal live --dry-run
```

Без `--dry-run` новый сигнал сохраняется в Supabase/SQLite и отправляется в Telegram. Нейтральные состояния не отправляются.

## GitHub Actions

Workflow **BTC trend signal** запускается:

- автоматически через 7 минут после закрытия каждой 4H-свечи UTC;
- вручную через `Actions → BTC trend signal → Run workflow`.

Для рабочего запуска нужны secrets:

- `TELEGRAM_BOT_TOKEN`;
- `TELEGRAM_CHAT_ID`;
- `SUPABASE_URL`;
- `SUPABASE_SECRET_KEY`.

SQL-схема находится в `supabase/migrations/0001_btc_signal.sql`.

## Формат сообщения

```text
BTCUSDT PERPETUAL · LONG/SHORT
Вход
Стоп-лосс
TP1 (2R)
TP2 (3R)
Уровень пробоя 4H
Цена включения трейлинга
Дистанция трейлинга
Источник и время подтверждения
```

## Исследовательские команды

Предыдущие backtest-модули сохранены для воспроизводимости:

```bash
btc-signal trend-v030-backtest
btc-signal trend-backtest
btc-signal short-backtest
btc-signal backtest --days 240
```

Рабочий `live` больше не использует прежний контртрендовый движок отскока от уровня. Он оставлен только как исторический исследовательский код.

Это программный генератор сигналов, а не персональная финансовая рекомендация.


### Operational entry-stage gates (0.4.2)

EARLY LONG/SHORT are suppressed when the completed 4H close is more than 3 ATR from EMA200, or the live entry is less than 1 ATR from the prior 20-bar / 180-bar high or low in the trade direction. The limits are applied **only to EARLY entries**, leaving frozen 20-bar breakouts and FAST TREND detection unchanged. Near-extreme checks use the latest completed 4H candle and prior highs/lows, never future bars.

Repeated same-side alerts are no longer released simply because 24 hours elapsed. The system examines the last **notified** entry within 30 days and requires a subsequent close-based countertrend pullback of at least 1.25 ATR with two consecutive countertrend closes before another alert in that direction. Opposite-side EARLY/FAST reversal confirmation is retained. The prior 24-hour time-based same-side suppression is replaced, not stacked. This entry-stage protection is an operational hypothesis and **has not been proven profitable by the frozen research backtests**.
