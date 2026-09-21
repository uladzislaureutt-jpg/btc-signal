create table if not exists public.btc_signals (
  id bigint generated always as identity primary key,
  dedup_key text not null unique,
  candle_time timestamptz not null,
  state text not null check (state in ('neutral', 'watch', 'signal', 'strong_signal', 'cancelled')),
  side text check (side in ('long', 'short')),
  score smallint not null check (score between 0 and 100),
  price numeric(20, 8) not null check (price > 0),
  entry_low numeric(20, 8),
  entry_high numeric(20, 8),
  stop numeric(20, 8),
  target_1 numeric(20, 8),
  target_2 numeric(20, 8),
  rr numeric(10, 4),
  provider text,
  strategy_version text not null,
  payload jsonb not null,
  notified boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists btc_signals_state_candle_time_idx
  on public.btc_signals (state, candle_time desc);

create index if not exists btc_signals_open_notifications_idx
  on public.btc_signals (candle_time desc)
  where notified = true;

alter table public.btc_signals enable row level security;

revoke all on table public.btc_signals from anon, authenticated;
revoke all on sequence public.btc_signals_id_seq from anon, authenticated;
grant select, insert, update on table public.btc_signals to service_role;
grant usage, select on sequence public.btc_signals_id_seq to service_role;

