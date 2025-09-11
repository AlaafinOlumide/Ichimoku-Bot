# PP MTF Signal Bot (v6) — News-aware + Performance Summary

### What it does
- **M5 entries**, **M15 trend**, **H1 setup**
- Filters: RSI+Stoch cross, ATR%, Relative Volume, Candlesticks (engulfing/pin bar), Bollinger, Fresh-entry window
- **Support/Resistance**: Daily pivots (from H1) + M15 swing highs/lows with ATR headroom
- **Confidence score (0–100)**, **Session tagging (Asia/London/NY)**, **ATR-based TP/SL suggestions**
- **ForexFactory news flag** within a window of high-impact events
- **CSV logging + TP/SL result tracking**
- **Daily Telegram summary** with performance (wins/losses, win-rate, avg R:R)

## Setup

### 1) Telegram
- Create a bot via **@BotFather** → get `TELEGRAM_BOT_TOKEN`
- Message your bot once (or add to a group)
- Get `TELEGRAM_CHAT_ID` via **@userinfobot** or `getUpdates`

### 2) Twelve Data
- Create account → copy your `TD_API_KEY`

### 3) Configure env
Copy `.env.example` → `.env` and fill keys. Example:
```
TD_API_KEY=YOUR_TWELVE_DATA_API_KEY
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
SYMBOLS=XAU/USD,USD/JPY
POLL_SECONDS=60
ATR_PERIOD=14
RSI_PERIOD=14
STO_K=14
STO_D=3
SIGNAL_EXPIRATION_BARS=3
SUMMARY_ENABLED=true
SUMMARY_UTC_HOUR=22
SUMMARY_MIN_SIGNALS=0
LOG_PATH=signals_log.csv
NEWS_WINDOW_MINS=30
```

### 4) Run
```
pip install -r requirements.txt
python bot.py
```

### 5) Deploy on Render
- Push to GitHub, create a **Worker** service, add env vars, deploy.

## News Flag (ForexFactory)
- Scrapes FF calendar for **High-impact (red)** events.
- If a signal occurs within ± `NEWS_WINDOW_MINS` of a relevant currency event, the alert includes a warning like:
  `⚠️ High-impact news: NFP (USD) in +15m`

## Performance Tracking
- Each signal is stored as an open position with ATR-based TP/SL.
- At each new M5 candle, the bot checks for **TP/SL hits** using high/low and logs a close row with `WIN` or `LOSS` and R:R.
- The **daily summary** includes total signals, average confidence, **wins/losses, win-rate, avg R:R**.

## ATR-based TP/SL (by symbol)
- XAU/USD: **SL 1×ATR**, **TP 1.6×ATR**
- USD/JPY: **SL 0.8×ATR**, **TP 1.4×ATR**
- Others: **SL 1×ATR**, **TP 1.5×ATR**

> Educational use only. Tune thresholds and test before relying on live signals.
