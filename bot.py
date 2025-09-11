import os
import time
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np

from config import (
    TD_API_KEY, TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN,
    SYMBOLS, POLL_SECONDS, ATR_PERIOD, RSI_PERIOD, STO_K, STO_D, SIGNAL_EXPIRATION_BARS,
    SUMMARY_ENABLED, SUMMARY_UTC_HOUR, SUMMARY_MIN_SIGNALS, LOG_PATH, NEWS_WINDOW_MINS
)
from data import fetch_series
from strategy import MTFStrategy
from telegram_client import send_message
from utils import now_utc_iso, append_csv_row, read_csv_df, load_json, save_json
from sr import daily_pivots_from_h1, swing_points, nearest_levels
from news import relevant_red_news

SR_HEADROOM = {
    'xau/usd': 0.80,
    'usd/jpy': 0.60,
    'default': 0.70,
}

SESSION_BLOCKS = [
    (0, 6, 'Asia'),
    (7, 12, 'London'),
    (13, 20, 'New York'),
    (21, 23, 'After-hours')
]

OPEN_POS_PATH = "open_positions.json"

def session_name(dt_utc):
    h = dt_utc.hour
    for start, end, name in SESSION_BLOCKS:
        if start <= h <= end:
            return name
    return 'Unknown'

def atr_based_levels(symbol: str, direction: str, price: float, atr_val: float):
    s = symbol.lower()
    if 'xau/usd' in s:
        sl_mult, tp_mult = 1.0, 1.6
    elif 'usd/jpy' in s:
        sl_mult, tp_mult = 0.8, 1.4
    else:
        sl_mult, tp_mult = 1.0, 1.5
    sl_dist = sl_mult * atr_val
    tp_dist = tp_mult * atr_val
    if direction == 'BUY':
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    return sl, tp, sl_dist, tp_dist

def _log_signal(symbol, direction, meta, price_close, session, confidence, sl, tp):
    row = {
        "time_utc": now_utc_iso(),
        "symbol": symbol,
        "direction": direction,
        "close": round(float(price_close), 6),
        "confidence": round(float(confidence), 2),
        "atr_pct": round(float(meta.get("atr_pct", float('nan'))), 6) if meta.get("atr_pct") is not None else None,
        "rsi": round(float(meta.get("rsi", float('nan'))), 3) if meta.get("rsi") is not None else None,
        "rvol": round(float(meta.get("rvol", float('nan'))), 3) if meta.get("rvol") is not None else None,
        "stoch_k": round(float(meta.get("stoch_k", float('nan'))), 3) if meta.get("stoch_k") is not None else None,
        "stoch_d": round(float(meta.get("stoch_d", float('nan'))), 3) if meta.get("stoch_d") is not None else None,
        "session": session,
        "sl": round(float(sl), 6),
        "tp": round(float(tp), 6),
        "result": "",  # PENDING/WIN/LOSS
        "exit_time": "",
        "rr": "",
    }
    append_csv_row(LOG_PATH, row, field_order=list(row.keys()))

def _check_tp_sl_hits(open_positions: dict, df_m5: pd.DataFrame, symbol: str):
    # Check the latest bar's high/low for TP/SL hits
    if symbol not in open_positions:
        return
    if not open_positions[symbol]:
        return
    latest = df_m5.iloc[-1]
    high = float(latest["high"])
    low = float(latest["low"])
    closed = []
    for pos in list(open_positions[symbol]):
        direction = pos["direction"]
        entry = float(pos["entry"])
        tp = float(pos["tp"])
        sl = float(pos["sl"])
        # For BUY: TP if high >= tp; SL if low <= sl
        hit = None
        if direction == "BUY":
            if high >= tp:
                hit = ("WIN", tp)
            elif low <= sl:
                hit = ("LOSS", sl)
        else:
            if low <= tp:
                hit = ("WIN", tp)
            elif high >= sl:
                hit = ("LOSS", sl)
        if hit:
            outcome, exit_price = hit
            pos["result"] = outcome
            pos["exit_time"] = now_utc_iso()
            rr = abs(exit_price - entry) / max(1e-9, abs(entry - sl))
            pos["rr"] = rr
            closed.append(pos)
            open_positions[symbol].remove(pos)
    return closed

def _maybe_send_daily_summary():
    if not SUMMARY_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import datetime as _dt
    now = _dt.datetime.utcnow()
    if now.hour < SUMMARY_UTC_HOUR:
        return
    df = read_csv_df(LOG_PATH)
    if df.empty:
        return
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df = df.dropna(subset=["time_utc"])
    today_df = df.loc[df["time_utc"].dt.date == now.date()]
    if today_df.empty or len(today_df) < SUMMARY_MIN_SIGNALS:
        return
    if (today_df["direction"] == "SUMMARY").any():
        return
    # Performance: match rows with result in same CSV (after pos closes we append a row with outcome)
    perf = today_df.dropna(subset=["result"])
    wins = int((perf["result"] == "WIN").sum())
    losses = int((perf["result"] == "LOSS").sum())
    total = len(today_df[today_df["direction"].isin(["BUY","SELL"])])
    avg_conf = float(today_df["confidence"].mean()) if "confidence" in today_df else float("nan")
    avg_rr = float(perf["rr"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().mean()) if "rr" in perf else float("nan")
    by_symbol = today_df.groupby("symbol").size().sort_values(ascending=False)
    by_dir = today_df.groupby("direction").size()
    top_syms = ", ".join([f"{k}({v})" for k, v in by_symbol.head(5).items()])
    buys = int(by_dir.get("BUY", 0)); sells = int(by_dir.get("SELL", 0))
    winrate = (wins / max(1, wins + losses)) * 100.0
    msg_lines = []
    msg_lines.append("<b>Daily Signal Summary</b> 📊")
    msg_lines.append(f"Date (UTC): {now.date().isoformat()}  |  From: {SUMMARY_UTC_HOUR}:00 → 23:59")
    msg_lines.append(f"Total signals: {total}")
    msg_lines.append(f"Average confidence: {avg_conf:.1f}%")
    msg_lines.append(f"BUY/SELL: {buys}/{sells}")
    msg_lines.append(f"Performance (ATR TP/SL): Wins {wins} | Losses {losses} | Win-rate {winrate:.0f}%")
    if not np.isnan(avg_rr):
        msg_lines.append(f"Avg R:R ~ {avg_rr:.2f}")
    if len(by_symbol) > 0:
        msg_lines.append(f"Top symbols: {top_syms}")
    send_message("\n".join(msg_lines))
    append_csv_row(LOG_PATH, {
        "time_utc": now_utc_iso(),
        "symbol": "SUMMARY",
        "direction": "SUMMARY",
        "close": "",
        "confidence": "",
        "atr_pct": "",
        "rsi": "",
        "rvol": "",
        "stoch_k": "",
        "stoch_d": "",
        "session": "",
        "sl": "",
        "tp": "",
        "result": "",
        "exit_time": "",
        "rr": "",
    })

def assemble_message(symbol: str, direction: str, meta: Dict, tf_refs: Dict, confidence: float, atr_val: float, sr_info: Dict | None = None, news_flags=None):
    h1, m15, m5 = tf_refs["H1"].iloc[-1], tf_refs["M15"].iloc[-1], tf_refs["M5"].iloc[-1]
    sess = session_name(tf_refs["M5"]["datetime"].iloc[-1].to_pydatetime())
    sl, tp, sl_dist, tp_dist = atr_based_levels(symbol, direction, float(m5['close']), float(atr_val))
    txt = []
    txt.append(f"<b>{symbol}</b> — <b>{direction}</b> signal 🚨")
    txt.append(f"Time: {now_utc_iso()}")
    txt.append("")
    txt.append(f"<b>Session</b>: {sess}")
    txt.append(f"<b>Confidence</b>: {confidence:.0f}%")
    txt.append("")
    txt.append("<b>Filters</b>")
    txt.append(f"• ATR%: {meta['atr_pct']:.4f}")
    txt.append(f"• RSI: {meta['rsi']:.1f}  |  Stoch K/D: {meta['stoch_k']:.1f}/{meta['stoch_d']:.1f}")
    txt.append(f"• RVOL: {meta['rvol']:.2f}")
    txt.append(f"• Candle: {'Bullish' if meta['bull_candle'] else ('Bearish' if meta['bear_candle'] else 'None')}")
    txt.append(f"• BB: mid {meta['bb_middle']:.3f} | up {meta['bb_upper']:.3f} | lo {meta['bb_lower']:.3f}")
    txt.append(f"• Fresh-window: {meta['fresh_window_bars']} bars")
    txt.append("")
    txt.append("<b>Last M5 Bar</b>")
    txt.append(f"O:{m5['open']:.3f} H:{m5['high']:.3f} L:{m5['low']:.3f} C:{m5['close']:.3f} Vol:{m5['volume']:.0f}")
    txt.append("")
    txt.append("<b>TP/SL (ATR-based)</b>")
    txt.append(f"• SL: {sl:.3f} (dist ~ {sl_dist:.3f})  |  TP: {tp:.3f} (dist ~ {tp_dist:.3f})")
    if sr_info:
        txt.append("")
        txt.append("<b>Nearest S/R</b>")
        if sr_info.get('nearest_res'):
            name, lvl, dist = sr_info['nearest_res']
            txt.append(f"• Resistance: {name} @ {lvl:.3f}  (Δ ~ {dist:.3f})")
        if sr_info.get('nearest_sup'):
            name, lvl, dist = sr_info['nearest_sup']
            txt.append(f"• Support: {name} @ {lvl:.3f}  (Δ ~ {dist:.3f})")
        if sr_info.get('pivot_day'):
            txt.append(f"• Pivots from: {sr_info['pivot_day']}")
    if news_flags:
        for nf in news_flags:
            mins = nf.get('mins_from_now')
            txt.append("")
            txt.append(f"⚠️ High-impact news: {nf.get('title')} ({nf.get('currency')}) in {mins:+d}m")
    txt.append("")
    txt.append("<b>Context</b>")
    txt.append(f"H1 Close: {h1['close']:.3f} | M15 Close: {m15['close']:.3f}")
    txt.append("— Trend: H1 & M15 aligned with signal direction")
    return "\n".join(txt), sl, tp

def main():
    if not TD_API_KEY:
        print("[ERROR] TD_API_KEY missing.")
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured; signals will print to console.")

    strat = MTFStrategy(rsi_period=RSI_PERIOD, sto_k=STO_K, sto_d=STO_D, atr_period=ATR_PERIOD, expiration_bars=SIGNAL_EXPIRATION_BARS)
    print(f"[INFO] Starting PP MTF Signal Bot. Symbols={SYMBOLS} Poll={POLL_SECONDS}s")

    last_bar_time: Dict[str, str] = {}
    open_positions = load_json(OPEN_POS_PATH, default={})

    while True:
        try:
            for symbol in SYMBOLS:
                df_m5 = fetch_series(symbol, "5min", TD_API_KEY, 300)
                df_m15 = fetch_series(symbol, "15min", TD_API_KEY, 300)
                df_h1 = fetch_series(symbol, "1h", TD_API_KEY, 300)

                if df_m5 is None or df_m15 is None or df_h1 is None:
                    print(f"[WARN] Data fetch failed for {symbol}")
                    continue

                # Check TP/SL hits for open positions on this symbol
                closed = _check_tp_sl_hits(open_positions, df_m5, symbol)
                if closed:
                    for pos in closed:
                        # Append a result row to CSV
                        append_csv_row(LOG_PATH, {
                            "time_utc": pos["exit_time"],
                            "symbol": symbol,
                            "direction": pos["result"],
                            "close": pos["entry"],
                            "confidence": "",
                            "atr_pct": "",
                            "rsi": "",
                            "rvol": "",
                            "stoch_k": "",
                            "stoch_d": "",
                            "session": pos.get("session",""),
                            "sl": pos["sl"],
                            "tp": pos["tp"],
                            "result": pos["result"],
                            "exit_time": pos["exit_time"],
                            "rr": pos["rr"],
                        })

                # Only act when a M5 bar closes
                last_dt = df_m5["datetime"].iloc[-1].isoformat()
                if last_bar_time.get(symbol) == last_dt:
                    continue
                last_bar_time[symbol] = last_dt

                bias_h1_up, bias_m15_up, bias_h1_dn, bias_m15_dn = strat.trend_bias(df_h1, df_m15)

                long_ok, short_ok, meta = strat.signal_m5(df_m5, symbol=symbol)

                # ATR absolute value
                try:
                    hl = (df_m5['high'] - df_m5['low']).abs()
                    hc = (df_m5['high'] - df_m5['close'].shift(1)).abs()
                    lc = (df_m5['low'] - df_m5['close'].shift(1)).abs()
                    tr = np.maximum.reduce([hl, hc, lc])
                    atr_val = float(tr.rolling(14).mean().iloc[-1])
                except Exception:
                    atr_val = float('nan')

                fire_long = long_ok and bias_h1_up and bias_m15_up
                fire_short = short_ok and bias_h1_dn and bias_m15_dn

                # S/R filtering
                piv = daily_pivots_from_h1(df_h1)
                sh, sl_sw = swing_points(df_m15, lookback=60, left=2, right=2)
                price = float(df_m5['close'].iloc[-1])
                nearest_res, nearest_sup = nearest_levels(price, piv, sh, sl_sw)
                sym_key = symbol.lower()
                head_mult = SR_HEADROOM.get(sym_key, SR_HEADROOM['default'])
                ok_headroom_long = True
                ok_headroom_short = True
                if nearest_res is not None and not (nearest_res[2] >= head_mult * (atr_val if atr_val == atr_val else 0.0)):
                    ok_headroom_long = False
                if nearest_sup is not None and not (nearest_sup[2] >= head_mult * (atr_val if atr_val == atr_val else 0.0)):
                    ok_headroom_short = False
                fire_long = fire_long and ok_headroom_long
                fire_short = fire_short and ok_headroom_short

                if fire_long or fire_short:
                    direction = "BUY" if fire_long else "SELL"
                    conf = float(meta.get('confidence', 50.0))
                    sr_info = {'nearest_res': nearest_res, 'nearest_sup': nearest_sup, 'pivot_day': piv.get('src_day') if piv else None}
                    news_flags = relevant_red_news(symbol, NEWS_WINDOW_MINS)
                    msg, sl_val, tp_val = assemble_message(symbol, direction, meta, {"H1": df_h1, "M15": df_m15, "M5": df_m5}, conf, atr_val, sr_info, news_flags)
                    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        send_message(msg)
                    print(msg)
                    # Log and store open position for TP/SL tracking
                    try:
                        sess = session_name(df_m5["datetime"].iloc[-1].to_pydatetime())
                        _log_signal(symbol, direction, meta, float(df_m5['close'].iloc[-1]), sess, conf, sl_val, tp_val)
                        if symbol not in open_positions:
                            open_positions[symbol] = []
                        open_positions[symbol].append({
                            "direction": direction,
                            "entry": float(df_m5['close'].iloc[-1]),
                            "sl": float(sl_val),
                            "tp": float(tp_val),
                            "session": sess,
                            "result": "",
                            "exit_time": "",
                            "rr": ""
                        })
                        save_json(OPEN_POS_PATH, open_positions)
                    except Exception as _e:
                        print(f"[WARN] log/store error: {_e}")
                else:
                    print(f"[{symbol}] {last_dt} — no aligned signal.")

            _maybe_send_daily_summary()
            time.sleep(POLL_SECONDS)
        except Exception as e:
            print(f"[ERROR] Loop exception: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
