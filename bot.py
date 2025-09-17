import os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

# ================== Config ==================
load_dotenv()
TD_API_KEY = os.getenv("TD_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DEBUG = os.getenv("DEBUG", "0") == "1"

# Fixed symbols (no slashes; works well on Twelve Data)
SYMBOLS = ["XAUUSD", "USDJPY"]

# Loop sleep (seconds)
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

# Cadences (seconds) — API credit control
CADENCE_M5_SEC  = int(os.getenv("CADENCE_M5_SEC",  "360"))   # 6 minutes
CADENCE_M15_SEC = int(os.getenv("CADENCE_M15_SEC", "900"))   # 15 minutes
CADENCE_H1_SEC  = int(os.getenv("CADENCE_H1_SEC",  "3600"))  # 60 minutes

# Filters
ATR_PERIOD = 14
RSI_PERIOD = 14
STO_K = 14
STO_D = 3
SIGNAL_EXPIRATION_BARS = 3  # strict mode freshness for TK cross & triggers

# "Minimum baseline" relaxor: if no signal for X minutes, loosen rules
RELAX_IF_NO_SIGNAL_MIN = int(os.getenv("RELAX_IF_NO_SIGNAL_MIN", "240"))  # 4 hours
RELAX_FRESH_BARS = int(os.getenv("RELAX_FRESH_BARS", "6"))                # relaxed TK cross window
RELAX_ATR_DISCOUNT = float(os.getenv("RELAX_ATR_DISCOUNT", "0.85"))       # 15% easier ATR in relaxed mode

# Per-symbol thresholds (strict)
THRESH = {
    "xauusd": {"atr_min_frac": 0.0012, "rvol_min": 1.20, "bb_overext": 0.25},
    "usdjpy": {"atr_min_frac": 0.0006, "rvol_min": 1.05, "bb_overext": 0.20},
    "default": {"atr_min_frac": 0.0008, "rvol_min": 1.10, "bb_overext": 0.20},
}

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ================== Data Fetch (with logging) ==================
BASE_URL = "https://api.twelvedata.com/time_series"
_api_calls_today = 0

def _fetch_once(symbol: str, interval: str, outputsize: int):
    global _api_calls_today
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TD_API_KEY,
        "outputsize": outputsize,
        "format": "JSON",
        "order": "ASC",
    }
    r = requests.get(BASE_URL, params=params, timeout=20)
    _api_calls_today += 1
    try:
        j = r.json()
    except Exception:
        print(f"[TD] Non-JSON response ({r.status_code}) for {symbol} {interval}: {r.text[:200]}")
        return None, None
    if "values" in j:
        return j["values"], None
    return None, j.get("message") or j.get("code") or j.get("status") or j

def fetch_series(symbol: str, interval: str, outputsize: int = 300):
    values, err = _fetch_once(symbol, interval, outputsize)
    if values is None:
        print(f"[TD] Fetch failed for {symbol} {interval}: {err}")
        return None
    df = pd.DataFrame(values)
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            print(f"[TD] Missing {c} for {symbol} {interval}")
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)

# ================== Indicators ==================
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def ema_slope(close, n, lookback):
    e = ema(close, n)
    return e, e.diff(lookback)

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100/(1+rs)
    return out.bfill().fillna(50)

def stochastic(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    K = 100 * (close - ll) / (hh - ll)
    D = K.rolling(d).mean()
    return K.bfill(), D.bfill()

def atr(high, low, close, n=14):
    hl = (high-low).abs()
    hc = (high-close.shift(1)).abs()
    lc = (low-close.shift(1)).abs()
    tr = pd.concat([hl,hc,lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rvol(volume, n=20):
    avg = volume.rolling(n).mean()
    return volume / avg.replace(0, np.nan)

def bollinger(close, n=20, mult=2.0):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    return ma, ma+mult*std, ma-mult*std

def bullish_engulfing(df):
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    return ((pc < po) & (c > o) & (c >= po) & (o <= pc)).fillna(False)

def bearish_engulfing(df):
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    return ((pc > po) & (c < o) & (c <= po) & (o >= pc)).fillna(False)

def pin_bar(df, bullish=True, body_ratio=0.3):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = h - l
    up_w = h - pd.concat([o, c], axis=1).max(axis=1)
    dn_w = pd.concat([o, c], axis=1).min(axis=1) - l
    small = body <= (rng * body_ratio)
    return (small & (dn_w > up_w * 2) & (c > o)) if bullish else (small & (up_w > dn_w * 2) & (c < o))

# ================== Ichimoku (correctly aligned) ==================
def ichimoku_core(df, tenkan=9, kijun=26, senkou=52):
    """
    Core Ichimoku lines WITHOUT forward displacement (good for decision logic).
    For plotting, you'd shift Span A/B forward; for logic, compare price to current spans.
    """
    high = df["high"]; low = df["low"]; close = df["close"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = (tenkan_sen + kijun_sen) / 2                # no shift
    span_b = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2  # no shift
    chikou  = close.shift(-26)  # chikou is still shifted back for confirmation vs price 26 ago
    return pd.DataFrame({"tenkan": tenkan_sen, "kijun": kijun_sen, "span_a": span_a, "span_b": span_b, "chikou": chikou})

def ichimoku_bias(df):
    ichi = ichimoku_core(df)
    cloud_top  = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).max(axis=1)
    cloud_bot  = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).min(axis=1)
    price = df["close"]
    up_bias   = (price > cloud_top) & (ichi["span_a"] > ichi["span_b"])
    down_bias = (price < cloud_bot) & (ichi["span_a"] < ichi["span_b"])
    return {"ichi": ichi, "up_bias": up_bias.fillna(False), "down_bias": down_bias.fillna(False),
            "cloud_top": cloud_top, "cloud_bot": cloud_bot}

def ichimoku_signal_strict(df, fresh_bars=3):
    """Strict M5 confirmation: fresh TK cross + price beyond cloud + Chikou confirm."""
    ichi = ichimoku_core(df)
    price = df["close"]
    cloud_top = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).max(axis=1)
    cloud_bot = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).min(axis=1)

    tk_up   = (ichi["tenkan"].shift(1) <= ichi["kijun"].shift(1)) & (ichi["tenkan"] > ichi["kijun"])
    tk_down = (ichi["tenkan"].shift(1) >= ichi["kijun"].shift(1)) & (ichi["tenkan"] < ichi["kijun"])
    recent_bull = bool(tk_up.tail(fresh_bars).any())
    recent_bear = bool(tk_down.tail(fresh_bars).any())

    chikou = ichi["chikou"]; price_26_ago = price.shift(26)

    bull_ok = recent_bull and bool(pd.notna(cloud_top.iloc[-1]) and price.iloc[-1] > cloud_top.iloc[-1]) \
              and bool(pd.notna(chikou.iloc[-1]) and pd.notna(price_26_ago.iloc[-1]) and chikou.iloc[-1] > price_26_ago.iloc[-1])
    bear_ok = recent_bear and bool(pd.notna(cloud_bot.iloc[-1]) and price.iloc[-1] < cloud_bot.iloc[-1]) \
              and bool(pd.notna(chikou.iloc[-1]) and pd.notna(price_26_ago.iloc[-1]) and chikou.iloc[-1] < price_26_ago.iloc[-1])

    meta = {
        "tenkan": float(ichi["tenkan"].iloc[-1]) if pd.notna(ichi["tenkan"].iloc[-1]) else None,
        "kijun": float(ichi["kijun"].iloc[-1]) if pd.notna(ichi["kijun"].iloc[-1]) else None,
        "cloud_top": float(cloud_top.iloc[-1]) if pd.notna(cloud_top.iloc[-1]) else None,
        "cloud_bot": float(cloud_bot.iloc[-1]) if pd.notna(cloud_bot.iloc[-1]) else None,
        "tk_up_recent": recent_bull, "tk_down_recent": recent_bear
    }
    return bull_ok, bear_ok, meta

def ichimoku_signal_relaxed(df, fresh_bars=6):
    """Relaxed M5: recent TK cross (wider window), price above Kijun OR above cloud (and vice versa), Chikou optional."""
    ichi = ichimoku_core(df)
    price = df["close"]
    cloud_top = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).max(axis=1)
    cloud_bot = pd.concat([ichi["span_a"], ichi["span_b"]], axis=1).min(axis=1)

    tk_up   = (ichi["tenkan"].shift(1) <= ichi["kijun"].shift(1)) & (ichi["tenkan"] > ichi["kijun"])
    tk_down = (ichi["tenkan"].shift(1) >= ichi["kijun"].shift(1)) & (ichi["tenkan"] < ichi["kijun"])
    recent_bull = bool(tk_up.tail(fresh_bars).any())
    recent_bear = bool(tk_down.tail(fresh_bars).any())

    above_kijun = price > ichi["kijun"]
    below_kijun = price < ichi["kijun"]

    bull_ok = recent_bull and bool(pd.notna(cloud_top.iloc[-1]) and (price.iloc[-1] > cloud_top.iloc[-1] or above_kijun.iloc[-1]))
    bear_ok = recent_bear and bool(pd.notna(cloud_bot.iloc[-1]) and (price.iloc[-1] < cloud_bot.iloc[-1] or below_kijun.iloc[-1]))

    meta = {
        "tenkan": float(ichi["tenkan"].iloc[-1]) if pd.notna(ichi["tenkan"].iloc[-1]) else None,
        "kijun": float(ichi["kijun"].iloc[-1]) if pd.notna(ichi["kijun"].iloc[-1]) else None,
        "cloud_top": float(cloud_top.iloc[-1]) if pd.notna(cloud_top.iloc[-1]) else None,
        "cloud_bot": float(cloud_bot.iloc[-1]) if pd.notna(cloud_bot.iloc[-1]) else None,
        "tk_up_recent": recent_bull, "tk_down_recent": recent_bear
    }
    return bull_ok, bear_ok, meta

# ================== Strategy ==================
def ema_trend_bias(df_h1, df_m15):
    ema_h1, slope_h1 = ema_slope(df_h1["close"], 200, 3)
    ema_m15, slope_m15 = ema_slope(df_m15["close"], 100, 3)
    up_h1 = (df_h1["close"] > ema_h1) & (slope_h1 > 0)
    dn_h1 = (df_h1["close"] < ema_h1) & (slope_h1 < 0)
    up_m15 = (df_m15["close"] > ema_m15) & (slope_m15 > 0)
    dn_m15 = (df_m15["close"] < ema_m15) & (slope_m15 < 0)
    return bool(up_h1.iloc[-1]), bool(up_m15.iloc[-1]), bool(dn_h1.iloc[-1]), bool(dn_m15.iloc[-1])

def combined_trend_bias(df_h1, df_m15, relaxed: bool):
    # Ichimoku trend
    h1_ichi = ichimoku_bias(df_h1)
    m15_ichi = ichimoku_bias(df_m15)
    up_h1_i = bool(h1_ichi["up_bias"].iloc[-1]); dn_h1_i = bool(h1_ichi["down_bias"].iloc[-1])
    up_m15_i = bool(m15_ichi["up_bias"].iloc[-1]); dn_m15_i = bool(m15_ichi["down_bias"].iloc[-1])

    # EMA trend
    up_h1_e, up_m15_e, dn_h1_e, dn_m15_e = ema_trend_bias(df_h1, df_m15)

    if relaxed:
        # Relaxed: allow Ichimoku OR EMA on both TFs in same direction
        up_ok   = ((up_h1_i and up_m15_i) or (up_h1_e and up_m15_e))
        down_ok = ((dn_h1_i and dn_m15_i) or (dn_h1_e and dn_m15_e))
    else:
        # Strict: require BOTH Ichimoku AND EMA alignment
        up_ok   = (up_h1_i and up_m15_i) and (up_h1_e and up_m15_e)
        down_ok = (dn_h1_i and dn_m15_i) and (dn_h1_e and dn_m15_e)

    return up_ok, down_ok, h1_ichi, m15_ichi

def signal_m5(df_m5, symbol, relaxed: bool):
    base = THRESH.get(symbol.lower(), THRESH["default"]).copy()
    th = base.copy()
    if relaxed:
        th["atr_min_frac"] *= RELAX_ATR_DISCOUNT  # slightly easier ATR requirement

    a = atr(df_m5["high"], df_m5["low"], df_m5["close"], ATR_PERIOD)
    atr_pct = a / df_m5["close"]
    r = rsi(df_m5["close"], RSI_PERIOD)
    K, D = stochastic(df_m5["high"], df_m5["low"], df_m5["close"], STO_K, STO_D)
    ma, bb_u, bb_l = bollinger(df_m5["close"], 20, 2.0)

    # Optional RVOL (FX feeds may lack true volume)
    rv = rvol(df_m5["volume"], 20)
    has_volume = not df_m5["volume"].isna().all() and (df_m5["volume"].fillna(0).sum() > 0)
    if has_volume:
        vol_ok = (rv >= th["rvol_min"]).fillna(False)
    else:
        rv = pd.Series([1.0] * len(df_m5), index=df_m5.index)
        vol_ok = pd.Series([True] * len(df_m5), index=df_m5.index)

    # Candles + oscillators
    bull_c = (bullish_engulfing(df_m5) | pin_bar(df_m5, bullish=True)).fillna(False)
    bear_c = (bearish_engulfing(df_m5) | pin_bar(df_m5, bullish=False)).fillna(False)
    bull_osc = (r < 55) & (K.shift(1) < D.shift(1)) & (K > D) & (r.diff() > 0)
    bear_osc = (r > 45) & (K.shift(1) > D.shift(1)) & (K < D) & (r.diff() < 0)
    atr_ok = (atr_pct > th["atr_min_frac"]).fillna(False)

    close = df_m5["close"]
    over = th["bb_overext"] * a
    buy_bb = (close > ma) & (close <= (bb_u + over))
    sell_bb = (close < ma) & (close >= (bb_l - over))

    bull_trigger = (bull_osc & bull_c).fillna(False)
    bear_trigger = (bear_osc & bear_c).fillna(False)
    fresh_bull = bool(bull_trigger.tail(SIGNAL_EXPIRATION_BARS if not relaxed else RELAX_FRESH_BARS).any())
    fresh_bear = bool(bear_trigger.tail(SIGNAL_EXPIRATION_BARS if not relaxed else RELAX_FRESH_BARS).any())

    # Ichimoku entry (strict vs relaxed)
    if relaxed:
        ichi_long, ichi_short, ichi_meta_m5 = ichimoku_signal_relaxed(df_m5, fresh_bars=RELAX_FRESH_BARS)
    else:
        ichi_long, ichi_short, ichi_meta_m5 = ichimoku_signal_strict(df_m5, fresh_bars=SIGNAL_EXPIRATION_BARS)

    i = -1
    long_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and buy_bb.iloc[i] and fresh_bull and ichi_long)
    short_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and sell_bb.iloc[i] and fresh_bear and ichi_short)

    if DEBUG:
        print(f"[DBG] M5 gates {symbol} relaxed={relaxed} | atr_ok={bool(atr_ok.iloc[i])} vol_ok={bool(vol_ok.iloc[i])} "
              f"bb_ok={'BUY' if buy_bb.iloc[i] else ('SELL' if sell_bb.iloc[i] else 'NO')} "
              f"fresh_bull={fresh_bull} fresh_bear={fresh_bear} ichi_long={ichi_long} ichi_short={ichi_short}")

    meta = {
        "mode": "RELAXED" if relaxed else "STRICT",
        "atr_pct": float(atr_pct.iloc[i]) if len(atr_pct) else None,
        "rsi": float(r.iloc[i]) if len(r) else None,
        "stoch_k": float(K.iloc[i]) if len(K) else None,
        "stoch_d": float(D.iloc[i]) if len(D) else None,
        "rvol": float(rv.iloc[i]) if len(rv) else None,
        "bull_candle": bool(bull_c.iloc[i]) if len(bull_c) else False,
        "bear_candle": bool(bear_c.iloc[i]) if len(bear_c) else False,
        "bb_middle": float(ma.iloc[i]) if len(ma) else None,
        "bb_upper": float(bb_u.iloc[i]) if len(bb_u) else None,
        "bb_lower": float(bb_l.iloc[i]) if len(bb_l) else None,
        "fresh_window_bars": int(SIGNAL_EXPIRATION_BARS if not relaxed else RELAX_FRESH_BARS),
        # Ichimoku M5 details
        "ichi_tenkan": ichi_meta_m5["tenkan"],
        "ichi_kijun": ichi_meta_m5["kijun"],
        "ichi_cloud_top": ichi_meta_m5["cloud_top"],
        "ichi_cloud_bot": ichi_meta_m5["cloud_bot"],
        "ichi_tk_up_recent": ichi_meta_m5["tk_up_recent"],
        "ichi_tk_down_recent": ichi_meta_m5["tk_down_recent"],
        "has_volume": bool(has_volume),
    }
    return long_ok, short_ok, meta

# ================== Telegram ==================
def send_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured"); print(text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"[Telegram] {e}")

def assemble_message(symbol, direction, meta, df_h1, df_m15, df_m5, mode_note):
    m5 = df_m5.iloc[-1]; h1 = df_h1.iloc[-1]; m15 = df_m15.iloc[-1]
    txt = []
    txt.append(f"<b>{symbol}</b> — <b>{direction}</b> signal 🚨  ({mode_note})")
    txt.append(f"Time: {now_utc_iso()}")
    txt.append("")
    txt.append("<b>Filters</b>")
    txt.append(f"• ATR%: {meta['atr_pct']:.4f}")
    txt.append(f"• RSI: {meta['rsi']:.1f}  |  Stoch K/D: {meta['stoch_k']:.1f}/{meta['stoch_d']:.1f}")
    if meta["has_volume"]:
        txt.append(f"• RVOL: {meta['rvol']:.2f}")
    else:
        txt.append("• RVOL: n/a (no volume in feed)")
    txt.append(f"• Candle: {'Bullish' if meta['bull_candle'] else ('Bearish' if meta['bear_candle'] else 'None')}")
    txt.append(f"• BB: mid {meta['bb_middle']:.3f} | up {meta['bb_upper']:.3f} | lo {meta['bb_lower']:.3f}")
    txt.append("")
    txt.append("<b>Ichimoku (M5)</b>")
    txt.append(f"• Tenkan/Kijun: {meta['ichi_tenkan']:.3f}/{meta['ichi_kijun']:.3f}  |  Cloud T/B: {meta['ichi_cloud_top']:.3f}/{meta['ichi_cloud_bot']:.3f}")
    tk_txt = "Up-cross" if meta['ichi_tk_up_recent'] else ("Down-cross" if meta['ichi_tk_down_recent'] else "No fresh cross")
    txt.append(f"• TK cross window: {tk_txt} (≤ {meta['fresh_window_bars']} bars)")
    txt.append("")
    txt.append("<b>Context</b>")
    txt.append(f"H1 Close: {h1['close']:.3f} | M15 Close: {m15['close']:.3f} | M5 Close: {m5['close']:.3f}")
    return "\n".join(txt)

# ================== Cadenced fetch cache ==================
cache = {}  # (symbol, interval) -> {"df": DataFrame, "last_fetch": ts}

def need_fetch(symbol: str, interval: str, cadence_sec: int) -> bool:
    now_ts = time.time()
    slot = cache.get((symbol, interval))
    return (slot is None) or ((now_ts - slot["last_fetch"]) >= cadence_sec)

def cached_fetch(symbol: str, interval: str, cadence_sec: int, outputsize: int):
    if need_fetch(symbol, interval, cadence_sec):
        df = fetch_series(symbol, interval, outputsize)
        if df is not None and len(df):
            cache[(symbol, interval)] = {"df": df, "last_fetch": time.time()}
            return df
        return cache[(symbol, interval)]["df"] if (symbol, interval) in cache else None
    return cache[(symbol, interval)]["df"]

# ================== Main Loop ==================
def main():
    global _api_calls_today
    if not TD_API_KEY:
        print("[ERROR] TD_API_KEY missing"); return

    print(f"[INFO] Ichimoku MTF Bot — Symbols={SYMBOLS} | Poll={POLL_SECONDS}s | Cadences M5={CADENCE_M5_SEC}s M15={CADENCE_M15_SEC}s H1={CADENCE_H1_SEC}s")
    start_ts = time.time()
    last_signal_ts = None

    while True:
        loop_start = time.time()
        try:
            # Determine mode based on time since last signal
            now_ts = time.time()
            minutes_since = 1e9 if last_signal_ts is None else (now_ts - last_signal_ts) / 60.0
            relaxed = minutes_since >= RELAX_IF_NO_SIGNAL_MIN
            mode_note = "RELAXED" if relaxed else "STRICT"

            for symbol in SYMBOLS:
                try:
                    df_m5  = cached_fetch(symbol, "5min",  CADENCE_M5_SEC,  200)
                    df_m15 = cached_fetch(symbol, "15min", CADENCE_M15_SEC, 200)
                    df_h1  = cached_fetch(symbol, "1h",    CADENCE_H1_SEC,  300)
                    if df_m5 is None or df_m15 is None or df_h1 is None:
                        print(f"[WARN] data fetch failed (cached) for {symbol}")
                        continue

                    up_trend, down_trend, h1_ichi, m15_ichi = combined_trend_bias(df_h1, df_m15, relaxed=relaxed)
                    long_ok, short_ok, meta = signal_m5(df_m5, symbol, relaxed=relaxed)

                    fire_long  = long_ok  and up_trend
                    fire_short = short_ok and down_trend

                    if DEBUG:
                        print(f"[DBG] Trend gates {symbol} relaxed={relaxed} | up_trend={up_trend} down_trend={down_trend}")

                    if fire_long or fire_short:
                        direction = "BUY" if fire_long else "SELL"
                        msg = assemble_message(symbol, direction, meta, df_h1, df_m15, df_m5, mode_note)
                        send_message(msg)
                        print(msg)
                        last_signal_ts = time.time()  # reset relax timer
                    else:
                        ts = df_m5["datetime"].iloc[-1].isoformat()
                        print(f"[{symbol}] {ts} — no aligned signal. Mode={mode_note}")

                except Exception as e_sym:
                    print(f"[ERROR] Symbol loop exception for {symbol}: {e_sym}")

            # crude projected daily usage
            elapsed = time.time() - start_ts
            if elapsed > 0:
                proj_daily = int(_api_calls_today * (86400 / max(1, elapsed)))
                print(f"[USAGE] Calls so far: {_api_calls_today} | Projected/day: ~{proj_daily} | Mode={mode_note}")

            # Sleep to next tick
            sleep_left = POLL_SECONDS - (time.time() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)

        except Exception as e:
            print(f"[ERROR] Loop exception: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
