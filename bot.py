import os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

# ========== Config ==========
load_dotenv()
TD_API_KEY = os.getenv("TD_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","XAU/USD,USD/JPY").split(",") if s.strip()]
POLL_SECONDS = int(os.getenv("POLL_SECONDS","60"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD","14"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD","14"))
STO_K = int(os.getenv("STO_K","14"))
STO_D = int(os.getenv("STO_D","3"))
SIGNAL_EXPIRATION_BARS = int(os.getenv("SIGNAL_EXPIRATION_BARS","3"))

# Per-symbol thresholds (simple defaults)
THRESH = {
    "xau/usd": {"atr_min_frac": 0.0012, "rvol_min": 1.20, "bb_overext": 0.25},
    "usd/jpy": {"atr_min_frac": 0.0006, "rvol_min": 1.05, "bb_overext": 0.20},
    "default": {"atr_min_frac": 0.0008, "rvol_min": 1.10, "bb_overext": 0.20},
}

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ========== Data ==========
BASE_URL = "https://api.twelvedata.com/time_series"

def fetch_series(symbol: str, interval: str, outputsize: int = 300):
    """
    Fetches OHLCV (if available) from Twelve Data.
    Some FX/metal feeds don't include 'volume' — we create a NaN column so downstream code won't KeyError.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TD_API_KEY,
        "outputsize": outputsize,
        "format": "JSON",
        "order": "ASC",
    }
    r = requests.get(BASE_URL, params=params, timeout=20)
    j = r.json()
    if "values" not in j:
        return None
    df = pd.DataFrame(j["values"])
    # Parse numeric columns that always exist
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            return None

    # Handle volume optionality
    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ========== Indicators ==========
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

# ========== Strategy (Lite) ==========
def trend_bias(df_h1, df_m15):
    ema_h1, slope_h1 = ema_slope(df_h1["close"], 200, 3)
    ema_m15, slope_m15 = ema_slope(df_m15["close"], 100, 3)
    up_h1 = (df_h1["close"] > ema_h1) & (slope_h1 > 0)
    dn_h1 = (df_h1["close"] < ema_h1) & (slope_h1 < 0)
    up_m15 = (df_m15["close"] > ema_m15) & (slope_m15 > 0)
    dn_m15 = (df_m15["close"] < ema_m15) & (slope_m15 < 0)
    return (
        bool(up_h1.iloc[-1] and not dn_h1.iloc[-1]),
        bool(up_m15.iloc[-1] and not dn_m15.iloc[-1]),
        bool(dn_h1.iloc[-1] and not up_h1.iloc[-1]),
        bool(dn_m15.iloc[-1] and not up_m15.iloc[-1]),
    )

def signal_m5(df_m5, symbol):
    th = THRESH.get(symbol.lower(), THRESH["default"])

    a = atr(df_m5["high"], df_m5["low"], df_m5["close"], ATR_PERIOD)
    atr_pct = a / df_m5["close"]
    r = rsi(df_m5["close"], RSI_PERIOD)
    K, D = stochastic(df_m5["high"], df_m5["low"], df_m5["close"], STO_K, STO_D)
    ma, bb_u, bb_l = bollinger(df_m5["close"], 20, 2.0)

    rv = rvol(df_m5["volume"], 20)
    has_volume = not df_m5["volume"].isna().all() and (df_m5["volume"].fillna(0).sum() > 0)
    if has_volume:
        vol_ok = (rv >= th["rvol_min"]).fillna(False)
    else:
        rv = pd.Series([1.0] * len(df_m5), index=df_m5.index)
        vol_ok = pd.Series([True] * len(df_m5), index=df_m5.index)

    bull_c = (bullish_engulfing(df_m5) | pin_bar(df_m5, bullish=True)).fillna(False)
    bear_c = (bearish_engulfing(df_m5) | pin_bar(df_m5, bullish=False)).fillna(False)

    bull_osc = (r < 55) & (K.shift(1) < D.shift(1)) & (K > D) & (r.diff() > 0)
    bear_osc = (r > 45) & (K.shift(1) > D.shift(1)) & (K < D) & (r.diff() < 0)

    atr_ok = (atr_pct > th["atr_min_frac"]).fillna(False)

    close = df_m5["close"]
    over = th["bb_overext"] * a
    buy_bb = (close > ma) & (close <= (bb_u + over))
    sell_bb = (close < ma) & (close >= (bb_l - over))

    recent_buy_zone = (
        (close <= ma) | (close <= ma + (bb_u - ma) * 0.25) | (close <= ma + a * 0.2)
    ).rolling(SIGNAL_EXPIRATION_BARS).apply(lambda x: (x > 0).any(), raw=True).astype(bool)

    recent_sell_zone = (
        (close >= ma) | (close >= ma - (ma - bb_l) * 0.25) | (close >= ma - a * 0.2)
    ).rolling(SIGNAL_EXPIRATION_BARS).apply(lambda x: (x > 0).any(), raw=True).astype(bool)

    bull_trigger = (bull_osc & bull_c).fillna(False)
    bear_trigger = (bear_osc & bear_c).fillna(False)
    fresh_bull = bool(bull_trigger.tail(SIGNAL_EXPIRATION_BARS).any())
    fresh_bear = bool(bear_trigger.tail(SIGNAL_EXPIRATION_BARS).any())

    i = -1
    long_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and buy_bb.iloc[i] and recent_buy_zone.iloc[i] and fresh_bull)
    short_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and sell_bb.iloc[i] and recent_sell_zone.iloc[i] and fresh_bear)

    meta = {
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
        "fresh_window_bars": int(SIGNAL_EXPIRATION_BARS),
        "has_volume": bool(has_volume),
    }
    return long_ok, short_ok, meta

# ========== Telegram ==========
def send_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured"); print(text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"[Telegram] {e}")

def assemble_message(symbol, direction, meta, df_h1, df_m15, df_m5):
    m5 = df_m5.iloc[-1]; h1 = df_h1.iloc[-1]; m15 = df_m15.iloc[-1]
    txt = []
    txt.append(f"<b>{symbol}</b> — <b>{direction}</b> signal 🚨")
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
    txt.append(f"• Fresh-window: {meta['fresh_window_bars']} bars")
    txt.append("")
    txt.append("<b>Last M5 Bar</b>")
    txt.append(f"O:{m5['open']:.3f} H:{m5['high']:.3f} L:{m5['low']:.3f} C:{m5['close']:.3f} Vol:{0 if pd.isna(m5['volume']) else int(m5['volume'])}")
    txt.append("")
    txt.append("<b>Context</b>")
    txt.append(f"H1 Close: {h1['close']:.3f} | M15 Close: {m15['close']:.3f}")
    txt.append("— Trend: H1 & M15 aligned with signal direction")
    return "\n".join(txt)

# ========== Main Loop ==========
def main():
    if not TD_API_KEY:
        print("[ERROR] TD_API_KEY missing"); return
    last_bar = {}

    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    df_m5 = fetch_series(symbol, "5min", 200)
                    df_m15 = fetch_series(symbol, "15min", 200)
                    df_h1 = fetch_series(symbol, "1h", 300)
                    if df_m5 is None or df_m15 is None or df_h1 is None:
                        print(f"[WARN] data fetch failed for {symbol}")
                        continue

                    last_dt = df_m5["datetime"].iloc[-1].isoformat()
                    if last_bar.get(symbol) == last_dt:
                        continue
                    last_bar[symbol] = last_dt

                    up_h1, up_m15, dn_h1, dn_m15 = trend_bias(df_h1, df_m15)
                    long_ok, short_ok, meta = signal_m5(df_m5, symbol)

                    fire_long = long_ok and up_h1 and up_m15
                    fire_short = short_ok and dn_h1 and dn_m15

                    if fire_long or fire_short:
                        direction = "BUY" if fire_long else "SELL"
                        msg = assemble_message(symbol, direction, meta, df_h1, df_m15, df_m5)
                        send_message(msg)
                        print(msg)
                    else:
                        print(f"[{symbol}] {last_dt} — no aligned signal.")
                except Exception as e_sym:
                    print(f"[ERROR] Symbol loop exception for {symbol}: {e_sym}")
            time.sleep(POLL_SECONDS)
        except Exception as e:
            print(f"[ERROR] Loop exception: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()