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
        rv = pd.Series([1.0] * len(df_m5), index=df_m5