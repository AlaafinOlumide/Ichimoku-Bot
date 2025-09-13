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

# Fixed symbols (no slashes, Twelve Data accepts these)
SYMBOLS = ["XAUUSD", "USDJPY"]

# Loop sleep: how often the main loop spins
POLL_SECONDS = 60

# Timeframe refresh cadences (in seconds)
CADENCE_M5_SEC  = 360    # 6 minutes
CADENCE_M15_SEC = 900    # 15 minutes
CADENCE_H1_SEC  = 3600   # 60 minutes

# Indicators/filters
ATR_PERIOD = 14
RSI_PERIOD = 14
STO_K = 14
STO_D = 3
SIGNAL_EXPIRATION_BARS = 3

# Per-symbol thresholds
THRESH = {
    "xauusd": {"atr_min_frac": 0.0012, "rvol_min": 1.20, "bb_overext": 0.25},
    "usdjpy": {"atr_min_frac": 0.0006, "rvol_min": 1.05, "bb_overext": 0.20},
    "default": {"atr_min_frac": 0.0008, "rvol_min": 1.10, "bb_overext": 0.20},
}

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ========== Data Fetch ==========
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
    return None, j.get("message") or j

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

# ========== Indicators ==========
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def ema_slope(close, n, lookback): e=ema(close,n); return e,e.diff(lookback)
def rsi(close,n=14): d=close.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean(); rs=up/dn.replace(0,np.nan); out=100-100/(1+rs); return out.bfill().fillna(50)
def stochastic(h,l,c,k=14,d=3): ll=l.rolling(k).min(); hh=h.rolling(k).max(); K=100*(c-ll)/(hh-ll); D=K.rolling(d).mean(); return K.bfill(),D.bfill()
def atr(h,l,c,n=14): hl=(h-l).abs(); hc=(h-c.shift(1)).abs(); lc=(l-c.shift(1)).abs(); tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.rolling(n).mean()
def rvol(v,n=20): avg=v.rolling(n).mean(); return v/avg.replace(0,np.nan)
def bollinger(c,n=20,m=2): ma=c.rolling(n).mean(); std=c.rolling(n).std(); return ma,ma+m*std,ma-m*std
def bullish_engulfing(df): o,c=df["open"],df["close"]; po,pc=o.shift(1),c.shift(1); return ((pc<po)&(c>o)&(c>=po)&(o<=pc)).fillna(False)
def bearish_engulfing(df): o,c=df["open"],df["close"]; po,pc=o.shift(1),c.shift(1); return ((pc>po)&(c<o)&(c<=po)&(o>=pc)).fillna(False)
def pin_bar(df,bullish=True,body_ratio=0.3): o,h,l,c=df["open"],df["high"],df["low"],df["close"]; body=(c-o).abs(); rng=h-l; up=h-pd.concat([o,c],axis=1).max(axis=1); dn=pd.concat([o,c],axis=1).min(axis=1)-l; small=body<=rng*body_ratio; return (small&(dn>up*2)&(c>o)) if bullish else (small&(up>dn*2)&(c<o))

# ========== Strategy ==========
def trend_bias(df_h1, df_m15):
    ema_h1,slope_h1=ema_slope(df_h1["close"],200,3)
    ema_m15,slope_m15=ema_slope(df_m15["close"],100,3)
    up_h1=(df_h1["close"]>ema_h1)&(slope_h1>0)
    dn_h1=(df_h1["close"]<ema_h1)&(slope_h1<0)
    up_m15=(df_m15["close"]>ema_m15)&(slope_m15>0)
    dn_m15=(df_m15["close"]<ema_m15)&(slope_m15<0)
    return bool(up_h1.iloc[-1]), bool(up_m15.iloc[-1]), bool(dn_h1.iloc[-1]), bool(dn_m15.iloc[-1])

def signal_m5(df_m5,symbol):
    th=THRESH.get(symbol.lower(),THRESH["default"])
    a=atr(df_m5["high"],df_m5["low"],df_m5["close"],ATR_PERIOD); atr_pct=a/df_m5["close"]
    r=rsi(df_m5["close"],RSI_PERIOD); K,D=stochastic(df_m5["high"],df_m5["low"],df_m5["close"],STO_K,STO_D)
    ma,bb_u,bb_l=bollinger(df_m5["close"],20,2.0)
    rv=rvol(df_m5["volume"],20); has_vol=not df_m5["volume"].isna().all()
    vol_ok=(rv>=th["rvol_min"]).fillna(False) if has_vol else pd.Series([True]*len(df_m5),index=df_m5.index)
    bull_c=(bullish_engulfing(df_m5)|pin_bar(df_m5,True)).fillna(False); bear_c=(bearish_engulfing(df_m5)|pin_bar(df_m5,False)).fillna(False)
    bull_osc=(r<55)&(K.shift(1)<D.shift(1))&(K>D)&(r.diff()>0)
    bear_osc=(r>45)&(K.shift(1)>D.shift(1))&(K<D)&(r.diff()<0)
    atr_ok=(atr_pct>th["atr_min_frac"]).fillna(False)
    close=df_m5["close"]; over=th["bb_overext"]*a
    buy_bb=(close>ma)&(close<=(bb_u+over)); sell_bb=(close<ma)&(close>=(bb_l-over))
    bull_trigger=(bull_osc&bull_c).fillna(False); bear_trigger=(bear_osc&bear_c).fillna(False)
    fresh_bull=bool(bull_trigger.tail(SIGNAL_EXPIRATION_BARS).any()); fresh_bear=bool(bear_trigger.tail(SIGNAL_EXPIRATION_BARS).any())
    i=-1; long_ok=bool(atr_ok.iloc[i] and vol_ok.iloc[i] and buy_bb.iloc[i] and fresh_bull); short_ok=bool(atr_ok.iloc[i] and vol_ok.iloc[i] and sell_bb.iloc[i] and fresh_bear)
    return long_ok,short_ok,{"atr_pct":float(atr_pct.iloc[i]),"rsi":float(r.iloc[i]),"stoch_k":float(K.iloc[i]),"stoch_d":float(D.iloc[i]),"rvol":float(rv.iloc[i]),"bull_candle":bool(bull_c.iloc[i]),"bear_candle":bool(bear_c.iloc[i]),"bb_middle":float(ma.iloc[i]),"bb_upper":float(bb_u.iloc[i]),"bb_lower":float(bb_l.iloc[i]),"fresh_window_bars":SIGNAL_EXPIRATION_BARS,"has_volume":has_vol}

# ========== Telegram ==========
def send_message(txt):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print("[WARN] Telegram not set"); print(txt); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"; payload={"chat_id":TELEGRAM_CHAT_ID,"text":txt,"parse_mode":"HTML"}; 
    try: requests.post(url,json=payload,timeout=15)
    except Exception as e: print(f"[Telegram] {e}")

# ========== Cache ==========
cache={}

def need_fetch(symbol,interval,cadence): now=time.time(); slot=cache.get((symbol,interval)); return not slot or (now-slot["last_fetch"])>=cadence
def cached_fetch(symbol,interval,cadence,outputsize):
    if need_fetch(symbol,interval,cadence):
        df=fetch_series(symbol,interval,outputsize)
        if df is not None: cache[(symbol,interval)]={"df":df,"last_fetch":time.time()}; return df
        return cache[(symbol,interval)]["df"] if (symbol,interval) in cache else None
    return cache[(symbol,interval)]["df"]

# ========== Main Loop ==========
def main():
    global _api_calls_today
    print(f"[INFO] Bot started with {SYMBOLS}, cadences: M5={CADENCE_M5_SEC}s, M15={CADENCE_M15_SEC}s, H1={CADENCE_H1_SEC}s")
    start=time.time()
    while True:
        try:
            for s in SYMBOLS:
                df_m5=cached_fetch(s,"5min",CADENCE_M5_SEC,200)
                df_m15=cached_fetch(s,"15min",CADENCE_M15_SEC,200)
                df_h1=cached_fetch(s,"1h",CADENCE_H1_SEC,300)
                if df_m5 is None or df_m15 is None or df_h1 is None: continue
                long_ok,short_ok,meta=signal_m5(df_m5,s)
                up_h1,up_m15,dn_h1,dn_m15=trend_bias(df_h1,df_m15)
                if long_ok and up_h1 and up_m15: send_message(f"{s} — BUY signal 🚨 | {now_utc_iso()}")
                elif short_ok and dn_h1 and dn_m15: send_message(f"{s} — SELL signal 🚨 | {now_utc_iso()}")
            elapsed=time.time()-start; proj=int(_api_calls_today*(86400/max(1,elapsed)))
            print(f"[USAGE] Calls so far={_api_calls_today}, Projected/day≈{proj}")
            time.sleep(POLL_SECONDS)
        except Exception as e: print(f"[ERROR] {e}"); time.sleep(POLL_SECONDS)

if __name__=="__main__": main()
