import pandas as pd
from indicators import atr as atr_ind, rsi as rsi_ind, stochastic, relative_volume

def atr_filter(df: pd.DataFrame, period: int, min_atr_frac: float = 0.001):
    a = atr_ind(df["high"], df["low"], df["close"], period)
    atr_pct = a / df["close"]
    return (atr_pct > min_atr_frac).fillna(False), atr_pct

def oscillator_confirmation(df: pd.DataFrame, rsi_period: int, sto_k: int, sto_d: int):
    r = rsi_ind(df["close"], rsi_period)
    k, d = stochastic(df["high"], df["low"], df["close"], sto_k, sto_d)
    bull = (r < 55) & (k.shift(1) < d.shift(1)) & (k > d) & (r.diff() > 0)
    bear = (r > 45) & (k.shift(1) > d.shift(1)) & (k < d) & (r.diff() < 0)
    return bull.fillna(False), bear.fillna(False), r, k, d

def volume_filter(df: pd.DataFrame, window: int = 20, min_rvol: float = 1.2):
    rv = relative_volume(df["volume"], window)
    return (rv >= min_rvol).fillna(False), rv
