import pandas as pd
import numpy as np

def daily_pivots_from_h1(df_h1: pd.DataFrame):
    if df_h1 is None or df_h1.empty or len(df_h1) < 24:
        return {}
    d = df_h1.copy()
    d['date'] = d['datetime'].dt.date
    days = sorted(d['date'].unique())
    prev_day = days[-2] if len(days) >= 2 else days[-1]
    day_df = d[d['date'] == prev_day]
    if day_df.empty:
        return {}
    H = float(day_df['high'].max())
    L = float(day_df['low'].min())
    C = float(day_df['close'].iloc[-1])
    P = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2*(P - L)
    S3 = L - 2*(H - P)
    return { 'P': P, 'R1': R1, 'R2': R2, 'R3': R3, 'S1': S1, 'S2': S2, 'S3': S3, 'src_day': str(prev_day) }

def swing_points(df: pd.DataFrame, lookback: int = 60, left: int = 2, right: int = 2):
    if df is None or df.empty:
        return [], []
    n = len(df)
    highs = []
    lows = []
    for i in range(max(left, right), n - right):
        window = df.iloc[i-left:i+right+1]
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        if np.isfinite(h) and h == window['high'].max():
            highs.append((df['datetime'].iloc[i], float(h)))
        if np.isfinite(l) and l == window['low'].min():
            lows.append((df['datetime'].iloc[i], float(l)))
    highs = highs[-lookback:]
    lows = lows[-lookback:]
    return highs, lows

def nearest_levels(price: float, pivots: dict, swing_highs, swing_lows):
    res_levels = []
    sup_levels = []
    if pivots:
        for k in ['R1','R2','R3']:
            if k in pivots and pivots[k] is not None:
                res_levels.append(('pivot_'+k, float(pivots[k])))
        for k in ['S1','S2','S3']:
            if k in pivots and pivots[k] is not None:
                sup_levels.append(('pivot_'+k, float(pivots[k])))
    for _, h in swing_highs:
        res_levels.append(('swing_high', float(h)))
    for _, l in swing_lows:
        sup_levels.append(('swing_low', float(l)))
    res_above = [(name, lvl, lvl - price) for name, lvl in res_levels if lvl >= price]
    sup_below = [(name, lvl, price - lvl) for name, lvl in sup_levels if lvl <= price]
    nearest_res = min(res_above, key=lambda x: x[2]) if res_above else None
    nearest_sup = min(sup_below, key=lambda x: x[2]) if sup_below else None
    return nearest_res, nearest_sup
