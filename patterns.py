import pandas as pd

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    cond = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    return cond.fillna(False)

def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    cond = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)
    return cond.fillna(False)

def pin_bar(df: pd.DataFrame, bullish=True, body_ratio=0.3):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    range_ = h - l
    upper_wick = h - df[["open","close"]].max(axis=1)
    lower_wick = df[["open","close"]].min(axis=1) - l
    small_body = body <= (range_ * body_ratio)
    if bullish:
        return small_body & (lower_wick > upper_wick*2) & (c > o)
    else:
        return small_body & (upper_wick > lower_wick*2) & (c < o)
