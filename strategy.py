import pandas as pd
import numpy as np
from indicators import ema_slope, bollinger, atr as atr_ind, rsi as rsi_ind, stochastic
from patterns import bullish_engulfing, bearish_engulfing, pin_bar

class MTFStrategy:
    def __init__(self, rsi_period=14, sto_k=14, sto_d=3, atr_period=14, expiration_bars=3):
        self.rsi_period = rsi_period
        self.sto_k = sto_k
        self.sto_d = sto_d
        self.atr_period = atr_period
        self.expiration_bars = max(1, int(expiration_bars))

        self.thresholds = {
            "xau/usd": {"atr_min_frac": 0.0012, "rvol_min": 1.20, "overext_mult": 0.25},
            "usd/jpy": {"atr_min_frac": 0.0006, "rvol_min": 1.05, "overext_mult": 0.20},
            "default": {"atr_min_frac": 0.0008, "rvol_min": 1.10, "overext_mult": 0.20},
        }

    def _get_th(self, symbol: str):
        return self.thresholds.get(symbol.lower(), self.thresholds["default"])

    def trend_bias(self, df_h1: pd.DataFrame, df_m15: pd.DataFrame):
        ema_h1, slope_h1 = ema_slope(df_h1["close"], period=200, lookback=3)
        ema_m15, slope_m15 = ema_slope(df_m15["close"], period=100, lookback=3)
        up_h1 = (df_h1["close"] > ema_h1) & (slope_h1 > 0)
        down_h1 = (df_h1["close"] < ema_h1) & (slope_h1 < 0)
        up_m15 = (df_m15["close"] > ema_m15) & (slope_m15 > 0)
        down_m15 = (df_m15["close"] < ema_m15) & (slope_m15 < 0)
        return (bool(up_h1.iloc[-1] and not down_h1.iloc[-1]),
                bool(up_m15.iloc[-1] and not down_m15.iloc[-1]),
                bool(down_h1.iloc[-1] and not up_h1.iloc[-1]),
                bool(down_m15.iloc[-1] and not up_m15.iloc[-1]))

    def _fresh_within(self, series_bool: pd.Series, bars: int) -> bool:
        return bool(series_bool.tail(bars).any())

    def signal_m5(self, df_m5: pd.DataFrame, symbol: str = ""):
        a = atr_ind(df_m5["high"], df_m5["low"], df_m5["close"], self.atr_period)
        atr_pct = a / df_m5["close"]
        r = rsi_ind(df_m5["close"], self.rsi_period)
        k, d = stochastic(df_m5["high"], df_m5["low"], df_m5["close"], self.sto_k, self.sto_d)
        ma, bb_u, bb_l = bollinger(df_m5["close"], 20, 2.0)

        bull_candle = bullish_engulfing(df_m5) | pin_bar(df_m5, bullish=True)
        bear_candle = bearish_engulfing(df_m5) | pin_bar(df_m5, bullish=False)

        bull_osc = (r < 55) & (k.shift(1) < d.shift(1)) & (k > d) & (r.diff() > 0)
        bear_osc = (r > 45) & (k.shift(1) > d.shift(1)) & (k < d) & (r.diff() < 0)

        rv = df_m5["volume"] / (df_m5["volume"].rolling(20).mean().replace(0, np.nan))

        th = self._get_th(symbol)
        atr_ok = (atr_pct > th["atr_min_frac"]).fillna(False)
        vol_ok = (rv >= th["rvol_min"]).fillna(False)

        close = df_m5["close"]
        buy_bb = (close > ma) & (close <= (bb_u + th["overext_mult"] * a))
        sell_bb = (close < ma) & (close >= (bb_l - th["overext_mult"] * a))

        recent_buy_zone = ((close <= ma) | (close <= (ma + (bb_u - ma) * 0.25)) | (close <= (ma + a * 0.2))).rolling(self.expiration_bars).apply(lambda x: (x>0).any(), raw=True).astype(bool)
        recent_sell_zone = ((close >= ma) | (close >= (ma - (ma - bb_l) * 0.25)) | (close >= (ma - a * 0.2))).rolling(self.expiration_bars).apply(lambda x: (x>0).any(), raw=True).astype(bool)

        bull_trigger = (bull_osc & bull_candle).fillna(False)
        bear_trigger = (bear_osc & bear_candle).fillna(False)

        fresh_bull = self._fresh_within(bull_trigger, self.expiration_bars)
        fresh_bear = self._fresh_within(bear_trigger, self.expiration_bars)

        i = -1
        long_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and buy_bb.iloc[i] and recent_buy_zone.iloc[i] and fresh_bull)
        short_ok = bool(atr_ok.iloc[i] and vol_ok.iloc[i] and sell_bb.iloc[i] and recent_sell_zone.iloc[i] and fresh_bear)

        # Confidence scoring
        atr_headroom = float((atr_pct.iloc[i] / th["atr_min_frac"]) * 25.0) if th["atr_min_frac"] else 0.0
        rvol_headroom = float((rv.iloc[i] / th["rvol_min"]) * 25.0) if th["rvol_min"] else 0.0
        mom = float(abs(k.iloc[i] - d.iloc[i])) if len(k) else 0.0
        mom_score = max(0.0, min(25.0, mom))
        candle_score = 25.0 if (bull_candle.iloc[i] or bear_candle.iloc[i]) else 10.0
        confidence = max(1.0, min(100.0, atr_headroom + rvol_headroom + mom_score + candle_score))

        row = {
            "atr_pct": float(atr_pct.iloc[i]) if len(atr_pct) else None,
            "rsi": float(r.iloc[i]) if len(r) else None,
            "stoch_k": float(k.iloc[i]) if len(k) else None,
            "stoch_d": float(d.iloc[i]) if len(d) else None,
            "rvol": float(rv.iloc[i]) if len(rv) else None,
            "bull_candle": bool(bull_candle.iloc[i]) if len(bull_candle) else False,
            "bear_candle": bool(bear_candle.iloc[i]) if len(bear_candle) else False,
            "bb_middle": float(ma.iloc[i]) if len(ma) else None,
            "bb_upper": float(bb_u.iloc[i]) if len(bb_u) else None,
            "bb_lower": float(bb_l.iloc[i]) if len(bb_l) else None,
            "fresh_window_bars": int(self.expiration_bars),
            "confidence": float(confidence),
        }
        return long_ok, short_ok, row
