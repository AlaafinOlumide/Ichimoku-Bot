import requests
import pandas as pd
from typing import Optional

BASE_URL = "https://api.twelvedata.com/time_series"

def fetch_series(symbol: str, interval: str, apikey: str, outputsize: int = 300) -> Optional[pd.DataFrame]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": apikey,
        "outputsize": outputsize,
        "format": "JSON",
        "order": "ASC"
    }
    resp = requests.get(BASE_URL, params=params, timeout=20)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
