import time, csv, os, json
from datetime import datetime, timezone

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sleep_secs(secs: int):
    try:
        time.sleep(secs)
    except KeyboardInterrupt:
        pass

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def pct_change(a, b):
    return (a - b) / b if b != 0 else 0.0

def append_csv_row(path: str, row: dict, field_order=None):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    write_header = not os.path.exists(path)
    fields = field_order or list(row.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)

def read_csv_df(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# Simple persistent store for open signals (to track TP/SL results)
def load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
