import os
from dotenv import load_dotenv
load_dotenv()

TD_API_KEY = os.getenv("TD_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAU/USD,USD/JPY").split(",") if s.strip()]

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
STO_K = int(os.getenv("STO_K", "14"))
STO_D = int(os.getenv("STO_D", "3"))

SIGNAL_EXPIRATION_BARS = int(os.getenv("SIGNAL_EXPIRATION_BARS", "3"))

SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "true").lower() in ("1","true","yes","y")
SUMMARY_UTC_HOUR = int(os.getenv("SUMMARY_UTC_HOUR", "22"))
SUMMARY_MIN_SIGNALS = int(os.getenv("SUMMARY_MIN_SIGNALS", "0"))
LOG_PATH = os.getenv("LOG_PATH", "signals_log.csv")

NEWS_WINDOW_MINS = int(os.getenv("NEWS_WINDOW_MINS", "30"))
