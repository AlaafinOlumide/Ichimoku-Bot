# Very lightweight ForexFactory calendar scraper (High impact only)
# Note: FF has dynamic content; this parser aims to catch common server-side HTML.
# If the structure changes, adjust the regex markers below.
import re, requests, datetime
from typing import List, Dict

FF_URL = "https://www.forexfactory.com/calendar"

def fetch_red_events_utc() -> List[Dict]:
    try:
        html = requests.get(FF_URL, timeout=20).text
    except Exception:
        return []
    # Basic parse: find rows with 'High' impact (red). ForexFactory often uses 'impact' icons or 'high' class
    # Extract date context and time, currency, title, impact.
    # WARNING: This is heuristic; refine if needed.
    events = []
    # split by day blocks
    day_blocks = re.split(r'<tr class="calendar__row--day.*?>', html)
    current_date = None
    for block in day_blocks:
        # find date like data-event-day or header: e.g., data-day="2025-09-08"
        mdate = re.search(r'data-day="(\d{4}-\d{2}-\d{2})"', block) or re.search(r'(\d{4}-\d{2}-\d{2})', block)
        if mdate:
            current_date = mdate.group(1)
        # find rows with high impact
        for m in re.finditer(r'<tr[^>]*?calendar__row.*?>.*?</tr>', block, re.S):
            row = m.group(0)
            if not re.search(r'impact(.*?)(High|HIGH|red)', row):
                continue
            # currency
            cur = re.search(r'currency[^>]*>([A-Z]{3})<', row)
            # title
            title = re.search(r'title[^>]*>(.*?)<', row) or re.search(r'event.*?>(.*?)<', row)
            # time (in site local, usually ET; many rows have data-time)
            dtime = re.search(r'data-time="(\d{2}:\d{2})"', row) or re.search(r'>(\d{1,2}:\d{2})<', row)
            if not (cur and title and dtime and current_date):
                continue
            hh, mm = dtime.group(1).split(':')
            # naive: assume time is in UTC on server (FF displays visitor-local; we need UTC. Some rows include data-tz)
            # If 'data-tz' with UTC exists, better. Otherwise we treat it as UTC for safety.
            dt_utc = f"{current_date} {hh}:{mm}Z"
            events.append({
                'date': current_date,
                'time_utc': dt_utc,
                'currency': cur.group(1),
                'title': re.sub(r'<.*?>','', title.group(1)).strip(),
                'impact': 'High'
            })
    return events

AFFECT_MAP = {
    # Base currency mapping for our pairs
    'XAU/USD': ['USD'],
    'USD/JPY': ['USD','JPY'],
}

def relevant_red_news(symbol: str, window_mins: int) -> List[Dict]:
    evs = fetch_red_events_utc()
    out = []
    try:
        from datetime import datetime
        now = datetime.utcnow()
    except Exception:
        return out
    affects = AFFECT_MAP.get(symbol.upper(), [])
    for e in evs:
        if e['currency'] not in affects:
            continue
        # parse e['time_utc'] as UTC
        try:
            dt = datetime.strptime(e['time_utc'].replace('Z',''), '%Y-%m-%d %H:%M')
        except Exception:
            continue
        delta = (dt - now).total_seconds() / 60.0
        # within +/- window
        if -window_mins <= delta <= window_mins:
            e['mins_from_now'] = int(round(delta))
            out.append(e)
    return out
