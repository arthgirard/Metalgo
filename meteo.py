import threading
import time
from datetime import datetime
from collections import Counter
from zoneinfo import ZoneInfo

import requests

import shop_schedule as sched

LAT = 45.183
LON = -73.417

# The shop's wall clock. Every timestamp in this project is shop-local, so
# the process timezone (which is whatever the host happens to be set to, and
# is UTC on most deployments) must never be allowed to leak into the data.
SHOP_TZ = ZoneInfo("America/Toronto")

# Open-Meteo wants an IANA name and returns naive local timestamps for it.
API_TIMEZONE = "America/Toronto"

# Neutral fallback (°C) used only when the API has never returned a reading
# yet. Chosen to be unremarkable so it doesn't nudge predictions either way.
FALLBACK_TEMPERATURE = 15.0

CACHE_DURATION_SECONDS = 600        # current conditions: 10 minutes
HOURLY_CACHE_SECONDS = 1800         # today's hourly curve: 30 minutes
WEEKLY_CACHE_SECONDS = 1800         # 7-day outlook: 30 minutes

# Score the model is trained on when the weather is genuinely unknown.
NEUTRAL_WEATHER_SCORE = 1

_lock = threading.Lock()
_weather_cache = {"timestamp": 0, "condition": "Indisponible", "factor": 1.0, "temperature": None}
_hourly_cache = {"timestamp": 0, "date": None, "hours": {}}
_weekly_cache = {"timestamp": 0, "date": None, "forecast": []}


def now_local():
    """Current time on the shop's wall clock, as a naive datetime.

    Naive on purpose: log timestamps are stored as naive local strings and
    compared with SQLite's `date()`/`strftime()`, which have no timezone
    concept. Anchoring to SHOP_TZ first and dropping the offset afterwards
    keeps that storage format while making the value independent of the
    host's own timezone.
    """
    return datetime.now(SHOP_TZ).replace(tzinfo=None)


def today_local():
    """Today's date on the shop's wall clock."""
    return now_local().date()


# ---------------------------------------------------------------------------
# Weather interpretation
# ---------------------------------------------------------------------------
# One table, three consumers. The sales factor (used by the naive fallback)
# and the model score (used as a training feature) used to live in two
# separate mappings — a factor->score function in app.py and a French
# label->score dict in train_model.py — that had to be kept in sync by hand.
# Editing the WMO bands without touching both would have silently mislabelled
# every training row. They are derived from this single table now.

def _band(code):
    """Maps a WMO weather code to (display label, sales factor, model score)."""
    code = int(code)
    if code in (0, 1):
        return "Ensoleillé", (1.2 if code == 0 else 1.1), 2
    if code == 2:
        return "Variable", 1.0, 1
    if code == 3:
        return "Nuageux", 1.0, 1
    if 45 <= code <= 48:
        return "Brouillard", 0.9, 1
    if 51 <= code <= 67:
        return "Pluie", 0.7, 0
    if 71 <= code <= 77:
        return "Neige", 0.6, 0
    if 80 <= code <= 82:
        return "Averses", 0.7, 0
    if code >= 95:
        return "Orage", 0.5, 0
    return "Variable", 1.0, 1


def interpret_weather_code(code):
    """Translates a WMO code to (description, sales multiplier factor)."""
    label, factor, _ = _band(code)
    return label, factor


def weather_score_from_code(code):
    """The model's `weather_score` feature (0 = bad, 1 = neutral, 2 = good)."""
    return _band(code)[2]


# Every label this table can produce, mapped back to its score. Built by
# sweeping the band function so it can never disagree with it.
LABEL_TO_SCORE = {}
for _code in range(0, 100):
    _label, _, _score = _band(_code)
    LABEL_TO_SCORE.setdefault(_label, _score)

# Labels written by older versions of the app that no longer occur, kept so
# historical rows in `logs.meteo_summary` stay usable for training.
LEGACY_LABEL_TO_SCORE = {"Orages": 0}

# Labels that mean "we don't know", as opposed to "the weather was average".
# These deliberately resolve to None rather than a neutral score so training
# can impute them from the rest of the day instead of inventing a fact.
UNKNOWN_LABELS = frozenset({"Indisponible", "Inconnu", ""})


def weather_score_from_label(label):
    """
    Score for a stored French weather label, or None if it is unknown.

    None means "no observation", not "average weather" — the caller decides
    how to impute it.
    """
    if label is None:
        return None
    label = str(label).strip()
    if label in UNKNOWN_LABELS:
        return None
    if label in LABEL_TO_SCORE:
        return LABEL_TO_SCORE[label]
    return LEGACY_LABEL_TO_SCORE.get(label)


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------

def get_current_weather():
    """
    Returns (condition: str, factor: float, temperature: float | None).

    temperature is in °C; None only if it has never been successfully fetched
    since the process started.
    """
    with _lock:
        if time.time() - _weather_cache["timestamp"] < CACHE_DURATION_SECONDS:
            return _weather_cache["condition"], _weather_cache["factor"], _weather_cache["temperature"]
        last_known_temperature = _weather_cache["temperature"]

    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
               f"&current=weather_code,temperature_2m&timezone={API_TIMEZONE}")
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        current = response.json()['current']
        temperature = current.get('temperature_2m')
        condition, factor = interpret_weather_code(current['weather_code'])

        with _lock:
            _weather_cache.update({"timestamp": time.time(), "condition": condition,
                                   "factor": factor, "temperature": temperature})
        return condition, factor, temperature

    except Exception as e:
        print(f"Weather API Error: {e}")
        # Keep serving the last known temperature instead of losing it outright;
        # only the condition/factor degrade to the "unavailable" default.
        return "Indisponible", 1.0, last_known_temperature


def _fetch_hourly(forecast_days):
    """Raw hourly weather_code/temperature_2m series, as (times, codes, temps)."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
           f"&hourly=weather_code,temperature_2m&timezone={API_TIMEZONE}"
           f"&forecast_days={forecast_days}")
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    hourly = response.json().get('hourly', {})
    return hourly.get('time', []), hourly.get('weather_code', []), hourly.get('temperature_2m', [])


def get_today_hourly_weather():
    """
    Today's weather hour by hour: {hour: {"code", "score", "temperature"}}.

    The intraday forecast used to apply the weather observed *right now* to
    every remaining hour of the day, so a sunny 10h would happily project a
    sunny 16h through a forecast rain band. Predicting per hour fixes that;
    an empty dict just means the caller falls back to current conditions.
    """
    today = today_local().strftime('%Y-%m-%d')
    with _lock:
        fresh = (time.time() - _hourly_cache["timestamp"] < HOURLY_CACHE_SECONDS
                 and _hourly_cache["date"] == today)
        if fresh:
            return dict(_hourly_cache["hours"])

    try:
        times, codes, temps = _fetch_hourly(forecast_days=1)
        hours = {}
        for t, code, temp in zip(times, codes, temps):
            dt = datetime.fromisoformat(t)
            if dt.strftime('%Y-%m-%d') != today or code is None:
                continue
            hours[dt.hour] = {"code": int(code),
                              "score": weather_score_from_code(code),
                              "temperature": temp}
        with _lock:
            _hourly_cache.update({"timestamp": time.time(), "date": today, "hours": hours})
        return dict(hours)

    except Exception as e:
        print(f"Hourly forecast API Error: {e}")
        return {}


def get_weekly_forecast():
    """
    Daily outlook for the next 8 days, aggregated over each day's own
    selling hours (which differ on Thursdays and Fridays).

    Each entry is {"date", "code", "description", "score", "temperature"},
    ordered by date. The representative code is the mode across the day's
    business hours and the temperature is their mean, matching the day-level
    aggregation training uses.
    """
    today = today_local().strftime('%Y-%m-%d')
    with _lock:
        # Keyed on the date as well as the age: a cache filled at 23:50 and
        # still "fresh" at 00:05 would otherwise hand back a list whose first
        # entry is yesterday, and callers select the upcoming days by date.
        fresh = (time.time() - _weekly_cache["timestamp"] < WEEKLY_CACHE_SECONDS
                 and _weekly_cache["date"] == today
                 and _weekly_cache["forecast"])
        if fresh:
            return list(_weekly_cache["forecast"])

    try:
        times, codes, temps = _fetch_hourly(forecast_days=8)

        daily = {}
        for t, code, temp in zip(times, codes, temps):
            dt = datetime.fromisoformat(t)
            # Aggregate over the hours this specific weekday is actually open,
            # rather than one hardcoded 10h-17h window for every day. Closed
            # days still get a summary over the default window so callers can
            # render a "Fermé" card that shows the weather anyway.
            window = sched.selling_hours(dt.weekday()) or range(sched.OPEN_HOUR, sched.CLOSE_HOUR)
            if dt.hour not in window:
                continue
            if code is None:
                continue
            day = daily.setdefault(dt.strftime('%Y-%m-%d'), {'codes': [], 'temps': []})
            day['codes'].append(code)
            if temp is not None:
                day['temps'].append(temp)

        forecast = []
        for date_str in sorted(daily):
            data = daily[date_str]
            if not data['codes']:
                continue
            most_common_code = Counter(data['codes']).most_common(1)[0][0]
            mean_temp = round(sum(data['temps']) / len(data['temps']), 1) if data['temps'] else None
            forecast.append({
                "date": date_str,
                "code": most_common_code,
                "description": interpret_weather_code(most_common_code)[0],
                "score": weather_score_from_code(most_common_code),
                "temperature": mean_temp,
            })

        with _lock:
            _weekly_cache.update({"timestamp": time.time(), "date": today, "forecast": forecast})
        return list(forecast)

    except Exception as e:
        print(f"Forecast API Error: {e}")
        return []
