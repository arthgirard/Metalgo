import requests
import time

LAT = 45.183
LON = -73.417

# Neutral fallback (°C) used only when the API has never returned a reading
# yet. Chosen to be unremarkable so it doesn't nudge predictions either way.
FALLBACK_TEMPERATURE = 15.0

# Cache to prevent rate-limiting and UI lag during rush hours
_weather_cache = {"timestamp": 0, "condition": "Indisponible", "factor": 1.0, "temperature": None}
CACHE_DURATION_SECONDS = 600  # 10 minutes

def interpret_weather_code(code):
    # Translates WMO code to (Description, Sales Multiplier Factor)
    if code in [0, 1]: return "Ensoleillé", 1.1 if code == 1 else 1.2
    if code in [2, 3]: return "Variable" if code == 2 else "Nuageux", 1.0
    if 45 <= code <= 48: return "Brouillard", 0.9
    if 51 <= code <= 67: return "Pluie", 0.7
    if 80 <= code <= 82: return "Averses", 0.7
    if 71 <= code <= 77: return "Neige", 0.6
    if code >= 95: return "Orage", 0.5 
    return "Variable", 1.0

def get_current_weather():
    """
    Returns (condition: str, factor: float, temperature: float | None).
    temperature is in °C; None only if it has never been successfully fetched
    since the process started.
    """
    global _weather_cache
    current_time = time.time()

    # Return cached weather if still valid
    if current_time - _weather_cache["timestamp"] < CACHE_DURATION_SECONDS:
        return _weather_cache["condition"], _weather_cache["factor"], _weather_cache["temperature"]

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=weather_code,temperature_2m&timezone=America%2FNew_York"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        current = response.json()['current']
        code = current['weather_code']
        temperature = current.get('temperature_2m')
        condition, factor = interpret_weather_code(code)

        # Update cache
        _weather_cache.update({"timestamp": current_time, "condition": condition, "factor": factor, "temperature": temperature})
        return condition, factor, temperature

    except Exception as e:
        print(f"Weather API Error: {e}")
        # Keep serving the last known temperature instead of losing it outright;
        # only the condition/factor degrade to the "unavailable" default.
        return "Indisponible", 1.0, _weather_cache["temperature"]

def get_weekly_forecast():
    # Fetches weather codes and mean temperature for the next 8 days.
    # Mean, not max: the model is trained on the average of each day's
    # actual logged temperatures (see train_model.py), not its peak, so
    # feeding it the day's max here would be a value it never saw an
    # equivalent of during training.
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=weather_code,temperature_2m_mean&timezone=America%2FNew_York&forecast_days=8"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        daily = response.json().get('daily', {})
        codes = daily.get('weather_code', [])
        temps = daily.get('temperature_2m_mean', [])
        return [
            {
                "date": d,
                "code": c,
                "description": interpret_weather_code(c)[0],
                "temperature": (temps[i] if i < len(temps) else None),
            }
            for i, (d, c) in enumerate(zip(daily.get('time', []), codes))
        ]
    except Exception as e:
        print(f"Forecast API Error: {e}")
        return []
