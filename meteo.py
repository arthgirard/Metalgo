import requests
import time

LAT = 45.183
LON = -73.417

# Cache to prevent rate-limiting and UI lag during rush hours
_weather_cache = {"timestamp": 0, "condition": "Indisponible", "factor": 1.0}
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
    global _weather_cache
    current_time = time.time()
    
    # Return cached weather if still valid
    if current_time - _weather_cache["timestamp"] < CACHE_DURATION_SECONDS:
        return _weather_cache["condition"], _weather_cache["factor"]

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=weather_code&timezone=America%2FNew_York"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        code = response.json()['current']['weather_code']
        condition, factor = interpret_weather_code(code)
        
        # Update cache
        _weather_cache.update({"timestamp": current_time, "condition": condition, "factor": factor})
        return condition, factor
        
    except Exception as e:
        print(f"Weather API Error: {e}")
        return "Indisponible", 1.0

def get_weekly_forecast():
    # Fetches weather codes for the next 8 days
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=weather_code&timezone=America%2FNew_York&forecast_days=8"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        daily = response.json().get('daily', {})
        return [
            {"date": d, "code": c, "description": interpret_weather_code(c)[0]}
            for d, c in zip(daily.get('time', []), daily.get('weather_code', []))
        ]
    except Exception as e:
        print(f"Forecast API Error: {e}")
        return []
