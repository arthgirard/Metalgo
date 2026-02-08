import requests

LAT = 45.183
LON = -73.417

def interpret_weather_code(code):
    """
    Translates WMO code (0-99) to (Description, Sales Factor).
    Description remains in French for frontend display.
    """
    # 0: Clear sky -> Excellent sales (+20%)
    if code == 0: 
        return "Ensoleillé", 1.2
    
    # 1-3: Partly cloudy -> Standard
    if 1 <= code <= 3: 
        return "Nuageux", 1.0 
    
    # 45-48: Fog -> Slight decrease (-10%)
    if 45 <= code <= 48: 
        return "Brouillard", 0.9
    
    # 51-67: Rain/Drizzle -> Significant decrease (-30%)
    if 51 <= code <= 67: 
        return "Pluie", 0.7
    
    # 80-82: Showers -> Significant decrease (-30%)
    if 80 <= code <= 82: 
        return "Averses", 0.7
    
    # 71-77: Snow -> Heavy decrease (-40%)
    if 71 <= code <= 77: 
        return "Neige", 0.6
    
    # 95+: Thunderstorm -> Heavy decrease (-50%)
    if code >= 95: 
        return "Orage", 0.5 
    
    return "Variable", 1.0

def get_current_weather():
    """
    Fetches current weather and returns (Description, Impact Factor).
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=weather_code&timezone=America%2FNew_York"
        response = requests.get(url, timeout=5) 
        data = response.json()
        
        code = data['current']['weather_code']
        return interpret_weather_code(code)
        
    except Exception as e:
        print(f"Error in Weather Module (Current): {e}")
        return "Indisponible", 1.0

def get_weekly_forecast():
    """
    Fetches weather codes for the next 8 days.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=weather_code&timezone=America%2FNew_York&forecast_days=8"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        results = []
        daily = data.get('daily', {})
        dates = daily.get('time', [])
        codes = daily.get('weather_code', [])
        
        for i in range(len(dates)):
            code = codes[i]
            desc, _ = interpret_weather_code(code)
            
            results.append({
                "date": dates[i],
                "code": code,
                "description": desc
            })
            
        return results
        
    except Exception as e:
        print(f"Error in Weather Module (Forecast): {e}")
        return []