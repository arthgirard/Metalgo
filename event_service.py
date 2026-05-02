import requests
import holidays
from datetime import date, timedelta
from dateutil.easter import easter

# QC Holidays (French for display)
qc_holidays = holidays.CA(subdiv='QC', language='fr')

# Fixed events (Month, Day) -> (Name, Multiplier)
FIXED_EVENTS = {
    (2, 14):  ("💖 St-Valentin", 1.4),
    (6, 24):  ("⚜️ St-Jean-Baptiste", 1.5),
    (7, 1):   ("🇨🇦 Fête du Canada", 1.3),
    (10, 31): ("🎃 Halloween", 1.3),
    (12, 24): ("🎄 Veille de Noël", 2.0),
    (12, 31): ("🎉 Sylvestre", 1.8),
}

# Cache for NHL schedules to avoid redundant API calls
_nhl_cache = {}

def get_season_string(date_obj):
    # NHL season usually starts in October. 
    # If month >= 8, season is current_year + next_year. Otherwise, prev_year + current_year.
    if date_obj.month >= 8:
        return f"{date_obj.year}{date_obj.year + 1}"
    else:
        return f"{date_obj.year - 1}{date_obj.year}"

def get_game_info(date_obj):
    # Returns tuple (is_game_day, is_playoff_game)
    season = get_season_string(date_obj)
    
    if season not in _nhl_cache:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/MTL/{season}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                _nhl_cache[season] = response.json().get('games', [])
            else:
                _nhl_cache[season] = []
        except Exception as e:
            print(f"Error fetching NHL schedule: {e}")
            _nhl_cache[season] = []

    date_str = date_obj.strftime("%Y-%m-%d")
    for game in _nhl_cache[season]:
        if game.get('gameDate') == date_str:
            # gameType: 2 is Regular Season, 3 is Playoffs
            is_playoff = 1 if game.get('gameType') == 3 else 0
            return 1, is_playoff
    
    return 0, 0

def get_special_event(date_obj):
    # Returns (Event Name, Multiplier)
    event_name = None
    multiplier = 1.0

    # 1. Fixed Dates
    date_key = (date_obj.month, date_obj.day)
    if date_key in FIXED_EVENTS:
        event_name, multiplier = FIXED_EVENTS[date_key]

    # 2. Mobile Dates
    if not event_name:
        easter_date = easter(date_obj.year)
        if date_obj == easter_date:
            event_name, multiplier = "🐰 Pâques", 1.6
        elif date_obj == easter_date - timedelta(days=1):
            event_name, multiplier = "🐰 Samedi de Pâques", 1.5
        elif date_obj.month == 2 and date_obj.weekday() == 6:
            feb_first = date(date_obj.year, 2, 1)
            offset = (6 - feb_first.weekday() + 7) % 7
            first_sunday = feb_first + timedelta(days=offset)
            super_bowl = first_sunday + timedelta(weeks=1)
            
            if date_obj == super_bowl:
                event_name, multiplier = "🏈 Super Bowl", 1.5

    # 3. Generic Holidays
    if not event_name and date_obj in qc_holidays:
        holiday_name = qc_holidays.get(date_obj)
        # Avoid duplicates if Christmas/New Year handled above
        if "Noël" not in holiday_name and "Jour de l'An" not in holiday_name:
            event_name, multiplier = f"🎉 {holiday_name}", 1.2

    # 4. Hockey Games Overlay
    is_game, is_playoff = get_game_info(date_obj)
    if is_game:
        hockey_str = "🏒 Match du CH (Séries)" if is_playoff else "🏒 Match du CH"
        
        # Append hockey event if another event already exists
        if event_name:
            event_name = f"{event_name} + {hockey_str}"
            multiplier = multiplier * (1.3 if is_playoff else 1.1)
        else:
            event_name = hockey_str
            multiplier = 1.3 if is_playoff else 1.1

    return event_name, multiplier
