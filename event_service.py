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

def get_special_event(date_obj):
    """
    Returns (Event Name, Multiplier).
    """
    # 1. Fixed Dates
    date_key = (date_obj.month, date_obj.day)
    if date_key in FIXED_EVENTS:
        return FIXED_EVENTS[date_key]

    # 2. Mobile Dates
    # Easter
    easter_date = easter(date_obj.year)
    if date_obj == easter_date:
        return "🐰 Pâques", 1.6
    
    if date_obj == easter_date - timedelta(days=1):
        return "🐰 Samedi de Pâques", 1.5

    # Super Bowl (2nd Sunday of Feb)
    if date_obj.month == 2 and date_obj.weekday() == 6:
        feb_first = date(date_obj.year, 2, 1)
        offset = (6 - feb_first.weekday() + 7) % 7
        first_sunday = feb_first + timedelta(days=offset)
        super_bowl = first_sunday + timedelta(weeks=1)
        
        if date_obj == super_bowl:
            return "🏈 Super Bowl", 1.5

    # 3. Generic Holidays
    if date_obj in qc_holidays:
        holiday_name = qc_holidays.get(date_obj)
        # Avoid duplicates if Christmas/New Year handled above
        if "Noël" not in holiday_name and "Jour de l'An" not in holiday_name:
            return f"🎉 {holiday_name}", 1.2

    return None, 1.0