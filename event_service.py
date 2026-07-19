import sqlite3
import requests
import holidays
from datetime import date, timedelta
from dateutil.easter import easter

# --- Quebec public holidays (French display names) ---
qc_holidays = holidays.CA(subdiv='QC', language='fr')

# ---------------------------------------------------------------------------
# Static event definitions
# ---------------------------------------------------------------------------

# Fixed calendar dates: (month, day) → (display_name, default_multiplier)
FIXED_EVENTS = {
    (2, 14):  ("💖 St-Valentin",       1.4),
    (6, 24):  ("⚜️ St-Jean-Baptiste",  2.0),
    (7, 1):   ("🇨🇦 Fête du Canada",   1.3),
    (10, 31): ("🎃 Halloween",          1.3),
    (12, 24): ("🎄 Veille de Noël",    1.2),
    (12, 31): ("🎉 Sylvestre",          1.5),
}

# Default NHL multipliers, used as priors before data is available.
NHL_DEFAULT_MULTIPLIERS = {
    "nhl_regular": 1.1,
    "nhl_playoff": 1.3,
}

# ---------------------------------------------------------------------------
# Bayesian smoothing factor
# ---------------------------------------------------------------------------
SMOOTHING_FACTOR = 3

# In-memory NHL schedule cache to avoid redundant API calls per process lifetime
_nhl_cache = {}


# NHL helpers
def get_season_string(date_obj):
    """
    Returns the NHL season identifier for a given date.
    The season is considered to start in August (e.g. Aug 2024 → '20242025').
    """
    if date_obj.month >= 8:
        return f"{date_obj.year}{date_obj.year + 1}"
    return f"{date_obj.year - 1}{date_obj.year}"


def get_game_info(date_obj):
    """
    Checks whether the Canadiens play on date_obj by querying the NHL API.
    Results are cached in memory for the lifetime of the process.

    Returns:
        (is_game_day: int, is_playoff: int)  — both are 0 or 1
    """
    season = get_season_string(date_obj)

    if season not in _nhl_cache:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/MTL/{season}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                _nhl_cache[season] = resp.json().get('games', [])
            else:
                _nhl_cache[season] = []
        except Exception as e:
            print(f"Error fetching NHL schedule: {e}")
            _nhl_cache[season] = []

    date_str = date_obj.strftime("%Y-%m-%d")
    for game in _nhl_cache[season]:
        if game.get('gameDate') == date_str:
            # gameType 2 = Regular Season, 3 = Playoffs
            is_playoff = 1 if game.get('gameType') == 3 else 0
            return 1, is_playoff

    return 0, 0


# Event key
def get_event_key(date_obj):
    """
    Returns a canonical, stable string key for the non-NHL special event on
    date_obj, or None if no special event occurs.

    These keys are stored in the daily_snapshots table and serve as the
    primary identifier when computing per-event learned multipliers.
    NHL games use separate keys ("nhl_regular" / "nhl_playoff") handled
    internally by _get_learned_multiplier.
    """
    # 1. Fixed calendar dates
    fixed_key = (date_obj.month, date_obj.day)
    if fixed_key in FIXED_EVENTS:
        return f"fixed_{date_obj.month:02d}-{date_obj.day:02d}"

    # 2. Mobile dates
    easter_date = easter(date_obj.year)
    if date_obj == easter_date:
        return "mobile_easter"
    if date_obj == easter_date - timedelta(days=1):
        return "mobile_easter_saturday"

    # Super Bowl: second Sunday of February
    if date_obj.month == 2 and date_obj.weekday() == 6:
        feb_first = date(date_obj.year, 2, 1)
        offset = (6 - feb_first.weekday() + 7) % 7
        first_sunday = feb_first + timedelta(days=offset)
        if date_obj == first_sunday + timedelta(weeks=1):
            return "mobile_super_bowl"

    # 3. Generic Quebec public holidays
    #    Christmas Eve and New Year's Eve are handled as fixed events above,
    #    so we explicitly exclude them here to avoid duplicate keys.
    if date_obj in qc_holidays:
        holiday_name = qc_holidays.get(date_obj, "")
        if "Noël" not in holiday_name and "Jour de l'An" not in holiday_name:
            return "qc_holiday"

    return None


# Learned multiplier engine
def _get_learned_multiplier(event_key, default_multiplier, db_path, is_nhl_key=False):
    """
    Queries the daily_snapshots table to derive a data-driven sales multiplier
    for the given event_key, then blends it with the hardcoded default prior.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # --- Fetch event-day statistics ---
        if is_nhl_key:
            is_playoff = 1 if event_key == "nhl_playoff" else 0
            c.execute("""
                SELECT AVG(total_250g + total_1kg + total_2kg),
                       COUNT(*),
                       GROUP_CONCAT(weekday)
                FROM daily_snapshots
                WHERE is_nhl_game = 1 AND is_nhl_playoff = ?
            """, (is_playoff,))
        else:
            c.execute("""
                SELECT AVG(total_250g + total_1kg + total_2kg),
                       COUNT(*),
                       GROUP_CONCAT(weekday)
                FROM daily_snapshots
                WHERE event_key = ?
            """, (event_key,))

        row = c.fetchone()
        event_avg    = row[0]
        n_events     = row[1] if row[1] else 0
        weekdays_raw = row[2]

        # Not enough data — return the prior unchanged
        if n_events == 0 or event_avg is None or weekdays_raw is None:
            conn.close()
            return default_multiplier

        # Determine which weekdays these events occurred on (for a fair baseline)
        weekdays     = list(set(int(w) for w in weekdays_raw.split(',')))
        placeholders = ','.join('?' * len(weekdays))

        # --- Fetch baseline: normal days (no event, no game) on the same weekday(s) ---
        c.execute(f"""
            SELECT AVG(total_250g + total_1kg + total_2kg)
            FROM daily_snapshots
            WHERE event_key IS NULL
              AND is_nhl_game = 0
              AND weekday IN ({placeholders})
        """, weekdays)

        baseline_row = c.fetchone()
        baseline_avg = baseline_row[0] if baseline_row and baseline_row[0] else None
        conn.close()

        if not baseline_avg or baseline_avg == 0:
            # No baseline available yet (e.g. only event days recorded so far)
            return default_multiplier

        # --- Bayesian blend ---
        learned_multiplier = event_avg / baseline_avg
        weight   = n_events / (n_events + SMOOTHING_FACTOR)
        blended  = weight * learned_multiplier + (1 - weight) * default_multiplier

        return round(blended, 3)

    except Exception as e:
        print(f"Error computing learned multiplier for '{event_key}': {e}")
        return default_multiplier


# Public API
def get_special_event(date_obj, db_path=None):
    """
    Returns the special event context for date_obj.

    Args:
        date_obj:        A datetime.date instance.
        db_path (str):   Path to the SQLite DB; None disables learned logic.

    Returns:
        (event_name: str | None, base_multiplier: float, nhl_multiplier: float)

        base_multiplier reflects ONLY calendar/holiday events (Valentine's,
        Halloween, Québec public holidays, etc). nhl_multiplier reflects the
        Canadiens game/playoff boost on its own, or 1.0 if there's no game.

        These are returned SEPARATELY on purpose: the RandomForest model is
        trained with is_game_day / is_playoff_game as features, so it has
        already learned the NHL effect from history. Callers feeding a
        prediction into that model should use base_multiplier alone —
        multiplying nhl_multiplier back in double-counts the game-day boost.
        Only combine them (base_multiplier * nhl_multiplier) for a caller
        that is NOT going through the ML model and has no other way of
        knowing today is a game day (e.g. a naive linear fallback).
    """
    event_name = None
    base_multiplier = 1.0
    event_key  = get_event_key(date_obj)

    # --- Non-NHL event ---
    if event_key:
        # Resolve display name and default multiplier from static definitions
        fixed_key = (date_obj.month, date_obj.day)
        if fixed_key in FIXED_EVENTS:
            event_name, default_mult = FIXED_EVENTS[fixed_key]
        elif event_key == "mobile_easter":
            event_name, default_mult = "🐰 Pâques", 1.6
        elif event_key == "mobile_easter_saturday":
            event_name, default_mult = "🐰 Samedi de Pâques", 1.5
        elif event_key == "mobile_super_bowl":
            event_name, default_mult = "🏈 Super Bowl", 1.5
        elif event_key == "qc_holiday":
            holiday_label = qc_holidays.get(date_obj, "Jour Férié")
            event_name, default_mult = f"🎉 {holiday_label}", 1.2
        else:
            event_name, default_mult = "🎉 Événement", 1.0

        base_multiplier = (
            _get_learned_multiplier(event_key, default_mult, db_path)
            if db_path else default_mult
        )

    # --- NHL game overlay (kept separate from base_multiplier — see docstring) ---
    nhl_multiplier = 1.0
    is_game, is_playoff = get_game_info(date_obj)
    if is_game:
        nhl_key      = "nhl_playoff" if is_playoff else "nhl_regular"
        nhl_default  = NHL_DEFAULT_MULTIPLIERS[nhl_key]
        nhl_label    = "🏒 Match du CH (Séries)" if is_playoff else "🏒 Match du CH"

        nhl_multiplier = (
            _get_learned_multiplier(nhl_key, nhl_default, db_path, is_nhl_key=True)
            if db_path else nhl_default
        )

        # Name stacking is purely cosmetic (for UI display) and independent
        # of how the two multipliers get combined numerically by the caller.
        event_name = f"{event_name} + {nhl_label}" if event_name else nhl_label

    return event_name, base_multiplier, nhl_multiplier
