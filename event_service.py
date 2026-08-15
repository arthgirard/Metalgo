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


# Event detection
def _fixed_event(date_obj):
    """The fixed-calendar event on this date, as (key, name, default), or None."""
    fixed_key = (date_obj.month, date_obj.day)
    if fixed_key not in FIXED_EVENTS:
        return None
    name, default_mult = FIXED_EVENTS[fixed_key]
    return f"fixed_{date_obj.month:02d}-{date_obj.day:02d}", name, default_mult


def _mobile_event(date_obj):
    """The moving-date event on this date, as (key, name, default), or None."""
    easter_date = easter(date_obj.year)
    if date_obj == easter_date:
        return "mobile_easter", "🐰 Pâques", 1.6
    if date_obj == easter_date - timedelta(days=1):
        return "mobile_easter_saturday", "🐰 Samedi de Pâques", 1.5

    # Super Bowl: second Sunday of February
    if date_obj.month == 2 and date_obj.weekday() == 6:
        feb_first = date(date_obj.year, 2, 1)
        offset = (6 - feb_first.weekday() + 7) % 7
        first_sunday = feb_first + timedelta(days=offset)
        if date_obj == first_sunday + timedelta(weeks=1):
            return "mobile_super_bowl", "🏈 Super Bowl", 1.5
    return None


def _holiday_event(date_obj):
    """
    A generic Quebec public holiday, as (key, name, default), or None.

    There used to be a name filter here dropping any holiday containing
    "Noël" or "Jour de l'An", meant to stop Christmas Eve and New Year's Eve
    from producing a second key alongside their fixed events. It could never
    have done that — 24 and 31 December are not Quebec public holidays, so no
    duplicate was possible — and what it actually did was erase Christmas Day
    itself, whose holiday name is "Jour de Noël". Boxing-day-week demand was
    being forecast as an ordinary Friday.

    New Year's Day survived only by accident: the library spells it "Jour de
    l'an" with a lowercase A and the filter tested for "Jour de l'An".

    Duplicates are now prevented structurally instead — get_all_events only
    consults this function when nothing more specific matched.
    """
    if date_obj not in qc_holidays:
        return None
    holiday_name = qc_holidays.get(date_obj, "")
    return "qc_holiday", f"🎉 {holiday_name or 'Jour Férié'}", 1.2


def get_all_events(date_obj):
    """
    Every non-NHL event on this date, most specific first.

    Fixed and mobile events can genuinely coincide — the 2027 Super Bowl falls
    on Saint-Valentin — and the old single-return chain made the second one
    vanish completely, so that day would have been forecast as an ordinary
    Valentine's. A generic public holiday is only reported when nothing more
    specific matched, since Saint-Jean is already both and naming it twice
    helps nobody.
    """
    events = [event for event in (_fixed_event(date_obj), _mobile_event(date_obj)) if event]
    if not events:
        holiday = _holiday_event(date_obj)
        if holiday:
            events.append(holiday)
    return events


def get_event_key(date_obj):
    """
    Returns a canonical, stable string key for the non-NHL special event on
    date_obj, or None if no special event occurs.

    These keys are stored in the daily_snapshots table and serve as the
    primary identifier when computing per-event learned multipliers. When two
    events coincide the most specific one is the key, so the keys already in
    daily_snapshots keep their meaning.

    NHL games use separate keys ("nhl_regular" / "nhl_playoff") handled
    internally by _get_learned_multiplier.
    """
    events = get_all_events(date_obj)
    return events[0][0] if events else None


# Learned multiplier engine
def _get_learned_multiplier(event_key, default_multiplier, db_path, is_nhl_key=False):
    """
    Queries the daily_snapshots table to derive a data-driven sales multiplier
    for the given event_key, then blends it with the hardcoded default prior.
    """
    conn = None
    try:
        # busy_timeout so a concurrent snapshot/retrain write can't turn this
        # into a "database is locked" error on a live prediction request.
        conn = sqlite3.connect(db_path, timeout=10.0)
        c = conn.cursor()

        # --- Fetch event-day statistics ---
        if is_nhl_key:
            is_playoff = 1 if event_key == "nhl_playoff" else 0
            # `event_key IS NULL` matters: the baseline below excludes both
            # holidays and game days, so counting a Saint-Jean-that-was-also-a
            # -game-day here would charge the whole holiday uplift to the
            # Canadiens and inflate the NHL multiplier.
            c.execute("""
                SELECT AVG(total_250g + total_1kg + total_2kg),
                       COUNT(*),
                       GROUP_CONCAT(weekday)
                FROM daily_snapshots
                WHERE is_nhl_game = 1 AND is_nhl_playoff = ? AND event_key IS NULL
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

    finally:
        if conn is not None:
            conn.close()


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

        These are returned SEPARATELY because the model knows about exactly
        one of them:

        * NHL — the RandomForest is trained with is_game_day /
          is_playoff_game as features and has learned the effect from ~10
          recorded game days. Callers going through the model must NOT
          multiply nhl_multiplier back in; that double-counts the boost.

        * Calendar events — deliberately NOT a model feature. A single binary
          is_special_event lumped Saint-Jean (≈2x) together with Halloween
          (≈1.3x), and with only a couple of event days ever recorded the
          model could never separate them anyway. The Bayesian blend below is
          far more sample-efficient at this data volume, so callers going
          through the model SHOULD apply base_multiplier themselves.

        A caller that is not going through the model at all (the naive linear
        fallback) has no other way of knowing today is a game day, and so
        applies both: base_multiplier * nhl_multiplier.
    """
    event_name = None
    base_multiplier = 1.0
    events = get_all_events(date_obj)

    # --- Non-NHL event ---
    if events:
        event_key = events[0][0]
        event_name = " + ".join(name for _, name, _ in events)
        # The strongest prior wins rather than the product: two events landing
        # on the same day don't multiply demand, the bigger occasion simply
        # dominates it. Multiplying a 1.5 Super Bowl by a 1.4 Saint-Valentin
        # would forecast a 2.1x day nobody has ever observed.
        default_mult = max(default for _, _, default in events)

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
