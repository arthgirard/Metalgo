"""
Single source of truth for the shop's opening schedule and for the two
weekday conventions this codebase has to live with.

Why this module exists
----------------------
The opening hours used to be re-derived in four places (`is_shop_open`,
`get_prediction`, `forecast_week_endpoint` in app.py, and the training grid
in train_model.py), each with its own literal `10` / `17` / `18`. They
happened to agree, but the hour windows used to *display* data did not:
the stats chart covered 10h-18h while the training grid covered 10h-16h on
most days, so real sales logged at 9h and 17h were silently discarded from
training. Everything now goes through the helpers below.

Weekday conventions
-------------------
Both of these are load-bearing and neither can be dropped:

  * Python  (`datetime.weekday()`)  -> Monday=0 ... Sunday=6
  * SQLite  (`strftime('%w', ...)`) -> Sunday=0 ... Saturday=6

`daily_snapshots.weekday` stores the Python convention. The ML model's
`weekday` feature uses the SQLite convention, because training reads it
straight out of SQL and changing it would invalidate every existing model.
Rather than pick a winner, every conversion goes through
`py_weekday_from_sql` / `sql_weekday_from_py` so the two can't silently
drift apart again.
"""

OPEN_HOUR = 10

# Regular closing hour, and the later one used on Thursdays and Fridays.
CLOSE_HOUR = 17
CLOSE_HOUR_LATE = 18

# Python convention (Monday=0) throughout this block.
CLOSED_WEEKDAY = 0                      # Monday: shop closed all day
LATE_WEEKDAYS = frozenset({3, 4})       # Thursday, Friday

# The widest window the shop can ever be open, used for display grids.
MAX_CLOSE_HOUR = CLOSE_HOUR_LATE

FORMATS = ("250g", "1kg", "2kg")

# French weekday names, indexed by the Python convention (Monday=0).
WEEKDAY_NAMES_FR = ("Lundi", "Mardi", "Mercredi", "Jeudi",
                    "Vendredi", "Samedi", "Dimanche")


def py_weekday_from_sql(sql_weekday):
    """Convert a SQLite `strftime('%w')` weekday (Sun=0) to Python's (Mon=0)."""
    return (int(sql_weekday) - 1) % 7


def sql_weekday_from_py(py_weekday):
    """Convert a Python weekday (Mon=0) to SQLite's `strftime('%w')` (Sun=0)."""
    return (int(py_weekday) + 1) % 7


def is_closed_day(py_weekday):
    """True if the shop is shut all day on this weekday (Python convention)."""
    return int(py_weekday) == CLOSED_WEEKDAY


def close_hour(py_weekday):
    """
    Closing hour for a weekday in the Python convention.

    This is the hour the shop *shuts*, not a selling hour: a close_hour of 17
    means the last selling hour is the 16h-17h slot.
    """
    return CLOSE_HOUR_LATE if int(py_weekday) in LATE_WEEKDAYS else CLOSE_HOUR


def close_hour_sql(sql_weekday):
    """Same as `close_hour`, for a weekday in the SQLite `%w` convention."""
    return close_hour(py_weekday_from_sql(sql_weekday))


def selling_hours(py_weekday):
    """
    The hours the shop actually sells during, as a range.

    Empty on closing days, and stops before `close_hour` because the closing
    hour is not itself a selling slot.
    """
    if is_closed_day(py_weekday):
        return range(0)
    return range(OPEN_HOUR, close_hour(py_weekday))


def is_open_at(dt):
    """True if the shop is open at the given (shop-local) datetime."""
    return dt.hour in selling_hours(dt.weekday())


def hour_buckets(py_weekday, observed_hours=()):
    """
    Sorted hour buckets for charts and for the training grid.

    Starts from the scheduled selling hours, then folds in any hour that
    actually has data. Sales logged outside opening hours are real — the shop
    serves a customer who walked in at 9h55, or finishes one at 17h05 — and
    dropping them made training quietly disagree with the charts. Including
    them costs an extra empty column only on the days they occur.
    """
    hours = set(selling_hours(py_weekday))
    hours.update(int(h) for h in observed_hours)
    return sorted(hours)


def hour_buckets_sql(sql_weekday, observed_hours=()):
    """Same as `hour_buckets`, for a weekday in the SQLite `%w` convention."""
    return hour_buckets(py_weekday_from_sql(sql_weekday), observed_hours)
