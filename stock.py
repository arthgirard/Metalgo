"""
What was bagged, what was repacked, and what that says about the day.

The shop bags once in the morning and never restocks, so a format that runs
out simply stops selling. Without knowing what was bagged, that looks
exactly like a quiet afternoon, and the forecast learns "we ran out" as
"nobody wanted it". The records here are what lets a day be recognised as a
sell-out.

Three records, all entered by staff:

  bagging_plan     bags of each format for a date, often saved days ahead
  conversions      2kg (or 1kg) bags opened and repacked into a smaller format
  closing_record   the end-of-day confirmation, with optional leftovers

From them and the logged sales, `format_status` derives the stock and
whether each format sold out. Only a confirmed sell-out (plan + closing +
sales reaching the stock while the shop kept selling other formats) is ever
used as censored demand; anything that does not add up is flagged instead.
"""

from datetime import datetime

import history
import shop_schedule as sched

FORMATS = list(sched.FORMATS)

# Bags produced per bag opened, by weight.
CONVERSION_YIELDS = {("2kg", "250g"): 8, ("2kg", "1kg"): 2, ("1kg", "250g"): 4}

# A format with this many bags or fewer left counts as sold out: a torn or
# miscounted bag shouldn't hide a sell-out.
SELLOUT_TOLERANCE = 1

# Other-format sales that must follow a format's last sale before its silence
# counts as a sell-out rather than the shop closing. A single straggler
# served after closing can neither create nor block one.
MIN_OTHER_SALES_AFTER = 3

_STAMP = "%Y-%m-%d %H:%M:%S"


def init_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS bagging_plan (
        date TEXT NOT NULL, format TEXT NOT NULL, bags INTEGER NOT NULL,
        recommended INTEGER, entered_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (date, format))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS conversions (
        id INTEGER PRIMARY KEY, date TEXT NOT NULL, entered_at TEXT NOT NULL,
        source TEXT NOT NULL, target TEXT NOT NULL,
        source_bags INTEGER NOT NULL, produced_bags INTEGER NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS closing_record (
        date TEXT PRIMARY KEY, saved_at TEXT NOT NULL,
        leftover_250g INTEGER, leftover_1kg INTEGER, leftover_2kg INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS forecast_log (
        id INTEGER PRIMARY KEY, target_date TEXT NOT NULL, format TEXT NOT NULL,
        issued_at TEXT NOT NULL, horizon_days INTEGER NOT NULL,
        expected REAL NOT NULL, recommended INTEGER NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversions_date ON conversions(date)")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def save_plan(conn, day, bags, recommended, now):
    """Upsert a day's plan. `entered_at` keeps the first save."""
    stamp = now.strftime(_STAMP)
    for fmt in FORMATS:
        rec = (recommended or {}).get(fmt)
        conn.execute("""
            INSERT INTO bagging_plan (date, format, bags, recommended, entered_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, format) DO UPDATE SET
                bags = excluded.bags,
                recommended = COALESCE(excluded.recommended, bagging_plan.recommended),
                updated_at = excluded.updated_at
        """, (day, fmt, int(bags[fmt]), rec, stamp, stamp))


def get_plans(conn, days):
    """{day: {"bags", "recommended", "updated_at"}} for the days that have a plan."""
    if not days:
        return {}
    marks = ",".join("?" * len(days))
    rows = conn.execute(f"SELECT date, format, bags, recommended, updated_at FROM bagging_plan "
                        f"WHERE date IN ({marks})", list(days)).fetchall()
    plans = {}
    for day, fmt, bags, rec, updated in rows:
        plan = plans.setdefault(day, {"bags": {}, "recommended": {}, "updated_at": updated})
        plan["bags"][fmt] = bags
        if rec is not None:
            plan["recommended"][fmt] = rec
        plan["updated_at"] = max(plan["updated_at"], updated)
    return plans


def get_plan(conn, day):
    plan = get_plans(conn, [day]).get(day)
    return plan["bags"] if plan else None


# ---------------------------------------------------------------------------
# Conversions and closing
# ---------------------------------------------------------------------------

def add_conversion(conn, day, source, target, source_bags, now):
    if (source, target) not in CONVERSION_YIELDS:
        raise ValueError(f"conversion {source} -> {target} is not allowed")
    if not isinstance(source_bags, int) or source_bags < 1:
        raise ValueError("at least one bag must be opened")
    produced = source_bags * CONVERSION_YIELDS[(source, target)]
    cur = conn.execute("INSERT INTO conversions (date, entered_at, source, target, source_bags, produced_bags) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (day, now.strftime(_STAMP), source, target, source_bags, produced))
    return cur.lastrowid


def delete_conversion(conn, conversion_id):
    return conn.execute("DELETE FROM conversions WHERE id = ?", (conversion_id,)).rowcount > 0


def get_conversions(conn, day):
    rows = conn.execute("SELECT id, entered_at, source, target, source_bags, produced_bags "
                        "FROM conversions WHERE date = ? ORDER BY id", (day,)).fetchall()
    keys = ("id", "entered_at", "source", "target", "source_bags", "produced_bags")
    return [dict(zip(keys, row)) for row in rows]


def save_closing(conn, day, leftovers, now):
    values = [None if leftovers is None else leftovers.get(fmt) for fmt in FORMATS]
    conn.execute("""
        INSERT INTO closing_record (date, saved_at, leftover_250g, leftover_1kg, leftover_2kg)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET saved_at = excluded.saved_at,
            leftover_250g = excluded.leftover_250g, leftover_1kg = excluded.leftover_1kg,
            leftover_2kg = excluded.leftover_2kg
    """, (day, now.strftime(_STAMP), *values))


def get_closing(conn, day):
    row = conn.execute("SELECT saved_at, leftover_250g, leftover_1kg, leftover_2kg "
                       "FROM closing_record WHERE date = ?", (day,)).fetchone()
    if row is None:
        return None
    leftovers = dict(zip(FORMATS, row[1:]))
    return {"saved_at": row[0],
            "leftovers": None if all(v is None for v in leftovers.values()) else leftovers}


# ---------------------------------------------------------------------------
# Derived stock and sell-outs
# ---------------------------------------------------------------------------

def available(plan, conversions):
    """Bags each format had to sell: planned, plus repacked in, minus opened."""
    out = {fmt: int(plan.get(fmt, 0)) for fmt in FORMATS}
    for conv in conversions:
        out[conv["source"]] -= conv["source_bags"]
        out[conv["target"]] += conv["produced_bags"]
    return out


def format_status(plan, conversions, times, closed, leftovers=None):
    """
    Per format: available, sold, remaining, and — once the day is closed —
    whether it sold out ("sold_out" / "incoherent" / "ok"). "incoherent" is
    reported as soon as it happens, so the counter can say so during the day.
    `leftovers` are the bags counted at closing, where staff typed them.
    """
    stock = available(plan, conversions)
    out = {}
    for fmt in FORMATS:
        sold_times = times.get(fmt, [])
        sold = len(sold_times)
        entry = {"available": stock[fmt], "sold": sold, "remaining": stock[fmt] - sold,
                 "status": None, "sold_out_at": None}
        if sold > stock[fmt]:
            # More sold than was ever there: early orders bagged separately,
            # yesterday's leftovers sold first, or a plan nobody corrected.
            entry["status"] = "incoherent"
        elif closed:
            left = (leftovers or {}).get(fmt)
            entry["status"] = _sellout(fmt, stock[fmt], sold_times, times, entry, left)
        out[fmt] = entry
    return out


def _sellout(fmt, stock, sold_times, times, entry, left=None):
    sold = len(sold_times)
    if sold < stock - SELLOUT_TOLERANCE:
        return "ok"
    if left is not None and left >= 1:
        # The tolerance is for a bag nobody counted; a bag someone did count
        # on the shelf at closing means the format never ran out.
        return "ok"
    if not sold_times:
        # Planned at zero: sold out before opening. One bag left unsold is
        # not a sell-out, tolerance or not.
        return "sold_out" if stock <= 0 else "ok"
    last = sold_times[-1]
    after = sum(1 for other, stamps in times.items() if other != fmt for t in stamps if t > last)
    if after < MIN_OTHER_SALES_AFTER:
        return "ok"
    entry["sold_out_at"] = last.strftime("%H:%M")
    return "sold_out"


def day_status(conn, day):
    """`format_status` for a stored day, or None when it has no plan."""
    plan = get_plan(conn, day)
    if plan is None:
        return None
    times = history.sales_times(conn, datetime.strptime(day, "%Y-%m-%d").date())
    closing = get_closing(conn, day)
    return format_status(plan, get_conversions(conn, day), times, closed=closing is not None,
                         leftovers=(closing or {}).get("leftovers"))


def confirmed_sellouts(conn, before):
    """{(date, format): bags sold} for every confirmed sell-out before `before`."""
    days = [row[0] for row in conn.execute(
        "SELECT c.date FROM closing_record c WHERE c.date < ? "
        "AND EXISTS (SELECT 1 FROM bagging_plan p WHERE p.date = c.date)",
        (before.strftime("%Y-%m-%d"),))]
    out = {}
    for day in days:
        for fmt, entry in (day_status(conn, day) or {}).items():
            if entry["status"] == "sold_out":
                out[(datetime.strptime(day, "%Y-%m-%d").date(), fmt)] = entry["sold"]
    return out


def record_forecast(conn, rows, issued_at):
    stamp = issued_at.strftime(_STAMP)
    conn.executemany(
        "INSERT INTO forecast_log (target_date, format, issued_at, horizon_days, expected, recommended) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(r["target_date"], r["format"], stamp, int(r["horizon_days"]), float(r["expected"]),
          int(r["recommended"])) for r in rows])
