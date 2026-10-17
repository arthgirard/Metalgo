"""
Every nightly forecast of expected sales, kept so the forecasts can later be
checked against what really sold.
"""

_STAMP = "%Y-%m-%d %H:%M:%S"


def init_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS forecast_log (
        id INTEGER PRIMARY KEY, target_date TEXT NOT NULL, format TEXT NOT NULL,
        issued_at TEXT NOT NULL, horizon_days INTEGER NOT NULL,
        expected REAL NOT NULL)""")
    # The first build also logged the 90th-percentile bagging recommendation,
    # as NOT NULL; it was dropped for leaning towards too much cheese.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(forecast_log)")}
    if "recommended" in columns:
        conn.execute("ALTER TABLE forecast_log DROP COLUMN recommended")


def record(conn, rows, issued_at):
    stamp = issued_at.strftime(_STAMP)
    conn.executemany(
        "INSERT INTO forecast_log (target_date, format, issued_at, horizon_days, expected) "
        "VALUES (?, ?, ?, ?, ?)",
        [(r["target_date"], r["format"], stamp, int(r["horizon_days"]), float(r["expected"]))
         for r in rows])
