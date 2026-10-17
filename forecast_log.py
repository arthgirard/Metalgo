"""
Every nightly recommendation, kept so the forecasts can later be checked
against what really sold.
"""

_STAMP = "%Y-%m-%d %H:%M:%S"


def init_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS forecast_log (
        id INTEGER PRIMARY KEY, target_date TEXT NOT NULL, format TEXT NOT NULL,
        issued_at TEXT NOT NULL, horizon_days INTEGER NOT NULL,
        expected REAL NOT NULL, recommended INTEGER NOT NULL)""")


def record(conn, rows, issued_at):
    stamp = issued_at.strftime(_STAMP)
    conn.executemany(
        "INSERT INTO forecast_log (target_date, format, issued_at, horizon_days, expected, recommended) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(r["target_date"], r["format"], stamp, int(r["horizon_days"]), float(r["expected"]),
          int(r["recommended"])) for r in rows])
