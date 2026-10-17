"""Shared test helpers for inserting rows straight into the test database."""

from datetime import date, timedelta


def _insert(app_module, timestamp, detail, action_type="VENTE"):
    with app_module.get_conn() as conn:
        conn.execute(
            "INSERT INTO logs (timestamp, action_type, detail, meteo_summary, temperature)"
            " VALUES (?, ?, ?, ?, ?)",
            (timestamp, action_type, detail, "Ensoleillé", 20.0))


def _count(app_module):
    with app_module.get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]


def _fill_day(app_module, day, n, fmt="250g", hour=11):
    """`n` sales of one format, ten seconds apart so each is its own customer."""
    for i in range(n):
        minute, second = divmod(i * 10, 60)
        _insert(app_module, f"{day} {hour:02d}:{minute:02d}:{second:02d}", fmt)


def _seed_sales(app_module, days=40, end=date(2026, 9, 30)):
    """`days` open days of sales ending at `end`, busier on weekends."""
    d, seeded = end, 0
    while seeded < days:
        if d.weekday() != 0:
            busy = 2 if d.weekday() >= 5 else 1
            for hour in range(10, 17):
                for i, fmt in enumerate(["250g"] * (2 * busy) + ["1kg"] * busy):
                    _insert(app_module, f"{d} {hour:02d}:{i * 5:02d}:00", fmt)
            if seeded % 3 == 0:
                _insert(app_module, f"{d} 15:30:00", "2kg")
            seeded += 1
        d -= timedelta(days=1)
