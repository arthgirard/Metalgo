"""Shared test helpers for inserting rows straight into the test database."""


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
