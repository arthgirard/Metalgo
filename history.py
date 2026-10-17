"""
Read-only views of the sales history, shaped for the day-level demand model.

Everything here counts every VENTE row of a day, whatever the hour. Sales
logged a little before opening or after closing are real customers, and an
earlier version of the training grid that kept only scheduled hours silently
dropped 26 of them.
"""

from datetime import datetime

import pandas as pd

import shop_schedule as sched

# A day with fewer than this many sales is treated as a partial/abandoned
# log rather than a genuinely quiet day, and excluded from fitting.
MIN_LOGS_THRESHOLD = 10

# Presses this close together are one customer. Clerks press quickly for a
# two-bag purchase, and a customer buying two formats gets both buttons in
# the same couple of seconds.
BASKET_GAP_SECONDS = 2

FORMATS = list(sched.FORMATS)


def _load_rows(conn, before, day=None):
    """VENTE rows as (ts, date, detail), oldest first."""
    if day is not None:
        query = ("SELECT timestamp, detail FROM logs WHERE action_type = 'VENTE' "
                 "AND date(timestamp) = ?")
        params = (day.strftime("%Y-%m-%d"),)
    else:
        query = ("SELECT timestamp, detail FROM logs WHERE action_type = 'VENTE' "
                 "AND date(timestamp) < ?")
        params = (before.strftime("%Y-%m-%d"),)
    df = pd.read_sql_query(query, conn, params=params)
    df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["date"] = df["ts"].dt.date
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _kept_days(rows):
    """Days that count as real trading days: enough sales, shop not closed."""
    counts = rows.groupby("date").size()
    return {d for d, n in counts.items() if n >= MIN_LOGS_THRESHOLD and not sched.is_closed_on(d)}


def load_daily_sales(conn, before):
    """
    One row per trading day strictly before `before`: date, weekday (Python
    convention), a count per format and the total. Sorted by date.
    """
    rows = _load_rows(conn, before)
    columns = ["date", "weekday", *FORMATS, "total"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows[rows["date"].isin(_kept_days(rows))]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    daily = rows.pivot_table(index="date", columns="detail", values="ts",
                             aggfunc="count", fill_value=0)
    daily = daily.reindex(columns=FORMATS, fill_value=0).astype(int)
    daily["total"] = daily[FORMATS].sum(axis=1)
    daily = daily.reset_index()
    daily["weekday"] = [d.weekday() for d in daily["date"]]
    return daily[columns].sort_values("date").reset_index(drop=True)


def _group_baskets(rows):
    """Basket number per row: a new basket after a gap or at a day change."""
    gap = rows["ts"].diff().dt.total_seconds()
    new = gap.isna() | (gap > BASKET_GAP_SECONDS) | (rows["date"] != rows["date"].shift())
    return new.cumsum()


def load_baskets(conn, before):
    """One row per customer before `before`: date and bags of each format."""
    rows = _load_rows(conn, before)
    if rows.empty:
        return pd.DataFrame(columns=["date", *FORMATS])
    rows = rows.assign(basket=_group_baskets(rows))
    baskets = rows.pivot_table(index="basket", columns="detail", values="ts",
                               aggfunc="count", fill_value=0)
    baskets = baskets.reindex(columns=FORMATS, fill_value=0).astype(int)
    baskets.insert(0, "date", rows.groupby("basket")["date"].first())
    return baskets.reset_index(drop=True)


def customers_on(conn, day):
    """Number of customers (baskets) on one day."""
    rows = _load_rows(conn, None, day=day)
    if rows.empty:
        return 0
    return int(_group_baskets(rows).nunique())


def _profile_group(py_weekday):
    return "late" if sched.close_hour(py_weekday) == sched.CLOSE_HOUR_LATE else "regular"


def hourly_profile(conn, before):
    """
    Share of a day's bags sold in each hour, pooled over trading days before
    `before`, separately for regular and late-closing days. Hours outside the
    schedule appear wherever sales happened in them. A group with no history
    falls back to an even spread over its scheduled hours.
    """
    rows = _load_rows(conn, before)
    if not rows.empty:
        rows = rows[rows["date"].isin(_kept_days(rows))]

    profile = {}
    for group, close in (("regular", sched.CLOSE_HOUR), ("late", sched.CLOSE_HOUR_LATE)):
        in_group = (rows[[_profile_group(d.weekday()) == group for d in rows["date"]]]
                    if not rows.empty else rows)
        if in_group.empty:
            hours = range(sched.OPEN_HOUR, close)
            profile[group] = {h: 1 / len(hours) for h in hours}
            continue
        counts = in_group["ts"].dt.hour.value_counts()
        total = counts.sum()
        profile[group] = {int(h): float(n / total) for h, n in sorted(counts.items())}
    return profile


def sales_times(conn, day):
    """Sorted sale timestamps per format for one day."""
    rows = _load_rows(conn, None, day=day)
    return {fmt: [ts.to_pydatetime() for ts in rows.loc[rows["detail"] == fmt, "ts"]]
            for fmt in FORMATS}
