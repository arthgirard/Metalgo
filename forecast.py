"""
Day-level demand model: how many bags of each format a given day will sell,
as a full predictive distribution so the planner can bag a safe quantity
rather than the average.

Why day level, and why this model
---------------------------------
Curds are bagged once, in the morning, and nothing is restocked. The number
that matters is the day's total per format, not its hourly shape. The old
per-hour random forest scored no better than a plain same-weekday mean
(walk-forward daily-total MAE 12.27 vs 12.21), because it had no notion of
the seasonal level: Tuesday-Thursday fell from about 60 bags a day in summer
to about 45 in October while Saturdays held.

The mean blends two simple estimates measured on the same walk-forward
(11.29 MAE together):

  A. recent level x weekday index x format share. The level is the last
     LEVEL_DAYS days with the weekday effect divided out, so it follows the
     season without needing a full year of history.
  B. the mean of the last SAME_WEEKDAY_N days with the same weekday.

The spread
----------
Customers buy in baskets (about 76% one bag, 20% two), so a day's count is
noisier than a Poisson count of bags would be: Var = mu * E[n^2]/E[n] for a
Poisson stream of customers buying n bags each. On top of that, days differ
for reasons the mean does not capture, which adds mu^2 / k. A negative
binomial with that mean and variance gives the quantiles. Even a perfect mean
would miss a typical 70-bag day by about 8.6 bags, so the margin is where
most of the planning value is.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import nbinom, poisson

import shop_schedule as sched

FORMATS = list(sched.FORMATS)

LEVEL_DAYS = 10
SAME_WEEKDAY_N = 3
BLEND_WEIGHT = 0.5

# Chance of selling out the planner accepts, per format. Generous on
# purpose: a leftover bag is sold the next morning or reground for
# restaurants, while a sell-out loses the customer.
SELLOUT_TARGET = 0.10

# Below this many trading days the weekday index and the level are noise.
MIN_FIT_DAYS = 8

# How many recent days the day-to-day noise (k) is measured over.
DISPERSION_DAYS = 30
DISPERSION_K_MIN, DISPERSION_K_MAX = 1.0, 1e6


@dataclass(frozen=True)
class DemandModel:
    as_of: date
    level: float
    weekday_index: dict
    share: dict
    same_weekday_mean: dict
    basket_factor: dict
    dispersion: dict
    profile: dict
    bags_per_customer: float
    extras: dict = field(default_factory=dict)


def _components(days):
    """level, weekday index, format share and same-weekday means from `days`."""
    totals = days["total"].astype(float)
    overall = totals.mean()
    idx = {int(w): float(m / overall) for w, m in totals.groupby(days["weekday"]).mean().items()}

    recent = days.tail(LEVEL_DAYS)
    level = float((recent["total"] / recent["weekday"].map(idx)).mean())
    recent_total = recent["total"].sum()
    share = {fmt: float(recent[fmt].sum() / recent_total) for fmt in FORMATS}

    same = {}
    for w, group in days.groupby("weekday"):
        tail = group.tail(SAME_WEEKDAY_N)
        for fmt in FORMATS:
            same[(int(w), fmt)] = float(tail[fmt].mean())
    return level, idx, share, same


def _blend(level, idx, share, same, weekday, fmt):
    a = level * idx.get(weekday, 1.0) * share[fmt]
    b = same.get((weekday, fmt), a)
    return BLEND_WEIGHT * a + (1 - BLEND_WEIGHT) * b


def _basket_factor(baskets):
    factors = {}
    for fmt in FORMATS:
        n = baskets[fmt].astype(float) if not baskets.empty else np.array([])
        first = float(np.mean(n)) if len(n) else 0.0
        factors[fmt] = float(np.mean(n ** 2) / first) if first > 0 else 1.0
    return factors


def _dispersion(days, basket_factor):
    """Day-to-day noise k per format, from one-step-ahead errors on recent days."""
    errors = {fmt: [] for fmt in FORMATS}
    start = max(MIN_FIT_DAYS, len(days) - DISPERSION_DAYS)
    for i in range(start, len(days)):
        prior, day = days.iloc[:i], days.iloc[i]
        level, idx, share, same = _components(prior)
        for fmt in FORMATS:
            mu = _blend(level, idx, share, same, int(day["weekday"]), fmt)
            errors[fmt].append((float(day[fmt]), mu))

    k = {}
    for fmt, pairs in errors.items():
        if not pairs:
            k[fmt] = DISPERSION_K_MAX
            continue
        y, mu = np.array(pairs).T
        excess = np.mean((y - mu) ** 2) - np.mean(mu * basket_factor[fmt])
        value = np.mean(mu ** 2) / excess if excess > 0 else DISPERSION_K_MAX
        k[fmt] = float(min(max(value, DISPERSION_K_MIN), DISPERSION_K_MAX))
    return k


def fit(daily, baskets, profile, as_of, event_days=frozenset()):
    """
    Fit on trading days strictly before `as_of`, leaving out `event_days`.
    Returns None when there are fewer than MIN_FIT_DAYS such days.
    """
    days = daily[(daily["date"] < as_of) & ~daily["date"].isin(set(event_days))]
    days = days.sort_values("date").reset_index(drop=True)
    if len(days) < MIN_FIT_DAYS:
        return None

    level, idx, share, same = _components(days)
    past_baskets = baskets[baskets["date"] < as_of] if not baskets.empty else baskets
    factor = _basket_factor(past_baskets)
    per_customer = (float(past_baskets[FORMATS].sum(axis=1).mean())
                    if not past_baskets.empty else 1.0)
    return DemandModel(
        as_of=as_of, level=level, weekday_index=idx, share=share,
        same_weekday_mean=same, basket_factor=factor,
        dispersion=_dispersion(days, factor), profile=profile,
        bags_per_customer=per_customer,
    )


def mean(model, target, fmt, event_factor=1.0):
    """Expected bags of `fmt` sold on `target`."""
    m = model
    return _blend(m.level, m.weekday_index, m.share, m.same_weekday_mean,
                  target.weekday(), fmt) * event_factor


def variance(model, fmt, mu):
    """Variance of a day's bags of `fmt` around a mean of `mu`."""
    return mu * model.basket_factor[fmt] + mu ** 2 / model.dispersion[fmt]


def quantile(mu, var, q):
    """The q-quantile of demand, in whole bags."""
    if mu <= 0:
        return 0
    if var <= mu:
        return int(poisson.ppf(q, mu))
    return int(nbinom.ppf(q, n=mu ** 2 / (var - mu), p=mu / var))


def recommend(model, target, fmt, event_factor=1.0, target_sellout=SELLOUT_TARGET):
    """
    Bags to prepare so the chance of selling out stays at `target_sellout`.

    Selling out is demand *reaching* the stock — the last bag going empties
    the shelf — so the answer is one more than the (1 - target) quantile:
    P(D >= q + 1) = 1 - P(D <= q) <= target. Without the +1 the walk-forward
    measured the shelf emptying on 14% of format-days against a 10% target.
    """
    mu = mean(model, target, fmt, event_factor)
    if mu <= 0:
        return 0
    return quantile(mu, variance(model, fmt, mu), 1 - target_sellout) + 1
