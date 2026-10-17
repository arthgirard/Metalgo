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

# Optional adjustments. The model can learn them whenever it is given the
# data; these switches decide whether the app and the nightly fit pass that
# data in. Each is turned on only when the walk-forward harness shows a gain.
# Measured 2026-10-17, last 44 trading days, daily-total MAE at D+0 / D+1 / D+2
# (`python backtest.py --trials`):
#
#   blend               11.29 / 11.46 / 11.62
#   + calendar events   11.27 / 11.44 / 11.60   kept: event days left out of
#                                               the fit, the event factor
#                                               applied to the day itself
#   + Canadiens games   11.85 / 12.03 / 12.20   dropped: every game day on
#                                               record is a spring weekend, so
#                                               the factor mostly re-adds season
#   + weather score     11.91 / 11.92 / 11.91   dropped
#   + temperature       11.52 / 11.70 / 11.84   dropped (55 days of samples)
#
# Re-run the trials before switching any of these on.
USE_EVENTS = True
USE_NHL = False
USE_WEATHER = False
USE_TEMPERATURE = False

# Shrinkage: a weather band or a temperature slope seen on few days stays
# close to "no effect".
WEATHER_SHRINK_DAYS = 10
TEMPERATURE_SHRINK_DAYS = 20


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


def _in_sample_ratio(days, level, idx, share, same):
    """Each fit day's total over the blend's expectation for it."""
    expected = [sum(_blend(level, idx, share, same, int(w), fmt) for fmt in FORMATS)
                for w in days["weekday"]]
    return days["total"].to_numpy(dtype=float) / np.array(expected)


def _weather_factors(days, ratio, weather_by_day):
    by_score = {}
    for d, r in zip(days["date"], ratio):
        score = weather_by_day.get(d)
        if score is not None:
            by_score.setdefault(int(score), []).append(r)
    factors = {}
    for score, ratios in by_score.items():
        n = len(ratios)
        factors[score] = float(1 + (np.mean(ratios) - 1) * n / (n + WEATHER_SHRINK_DAYS))
    return factors


def _temperature_slope(days, ratio, temperature_by_day):
    points = [(temperature_by_day[d], np.log(r)) for d, r in zip(days["date"], ratio)
              if temperature_by_day.get(d) is not None and r > 0]
    if len(points) < 3:
        return None
    t, y = np.array(points).T
    t_mean = float(t.mean())
    spread = float(((t - t_mean) ** 2).sum())
    if spread == 0:
        return None
    slope = float(((t - t_mean) * (y - y.mean())).sum() / spread)
    n = len(points)
    return {"t_mean": t_mean, "slope": slope * n / (n + TEMPERATURE_SHRINK_DAYS)}


def fit(daily, baskets, profile, as_of, event_days=frozenset(),
        weather_by_day=None, temperature_by_day=None):
    """
    Fit on trading days strictly before `as_of`, leaving out `event_days`.
    Returns None when there are fewer than MIN_FIT_DAYS such days.

    `weather_by_day` ({date: score}) and `temperature_by_day` ({date: °C})
    are optional; given, the model learns a multiplier for each.
    """
    days = daily[(daily["date"] < as_of) & ~daily["date"].isin(set(event_days))]
    days = days.sort_values("date").reset_index(drop=True)
    if len(days) < MIN_FIT_DAYS:
        return None

    past_baskets = baskets[baskets["date"] < as_of] if not baskets.empty else baskets
    factor = _basket_factor(past_baskets)

    level, idx, share, same = _components(days)
    per_customer = (float(past_baskets[FORMATS].sum(axis=1).mean())
                    if not past_baskets.empty else 1.0)
    extras = {}
    if weather_by_day or temperature_by_day:
        ratio = _in_sample_ratio(days, level, idx, share, same)
        if weather_by_day:
            extras["weather"] = _weather_factors(days, ratio, weather_by_day)
        if temperature_by_day:
            slope = _temperature_slope(days, ratio, temperature_by_day)
            if slope:
                extras["temperature"] = slope
    return DemandModel(
        as_of=as_of, level=level, weekday_index=idx, share=share,
        same_weekday_mean=same, basket_factor=factor,
        dispersion=_dispersion(days, factor), profile=profile,
        bags_per_customer=per_customer, extras=extras,
    )


def mean(model, target, fmt, event_factor=1.0, weather_score=None, temperature=None):
    """Expected bags of `fmt` sold on `target`."""
    m = model
    mu = _blend(m.level, m.weekday_index, m.share, m.same_weekday_mean,
                target.weekday(), fmt) * event_factor
    if weather_score is not None and "weather" in m.extras:
        mu *= m.extras["weather"].get(int(weather_score), 1.0)
    if temperature is not None and "temperature" in m.extras:
        t = m.extras["temperature"]
        mu *= float(np.exp(t["slope"] * (temperature - t["t_mean"])))
    return mu


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


def recommend(model, target, fmt, event_factor=1.0, target_sellout=SELLOUT_TARGET,
              weather_score=None, temperature=None):
    """
    Bags to prepare so the chance of selling out stays at `target_sellout`.

    Selling out is demand *reaching* the stock — the last bag going empties
    the shelf — so the answer is one more than the (1 - target) quantile:
    P(D >= q + 1) = 1 - P(D <= q) <= target. Without the +1 the walk-forward
    measured the shelf emptying on 14% of format-days against a 10% target.
    """
    mu = mean(model, target, fmt, event_factor, weather_score=weather_score,
              temperature=temperature)
    if mu <= 0:
        return 0
    return quantile(mu, variance(model, fmt, mu), 1 - target_sellout) + 1


# ---------------------------------------------------------------------------
# Intraday
# ---------------------------------------------------------------------------

# How many expected customers the observed count must out-weigh before the
# day's pace is trusted. Counted in customers, not bags: one five-bag order
# swung the old bag-based ratio, and the walk-forward measured that
# correction making the 14h forecast worse than no correction at all.
# Walk-forward on data.db (44 days, 2026-10-17), day-total MAE vs no
# correction 7.30 at 12h / 6.05 at 14h: 10 -> 6.43 / 6.18, 20 -> 6.61 /
# 6.02, 40 -> 6.93 / 5.95. 40 is the first value no worse than no correction
# for every format at both cuts (`python backtest.py --intraday 12 14`).
CUSTOMER_SMOOTHING = 40.0
RATIO_MIN, RATIO_MAX = 0.5, 2.0


def _done_share(profile, now):
    """Share of the day's bags normally sold before `now`, out-of-hours included."""
    done = sum(share for hour, share in profile.items() if hour < now.hour)
    done += profile.get(now.hour, 0.0) * now.minute / 60
    return min(max(done, 0.0), 1.0)


def intraday(model, day, now, sold, customers_so_far, mean_kwargs=None,
             smoothing=CUSTOMER_SMOOTHING):
    """
    The day's total per format given what has sold by `now`.

    The rest of the day is the model's expectation for the hours left, scaled
    by how busy the day has been so far — measured in customers against the
    customers expected by now, and trusted more as the day goes on. Returns
    {"multiplier", "done_share", "formats": {fmt: {"remaining", "point",
    "low", "high"}}}; low/high are the 10th/90th percentiles.
    """
    mean_kwargs = mean_kwargs or {}
    mu = {fmt: mean(model, day, fmt, **mean_kwargs) for fmt in FORMATS}
    group = "late" if sched.close_hour(day.weekday()) == sched.CLOSE_HOUR_LATE else "regular"
    done = _done_share(model.profile[group], now)

    expected_customers = sum(mu.values()) / (model.bags_per_customer or 1.0) * done
    ratio = (min(max(customers_so_far / expected_customers, RATIO_MIN), RATIO_MAX)
             if expected_customers > 0.1 else 1.0)
    confidence = done * expected_customers / (expected_customers + smoothing)
    multiplier = confidence * ratio + 1 - confidence

    formats = {}
    for fmt in FORMATS:
        remaining = mu[fmt] * (1 - done) * multiplier
        var = variance(model, fmt, remaining)
        point = sold[fmt] + int(round(remaining))
        low = sold[fmt] + quantile(remaining, var, 0.10)
        high = sold[fmt] + quantile(remaining, var, 0.90)
        formats[fmt] = {"remaining": float(remaining), "point": point,
                        "low": min(low, point), "high": max(high, point)}
    return {"multiplier": multiplier, "done_share": done, "formats": formats}
