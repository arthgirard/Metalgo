import random
from datetime import date, timedelta

import pandas as pd
import pytest

import forecast

FORMATS = ("250g", "1kg", "2kg")
PROFILE = {"regular": {h: 1 / 7 for h in range(10, 17)},
           "late": {h: 1 / 8 for h in range(10, 18)}}


def _daily(start, days, per_day, weekday_factor=None):
    """
    `days` consecutive open days (Mondays skipped) from `start`. `per_day` is
    {fmt: bags} or a callable i -> {fmt: bags}; `weekday_factor` scales by
    Python weekday.
    """
    rows, d, i = [], start, 0
    while i < days:
        if d.weekday() != 0:
            counts = per_day(i) if callable(per_day) else per_day
            factor = (weekday_factor or {}).get(d.weekday(), 1.0)
            row = {fmt: int(round(counts.get(fmt, 0) * factor)) for fmt in FORMATS}
            rows.append({"date": d, "weekday": d.weekday(), **row, "total": sum(row.values())})
            i += 1
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _baskets(daily, size=1):
    """Every bag in its own basket (size=1) or in baskets of `size` same-format bags."""
    rows = []
    for r in daily.itertuples(index=False):
        for fmt in FORMATS:
            n = getattr(r, f"_{FORMATS.index(fmt) + 2}")
            full, rest = divmod(n, size)
            for k in [size] * full + ([rest] if rest else []):
                rows.append({"date": r.date, **{f: (k if f == fmt else 0) for f in FORMATS}})
    return pd.DataFrame(rows, columns=["date", *FORMATS])


def _next_open(d):
    d += timedelta(days=1)
    return d + timedelta(days=1) if d.weekday() == 0 else d


START = date(2026, 8, 4)    # a Tuesday


def _fit(daily, baskets=None, **kwargs):
    as_of = _next_open(daily["date"].max())
    return forecast.fit(daily, _baskets(daily) if baskets is None else baskets,
                        PROFILE, as_of, **kwargs), as_of


class TestMean:
    def test_flat_history_predicts_the_flat_level(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, as_of = _fit(daily)
        assert forecast.mean(model, as_of, "250g") == pytest.approx(40, rel=0.02)
        assert forecast.mean(model, as_of, "2kg") == pytest.approx(3, rel=0.05)

    def test_level_follows_a_recent_drop(self):
        daily = _daily(START, 30, lambda i: {"250g": 30, "1kg": 24, "2kg": 6} if i < 20
                       else {"250g": 20, "1kg": 16, "2kg": 4})
        model, as_of = _fit(daily)
        # Halfway between the old and new level would be 25 for 250g.
        assert forecast.mean(model, as_of, "250g") < 25

    def test_saturday_stays_above_tuesday(self):
        daily = _daily(START, 36, {"250g": 40, "1kg": 25, "2kg": 3},
                       weekday_factor={5: 1.8, 1: 0.7})
        model, as_of = _fit(daily)
        saturday = as_of + timedelta(days=(5 - as_of.weekday()) % 7)
        tuesday = as_of + timedelta(days=(1 - as_of.weekday()) % 7)
        assert forecast.mean(model, saturday, "250g") > 1.5 * forecast.mean(model, tuesday, "250g")

    def test_event_factor_scales_the_mean(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, as_of = _fit(daily)
        assert forecast.mean(model, as_of, "250g", event_factor=1.5) == pytest.approx(
            1.5 * forecast.mean(model, as_of, "250g"))

    def test_event_days_are_left_out_of_the_fit(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        spike = daily.index[-2]
        daily.loc[spike, ["250g", "total"]] = [400, 428]
        with_spike, as_of = _fit(daily)
        without, _ = _fit(daily, event_days={daily.loc[spike, "date"]})
        assert forecast.mean(without, as_of, "250g") == pytest.approx(40, rel=0.02)
        assert forecast.mean(with_spike, as_of, "250g") > 45

    def test_too_little_history_returns_none(self):
        daily = _daily(START, 5, {"250g": 40, "1kg": 25, "2kg": 3})
        model, _ = _fit(daily)
        assert model is None

    def test_only_days_before_as_of_are_used(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        later = _daily(_next_open(daily["date"].max()), 5, {"250g": 400, "1kg": 250, "2kg": 30})
        both = pd.concat([daily, later], ignore_index=True)
        as_of = later["date"].min()
        model = forecast.fit(both, _baskets(both), PROFILE, as_of)
        assert forecast.mean(model, as_of, "250g") == pytest.approx(40, rel=0.02)


class TestSpread:
    def test_recommendation_is_above_the_mean_and_monotone_in_target(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, d = _fit(daily)
        mu = forecast.mean(model, d, "250g")
        assert forecast.recommend(model, d, "250g") > mu
        assert (forecast.recommend(model, d, "250g", target_sellout=0.05)
                >= forecast.recommend(model, d, "250g", target_sellout=0.20))

    def test_chance_of_emptying_the_shelf_stays_within_target(self):
        """
        Selling out means demand reaches the stock, not only exceeds it:
        the last bag going is the shelf going empty. So P(D >= recommended)
        must be at most the target.
        """
        from scipy.stats import nbinom, poisson
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, d = _fit(daily, _baskets(daily, size=2))
        for fmt in FORMATS:
            mu = forecast.mean(model, d, fmt)
            var = forecast.variance(model, fmt, mu)
            rec = forecast.recommend(model, d, fmt)
            sf = (poisson.sf(rec - 1, mu) if var <= mu
                  else nbinom.sf(rec - 1, n=mu ** 2 / (var - mu), p=mu / var))
            assert sf <= forecast.SELLOUT_TARGET + 1e-9, fmt

    def test_default_target_is_ten_percent(self):
        assert forecast.SELLOUT_TARGET == 0.10

    def test_baskets_widen_the_spread(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        singles, _ = _fit(daily, _baskets(daily, size=1))
        pairs, _ = _fit(daily, _baskets(daily, size=2))
        assert pairs.basket_factor["250g"] == pytest.approx(2.0)
        assert forecast.variance(pairs, "250g", 40) > forecast.variance(singles, "250g", 40)

    def test_day_to_day_noise_widens_the_spread(self):
        steady = _daily(START, 40, {"250g": 40, "1kg": 25, "2kg": 3})
        # Seeded, not alternating: an alternating pattern lines up with the
        # six-day trading week and becomes a weekday effect the mean explains.
        rng = random.Random(7)
        noise = [rng.choice((20, 60)) for _ in range(40)]
        noisy = _daily(START, 40, lambda i: {"250g": noise[i], "1kg": 25, "2kg": 3})
        m_steady, _ = _fit(steady)
        m_noisy, _ = _fit(noisy)
        assert forecast.variance(m_noisy, "250g", 40) > forecast.variance(m_steady, "250g", 40)

    def test_a_format_never_sold_recommends_zero(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 0})
        model, d = _fit(daily)
        assert forecast.recommend(model, d, "2kg") == 0

    def test_quantile_handles_poisson_and_overdispersed_cases(self):
        assert forecast.quantile(0, 0, 0.9) == 0
        assert forecast.quantile(40, 40, 0.9) == 48          # Poisson(40) 90th percentile
        assert forecast.quantile(40, 120, 0.9) > 48

    def test_profile_and_bags_per_customer_are_kept(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, _ = _fit(daily, _baskets(daily, size=2))
        assert model.profile == PROFILE
        assert model.bags_per_customer == pytest.approx(2.0, rel=0.05)


class TestWeatherAndTemperature:
    """
    Optional adjustments. Each is only switched on in the app if the
    walk-forward harness shows it helps; the switches live in forecast.py.
    """

    @staticmethod
    def _rainy_history():
        rng = random.Random(3)
        scores = [rng.choice((0, 2, 2, 2)) for _ in range(40)]
        daily = _daily(START, 40, lambda i: {"250g": 40 if scores[i] else 24,
                                             "1kg": 25 if scores[i] else 15, "2kg": 3})
        return daily, dict(zip(daily["date"], scores))

    def test_measured_switches(self):
        """
        Set from the walk-forward on 2026-10-17 (last 44 trading days, D+0):
        blend 11.29, +events 11.27, +events+NHL 11.85, +weather 11.91,
        +temperature 11.52 bags daily-total MAE. Only events earned a place.
        """
        assert forecast.USE_EVENTS is True
        assert forecast.USE_NHL is False
        assert forecast.USE_WEATHER is False
        assert forecast.USE_TEMPERATURE is False

    def test_weather_factor_learns_a_rainy_dip(self):
        daily, weather = self._rainy_history()
        model, d = _fit(daily, weather_by_day=weather)
        assert forecast.mean(model, d, "250g", weather_score=0) < 0.85 * forecast.mean(
            model, d, "250g", weather_score=2)

    def test_weather_factor_is_shrunk_with_few_days(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        rainy = daily["date"].iloc[5]
        daily.loc[daily["date"] == rainy, ["250g", "1kg", "total"]] = [20, 12, 35]
        weather = {d: (0 if d == rainy else 2) for d in daily["date"]}
        model, d = _fit(daily, weather_by_day=weather)
        ratio = forecast.mean(model, d, "250g", weather_score=0) / forecast.mean(model, d, "250g")
        # One rainy day at about half the usual: shrunk by 1 / (1 + 10).
        assert 0.9 < ratio < 1.0

    def test_no_weather_given_means_no_factor(self):
        daily = _daily(START, 30, {"250g": 40, "1kg": 25, "2kg": 3})
        model, d = _fit(daily)
        assert forecast.mean(model, d, "250g", weather_score=0) == forecast.mean(model, d, "250g")

    def test_temperature_factor_follows_warm_days(self):
        rng = random.Random(5)
        temps = [rng.uniform(5, 30) for _ in range(40)]
        daily = _daily(START, 40, lambda i: {"250g": round(20 + temps[i]), "1kg": 25, "2kg": 3})
        model, d = _fit(daily, temperature_by_day=dict(zip(daily["date"], temps)))
        assert forecast.mean(model, d, "250g", temperature=28) > forecast.mean(model, d, "250g", temperature=8)

    def test_unknown_temperature_means_no_factor(self):
        rng = random.Random(5)
        temps = [rng.uniform(5, 30) for _ in range(40)]
        daily = _daily(START, 40, lambda i: {"250g": round(20 + temps[i]), "1kg": 25, "2kg": 3})
        model, d = _fit(daily, temperature_by_day=dict(zip(daily["date"], temps)))
        plain = forecast.fit(daily, _baskets(daily), PROFILE, d)
        assert forecast.mean(model, d, "250g", temperature=None) == pytest.approx(
            forecast.mean(plain, d, "250g"))

    def test_recommendation_follows_the_weather(self):
        daily, weather = self._rainy_history()
        model, d = _fit(daily, weather_by_day=weather)
        assert (forecast.recommend(model, d, "250g", weather_score=0)
                < forecast.recommend(model, d, "250g", weather_score=2))
