from datetime import datetime

import pytest

import shop_schedule as sched


# datetime.weekday(): Monday=0 ... Sunday=6
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)


class TestWeekdayConversion:
    """
    The two conventions in play, and the fact that they round-trip.

    `daily_snapshots.weekday` is Python's (Mon=0) while the model's `weekday`
    feature is SQLite's `%w` (Sun=0). Getting these backwards shifts every
    prediction by a day without raising anything.
    """

    @pytest.mark.parametrize("py_weekday, sql_weekday", [
        (MONDAY, 1), (TUESDAY, 2), (WEDNESDAY, 3), (THURSDAY, 4),
        (FRIDAY, 5), (SATURDAY, 6), (SUNDAY, 0),
    ])
    def test_matches_strftime(self, py_weekday, sql_weekday):
        assert sched.sql_weekday_from_py(py_weekday) == sql_weekday
        assert sched.py_weekday_from_sql(sql_weekday) == py_weekday

    @pytest.mark.parametrize("py_weekday", range(7))
    def test_round_trips(self, py_weekday):
        assert sched.py_weekday_from_sql(sched.sql_weekday_from_py(py_weekday)) == py_weekday

    def test_agrees_with_real_dates(self):
        # 2026-08-14 is a Friday.
        date = datetime(2026, 8, 14)
        assert date.weekday() == FRIDAY
        assert sched.sql_weekday_from_py(date.weekday()) == int(date.strftime('%w'))


class TestCloseHour:
    @pytest.mark.parametrize("py_weekday, expected", [
        (TUESDAY, 17), (WEDNESDAY, 17), (SATURDAY, 17), (SUNDAY, 17),
        (THURSDAY, 18), (FRIDAY, 18),
    ])
    def test_late_days_close_at_18(self, py_weekday, expected):
        assert sched.close_hour(py_weekday) == expected

    def test_sql_variant_agrees(self):
        for py_weekday in range(7):
            sql_weekday = sched.sql_weekday_from_py(py_weekday)
            assert sched.close_hour_sql(sql_weekday) == sched.close_hour(py_weekday)


class TestSellingHours:
    def test_monday_has_none(self):
        assert list(sched.selling_hours(MONDAY)) == []

    def test_regular_day_stops_before_closing(self):
        # close_hour 17 means the last selling slot is 16h-17h.
        assert list(sched.selling_hours(TUESDAY)) == list(range(10, 17))

    def test_late_day_gets_the_extra_hour(self):
        assert list(sched.selling_hours(THURSDAY)) == list(range(10, 18))


class TestIsOpenAt:
    @pytest.mark.parametrize("dt, expected", [
        (datetime(2026, 8, 17, 12, 0), False),   # Monday, closed all day
        (datetime(2026, 8, 18, 9, 59), False),   # Tuesday, before opening
        (datetime(2026, 8, 18, 10, 0), True),    # Tuesday, opening hour
        (datetime(2026, 8, 18, 16, 59), True),   # Tuesday, last selling hour
        (datetime(2026, 8, 18, 17, 0), False),   # Tuesday, closing hour itself
        (datetime(2026, 8, 20, 17, 30), True),   # Thursday, late closing
        (datetime(2026, 8, 20, 18, 0), False),   # Thursday, shut
        (datetime(2026, 8, 16, 16, 30), True),   # Sunday, open
    ])
    def test_boundaries(self, dt, expected):
        assert sched.is_open_at(dt) is expected


class TestHourBuckets:
    def test_defaults_to_the_scheduled_window(self):
        assert sched.hour_buckets(TUESDAY) == list(range(10, 17))

    def test_folds_in_out_of_hours_sales(self):
        # The live database holds sales at 9h, at 17h on regular days and at
        # 18h; the old fixed grid discarded all 26 of them.
        assert sched.hour_buckets(TUESDAY, [9, 17]) == [9] + list(range(10, 17)) + [17]

    def test_ignores_duplicates_and_stays_sorted(self):
        assert sched.hour_buckets(TUESDAY, [12, 12, 9]) == [9] + list(range(10, 17))

    def test_closed_day_still_reports_observed_hours(self):
        assert sched.hour_buckets(MONDAY, [11]) == [11]


class TestModelHyperparameters:
    """
    Guards for two settings that were measured, not guessed. Both were
    changed because the forest scored WORSE than a same-weekday-and-hour
    average (ratio 0.91); see the notes in train_model.py.
    """

    def test_depth_is_capped(self):
        import train_model
        # Unconstrained, the forest memorised individual days: training error
        # ran at less than half the holdout error.
        assert train_model.RF_PARAMS.get("max_depth") is not None
        assert train_model.RF_PARAMS["max_depth"] <= 6

    def test_fit_and_evaluation_share_hyperparameters(self):
        """Otherwise the reported metrics describe a model nobody ships."""
        import inspect, train_model
        for fn in (train_model._fit, train_model._evaluate):
            source = inspect.getsource(fn)
            assert "RandomForestRegressor(**RF_PARAMS)" in source, fn.__name__

    def test_day_mean_temperature_is_not_a_feature(self):
        """
        Stored as a daily mean it was a unique fingerprint per day (61 days,
        61 distinct values, smallest gap 0.046 °C) rather than a weather
        signal. It may return once per-hour readings accumulate.
        """
        import train_model
        assert "temperature" not in train_model.FEATURE_COLUMNS
