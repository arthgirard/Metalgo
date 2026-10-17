from datetime import date, datetime

import pytest

from helpers import _fill_day, _insert


@pytest.fixture
def history(app_module):
    import history as history_
    return history_


class TestDailySales:
    def test_out_of_hours_sales_count_toward_the_day(self, app_module, history):
        """
        A customer served at 9h55 or 17h05 is a real sale. Dropping it made
        training quietly disagree with the charts once already.
        """
        # 2026-08-18 is a Tuesday: 10h-17h scheduled.
        _fill_day(app_module, "2026-08-18", n=12)
        _insert(app_module, "2026-08-18 09:55:00", "250g")
        _insert(app_module, "2026-08-18 17:05:00", "1kg")
        with app_module.get_conn() as conn:
            df = history.load_daily_sales(conn, date(2026, 8, 19))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["date"] == date(2026, 8, 18)
        assert row["weekday"] == 1
        assert int(row["total"]) == 14
        assert (int(row["250g"]), int(row["1kg"]), int(row["2kg"])) == (13, 1, 0)

    def test_days_below_threshold_and_today_are_excluded(self, app_module, history):
        _fill_day(app_module, "2026-08-18", n=9)          # a partial log
        _fill_day(app_module, "2026-08-19", n=12)         # the cut-off day itself
        with app_module.get_conn() as conn:
            df = history.load_daily_sales(conn, date(2026, 8, 19))
        assert df.empty

    def test_conversion_rows_are_not_sales(self, app_module, history):
        _fill_day(app_module, "2026-08-18", n=12)
        _insert(app_module, "2026-08-18 15:00:00", "2kg → 250g", action_type="CONVERSION")
        with app_module.get_conn() as conn:
            df = history.load_daily_sales(conn, date(2026, 8, 19))
        assert int(df.iloc[0]["total"]) == 12

    def test_microsecond_timestamps_parse(self, app_module, history):
        """Rows from before the server-side stamp carry microseconds."""
        for i in range(12):
            _insert(app_module, f"2026-02-07 10:{i:02d}:38.338266", "250g")
        with app_module.get_conn() as conn:
            df = history.load_daily_sales(conn, date(2026, 2, 8))
        assert int(df.iloc[0]["total"]) == 12


class TestBaskets:
    def test_presses_within_two_seconds_form_one_basket(self, app_module, history):
        for ts, fmt in [("11:00:00", "250g"), ("11:00:00", "250g"), ("11:00:02", "1kg"),
                        ("11:00:30", "250g")]:
            _insert(app_module, f"2026-08-18 {ts}", fmt)
        with app_module.get_conn() as conn:
            b = history.load_baskets(conn, date(2026, 8, 19))
        assert len(b) == 2
        assert b.iloc[0][["250g", "1kg", "2kg"]].tolist() == [2, 1, 0]
        assert b.iloc[1][["250g", "1kg", "2kg"]].tolist() == [1, 0, 0]

    def test_baskets_never_span_two_days(self, app_module, history):
        _insert(app_module, "2026-08-18 23:59:59", "250g")
        _insert(app_module, "2026-08-19 00:00:00", "250g")
        with app_module.get_conn() as conn:
            b = history.load_baskets(conn, date(2026, 8, 20))
        assert len(b) == 2

    def test_customers_on_one_day(self, app_module, history):
        for ts in ("11:00:00", "11:00:01", "11:05:00", "12:00:00"):
            _insert(app_module, f"2026-08-18 {ts}", "1kg")
        with app_module.get_conn() as conn:
            assert history.customers_on(conn, date(2026, 8, 18)) == 3
            assert history.customers_on(conn, date(2026, 8, 19)) == 0


class TestHourlyProfile:
    def test_includes_out_of_hours_and_sums_to_one(self, app_module, history):
        _fill_day(app_module, "2026-08-18", n=12)                  # Tuesday, regular close
        _insert(app_module, "2026-08-18 09:50:00", "250g")
        with app_module.get_conn() as conn:
            profile = history.hourly_profile(conn, date(2026, 8, 19))
        assert set(profile) == {"regular", "late"}
        assert 9 in profile["regular"]
        assert profile["regular"][9] == pytest.approx(1 / 13)
        assert sum(profile["regular"].values()) == pytest.approx(1.0)

    def test_late_days_get_their_own_profile(self, app_module, history):
        _fill_day(app_module, "2026-08-20", n=12, hour=17)         # Thursday, late close
        with app_module.get_conn() as conn:
            profile = history.hourly_profile(conn, date(2026, 8, 21))
        assert profile["late"] == {17: pytest.approx(1.0)}


class TestSalesTimes:
    def test_sorted_times_per_format(self, app_module, history):
        _insert(app_module, "2026-08-18 12:00:00", "1kg")
        _insert(app_module, "2026-08-18 11:00:00", "1kg")
        with app_module.get_conn() as conn:
            times = history.sales_times(conn, date(2026, 8, 18))
        assert times["1kg"] == [datetime(2026, 8, 18, 11), datetime(2026, 8, 18, 12)]
        assert times["250g"] == [] and times["2kg"] == []


class TestDailyWeather:
    def test_sampler_readings_win_over_sale_labels(self, app_module, history):
        _fill_day(app_module, "2026-08-18", n=12)       # sale labels say "Ensoleillé" (2)
        with app_module.get_conn() as conn:
            for ts, score, temp in (("2026-08-18 10:15:00", 0, 14.0),
                                    ("2026-08-18 11:15:00", 0, 16.0),
                                    ("2026-08-18 19:00:00", 2, 30.0)):   # after hours: ignored
                conn.execute("INSERT INTO weather_log (timestamp, condition, weather_score, temperature)"
                             " VALUES (?, 'Pluie', ?, ?)", (ts, score, temp))
            weather = history.daily_weather(conn, date(2026, 8, 19))
        assert weather[date(2026, 8, 18)] == {"score": 0, "temperature": pytest.approx(15.0)}

    def test_sale_labels_fill_in_before_the_sampler_existed(self, app_module, history):
        _fill_day(app_module, "2026-08-18", n=12)
        with app_module.get_conn() as conn:
            weather = history.daily_weather(conn, date(2026, 8, 19))
        assert weather[date(2026, 8, 18)] == {"score": 2, "temperature": None}
