from datetime import date, datetime

import pytest

from helpers import _insert

NOW = datetime(2026, 10, 15, 9, 0, 0)


@pytest.fixture
def stock(app_module):
    import stock as stock_
    return stock_


def _times(**per_format):
    """{fmt: ["HH:MM", ...]} -> sorted datetimes on 2026-10-15, every format present."""
    out = {fmt: [] for fmt in ("250g", "1kg", "2kg")}
    for key, stamps in per_format.items():
        fmt = {"g250": "250g", "kg1": "1kg", "kg2": "2kg"}[key]
        out[fmt] = sorted(datetime.strptime(f"2026-10-15 {s}", "%Y-%m-%d %H:%M") for s in stamps)
    return out


PLAN = {"250g": 10, "1kg": 20, "2kg": 3}


class TestAvailable:
    def test_conversion_moves_stock_by_weight(self, stock):
        av = stock.available({"250g": 40, "1kg": 20, "2kg": 6},
                             [{"source": "2kg", "target": "250g", "source_bags": 2, "produced_bags": 16}])
        assert av == {"250g": 56, "1kg": 20, "2kg": 4}

    def test_yields(self, stock):
        assert stock.CONVERSION_YIELDS == {("2kg", "250g"): 8, ("2kg", "1kg"): 2, ("1kg", "250g"): 4}


class TestFormatStatus:
    def test_sold_out_when_stock_reached_and_others_keep_selling(self, stock):
        times = _times(g250=[f"{10 + i // 4}:{(i % 4) * 15:02d}" for i in range(9)] + ["14:00"],
                       kg1=["14:10", "14:20", "15:00", "15:30", "16:00"])
        st = stock.format_status(PLAN, [], times, closed=True)
        assert st["250g"]["status"] == "sold_out"
        assert st["250g"]["sold_out_at"] == "14:00"
        assert st["250g"]["remaining"] == 0

    def test_within_one_bag_still_counts(self, stock):
        times = _times(g250=["11:00"] * 9, kg1=["12:00", "13:00", "14:00"])
        assert stock.format_status(PLAN, [], times, closed=True)["250g"]["status"] == "sold_out"

    def test_one_after_close_sale_does_not_make_a_sellout(self, stock):
        times = _times(kg2=["11:00", "14:00", "16:58"], kg1=["17:05"])
        assert stock.format_status(PLAN, [], times, closed=True)["2kg"]["status"] == "ok"

    def test_out_of_hours_sales_count_toward_a_sellout(self, stock):
        times = _times(kg2=["09:55", "11:00", "13:00"], kg1=["13:30", "16:00", "17:05"])
        st = stock.format_status(PLAN, [], times, closed=True)
        assert st["2kg"]["status"] == "sold_out" and st["2kg"]["sold_out_at"] == "13:00"

    def test_sales_beyond_stock_are_incoherent_not_sold_out(self, stock):
        times = _times(g250=["11:00"] * 12, kg1=["15:00"] * 5)
        assert stock.format_status(PLAN, [], times, closed=True)["250g"]["status"] == "incoherent"

    def test_format_left_unsold_is_ok(self, stock):
        times = _times(g250=["11:00"] * 4, kg1=["15:00"] * 5)
        assert stock.format_status(PLAN, [], times, closed=True)["250g"]["status"] == "ok"

    def test_a_format_planned_at_zero_is_sold_out_from_the_start(self, stock):
        st = stock.format_status({"250g": 10, "1kg": 20, "2kg": 0}, [], _times(), closed=True)
        assert st["2kg"]["status"] == "sold_out" and st["2kg"]["sold_out_at"] is None

    def test_one_bag_planned_and_none_sold_is_not_a_sellout(self, stock):
        st = stock.format_status({"250g": 10, "1kg": 20, "2kg": 1}, [], _times(kg1=["11:00"] * 5),
                                 closed=True)
        assert st["2kg"]["status"] == "ok"

    def test_a_leftover_typed_at_closing_vetoes_the_sellout(self, stock):
        times = _times(kg2=["11:00", "12:00"], kg1=["14:00", "15:00", "16:00"])
        st = stock.format_status(PLAN, [], times, closed=True, leftovers={"2kg": 1})
        assert st["2kg"]["status"] == "ok" and st["2kg"]["sold_out_at"] is None

    def test_a_leftover_of_zero_keeps_the_sellout(self, stock):
        times = _times(kg2=["11:00", "12:00"], kg1=["14:00", "15:00", "16:00"])
        st = stock.format_status(PLAN, [], times, closed=True, leftovers={"2kg": 0})
        assert st["2kg"]["status"] == "sold_out"

    def test_closing_leftovers_reach_day_status(self, app_module, stock):
        for stamp in ("11:00", "12:00"):
            _insert(app_module, f"2026-10-15 {stamp}:00", "2kg")
        for stamp in ("14:00", "15:00", "16:00"):
            _insert(app_module, f"2026-10-15 {stamp}:00", "1kg")
        with app_module.get_conn() as conn:
            stock.save_plan(conn, "2026-10-15", PLAN, None, NOW)
            stock.save_closing(conn, "2026-10-15", {"250g": None, "1kg": None, "2kg": 1}, NOW)
            assert stock.day_status(conn, "2026-10-15")["2kg"]["status"] == "ok"
            assert stock.confirmed_sellouts(conn, date(2026, 10, 16)) == {}

    def test_sales_beyond_stock_are_flagged_before_closing(self, stock):
        times = _times(g250=["11:00"] * 12)
        st = stock.format_status(PLAN, [], times, closed=False)
        assert st["250g"]["status"] == "incoherent" and st["1kg"]["status"] is None

    def test_no_closing_record_means_no_status_but_live_stock(self, stock):
        times = _times(g250=["11:00"] * 6)
        st = stock.format_status(PLAN, [], times, closed=False)
        assert st["250g"]["status"] is None and st["250g"]["remaining"] == 4

    def test_conversions_count_toward_the_sellout(self, stock):
        conv = [{"source": "2kg", "target": "250g", "source_bags": 1, "produced_bags": 8}]
        times = _times(g250=["11:00"] * 18, kg1=["15:00"] * 4)
        st = stock.format_status(PLAN, conv, times, closed=True)
        assert st["250g"]["available"] == 18 and st["250g"]["status"] == "sold_out"
        assert st["2kg"]["available"] == 2


class TestRecords:
    def test_plan_saved_ahead_keeps_first_entered_at(self, app_module, stock):
        with app_module.get_conn() as conn:
            stock.save_plan(conn, "2026-10-17", PLAN, {"250g": 11, "1kg": 21, "2kg": 4}, NOW)
            stock.save_plan(conn, "2026-10-17", {"250g": 12, "1kg": 20, "2kg": 3}, None,
                            datetime(2026, 10, 17, 8, 30))
            assert stock.get_plan(conn, "2026-10-17") == {"250g": 12, "1kg": 20, "2kg": 3}
            row = conn.execute("SELECT entered_at, updated_at FROM bagging_plan "
                               "WHERE date = '2026-10-17' AND format = '250g'").fetchone()
            plans = stock.get_plans(conn, ["2026-10-17", "2026-10-18"])
        assert row == ("2026-10-15 09:00:00", "2026-10-17 08:30:00")
        assert set(plans) == {"2026-10-17"}
        assert plans["2026-10-17"]["recommended"] == {"250g": 11, "1kg": 21, "2kg": 4}
        assert plans["2026-10-17"]["updated_at"] == "2026-10-17 08:30:00"

    def test_unknown_conversion_pair_is_refused(self, app_module, stock):
        with app_module.get_conn() as conn:
            with pytest.raises(ValueError):
                stock.add_conversion(conn, "2026-10-15", "250g", "1kg", 1, NOW)
            with pytest.raises(ValueError):
                stock.add_conversion(conn, "2026-10-15", "2kg", "250g", 0, NOW)
            assert stock.get_conversions(conn, "2026-10-15") == []

    def test_conversion_round_trip(self, app_module, stock):
        with app_module.get_conn() as conn:
            conv_id = stock.add_conversion(conn, "2026-10-15", "2kg", "1kg", 3, NOW)
            [conv] = stock.get_conversions(conn, "2026-10-15")
            assert (conv["id"], conv["produced_bags"]) == (conv_id, 6)
            assert stock.delete_conversion(conn, conv_id) is True
            assert stock.delete_conversion(conn, conv_id) is False

    def test_closing_round_trip(self, app_module, stock):
        with app_module.get_conn() as conn:
            assert stock.get_closing(conn, "2026-10-15") is None
            stock.save_closing(conn, "2026-10-15", {"250g": 2, "1kg": 0, "2kg": 1}, NOW)
            closing = stock.get_closing(conn, "2026-10-15")
        assert closing["leftovers"] == {"250g": 2, "1kg": 0, "2kg": 1}
        assert closing["saved_at"] == "2026-10-15 09:00:00"

    def test_day_status_is_none_without_a_plan(self, app_module, stock):
        with app_module.get_conn() as conn:
            assert stock.day_status(conn, "2026-10-15") is None

    def test_confirmed_sellouts_lists_only_sold_out_days(self, app_module, stock):
        for i in range(10):
            _insert(app_module, f"2026-10-14 11:{i:02d}:00", "250g")
        for i in range(4):
            _insert(app_module, f"2026-10-14 15:{i:02d}:00", "1kg")
        for i in range(10):
            _insert(app_module, f"2026-10-15 11:{i:02d}:00", "250g")      # closed record missing
        with app_module.get_conn() as conn:
            stock.save_plan(conn, "2026-10-14", PLAN, None, NOW)
            stock.save_plan(conn, "2026-10-15", PLAN, None, NOW)
            stock.save_closing(conn, "2026-10-14", None, NOW)
            result = stock.confirmed_sellouts(conn, date(2026, 10, 16))
        assert result == {(date(2026, 10, 14), "250g"): 10}

    def test_a_past_day_with_a_plan_counts_without_a_closing(self, app_module, stock):
        """Clerks often skip the closing; the plan and the sales are enough."""
        for i in range(10):
            _insert(app_module, f"2026-10-14 11:{i:02d}:00", "250g")
        for i in range(4):
            _insert(app_module, f"2026-10-14 15:{i:02d}:00", "1kg")
        with app_module.get_conn() as conn:
            stock.save_plan(conn, "2026-10-14", PLAN, None, NOW)
            assert stock.confirmed_sellouts(conn, date(2026, 10, 16)) == {(date(2026, 10, 14), "250g"): 10}
            assert stock.day_status(conn, "2026-10-14", ended=True)["250g"]["status"] == "sold_out"
            assert stock.day_status(conn, "2026-10-14")["250g"]["status"] is None

    def test_forecast_record(self, app_module, stock):
        with app_module.get_conn() as conn:
            stock.record_forecast(conn, [{"target_date": "2026-10-17", "format": "250g",
                                          "horizon_days": 2, "expected": 41.5, "recommended": 52}], NOW)
            rows = conn.execute("SELECT target_date, format, issued_at, horizon_days, expected, recommended "
                                "FROM forecast_log").fetchall()
        assert rows == [("2026-10-17", "250g", "2026-10-15 09:00:00", 2, 41.5, 52)]
