from datetime import date, datetime, timedelta

import pytest

from helpers import _insert, _seed_sales

NOW = datetime(2026, 10, 22, 9, 0, 0)        # a Thursday morning
BAGS = {"250g": 45, "1kg": 35, "2kg": 6}


def _at(app_module, monkeypatch, now):
    monkeypatch.setattr(app_module, "now_local", lambda: now)
    monkeypatch.setattr(app_module, "today_local", lambda: now.date())


@pytest.fixture
def fitted(app_module, monkeypatch):
    """A version-2 model trained on 30 days of history ending yesterday."""
    import train_model
    monkeypatch.setattr(train_model, "now_local", lambda: NOW)
    _seed_sales(app_module, days=30, end=(NOW - timedelta(days=1)).date())
    train_model.train_model(app_module.DB_NAME, app_module.MODEL_PATH)
    _at(app_module, monkeypatch, NOW)
    return app_module


def _day(client, day):
    return next(d for d in client.get('/api/plan').get_json() if d["date"] == day)


def test_plan_lists_today_and_skips_closed_days(client, app_module, monkeypatch):
    _at(app_module, monkeypatch, datetime(2026, 12, 19, 10, 0))
    dates = [d["date"] for d in client.get('/api/plan').get_json()]
    assert dates == [f"2026-12-{d}" for d in (19, 20, 22, 23, 24, 26)]


def test_each_day_carries_a_recommendation_above_the_expected(client, fitted):
    day = _day(client, "2026-10-24")
    assert day["date_affichee"] == "Samedi 24"
    for fmt in ("250g", "1kg"):
        assert day["recommande"][fmt] > day["attendu"][fmt] > 0
    assert day["plan"] is None and day["changement"] is None


def test_saving_a_plan_two_days_ahead(client, fitted):
    r = client.post('/api/plan', json={"date": "2026-10-24", "bags": BAGS})
    assert r.status_code == 200
    day = _day(client, "2026-10-24")
    assert day["plan"] == BAGS
    assert day["plan_maj"] == "2026-10-22 09:00:00"


def test_saved_plan_is_never_rewritten_but_change_is_flagged(client, fitted):
    client.post('/api/plan', json={"date": "2026-10-24", "bags": BAGS})
    current = _day(client, "2026-10-24")["recommande"]
    with fitted.get_conn() as conn:
        # As if the recommendation had been 20 bags lower when the plan was typed.
        conn.execute("UPDATE bagging_plan SET recommended = recommended - 20 "
                     "WHERE date = '2026-10-24' AND format = '250g'")
    day = _day(client, "2026-10-24")
    assert day["plan"] == BAGS
    assert day["changement"] == {"250g": current["250g"]}


def test_small_recommendation_drift_is_not_flagged(client, fitted):
    client.post('/api/plan', json={"date": "2026-10-24", "bags": BAGS})
    with fitted.get_conn() as conn:
        conn.execute("UPDATE bagging_plan SET recommended = recommended - 1 "
                     "WHERE date = '2026-10-24' AND format = '250g'")
    assert _day(client, "2026-10-24")["changement"] is None


@pytest.mark.parametrize("body", [
    {"date": "2026-10-21", "bags": BAGS},                                   # yesterday
    {"date": "2026-10-30", "bags": BAGS},                                   # 8 days ahead
    {"date": "2026-10-26", "bags": BAGS},                                   # a Monday
    {"date": "26-10-2026", "bags": BAGS},                                   # bad format
    {"date": "2026-02-30", "bags": BAGS},                                   # not a date
    {"date": "2026-10-24", "bags": {"250g": 45, "1kg": 35}},                # missing format
    {"date": "2026-10-24", "bags": {"250g": -1, "1kg": 35, "2kg": 6}},      # negative
    {"date": "2026-10-24", "bags": {"250g": 4.5, "1kg": 35, "2kg": 6}},     # not whole
    {"date": "2026-10-24", "bags": {"250g": True, "1kg": 35, "2kg": 6}},    # a boolean
    {"date": "2026-10-24", "bags": {"250g": 1000, "1kg": 35, "2kg": 6}},    # absurd
    {"date": "2026-10-24", "bags": {"250g": "45", "1kg": 35, "2kg": 6}},    # a string
    {"date": "2026-10-24"},
    [],
])
def test_rejects_bad_plans(client, app_module, monkeypatch, body):
    _at(app_module, monkeypatch, NOW)
    r = client.post('/api/plan', json=body)
    assert r.status_code == 400
    assert r.get_json()["status"] == "error"
    with app_module.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM bagging_plan").fetchone()[0] == 0


def test_today_can_be_planned(client, fitted):
    assert client.post('/api/plan', json={"date": "2026-10-22", "bags": BAGS}).status_code == 200


def test_no_model_still_lists_days(client, app_module, monkeypatch):
    _at(app_module, monkeypatch, NOW)
    days = client.get('/api/plan').get_json()
    assert days and days[0]["date"] == "2026-10-22"
    assert days[0]["attendu"] == {} and days[0]["recommande"] == {}
    assert client.post('/api/plan', json={"date": "2026-10-24", "bags": BAGS}).status_code == 200


def test_last_week_shows_conversions_and_sellouts(client, fitted):
    last_saturday = "2026-10-17"
    with fitted.get_conn() as conn:
        conn.execute("DELETE FROM logs WHERE date(timestamp) = ?", (last_saturday,))
    for i in range(6):
        _insert(fitted, f"{last_saturday} 11:0{i}:00", "2kg")
    for i in range(5):
        _insert(fitted, f"{last_saturday} 14:0{i}:00", "250g")
    import stock
    with fitted.get_conn() as conn:
        stock.save_plan(conn, last_saturday, {"250g": 40, "1kg": 30, "2kg": 8}, None, NOW)
        stock.add_conversion(conn, last_saturday, "2kg", "250g", 2, NOW)
        stock.save_closing(conn, last_saturday, None, NOW)
    week = _day(client, "2026-10-24")["semaine_derniere"]
    assert week["date"] == last_saturday
    assert [(c["source"], c["target"], c["source_bags"]) for c in week["conversions"]] == [("2kg", "250g", 2)]
    assert week["ruptures"] == ["2kg"]
    assert week["incoherent"] == []
