from datetime import datetime, timedelta

import pytest

from helpers import _seed_sales

NOW = datetime(2026, 10, 22, 9, 0, 0)        # a Thursday morning


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
    return next(d for d in client.get('/api/forecast_week').get_json() if d["date"] == day)


def test_lists_the_next_seven_days_without_today_or_closed_days(client, app_module, monkeypatch):
    _at(app_module, monkeypatch, datetime(2026, 12, 19, 10, 0))
    dates = [d["date"] for d in client.get('/api/forecast_week').get_json()]
    assert dates == [f"2026-12-{d}" for d in (20, 22, 23, 24, 26)]


def test_each_day_carries_a_recommendation_above_the_expected(client, fitted):
    day = _day(client, "2026-10-24")
    assert day["date_affichee"] == "Samedi 24"
    assert set(day) == {"date", "date_affichee", "evenement", "attendu", "recommande"}
    for fmt in ("250g", "1kg"):
        assert day["recommande"][fmt] > day["attendu"][fmt] > 0


def test_no_model_still_lists_days(client, app_module, monkeypatch):
    _at(app_module, monkeypatch, NOW)
    days = client.get('/api/forecast_week').get_json()
    assert days and days[0]["date"] == "2026-10-23"
    assert days[0]["attendu"] == {} and days[0]["recommande"] == {}


@pytest.mark.parametrize("method, url", [
    ("get", "/api/plan"), ("post", "/api/plan"), ("get", "/api/stock"),
    ("post", "/api/conversion"), ("delete", "/api/conversion/1"), ("post", "/api/closing"),
])
def test_plan_and_closing_endpoints_are_gone(client, method, url):
    assert getattr(client, method)(url, json={}).status_code in (404, 405)
