from datetime import datetime

import pytest

from helpers import _insert

NOW = datetime(2026, 10, 22, 15, 0, 0)       # a Thursday afternoon
TODAY = "2026-10-22"
PLAN = {"250g": 10, "1kg": 20, "2kg": 6}


@pytest.fixture
def today(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "now_local", lambda: NOW)
    monkeypatch.setattr(app_module, "today_local", lambda: NOW.date())
    return app_module


def _plan(app_module, bags=PLAN):
    import stock
    with app_module.get_conn() as conn:
        stock.save_plan(conn, TODAY, bags, None, NOW)


def test_stock_without_a_plan(client, today):
    body = client.get('/api/stock').get_json()
    assert body["date"] == TODAY and body["plan"] is False
    assert body["formats"] == {} and body["conversions"] == [] and body["cloture"] is None


def test_stock_with_a_plan_counts_sales(client, today):
    _plan(today)
    for i in range(4):
        _insert(today, f"{TODAY} 11:0{i}:00", "250g")
    body = client.get('/api/stock').get_json()
    assert body["plan"] is True
    assert body["formats"]["250g"]["remaining"] == 6
    assert body["formats"]["250g"]["status"] is None


def test_conversion_round_trip(client, today):
    _plan(today)
    r = client.post('/api/conversion', json={"source": "2kg", "target": "250g", "source_bags": 2})
    assert r.status_code == 200
    conv_id = r.get_json()["id"]
    formats = client.get('/api/stock').get_json()["formats"]
    assert (formats["250g"]["available"], formats["2kg"]["available"]) == (26, 4)

    assert client.delete(f'/api/conversion/{conv_id}').status_code == 200
    formats = client.get('/api/stock').get_json()["formats"]
    assert (formats["250g"]["available"], formats["2kg"]["available"]) == (10, 6)


@pytest.mark.parametrize("body", [
    {"source": "250g", "target": "1kg", "source_bags": 1},
    {"source": "2kg", "target": "250g", "source_bags": 0},
    {"source": "2kg", "target": "250g", "source_bags": True},
    {"source": "2kg", "target": "250g", "source_bags": "2"},
    {"source": "2kg", "target": "250g"},
    [],
])
def test_bad_conversions_are_refused(client, today, body):
    assert client.post('/api/conversion', json=body).status_code == 400
    assert client.get('/api/stock').get_json()["conversions"] == []


def test_conversion_from_another_day_cannot_be_deleted(client, today):
    import stock
    with today.get_conn() as conn:
        old = stock.add_conversion(conn, "2026-10-21", "2kg", "1kg", 1, NOW)
    assert client.delete(f'/api/conversion/{old}').status_code == 404
    assert client.delete('/api/conversion/9999').status_code == 404


def test_closing_then_stock_reports_sellout(client, today):
    _plan(today)
    for i in range(10):
        _insert(today, f"{TODAY} 11:0{i}:00", "250g")
    for i in range(4):
        _insert(today, f"{TODAY} 14:0{i}:00", "1kg")
    assert client.post('/api/closing', json={"leftovers": None}).status_code == 200
    body = client.get('/api/stock').get_json()
    assert body["cloture"]["saved_at"] == "2026-10-22 15:00:00"
    assert body["formats"]["250g"]["status"] == "sold_out"
    assert body["formats"]["250g"]["sold_out_at"] == "11:09"


def test_closing_can_be_edited(client, today):
    client.post('/api/closing', json={"leftovers": None})
    client.post('/api/closing', json={"leftovers": {"250g": 2, "1kg": 0, "2kg": 1}})
    assert client.get('/api/stock').get_json()["cloture"]["leftovers"] == {"250g": 2, "1kg": 0, "2kg": 1}


@pytest.mark.parametrize("body", [
    {"leftovers": {"250g": -1, "1kg": 0, "2kg": 0}},
    {"leftovers": {"250g": 1.5, "1kg": 0, "2kg": 0}},
    {"leftovers": {"500g": 1}},
    {"leftovers": "beaucoup"},
    [],
])
def test_bad_closings_are_refused(client, today, body):
    assert client.post('/api/closing', json=body).status_code == 400
    assert client.get('/api/stock').get_json()["cloture"] is None
