import os
from datetime import datetime, timedelta

import joblib
import pytest

from helpers import _seed_sales

NOW = datetime(2026, 10, 17, 3, 0, 0)       # a Saturday, nightly retrain time


@pytest.fixture
def trainer(app_module, monkeypatch):
    import train_model
    monkeypatch.setattr(train_model, "now_local", lambda: NOW)
    _seed_sales(app_module, days=30, end=(NOW - timedelta(days=1)).date())
    return train_model


def _train(app_module, trainer):
    return trainer.train_model(app_module.DB_NAME, app_module.MODEL_PATH)


def test_fit_saves_a_version_two_payload(app_module, trainer):
    metrics = _train(app_module, trainer)
    payload = joblib.load(app_module.MODEL_PATH)
    assert payload["version"] == 2
    assert payload["switches"] == {"events": True, "nhl": False, "weather": False, "temperature": False}
    assert metrics["status"] == "ok"
    assert set(metrics["gate"]) >= {"mae", "baseline_mae", "n_days"}
    assert app_module.get_demand_model() is not None


def test_rejected_fit_keeps_the_previous_model(app_module, trainer, monkeypatch):
    _train(app_module, trainer)
    before = os.path.getmtime(app_module.MODEL_PATH), open(app_module.MODEL_PATH, "rb").read()
    monkeypatch.setattr(trainer, "_light_gate",
                        lambda *a, **k: {"mae": 20.0, "baseline_mae": 10.0, "n_days": 14})
    metrics = _train(app_module, trainer)
    assert metrics["status"] == "rejected"
    assert (os.path.getmtime(app_module.MODEL_PATH), open(app_module.MODEL_PATH, "rb").read()) == before


def test_first_fit_is_saved_even_if_it_loses(app_module, trainer, monkeypatch):
    monkeypatch.setattr(trainer, "_light_gate",
                        lambda *a, **k: {"mae": 20.0, "baseline_mae": 10.0, "n_days": 14})
    metrics = _train(app_module, trainer)
    assert os.path.exists(app_module.MODEL_PATH)
    assert metrics["status"] == "ok"


def test_too_little_history_saves_nothing(app_module, monkeypatch):
    import train_model
    monkeypatch.setattr(train_model, "now_local", lambda: NOW)
    _seed_sales(app_module, days=5, end=(NOW - timedelta(days=1)).date())
    assert train_model.train_model(app_module.DB_NAME, app_module.MODEL_PATH) is None
    assert not os.path.exists(app_module.MODEL_PATH)


def test_forecast_log_gets_the_upcoming_open_days(app_module, trainer):
    import shop_schedule as sched
    _train(app_module, trainer)
    with app_module.get_conn() as conn:
        rows = conn.execute("SELECT target_date, format, issued_at, horizon_days, expected "
                            "FROM forecast_log").fetchall()
    days = sched.upcoming_open_days(NOW.date())
    assert {r[0] for r in rows} == {d.strftime("%Y-%m-%d") for d in days}
    assert len(rows) == 3 * len(days)
    assert {r[2] for r in rows} == {"2026-10-17 03:00:00"}
    assert all(0 <= r[3] <= 7 for r in rows)
    assert all(r[4] > 0 for r in rows)


def test_forecast_log_drops_the_old_recommended_column(app_module, trainer):
    """Expected sales only: the 90th-percentile recommendation is no longer logged."""
    with app_module.get_conn() as conn:
        conn.execute("DROP TABLE forecast_log")
        conn.execute("""CREATE TABLE forecast_log (
            id INTEGER PRIMARY KEY, target_date TEXT NOT NULL, format TEXT NOT NULL,
            issued_at TEXT NOT NULL, horizon_days INTEGER NOT NULL,
            expected REAL NOT NULL, recommended INTEGER NOT NULL)""")
        conn.execute("INSERT INTO forecast_log (target_date, format, issued_at, horizon_days, "
                     "expected, recommended) VALUES ('2026-10-16', '1kg', '2026-10-16 03:00:00', "
                     "0, 30.5, 42)")
    _train(app_module, trainer)
    with app_module.get_conn() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(forecast_log)")}
        kept = conn.execute("SELECT expected FROM forecast_log WHERE target_date = '2026-10-16'").fetchall()
    assert "recommended" not in columns
    assert kept == [(30.5,)]


def test_old_forest_payload_is_ignored(app_module):
    joblib.dump({"models": {"250g": object()}, "feature_columns": ["weekday", "hour"]},
                app_module.MODEL_PATH)
    assert app_module.get_demand_model() is None
    joblib.dump({"250g": object()}, app_module.MODEL_PATH)
    assert app_module.get_demand_model() is None


def test_no_model_file_means_no_model(app_module):
    assert app_module.get_demand_model() is None


def test_a_fit_within_noise_of_the_baseline_is_still_saved(app_module, trainer, monkeypatch):
    """
    Over 14 days the gap between two decent forecasts is mostly noise. On
    the live data the new model trailed the weekday mean by 10.85 vs 10.37
    there while winning the 44-day walk-forward; rejecting on that would
    freeze the model every night. Only a clearly broken fit is refused.
    """
    _train(app_module, trainer)
    monkeypatch.setattr(trainer, "_light_gate",
                        lambda *a, **k: {"mae": 10.85, "baseline_mae": 10.37, "n_days": 14})
    assert _train(app_module, trainer)["status"] == "ok"


def test_command_line_run_honours_the_app_environment(app_module, trainer, monkeypatch, tmp_path):
    """`python train_model.py` must train the database the app is pointed at,
    not whatever data.db sits in the working directory."""
    target = tmp_path / "elsewhere.pkl"
    monkeypatch.setenv("METALGO_DB", app_module.DB_NAME)
    monkeypatch.setenv("METALGO_MODEL", str(target))
    monkeypatch.chdir(tmp_path)
    trainer.main()
    assert joblib.load(target)["version"] == 2
    assert not (tmp_path / "data.db").exists() and not (tmp_path / "model.pkl").exists()
