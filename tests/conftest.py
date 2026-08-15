import os
import sys
from pathlib import Path

# Configure the app before importing it: app.py reads these at import time and
# runs init_db()/start_scheduler() there, so the environment has to be right
# first or the tests would touch the real data.db and start background threads.
_TEST_DIR = Path(__file__).resolve().parent
_ROOT = _TEST_DIR.parent
sys.path.insert(0, str(_ROOT))

os.environ["METALGO_NO_SCHEDULER"] = "1"

import pytest  # noqa: E402


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    """
    A freshly imported app bound to a throwaway database.

    Imported inside the fixture rather than at module scope so each test gets
    its own database file and its own module-level caches.
    """
    db_path = tmp_path / "test.db"
    model_path = tmp_path / "model.pkl"
    monkeypatch.setenv("METALGO_DB", str(db_path))
    monkeypatch.setenv("METALGO_MODEL", str(model_path))
    monkeypatch.setenv("METALGO_NO_SCHEDULER", "1")

    for name in ("app", "train_model", "event_service", "meteo", "shop_schedule"):
        sys.modules.pop(name, None)

    import meteo

    # No network in tests: pin the weather and the NHL schedule.
    monkeypatch.setattr(meteo, "get_current_weather", lambda: ("Ensoleillé", 1.2, 21.0))
    monkeypatch.setattr(meteo, "get_today_hourly_weather", lambda: {})
    monkeypatch.setattr(meteo, "get_weekly_forecast", lambda: [])

    import event_service
    monkeypatch.setattr(event_service, "get_game_info", lambda d: (0, 0))

    import app as app_module_
    monkeypatch.setattr(app_module_, "get_current_weather", lambda: ("Ensoleillé", 1.2, 21.0))
    monkeypatch.setattr(app_module_, "get_today_hourly_weather", lambda: {})
    monkeypatch.setattr(app_module_, "get_game_info", lambda d: (0, 0))
    return app_module_


@pytest.fixture
def client(app_module):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()
