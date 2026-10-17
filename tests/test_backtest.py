from datetime import date, timedelta

import pytest

from helpers import _insert


def _seed(app_module, days=40, end=date(2026, 9, 30)):
    """`days` open days of sales ending at `end`, busier on weekends."""
    d, seeded = end, 0
    while seeded < days:
        if d.weekday() != 0:
            busy = 2 if d.weekday() >= 5 else 1
            for hour in range(10, 17):
                for i, fmt in enumerate(["250g"] * (2 * busy) + ["1kg"] * busy):
                    _insert(app_module, f"{d} {hour:02d}:{i * 5:02d}:00", fmt)
            if seeded % 3 == 0:
                _insert(app_module, f"{d} 15:30:00", "2kg")
            seeded += 1
        d -= timedelta(days=1)


@pytest.fixture
def backtest(app_module):
    import backtest as backtest_
    return backtest_


def test_harness_reports_every_candidate_and_horizon(app_module, backtest):
    _seed(app_module)
    out = backtest.run(app_module.DB_NAME, test_days=5, horizons=(0, 2))
    assert set(out) == {"forest", "weekday_mean", "blend"}
    assert set(out["blend"]) == {0, 2}
    result = out["blend"][0]
    assert set(result["mae"]) == {"250g", "1kg", "2kg", "all"}
    assert 0 <= result["sellout_rate"] <= 1
    assert 0 <= result["exceed_q90"] <= 1
    assert result["leftover"] >= 0
    assert out["forest"][0]["exceed_q90"] is None
    assert out["weekday_mean"][0]["exceed_q90"] is None


def test_horizon_two_never_uses_the_two_previous_days(app_module, backtest, monkeypatch):
    _seed(app_module)
    import forecast
    seen = []
    original = forecast.mean

    def spy(model, target, fmt, *args, **kwargs):
        seen.append((model.as_of, target))
        return original(model, target, fmt, *args, **kwargs)

    monkeypatch.setattr(forecast, "mean", spy)
    backtest.run(app_module.DB_NAME, test_days=5, horizons=(2,), candidates=("blend",))
    assert seen
    for as_of, target in seen:
        assert as_of <= target - timedelta(days=2)


def test_steady_history_is_predicted_closely(app_module, backtest):
    _seed(app_module)
    out = backtest.run(app_module.DB_NAME, test_days=5, horizons=(0,), candidates=("blend",))
    # Seeded weekdays sell 21 bags (+ a stray 2kg), weekends 42.
    assert out["blend"][0]["mae"]["all"] < 2


def test_trial_candidates_run(app_module, backtest):
    _seed(app_module)
    trials = ("blend+events", "blend+events+nhl", "blend+weather", "blend+temperature")
    out = backtest.run(app_module.DB_NAME, test_days=5, horizons=(0,), candidates=("blend", *trials))
    for name in ("blend+events", "blend+events+nhl", "blend+weather"):
        assert out[name][0]["mae"]["all"] == pytest.approx(out["blend"][0]["mae"]["all"], abs=0.5)
    # No weather_log readings in the seeded data: nothing to test temperature on.
    assert out["blend+temperature"][0] is None
