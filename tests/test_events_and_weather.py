import sqlite3
from datetime import date

import pytest

import event_service
import meteo


@pytest.fixture(autouse=True)
def no_nhl_network(monkeypatch):
    """
    get_special_event consults the NHL schedule, which is a live HTTP call.
    Stub it so these tests stay hermetic and deterministic.
    """
    monkeypatch.setattr(event_service, "get_game_info", lambda date_obj: (0, 0))
    monkeypatch.setattr(event_service, "_nhl_cache", {})


class TestEventKeys:
    def test_fixed_dates(self):
        assert event_service.get_event_key(date(2026, 6, 24)) == "fixed_06-24"
        assert event_service.get_event_key(date(2026, 2, 14)) == "fixed_02-14"

    def test_ordinary_day_has_no_key(self):
        # A Wednesday in the middle of August, no holiday, no game.
        assert event_service.get_event_key(date(2026, 8, 19)) is None

    @pytest.mark.parametrize("year, expected", [
        (2024, date(2024, 2, 11)),
        (2025, date(2025, 2, 9)),
        (2026, date(2026, 2, 8)),
    ])
    def test_super_bowl_is_the_second_sunday_of_february(self, year, expected):
        assert event_service.get_event_key(expected) == "mobile_super_bowl"
        # The first Sunday must not match.
        from datetime import timedelta
        assert event_service.get_event_key(expected - timedelta(weeks=1)) != "mobile_super_bowl"

    def test_coinciding_events_are_both_reported(self):
        """
        In 2027 the Super Bowl lands on Saint-Valentin. The fixed date stays
        the stored key, but the day must not be forecast as an ordinary
        Valentine's with the Super Bowl silently dropped.
        """
        collision = date(2027, 2, 14)
        keys = [key for key, _, _ in event_service.get_all_events(collision)]
        assert keys == ["fixed_02-14", "mobile_super_bowl"]
        # Stable key for daily_snapshots, so existing rows keep their meaning.
        assert event_service.get_event_key(collision) == "fixed_02-14"

        name, base, _ = event_service.get_special_event(collision)
        assert "St-Valentin" in name and "Super Bowl" in name
        # The stronger prior wins; the two are not multiplied together.
        assert base == pytest.approx(1.5)

    def test_a_lone_fixed_event_is_unaffected(self):
        name, base, _ = event_service.get_special_event(date(2026, 2, 14))
        assert "St-Valentin" in name and "Super Bowl" not in name
        assert base == pytest.approx(1.4)

    def test_saint_jean_is_not_named_twice(self):
        # It is both a fixed event and a Quebec public holiday.
        name, _, _ = event_service.get_special_event(date(2026, 6, 24))
        assert name.count("+") == 0

    @pytest.mark.parametrize("day", [date(2026, 12, 25), date(2027, 12, 25)])
    def test_christmas_day_is_detected(self, day):
        """
        Christmas Day used to fall through every branch and come back with no
        event at all: the public-holiday filter tested for "Noël", which
        matches the library's own name for it, "Jour de Noël".
        """
        assert event_service.get_event_key(day) == "qc_holiday"
        name, base, _ = event_service.get_special_event(day)
        assert "Noël" in name
        assert base > 1.0

    def test_new_years_day_is_detected(self):
        # Survived the old filter only because the library spells it
        # "Jour de l'an" and the filter tested "Jour de l'An".
        assert event_service.get_event_key(date(2027, 1, 1)) == "qc_holiday"

    def test_christmas_eve_still_uses_its_fixed_key(self):
        # 24 December is not a public holiday, so it was never at risk of a
        # duplicate — it is a fixed event and must stay one.
        assert event_service.get_event_key(date(2026, 12, 24)) == "fixed_12-24"
        assert event_service.get_event_key(date(2026, 12, 31)) == "fixed_12-31"

    @pytest.mark.parametrize("year, easter_day", [
        (2026, date(2026, 4, 5)),
        (2027, date(2027, 3, 28)),
    ])
    def test_easter_and_the_saturday_before(self, year, easter_day):
        from datetime import timedelta
        assert event_service.get_event_key(easter_day) == "mobile_easter"
        assert event_service.get_event_key(easter_day - timedelta(days=1)) == "mobile_easter_saturday"


class TestLearnedMultiplier:
    """
    The Bayesian blend, including the NHL/holiday overlap that used to be
    double-counted.
    """

    @staticmethod
    def _make_db(path, rows):
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE daily_snapshots (
            date TEXT PRIMARY KEY, weekday INTEGER NOT NULL, event_key TEXT,
            event_name TEXT, is_nhl_game INTEGER DEFAULT 0, is_nhl_playoff INTEGER DEFAULT 0,
            total_250g INTEGER DEFAULT 0, total_1kg INTEGER DEFAULT 0, total_2kg INTEGER DEFAULT 0)""")
        conn.executemany(
            "INSERT INTO daily_snapshots (date, weekday, event_key, is_nhl_game, is_nhl_playoff, total_250g, total_1kg, total_2kg)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

    def test_returns_the_prior_without_data(self, tmp_path):
        db = tmp_path / "snap.db"
        self._make_db(db, [])
        assert event_service._get_learned_multiplier("fixed_02-14", 1.4, str(db)) == 1.4

    def test_blends_towards_observation(self, tmp_path):
        db = tmp_path / "snap.db"
        rows = [(f"2026-01-{d:02d}", 6, None, 0, 0, 100, 0, 0) for d in range(1, 5)]
        rows.append(("2026-02-14", 6, "fixed_02-14", 0, 0, 200, 0, 0))
        self._make_db(db, rows)

        # One event day worth 2x the baseline, blended against a prior of 1.4
        # with SMOOTHING_FACTOR 3: weight 1/4 on the observation.
        result = event_service._get_learned_multiplier("fixed_02-14", 1.4, str(db))
        expected = 0.25 * 2.0 + 0.75 * 1.4
        assert result == pytest.approx(expected, abs=0.001)

    def test_nhl_ignores_days_that_are_also_holidays(self, tmp_path):
        """
        A game day that is also a holiday must not be counted towards the NHL
        multiplier: the baseline excludes holidays, so charging the holiday
        uplift to the Canadiens inflated the factor.
        """
        db = tmp_path / "snap.db"
        rows = [(f"2026-01-{d:02d}", 6, None, 0, 0, 100, 0, 0) for d in range(1, 5)]
        # Plain game day at 1.1x, plus a holiday+game day at a wild 3x.
        rows.append(("2026-03-01", 6, None, 1, 0, 110, 0, 0))
        rows.append(("2026-06-24", 6, "fixed_06-24", 1, 0, 300, 0, 0))
        self._make_db(db, rows)

        result = event_service._get_learned_multiplier(
            "nhl_regular", 1.1, str(db), is_nhl_key=True)

        # Only the plain game day counts: 110/100 = 1.1, blended with the 1.1
        # prior, which stays 1.1. Including the holiday would have pushed the
        # observation to 2.05 and the blend well above 1.1.
        assert result == pytest.approx(1.1, abs=0.001)


class TestWeatherScoring:
    """
    The score used to live in two hand-synced tables: a factor->score function
    in app.py and a French label->score dict in train_model.py.
    """

    @pytest.mark.parametrize("code, label, score", [
        (0, "Ensoleillé", 2), (1, "Ensoleillé", 2), (2, "Variable", 1),
        (3, "Nuageux", 1), (45, "Brouillard", 1), (61, "Pluie", 0),
        (73, "Neige", 0), (81, "Averses", 0), (95, "Orage", 0),
    ])
    def test_code_label_and_score_agree(self, code, label, score):
        assert meteo.interpret_weather_code(code)[0] == label
        assert meteo.weather_score_from_code(code) == score
        assert meteo.weather_score_from_label(label) == score

    def test_label_scores_are_derived_from_the_code_bands(self):
        """Every label the bands can emit must resolve back to the same score."""
        for code in range(0, 100):
            label, _, score = meteo._band(code)
            assert meteo.weather_score_from_label(label) == score

    def test_score_still_matches_the_old_factor_thresholds(self):
        """
        The retired rule was: factor < 0.85 -> 0, factor >= 1.1 -> 2, else 1.
        Keeping the new table consistent with it means existing models and
        existing history stay comparable.
        """
        for code in range(0, 100):
            _, factor, score = meteo._band(code)
            expected = 0 if factor < 0.85 else (2 if factor >= 1.1 else 1)
            assert score == expected, f"code {code}"

    @pytest.mark.parametrize("label", ["Indisponible", "Inconnu", "", None])
    def test_unknown_means_none_not_average(self, label):
        # None so callers can impute; a neutral score would be a fabricated
        # observation that training treats as ground truth.
        assert meteo.weather_score_from_label(label) is None

    def test_legacy_label_still_resolves(self):
        assert meteo.weather_score_from_label("Orages") == 0
