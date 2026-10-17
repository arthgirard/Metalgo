
import pytest
from datetime import datetime, timedelta

from helpers import _count, _insert, _train_as_of


class TestLogValidation:
    """
    /api/log is unauthenticated, so the whitelist is the only thing standing
    between shop wifi and the database — and between a stored payload and
    every tablet's history feed.
    """

    def test_accepts_a_valid_sale(self, client, app_module):
        assert client.post('/api/log', json={'type': 'VENTE', 'detail': '250g'}).status_code == 200
        assert _count(app_module) == 1

    def test_rejects_an_unknown_format(self, client, app_module):
        resp = client.post('/api/log', json={'type': 'VENTE', 'detail': '500g'})
        assert resp.status_code == 400
        assert _count(app_module) == 0

    def test_rejects_a_script_payload(self, client, app_module):
        resp = client.post('/api/log', json={
            'type': 'VENTE', 'detail': '<img src=x onerror=alert(1)>'})
        assert resp.status_code == 400
        assert _count(app_module) == 0

    def test_rejects_an_unknown_action_type(self, client, app_module):
        assert client.post('/api/log', json={'type': 'CONVERSION', 'detail': '250g'}).status_code == 400
        assert _count(app_module) == 0

    def test_rejects_an_empty_body(self, client, app_module):
        assert client.post('/api/log').status_code == 400
        assert _count(app_module) == 0


class TestClientTimestamp:
    """
    Sales replayed from a tablet's offline queue carry the time they actually
    happened; stamping them at flush time would pile a morning onto one
    afternoon minute and skew the hourly distribution the model trains on.
    """

    def test_a_recent_client_time_is_honoured(self, client, app_module):
        earlier = (app_module.now_local() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        client.post('/api/log', json={'type': 'VENTE', 'detail': '1kg', 'client_time': earlier})
        with app_module.get_conn() as conn:
            assert conn.execute("SELECT timestamp FROM logs").fetchone()[0] == earlier

    def test_a_future_client_time_is_ignored(self, app_module):
        now = datetime(2026, 8, 14, 12, 0, 0)
        future = (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        assert app_module._resolve_timestamp(future, now) == now

    def test_a_stale_client_time_is_ignored(self, app_module):
        now = datetime(2026, 8, 14, 12, 0, 0)
        ancient = (now - timedelta(days=9)).strftime("%Y-%m-%d %H:%M:%S")
        assert app_module._resolve_timestamp(ancient, now) == now

    def test_small_clock_skew_is_tolerated(self, app_module):
        now = datetime(2026, 8, 14, 12, 0, 0)
        slightly_fast = now + timedelta(seconds=30)
        assert app_module._resolve_timestamp(
            slightly_fast.strftime("%Y-%m-%d %H:%M:%S"), now) == slightly_fast

    def test_garbage_falls_back_to_the_server_clock(self, app_module):
        now = datetime(2026, 8, 14, 12, 0, 0)
        for junk in ["not-a-date", "2026-08-14", "'; DROP TABLE logs;--", 12345, None]:
            assert app_module._resolve_timestamp(junk, now) == now


class TestUndo:
    """
    Undo used to delete the globally-last row of any kind: pressing it before
    the first sale of the day silently deleted yesterday's closing sale.
    """

    def test_removes_todays_last_sale(self, client, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        _insert(app_module, f"{today} 10:00:00", "250g")
        _insert(app_module, f"{today} 11:00:00", "1kg")

        resp = client.post('/api/undo')
        assert resp.status_code == 200
        assert resp.get_json()['status'] == 'success'
        with app_module.get_conn() as conn:
            remaining = [r[0] for r in conn.execute("SELECT detail FROM logs")]
        assert remaining == ["250g"]

    def test_never_reaches_back_to_a_previous_day(self, client, app_module):
        yesterday = (app_module.now_local() - timedelta(days=1)).strftime("%Y-%m-%d")
        _insert(app_module, f"{yesterday} 16:00:00", "2kg")

        resp = client.post('/api/undo')
        assert resp.get_json()['status'] == 'error'
        assert _count(app_module) == 1

    def test_ignores_retired_conversion_rows(self, client, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        _insert(app_module, f"{today} 10:00:00", "250g")
        _insert(app_module, f"{today} 12:00:00", "2kg → 1 kg", action_type="CONVERSION")

        assert client.post('/api/undo').get_json()['status'] == 'success'
        with app_module.get_conn() as conn:
            remaining = sorted(r[0] for r in conn.execute("SELECT detail FROM logs"))
        assert remaining == ["2kg → 1 kg"]


class TestSnapshotPruning:
    """
    daily_snapshots was insert-or-update only, so a day whose sales were later
    deleted kept its totals forever and went on skewing the learned baselines.
    """

    def test_orphan_snapshots_are_removed(self, app_module):
        today = app_module.now_local()
        real_day = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        for hour in range(10, 16):
            _insert(app_module, f"{real_day} {hour}:00:00", "250g")

        with app_module.get_conn() as conn:
            conn.execute(
                "INSERT INTO daily_snapshots (date, weekday, total_250g, total_1kg, total_2kg)"
                " VALUES ('2020-05-31', 6, 2, 6, 0)")

        app_module.snapshot_completed_days()

        with app_module.get_conn() as conn:
            dates = [r[0] for r in conn.execute("SELECT date FROM daily_snapshots")]
        assert "2020-05-31" not in dates
        assert real_day in dates

    def test_todays_partial_day_is_not_snapshotted(self, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        _insert(app_module, f"{today} 10:00:00", "250g")
        app_module.snapshot_completed_days()
        with app_module.get_conn() as conn:
            dates = [r[0] for r in conn.execute("SELECT date FROM daily_snapshots")]
        assert dates == []


class TestConnectionHandling:
    def test_connections_are_closed_on_exit(self, app_module):
        """
        sqlite3's own context manager commits but does not close. Under WAL a
        lingering reader holds back checkpoints and lets the -wal file grow.
        """
        with app_module.get_conn() as conn:
            conn.execute("SELECT 1")
        with pytest.raises(Exception):
            conn.execute("SELECT 1")          # already closed

    def test_a_failing_block_rolls_back_and_still_closes(self, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        with pytest.raises(RuntimeError):
            with app_module.get_conn() as conn:
                conn.execute(
                    "INSERT INTO logs (timestamp, action_type, detail) VALUES (?, 'VENTE', '250g')",
                    (f"{today} 10:00:00",))
                raise RuntimeError("boom")

        assert _count(app_module) == 0
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_wal_mode_is_enabled(self, app_module):
        with app_module.get_conn() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


class TestHistoryDay:
    def test_rejects_a_malformed_date(self, client):
        assert client.get('/api/history_day/not-a-date').status_code == 400
        assert client.get('/api/history_day/2026-13-99').status_code == 400

    def test_buckets_follow_the_days_real_hours(self, client, app_module):
        # 2026-08-13 is a Thursday, so the shop is open until 18h.
        _insert(app_module, "2026-08-13 17:30:00", "250g")
        hours = client.get('/api/history_day/2026-08-13').get_json()
        assert list(hours) == [f"{h:02d}" for h in range(10, 18)]
        assert hours["17"]["250g"] == 1

    def test_out_of_hours_sales_are_still_shown(self, client, app_module):
        # 2026-08-12 is a Wednesday, closing at 17h. A 9h sale is real and
        # used to be invisible in both the chart and the training grid.
        _insert(app_module, "2026-08-12 09:15:00", "1kg")
        hours = client.get('/api/history_day/2026-08-12').get_json()
        assert "09" in hours
        assert hours["09"]["1kg"] == 1


class TestStatusAndStats:
    def test_status_reports_a_boolean(self, client):
        body = client.get('/api/status').get_json()
        assert isinstance(body['ouvert'], bool)
        assert body['message'] in ("Ouvert", "Fermé")

    def test_stats_totals_the_mass_correctly(self, client, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        _insert(app_module, f"{today} 10:00:00", "250g")   # 0.25
        _insert(app_module, f"{today} 10:30:00", "1kg")    # 1.00
        _insert(app_module, f"{today} 11:00:00", "2kg")    # 2.00
        body = client.get('/api/stats').get_json()
        assert body['total_mass'] == "3.25 kg"
        assert (body['c250'], body['c1kg'], body['c2kg']) == (1, 1, 1)

    def test_conversion_rows_are_not_counted_as_sales(self, client, app_module):
        today = app_module.now_local().strftime("%Y-%m-%d")
        _insert(app_module, f"{today} 10:00:00", "2kg → 1 kg", action_type="CONVERSION")
        body = client.get('/api/stats').get_json()
        assert (body['c250'], body['c1kg'], body['c2kg']) == (0, 0, 0)


class TestPrediction:
    @pytest.fixture(autouse=True)
    def model(self, app_module, monkeypatch):
        _train_as_of(app_module, monkeypatch, datetime(2026, 8, 18, 3, 0, 0))

    def test_closed_day_returns_the_same_shape_as_an_open_day(self, app_module, client, monkeypatch):
        """
        The Monday early-return used to omit previsions_min/max and
        temperature, leaving the UI reading undefined fields.
        """
        monday = datetime(2026, 8, 17, 12, 0, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: monday)
        closed = client.get('/api/prediction').get_json()

        tuesday = datetime(2026, 8, 18, 12, 0, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: tuesday)
        app_module._prediction_cache.update({"key": None, "value": None})
        open_day = client.get('/api/prediction').get_json()

        assert set(closed) >= set(open_day)
        assert closed['heures_restantes'] == 0
        assert closed['debug_info'] == "Fermé"

    def test_total_includes_what_is_already_sold(self, app_module, client, monkeypatch):
        tuesday_afternoon = datetime(2026, 8, 18, 14, 30, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: tuesday_afternoon)
        today = tuesday_afternoon.strftime("%Y-%m-%d")
        for hour in range(10, 14):
            _insert(app_module, f"{today} {hour}:00:00", "250g")

        body = client.get('/api/prediction').get_json()
        # The headline number is the whole day, so it can never fall below
        # the bags already sold.
        assert body['previsions']['250g'] >= 4
        assert body['previsions_min']['250g'] >= 4

    def test_range_brackets_the_point_estimate(self, app_module, client, monkeypatch):
        tuesday = datetime(2026, 8, 18, 11, 0, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: tuesday)
        body = client.get('/api/prediction').get_json()
        for fmt in ("250g", "1kg", "2kg"):
            assert body['previsions_min'][fmt] <= body['previsions'][fmt] <= body['previsions_max'][fmt]

    def test_band_narrows_through_the_day(self, app_module, client, monkeypatch):
        def width(now):
            monkeypatch.setattr(app_module, "now_local", lambda: now)
            app_module._prediction_cache.update({"key": None, "value": None})
            body = client.get('/api/prediction').get_json()
            return body['previsions_max']['250g'] - body['previsions_min']['250g']
        assert width(datetime(2026, 8, 18, 11, 0)) > width(datetime(2026, 8, 18, 15, 0))

    def test_band_never_dips_below_what_is_already_sold(self, app_module, client, monkeypatch):
        """
        The day's total cannot come in under the bags already in customers'
        hands, however wide the error band is.
        """
        afternoon = datetime(2026, 8, 18, 16, 0, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: afternoon)
        today = afternoon.strftime("%Y-%m-%d")
        for hour in range(10, 16):
            for i in range(5):
                _insert(app_module, f"{today} {hour}:{i * 10:02d}:00", "250g")

        body = client.get('/api/prediction').get_json()
        assert body['previsions_min']['250g'] >= 30

    def test_payload_has_the_prediction_keys_only(self, app_module, client, monkeypatch):
        monkeypatch.setattr(app_module, "now_local", lambda: datetime(2026, 8, 18, 14, 0, 0))
        assert set(client.get('/api/prediction').get_json()) == {
            "heures_restantes", "meteo", "previsions", "previsions_min", "previsions_max",
            "evenement", "debug_info"}


class TestPredictionWithoutModel:
    def test_mirrors_what_is_sold(self, app_module, client, monkeypatch):
        tuesday = datetime(2026, 8, 18, 14, 0, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: tuesday)
        for hour in range(10, 14):
            _insert(app_module, f"2026-08-18 {hour}:00:00", "1kg")
        body = client.get('/api/prediction').get_json()
        assert body['previsions'] == {"250g": 0, "1kg": 4, "2kg": 0}
        assert body['previsions_min'] == body['previsions_max'] == body['previsions']
        assert body['debug_info'] == "Modèle manquant"


class TestPredictionCache:
    def test_is_cached_within_the_same_minute(self, app_module, client, monkeypatch):
        frozen = datetime(2026, 8, 18, 14, 30, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: frozen)

        calls = []
        original = app_module._build_prediction

        def counting(now, sales):
            calls.append(1)
            return original(now, sales)

        monkeypatch.setattr(app_module, "_build_prediction", counting)
        client.get('/api/prediction')
        client.get('/api/prediction')
        client.get('/api/prediction')
        assert len(calls) == 1

    def test_a_new_sale_busts_the_cache(self, app_module, client, monkeypatch):
        frozen = datetime(2026, 8, 18, 14, 30, 0)
        monkeypatch.setattr(app_module, "now_local", lambda: frozen)

        calls = []
        original = app_module._build_prediction
        monkeypatch.setattr(app_module, "_build_prediction",
                            lambda now, sales: (calls.append(1), original(now, sales))[1])

        client.get('/api/prediction')
        _insert(app_module, frozen.strftime("%Y-%m-%d %H:%M:%S"), "250g")
        client.get('/api/prediction')
        assert len(calls) == 2
