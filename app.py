import os
import re
import sqlite3
import threading
import time as time_module
from contextlib import contextmanager
from datetime import datetime, timedelta

import joblib
from flask import Flask, jsonify, render_template, request

import shop_schedule as sched
from event_service import get_event_key, get_game_info, get_special_event
from meteo import (get_current_weather, get_weekly_forecast, now_local, today_local,
                   weather_score_from_label)
import forecast
import forecast_log
import history
from train_model import PAYLOAD_VERSION, day_inputs, train_model

app = Flask(__name__)

DB_NAME = os.environ.get("METALGO_DB", "data.db")
MODEL_PATH = os.environ.get("METALGO_MODEL", "model.pkl")

# Debug defaults OFF. It used to be hardcoded on while binding 0.0.0.0, which
# exposes the Werkzeug interactive debugger — and therefore arbitrary code
# execution — to every device on the shop's network.
DEBUG = os.environ.get("METALGO_DEBUG") == "1"
HOST = os.environ.get("METALGO_HOST", "0.0.0.0")
PORT = int(os.environ.get("METALGO_PORT", "5000"))

# Whitelists for /api/log. The endpoint is unauthenticated by design (staff
# tablets on the shop LAN), so the least it can do is refuse to store
# anything it didn't ask for. The live database still carries the evidence of
# not doing this: three different spellings of the same conversion label.
VALID_ACTION_TYPES = frozenset({"VENTE"})
VALID_DETAILS = frozenset(sched.FORMATS)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

# How far back a client-supplied timestamp may reach. Long enough to cover a
# tablet that spent a whole shift offline, short enough that a stale queue
# can't rewrite last week.
MAX_BACKDATE_HOURS = 24
# Slack for a tablet clock running slightly fast.
MAX_CLOCK_SKEW_MINUTES = 2

# Nightly retrain time (24h). Chosen well after closing and well before
# opening so it never contends with `/api/log` writes during business hours.
NIGHTLY_RETRAIN_HOUR = 3

# How often the background sampler records the weather while the shop is open.
WEATHER_SAMPLE_INTERVAL_SECONDS = 900

_model_lock = threading.Lock()
_retrain_lock = threading.Lock()
_prediction_lock = threading.Lock()

_cached_payload = None
_cached_model_mtime = 0.0
_prediction_cache = {"key": None, "value": None}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """
    A connection with WAL and a busy timeout, committed and CLOSED on exit.

    Under the default rollback journal a writer blocks every reader, and the
    tablets poll several endpoints every few seconds — so a retrain rewriting
    62 snapshot rows could surface as "database is locked" mid-service. WAL
    lets readers carry on during a write, and the timeout absorbs the rest.

    Closing matters more under WAL than it did before: sqlite3's own context
    manager commits but does not close, so `with sqlite3.connect(...)` left
    the connection alive until garbage collection. A lingering reader holds
    back WAL checkpoints and lets the -wal file grow unbounded.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(conn, table, column, col_type):
    """
    Additive, idempotent schema migration: adds `column` to `table` only if
    it's missing. Never touches existing rows — on an existing DB the new
    column just comes back NULL for old rows, no data is deleted or moved.
    Safe to call on every startup; a no-op once the column already exists.
    """
    existing_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        print(f"Migrated: added column '{column}' to '{table}' (existing rows keep NULL there).")


def init_db():
    # initialize tables for logs, daily snapshots and weather samples
    with get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, timestamp TEXT, action_type TEXT, detail TEXT, meteo_summary TEXT, temperature REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS daily_snapshots (date TEXT PRIMARY KEY, weekday INTEGER NOT NULL, event_key TEXT, event_name TEXT, is_nhl_game INTEGER DEFAULT 0, is_nhl_playoff INTEGER DEFAULT 0, total_250g INTEGER DEFAULT 0, total_1kg INTEGER DEFAULT 0, total_2kg INTEGER DEFAULT 0)")
        # Weather sampled on a timer rather than on a sale. See
        # _weather_sampler_loop for why that distinction matters.
        conn.execute("CREATE TABLE IF NOT EXISTS weather_log (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, condition TEXT, weather_score INTEGER, temperature REAL)")
        # Covers upgrading a pre-existing data.db that predates the
        # `temperature` column, without touching any of its existing rows.
        _ensure_column(conn, "logs", "temperature", "REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_date ON logs(action_type, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_log_ts ON weather_log(timestamp)")
        forecast_log.init_table(conn)


def snapshot_completed_days():
    """
    Aggregate past sales into `daily_snapshots` before retraining.

    Also prunes snapshots whose sales have since been deleted. The table was
    insert-or-update only, so a day whose logs were later removed kept its
    stale totals forever and went on skewing the learned event baselines —
    the live database still carried one such orphan (2026-05-31, six 1kg and
    two 250g recorded against a day with no logs at all).
    """
    today_str = today_local().strftime("%Y-%m-%d")
    with get_conn() as conn:
        conn.execute("""
            DELETE FROM daily_snapshots
            WHERE date NOT IN (SELECT date(timestamp) FROM logs WHERE action_type = 'VENTE')
        """)
        rows = conn.execute("SELECT date(timestamp), SUM(CASE WHEN detail = '250g' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '1kg' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '2kg' THEN 1 ELSE 0 END) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) != ? GROUP BY date(timestamp)", (today_str,)).fetchall()
        for date_str, t_250g, t_1kg, t_2kg in rows:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            event_name, _, _ = get_special_event(date_obj)
            is_game, is_playoff = get_game_info(date_obj)
            conn.execute("INSERT INTO daily_snapshots (date, weekday, event_key, event_name, is_nhl_game, is_nhl_playoff, total_250g, total_1kg, total_2kg) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET total_250g=excluded.total_250g, total_1kg=excluded.total_1kg, total_2kg=excluded.total_2kg, event_key=excluded.event_key, event_name=excluded.event_name, is_nhl_game=excluded.is_nhl_game, is_nhl_playoff=excluded.is_nhl_playoff", (date_str, date_obj.weekday(), get_event_key(date_obj), event_name, is_game, is_playoff, t_250g or 0, t_1kg or 0, t_2kg or 0))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _normalise_payload(payload):
    """
    Accepts only the current payload shape: a fitted forecast.DemandModel.

    Older model.pkl files hold per-hour random forests (or, older still, a
    bare {format: estimator} dict). Nothing reads those any more, so they are
    refused outright and the next retrain replaces them.
    """
    if (not isinstance(payload, dict) or payload.get("version") != PAYLOAD_VERSION
            or not isinstance(payload.get("demand_model"), forecast.DemandModel)):
        print("Ignoring legacy model.pkl; a retrain will replace it.")
        return None
    return payload


def get_model_payload():
    """Load the model payload from disk, reusing the cache unless it changed."""
    global _cached_payload, _cached_model_mtime
    try:
        mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        return None

    with _model_lock:
        if _cached_payload is not None and mtime <= _cached_model_mtime:
            return _cached_payload

    try:
        payload = _normalise_payload(joblib.load(MODEL_PATH))
    except Exception as e:
        print(f"Could not load {MODEL_PATH}: {e}")
        return None
    if payload is None:
        return None

    with _model_lock:
        _cached_payload, _cached_model_mtime = payload, mtime
    return payload


def get_demand_model():
    payload = get_model_payload()
    return payload["demand_model"] if payload else None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _today_sales(conn, date_str):
    return dict(conn.execute(
        "SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail",
        (date_str,)).fetchall())


def _hourly_breakdown(rows, py_weekday):
    """Hour -> per-format counts, over the scheduled hours plus any observed."""
    buckets = sched.hour_buckets(py_weekday, [int(h) for h, _, _ in rows])
    hourly = {f"{h:02d}": {fmt: 0 for fmt in sched.FORMATS} for h in buckets}
    for hour, fmt, count in rows:
        if hour in hourly and fmt in hourly[hour]:
            hourly[hour][fmt] = count
    return hourly


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', est_ouvert=sched.is_open_at(now_local()))


@app.route('/api/status')
def get_status():
    open_status = sched.is_open_at(now_local())
    return jsonify({"ouvert": open_status, "message": "Ouvert" if open_status else "Fermé"})


def _resolve_timestamp(raw, now):
    """
    Pick the timestamp for a logged sale.

    A tablet that lost wifi queues its taps locally and replays them when the
    connection comes back, so it sends the moment the sale actually happened.
    Stamping those at flush time instead would pile a whole morning's sales
    onto one afternoon minute and quietly corrupt the hourly distribution the
    model is trained on. Anything missing, malformed, in the future or older
    than a day falls back to the server clock.
    """
    if not isinstance(raw, str) or not _DATETIME_RE.match(raw):
        return now
    try:
        client_time = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return now
    if client_time > now + timedelta(minutes=MAX_CLOCK_SKEW_MINUTES):
        return now
    if client_time < now - timedelta(hours=MAX_BACKDATE_HOURS):
        return now
    return client_time


@app.route('/api/log', methods=['POST'])
def log_action():
    # log a new sale entry with current weather (and temperature) context
    data = request.get_json(silent=True) or {}
    action_type, detail = data.get('type'), data.get('detail')
    if action_type not in VALID_ACTION_TYPES or detail not in VALID_DETAILS:
        return jsonify({"status": "error", "message": "Type ou format invalide"}), 400

    timestamp = _resolve_timestamp(data.get('client_time'), now_local())
    condition, _, temperature = get_current_weather()
    with get_conn() as conn:
        conn.execute("INSERT INTO logs (timestamp, action_type, detail, meteo_summary, temperature) VALUES (?, ?, ?, ?, ?)",
                     (timestamp.strftime("%Y-%m-%d %H:%M:%S"), action_type, detail, condition, temperature))
    return jsonify({"status": "success"})


@app.route('/api/undo', methods=['POST'])
def undo_last_action():
    """
    Revert today's most recent sale.

    Scoped to today AND to VENTE on purpose. It used to delete the
    globally-last row of any kind, so pressing "Annuler" before the first
    sale of the day silently deleted *yesterday's* closing sale — with no
    confirmation and no way back.
    """
    today_str = today_local().strftime("%Y-%m-%d")
    with get_conn() as conn:
        last_row = conn.execute(
            "SELECT id, detail FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? ORDER BY id DESC LIMIT 1",
            (today_str,)).fetchone()
        if last_row:
            conn.execute("DELETE FROM logs WHERE id = ?", (last_row[0],))
            return jsonify({"status": "success", "message": f"{last_row[1]} annulé"})
    return jsonify({"status": "error", "message": "Aucune vente aujourd'hui"})


@app.route('/api/stats')
def get_stats():
    # retrieve current day statistics and hourly chart data
    now = now_local()
    today = now.strftime("%Y-%m-%d")
    with get_conn() as conn:
        sales = _today_sales(conn, today)
        peak_hour = conn.execute("SELECT strftime('%H', timestamp) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp) ORDER BY COUNT(*) DESC LIMIT 1", (today,)).fetchone()
        hourly_raw = conn.execute("SELECT strftime('%H', timestamp), detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp), detail", (today,)).fetchall()

    stats = {fmt: sales.get(fmt, 0) for fmt in sched.FORMATS}
    return jsonify({
        "c250": stats["250g"],
        "c1kg": stats["1kg"],
        "c2kg": stats["2kg"],
        "peak_hour": f"{peak_hour[0]}h00" if peak_hour else "--",
        "total_mass": f"{(stats['250g'] * 0.25) + stats['1kg'] + (stats['2kg'] * 2):.2f} kg",
        "hourly_data": _hourly_breakdown(hourly_raw, now.weekday()),
    })


@app.route('/api/history')
def get_history():
    # fetch the last five actions for the history feed
    with get_conn() as conn:
        rows = conn.execute("SELECT action_type, detail, timestamp FROM logs ORDER BY id DESC LIMIT 5").fetchall()
    return jsonify([{"type": r[0], "detail": r[1],
                     "heure": datetime.strptime(r[2].split('.')[0], "%Y-%m-%d %H:%M:%S").strftime("%H:%M")}
                    for r in rows])


@app.route('/api/history_days')
def get_history_days():
    # summary of every day with reported sales (most recent first), for the Historique page
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT date(timestamp),
                   SUM(CASE WHEN detail = '250g' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN detail = '1kg' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN detail = '2kg' THEN 1 ELSE 0 END)
            FROM logs
            WHERE action_type = 'VENTE'
            GROUP BY date(timestamp)
            ORDER BY date(timestamp) DESC
        """).fetchall()
    return jsonify([{"date": r[0], "c250": r[1], "c1kg": r[2], "c2kg": r[3]} for r in rows])


@app.route('/api/history_day/<date_str>')
def get_history_day(date_str):
    # hourly breakdown for one specific day, used when a day is expanded
    if not _DATE_RE.match(date_str):
        return jsonify({"error": "Date invalide"}), 400
    try:
        py_weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    except ValueError:
        return jsonify({"error": "Date invalide"}), 400

    with get_conn() as conn:
        rows = conn.execute("SELECT strftime('%H', timestamp), detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp), detail", (date_str,)).fetchall()
    return jsonify(_hourly_breakdown(rows, py_weekday))


@app.route('/api/prediction')
def get_prediction():
    # calculate predictions for the remainder of the day
    now = now_local()
    today_str = now.strftime("%Y-%m-%d")

    with get_conn() as conn:
        real_sales = _today_sales(conn, today_str)

    # Recomputing three 100-tree ensembles on every 3s poll was pure waste:
    # nothing about the answer changes until the clock ticks over a minute or
    # a sale is logged. Both are in the key, so a new sale still refreshes
    # immediately.
    cache_key = (now.strftime("%Y-%m-%d %H:%M"), tuple(sorted(real_sales.items())))

    # The lock deliberately spans the computation, not just the cache lookup.
    # Several tablets poll in lockstep, so a plain check-then-compute lets
    # every one of them miss the same cold cache and build its own three
    # 100-tree ensembles at once. Serialising means the first request computes
    # and the rest return its result.
    with _prediction_lock:
        if _prediction_cache["key"] == cache_key:
            return jsonify(_prediction_cache["value"])
        response = _build_prediction(now, real_sales)
        _prediction_cache.update({"key": cache_key, "value": response})
    return jsonify(response)


def _build_prediction(now, real_sales):
    weather_cond, _, _ = get_current_weather()
    formats = list(sched.FORMATS)
    sold = {fmt: real_sales.get(fmt, 0) for fmt in formats}

    if sched.is_closed_on(now.date()):
        # Every key the open-day branch returns, so the client never has to
        # deal with a differently-shaped payload on Mondays.
        return {"heures_restantes": 0, "meteo": weather_cond,
                "previsions": dict(sold), "previsions_min": dict(sold), "previsions_max": dict(sold),
                "evenement": None, "debug_info": "Fermé"}

    open_hour, close_hour = sched.OPEN_HOUR, sched.close_hour(now.weekday())
    start_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    end_day = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    time_left = ((end_day - start_day) if now < start_day else max(end_day - now, timedelta(0)))
    event_name = get_special_event(now.date(), db_path=DB_NAME)[0]

    with get_conn() as conn:
        customers = history.customers_on(conn, now.date())

    model = get_demand_model()
    if model is None:
        predictions = dict(sold)
        low = high = dict(sold)
        debug_msg = "Modèle manquant"
    else:
        out = forecast.intraday(model, now.date(), now, sold, customers,
                                mean_kwargs=day_inputs(now.date(), DB_NAME, get_weekly_forecast()))
        predictions = {fmt: f["point"] for fmt, f in out["formats"].items()}
        low = {fmt: f["low"] for fmt, f in out["formats"].items()}
        high = {fmt: f["high"] for fmt, f in out["formats"].items()}
        # Read by the trend badge: the pace of the day against the model.
        debug_msg = f"{int(round(out['multiplier'] * 100))}%"

    return {
        "heures_restantes": round(time_left.total_seconds() / 3600, 1),
        "meteo": weather_cond,
        "previsions": predictions,
        "previsions_min": low,
        "previsions_max": high,
        "evenement": event_name,
        "debug_info": debug_msg,
    }


# ---------------------------------------------------------------------------
# The week ahead
# ---------------------------------------------------------------------------

def _recommendations(model, day, weekly):
    """(expected, recommended) per format for `day`, or two empty dicts without a model."""
    if model is None:
        return {}, {}
    kwargs = day_inputs(day, DB_NAME, weekly)
    expected = {fmt: round(forecast.mean(model, day, fmt, **kwargs), 1) for fmt in sched.FORMATS}
    recommended = {fmt: forecast.recommend(model, day, fmt, **kwargs) for fmt in sched.FORMATS}
    return expected, recommended


@app.route('/api/forecast_week')
def forecast_week():
    """
    Today and the following seven calendar days that are open, each with the
    expected sales and the bags to prepare per format.

    The planner often prepares a whole weekend two days ahead, so nothing
    here assumes anyone looks at it on the morning itself.
    """
    today = today_local()
    model = get_demand_model()
    weekly = get_weekly_forecast() if (forecast.USE_WEATHER or forecast.USE_TEMPERATURE) else []
    result = []
    for day in sched.upcoming_open_days(today):
        expected, recommended = _recommendations(model, day, weekly)
        result.append({
            "date": day.strftime("%Y-%m-%d"),
            "date_affichee": f"{sched.WEEKDAY_NAMES_FR[day.weekday()]} {day.day}",
            "evenement": get_special_event(day, db_path=DB_NAME)[0],
            "attendu": expected,
            "recommande": recommended,
        })
    return jsonify(result)


@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    # Manual on-demand recalibration — e.g. right after an unusual event you
    # want reflected immediately. A full retrain also runs automatically every
    # night (see start_scheduler below), so this is a supplement to that, not
    # the only way it happens.
    if not _retrain_lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "Calibrage déjà en cours"}), 409
    try:
        snapshot_completed_days()
        metrics = train_model(db_name=DB_NAME, model_path=MODEL_PATH)
        return jsonify({"status": "success", "message": "Calibrage terminé",
                        "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        _retrain_lock.release()


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

def record_weather_sample():
    """Store one weather observation. Returns True if a row was written."""
    condition, _, temperature = get_current_weather()
    score = weather_score_from_label(condition)
    if score is None:
        # API unavailable. Recording a neutral guess here would be worse than
        # recording nothing: training treats these rows as ground truth.
        return False
    with get_conn() as conn:
        conn.execute("INSERT INTO weather_log (timestamp, condition, weather_score, temperature) VALUES (?, ?, ?, ?)",
                     (now_local().strftime("%Y-%m-%d %H:%M:%S"), condition, score, temperature))
    return True


def _weather_sampler_loop():
    """
    Sample the weather on a timer while the shop is open.

    `logs.meteo_summary` only ever gets written when something sells, so the
    weather feature was sampled in proportion to sales: a washout morning
    with no customers contributed no rain observations at all, and the day's
    modal weather skewed towards whatever it was doing during the busy hours.
    A timer doesn't care whether anyone bought anything.
    """
    while True:
        try:
            if sched.is_open_at(now_local()):
                record_weather_sample()
        except Exception as e:
            print(f"[{now_local()}] Weather sampling failed: {e}")
        time_module.sleep(WEATHER_SAMPLE_INTERVAL_SECONDS)


def _seconds_until(hour, minute=0):
    now = now_local()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _run_retrain_safely(label):
    if not _retrain_lock.acquire(blocking=False):
        print(f"[{now_local()}] {label} retrain skipped, another one is running.")
        return
    try:
        print(f"[{now_local()}] {label} retrain starting...")
        snapshot_completed_days()
        train_model(db_name=DB_NAME, model_path=MODEL_PATH)
        print(f"[{now_local()}] {label} retrain complete.")
    except Exception as e:
        print(f"[{now_local()}] {label} retrain failed: {e}")
    finally:
        _retrain_lock.release()


def _nightly_retrain_loop():
    # Catch up once immediately (covers downtime, or data that piled up since
    # the last run), then settle into a fixed nightly schedule.
    _run_retrain_safely("Startup catch-up")
    while True:
        time_module.sleep(_seconds_until(NIGHTLY_RETRAIN_HOUR))
        _run_retrain_safely("Nightly")


def start_scheduler():
    """
    Start the background threads.

    Set METALGO_NO_SCHEDULER=1 to suppress them — the test suite does, and it
    is also what you want for a one-off management command.

    There is no reloader guard here any more. The old one keyed off
    WERKZEUG_RUN_MAIN, which only exists under Flask's debug reloader and is
    absent under gunicorn; combined with the whole block living behind
    `if __name__ == '__main__'`, a WSGI deployment silently got no scheduler,
    no schema migration and no tables at all. The reloader is switched off at
    the bottom of this file instead, so there is only ever one process.
    """
    if os.environ.get("METALGO_NO_SCHEDULER") == "1":
        return
    threading.Thread(target=_nightly_retrain_loop, daemon=True).start()
    threading.Thread(target=_weather_sampler_loop, daemon=True).start()
    print(f"Background threads started (retrain {NIGHTLY_RETRAIN_HOUR:02d}:00 daily, "
          f"weather every {WEATHER_SAMPLE_INTERVAL_SECONDS}s while open).")


# Run at import, not under `if __name__ == '__main__'`, so that gunicorn and
# any other WSGI server get a migrated schema and the background jobs too.
init_db()
start_scheduler()

if __name__ == '__main__':
    # use_reloader=False: the reloader forks a second process that would run
    # its own copy of the background threads.
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
