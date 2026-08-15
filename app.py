import math
import os
import re
import sqlite3
import threading
import time as time_module
from contextlib import contextmanager
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import shop_schedule as sched
from event_service import get_event_key, get_game_info, get_special_event
from meteo import (FALLBACK_TEMPERATURE, NEUTRAL_WEATHER_SCORE, get_current_weather,
                   get_today_hourly_weather, get_weekly_forecast, now_local, today_local,
                   weather_score_from_label)
from train_model import FEATURE_COLUMNS, train_model

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

# "Virtual" prior sales the intraday correction ratio must out-weigh before it's
# trusted (same smoothing idea as SMOOTHING_FACTOR in event_service.py).
# Also scaled by how much of the day has elapsed in get_prediction, so a
# popular format can't earn high confidence just from raw predicted volume
# in the first hour — see the time_fraction comment there.
CORRECTION_SMOOTHING = 3.0

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
    Accepts the current payload shape and rejects anything trained on a
    different feature set.

    Older model.pkl files were a bare {format: estimator} dict fitted with an
    `is_special_event` column that no longer exists. Feeding today's frames
    to one of those produces silent nonsense rather than an error, so they
    are refused outright and the next retrain replaces them.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    if "models" not in payload:
        print("Ignoring legacy model.pkl (pre-metrics format); a retrain will replace it.")
        return None
    if list(payload.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        print("Ignoring model.pkl trained on a different feature set; a retrain will replace it.")
        return None
    if not isinstance(payload.get("models"), dict) or not payload["models"]:
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


def get_models():
    payload = get_model_payload()
    return payload["models"] if payload else None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_total(model, feature_rows, weights):
    """
    The weighted SUM of `model`'s predictions over feature_rows.

    This used to also return a low/high band taken from the spread across the
    forest's individual trees. That spread measures how much the trees
    disagree with each other, which is not the same thing as how wrong they
    tend to be — and once the trees were depth-capped they agreed almost
    perfectly, collapsing the displayed range to about ±1 bag against a
    measured daily error near 8. The band now comes from held-out error
    instead; see _uncertainty.
    """
    if not feature_rows:
        return 0.0

    # Pass the DataFrame itself, not .values: the forest was fitted on named
    # columns and warns on a bare array. Selecting FEATURE_COLUMNS also
    # guarantees the order matches training, which sklearn does not do for
    # you. One vectorised predict() beats looping the trees in Python.
    X = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    return float((model.predict(X) * np.array(weights)).sum())


def _uncertainty(payload, fmt, remaining_fraction):
    """
    Half-width of the range shown under each forecast, in bags, or None when
    the model has never been scored.

    `daily_total_mae` is the mean absolute error on a whole day's total for
    this format, measured on the chronological holdout — an empirical answer
    to "how far out is this number usually?".

    It is scaled down as the day progresses, because the hours already sold
    are counted exactly and only the remaining ones can still be wrong. The
    square root is the standard way to aggregate per-hour errors: at opening
    the full day is at stake and the band is the full measured error; with
    half the day left it is about 71% of it; at closing time it is zero.
    """
    metrics = (payload or {}).get("metrics") or {}
    per_format = metrics.get("per_format") or {}
    daily_mae = (per_format.get(fmt) or {}).get("daily_total_mae")
    if not daily_mae:
        return None
    return daily_mae * math.sqrt(max(0.0, min(remaining_fraction, 1.0)))


def _feature_rows(hours, sql_weekday, hourly_weather, default_score, default_temp, is_game, is_playoff):
    """
    Build model input for a list of hours, using that hour's own weather.

    The intraday forecast used to stamp the weather observed *right now*
    across every remaining hour, so a sunny 10h projected a sunny 16h
    straight through a forecast rain band. Where the hourly forecast is
    available each hour now carries its own score and temperature; where it
    isn't, the current-conditions defaults stand in.

    `temperature` is still populated even though it is not currently in
    FEATURE_COLUMNS — _predict_total selects the feature columns it
    needs and ignores the rest. Keeping it here means reinstating the
    feature is a one-line change in train_model once the weather sampler
    has accumulated enough per-hour readings. See the note there.
    """
    rows = []
    for hour in hours:
        observed = hourly_weather.get(hour) or {}
        temperature = observed.get("temperature")
        rows.append({
            'weekday': sql_weekday,
            'hour': hour,
            'weather_score': observed.get("score", default_score),
            'is_game_day': is_game,
            'is_playoff_game': is_playoff,
            'temperature': default_temp if temperature is None else temperature,
        })
    return rows


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
    weather_cond, weather_factor, temperature = get_current_weather()
    temp_for_model = temperature if temperature is not None else FALLBACK_TEMPERATURE
    current_score = weather_score_from_label(weather_cond)
    if current_score is None:
        current_score = NEUTRAL_WEATHER_SCORE

    formats = list(sched.FORMATS)
    sold = {fmt: real_sales.get(fmt, 0) for fmt in formats}

    if sched.is_closed_day(now.weekday()):
        # Every key the open-day branch returns, so the client never has to
        # deal with a differently-shaped payload on Mondays.
        return {"heures_restantes": 0, "meteo": weather_cond,
                "previsions": dict(sold), "previsions_min": dict(sold), "previsions_max": dict(sold),
                "evenement": None, "debug_info": "Fermé"}

    open_hour, close_hour = sched.OPEN_HOUR, sched.close_hour(now.weekday())
    start_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    end_day = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    mode, time_left = (("PLANNING", (end_day - start_day).total_seconds() / 3600) if now < start_day
                       else ("LIVE", max(0, (end_day - now).total_seconds() / 3600)))
    elapsed_hours = 0 if mode == "PLANNING" else (now - start_day).total_seconds() / 3600
    business_hours = max(close_hour - open_hour, 1)

    event_name, base_event_factor, nhl_factor = get_special_event(now.date(), db_path=DB_NAME)
    is_game, is_playoff = get_game_info(now.date())
    payload = get_model_payload()
    models = payload["models"] if payload else None
    hourly_weather = get_today_hourly_weather()
    sql_weekday = sched.sql_weekday_from_py(now.weekday())

    predictions, predictions_low, predictions_high = {}, {}, {}
    debug_msg = "Prêt"

    if models:
        try:
            fmt_multipliers = {fmt: 1.0 for fmt in formats}
            if mode == "LIVE":
                # Fraction of the business day elapsed so far. A popular
                # format's predicted volume for just the first hour can
                # already be large, which used to make confidence (below)
                # jump to a high value almost immediately and let one busy
                # or slow opening stretch swing the whole remaining day's
                # forecast. Multiplying by time_fraction forces confidence
                # to build up gradually over the day regardless of format.
                time_fraction = min(elapsed_hours / business_hours, 1.0)
                # Clamped to the closing hour: after hours this range used to
                # run on to 18h, 19h and beyond, feeding the model hours it
                # was never trained on and corrupting the ratio.
                last_elapsed_hour = min(now.hour, close_hour - 1)
                past_hours = list(range(open_hour, last_elapsed_hour + 1))
                past_rows = _feature_rows(past_hours, sql_weekday, hourly_weather,
                                          current_score, temp_for_model, is_game, is_playoff)
                past_weights = [(now.minute / 60 if h == now.hour else 1) for h in past_hours]

                for fmt in formats:
                    if fmt not in models:
                        continue
                    past_pred = _predict_total(models[fmt], past_rows, past_weights)
                    if past_pred > 0.1:
                        # Blend the empirical ratio into the prior instead of replacing it outright.
                        empirical_ratio = max(0.3, min(sold[fmt] / past_pred, 3.0))
                        confidence = time_fraction * (past_pred / (past_pred + CORRECTION_SMOOTHING))
                        blended_ratio = confidence * empirical_ratio + (1 - confidence) * 1.0
                        fmt_multipliers[fmt] = max(0.5, min(blended_ratio, 3.0))

            start_h = open_hour if mode == "PLANNING" else now.hour
            future_hours = [h for h in range(start_h, close_hour)]
            future_rows = _feature_rows(future_hours, sql_weekday, hourly_weather,
                                        current_score, temp_for_model, is_game, is_playoff)
            future_weights = [(max(0, 60 - now.minute) / 60 if mode == "LIVE" and h == now.hour else 1)
                              for h in future_hours]
            # How much of a full trading day is still ahead, used to shrink
            # the error band as the day gets locked in.
            remaining_fraction = sum(future_weights) / business_hours

            for fmt in formats:
                if fmt not in models:
                    predictions[fmt] = predictions_low[fmt] = predictions_high[fmt] = sold[fmt]
                    continue
                point = _predict_total(models[fmt], future_rows, future_weights)
                # `is_special_event` is no longer a model feature, so the
                # learned calendar-event factor is applied here. The NHL
                # factor deliberately is not: is_game_day / is_playoff_game
                # are still features, and multiplying it back in would
                # double-count the game-day boost.
                scale = fmt_multipliers[fmt] * base_event_factor
                total = sold[fmt] + point * scale
                predictions[fmt] = int(round(total))

                band = _uncertainty(payload, fmt, remaining_fraction)
                if band is None:
                    predictions_low[fmt] = predictions_high[fmt] = predictions[fmt]
                else:
                    # The floor is what has already been sold: the day's total
                    # cannot come in under the bags that are in customers'
                    # hands.
                    predictions_low[fmt] = max(sold[fmt], int(round(total - band)))
                    predictions_high[fmt] = max(predictions_low[fmt], int(round(total + band)))
            debug_msg = f"{int((sum(fmt_multipliers.values()) / len(fmt_multipliers)) * 100)}%"
        except Exception as e:
            print(f"Prediction failed, falling back to linear extrapolation: {e}")
            models, debug_msg = None, "Erreur"

    if not models:
        # No trained model available: fall back to naive linear extrapolation,
        # which has no is_game_day or event feature of its own, so the FULL combined
        # factor (holiday * NHL) belongs here. No tree ensemble to draw a
        # range from, so min/max just mirror the point estimate.
        full_event_factor = base_event_factor * nhl_factor
        for fmt in formats:
            extra = (int(round((sold[fmt] / elapsed_hours) * time_left * weather_factor * full_event_factor))
                     if mode == "LIVE" and elapsed_hours > 0.1 else 0)
            predictions[fmt] = sold[fmt] + max(0, extra)
            predictions_low[fmt] = predictions_high[fmt] = predictions[fmt]

    return {
        "heures_restantes": round(time_left, 1),
        "meteo": weather_cond,
        "previsions": predictions,
        "previsions_min": predictions_low,
        "previsions_max": predictions_high,
        "evenement": event_name,
        "debug_info": debug_msg,
    }


@app.route('/api/forecast_week')
def forecast_week_endpoint():
    # forecast volumes for the next 7 days based on weather and events
    payload = get_model_payload()
    if not payload:
        return jsonify({"error": "Modèle manquant"})
    models = payload["models"]

    try:
        today_str = today_local().strftime("%Y-%m-%d")
        # Selected by date rather than by list position: slicing [1:8] assumed
        # the API's first entry is always today, which is only true as long as
        # nothing shifts the series.
        upcoming = [d for d in get_weekly_forecast() if d["date"] > today_str][:7]

        weekly_results = []
        for day_data in upcoming:
            dt = datetime.strptime(day_data['date'], "%Y-%m-%d")
            event_name, base_event_factor, _ = get_special_event(dt.date(), db_path=DB_NAME)
            is_game, is_playoff = get_game_info(dt.date())
            closed = sched.is_closed_day(dt.weekday())

            day_temp = day_data.get('temperature')
            if day_temp is None:
                day_temp = FALLBACK_TEMPERATURE
            score = day_data.get('score')
            if score is None:
                score = NEUTRAL_WEATHER_SCORE

            day_stats = {
                "date": day_data['date'],
                "date_affichee": f"{sched.WEEKDAY_NAMES_FR[dt.weekday()]} {dt.day}",
                "meteo": day_data['description'],
                "temperature": day_data.get('temperature'),
                "totals": {},
                "ferme": closed,
                "event": event_name,
            }

            if not closed:
                hours = list(sched.selling_hours(dt.weekday()))
                rows = _feature_rows(hours, sched.sql_weekday_from_py(dt.weekday()), {},
                                     score, day_temp, is_game, is_playoff)
                weights = [1] * len(hours)
                for fmt in sched.FORMATS:
                    if fmt not in models:
                        day_stats["totals"][fmt] = 0
                        continue
                    # round, not truncate — int() was shaving up to a full bag
                    # off every format of every day.
                    raw = _predict_total(models[fmt], rows, weights) * base_event_factor
                    day_stats["totals"][fmt] = max(0, int(round(raw)))
                    # No error band here on purpose. A whole day ahead carries
                    # the full measured error, which is wide enough relative to
                    # the forecast that it crowded out the number it was meant
                    # to qualify. The daily view still shows one, where it
                    # narrows through the day and actually informs a decision.

            weekly_results.append(day_stats)
        return jsonify(weekly_results)
    except Exception as e:
        print(f"Weekly forecast failed: {e}")
        return jsonify({"error": str(e)})


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
