import os
import sqlite3
import threading
import time as time_module
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from meteo import get_current_weather, get_weekly_forecast, interpret_weather_code, FALLBACK_TEMPERATURE
from event_service import get_special_event, get_game_info, get_event_key
from train_model import train_model, FEATURE_COLUMNS

app = Flask(__name__)
DB_NAME = "data.db"

# "Virtual" prior sales the intraday correction ratio must out-weigh before it's
# trusted (same smoothing idea as SMOOTHING_FACTOR in event_service.py).
# Also scaled by how much of the day has elapsed in get_prediction, so a
# popular format can't earn high confidence just from raw predicted volume
# in the first hour — see the time_fraction comment there.
CORRECTION_SMOOTHING = 3.0

# Nightly retrain time (24h). Chosen well after closing and well before
# opening so it never contends with `/api/log` writes during business hours.
NIGHTLY_RETRAIN_HOUR = 3

# cache models in memory to avoid disk operations
_cached_models = None
_cached_model_mtime = 0

def get_models():
    # load model from disk only if modified
    global _cached_models, _cached_model_mtime
    if not os.path.exists('model.pkl'): return None
    current_mtime = os.path.getmtime('model.pkl')
    if _cached_models is None or current_mtime > _cached_model_mtime:
        _cached_models = joblib.load('model.pkl')
        _cached_model_mtime = current_mtime
    return _cached_models

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
    # initialize tables for logs and daily snapshots
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, timestamp TEXT, action_type TEXT, detail TEXT, meteo_summary TEXT, temperature REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS daily_snapshots (date TEXT PRIMARY KEY, weekday INTEGER NOT NULL, event_key TEXT, event_name TEXT, is_nhl_game INTEGER DEFAULT 0, is_nhl_playoff INTEGER DEFAULT 0, total_250g INTEGER DEFAULT 0, total_1kg INTEGER DEFAULT 0, total_2kg INTEGER DEFAULT 0)")
        # Covers upgrading a pre-existing data.db that predates the
        # `temperature` column, without touching any of its existing rows.
        _ensure_column(conn, "logs", "temperature", "REAL")

def snapshot_completed_days():
    # aggregate past sales into snapshots before retraining
    today_str = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT date(timestamp), SUM(CASE WHEN detail = '250g' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '1kg' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '2kg' THEN 1 ELSE 0 END) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) != ? GROUP BY date(timestamp)", (today_str,)).fetchall()
        for date_str, t_250g, t_1kg, t_2kg in rows:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            event_name, _, _ = get_special_event(date_obj)
            is_game, is_playoff = get_game_info(date_obj)
            conn.execute("INSERT INTO daily_snapshots (date, weekday, event_key, event_name, is_nhl_game, is_nhl_playoff, total_250g, total_1kg, total_2kg) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET total_250g=excluded.total_250g, total_1kg=excluded.total_1kg, total_2kg=excluded.total_2kg, event_key=excluded.event_key, event_name=excluded.event_name, is_nhl_game=excluded.is_nhl_game, is_nhl_playoff=excluded.is_nhl_playoff", (date_str, date_obj.weekday(), get_event_key(date_obj), event_name, is_game, is_playoff, t_250g or 0, t_1kg or 0, t_2kg or 0))

def weather_to_score(factor):
    # convert weather multiplier to numerical score
    if factor < 0.85: return 0
    if factor >= 1.1: return 2
    return 1

def is_shop_open(dt):
    # evaluate if shop is currently open based on day and time
    day, hour = dt.weekday(), dt.hour
    if day == 0: return False
    return 10 <= hour < (18 if day in [3, 4] else 17)

def _predict_with_range(model, feature_rows, weights):
    """
    Predicts the weighted SUM of `model` over feature_rows, returning
    (point_estimate, low, high).

    The range comes from the spread across the RandomForest's individual
    trees. Each tree's predictions are summed across all the hours first,
    and percentiles are taken across trees afterwards — that keeps a given
    tree's own (correlated) view consistent across the whole remaining day,
    rather than combining independent per-hour spreads, which would over- or
    understate how uncertain the day's total actually is.
    """
    if not feature_rows:
        return 0.0, 0.0, 0.0
    
    # Keep the DataFrame step to ensure columns are perfectly aligned
    X_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    
    # Strip the feature names (convert to NumPy array) before passing to the inner trees 
    # to prevent the UserWarning.
    X_array = X_df.values
    
    per_tree = np.array([tree.predict(X_array) for tree in model.estimators_])  # (n_trees, n_rows)
    totals_per_tree = (per_tree * np.array(weights)).sum(axis=1)
    
    return float(totals_per_tree.mean()), float(np.percentile(totals_per_tree, 20)), float(np.percentile(totals_per_tree, 80))

@app.route('/')
def index():
    return render_template('index.html', est_ouvert=is_shop_open(datetime.now()))

@app.route('/api/status')
def get_status():
    open_status = is_shop_open(datetime.now())
    return jsonify({"ouvert": open_status, "message": "Ouvert" if open_status else "Fermé"})

@app.route('/api/log', methods=['POST'])
def log_action():
    # log a new sale entry with current weather (and temperature) context
    data = request.json
    condition, _, temperature = get_current_weather()
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO logs (timestamp, action_type, detail, meteo_summary, temperature) VALUES (?, ?, ?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data.get('type'), data.get('detail'), condition, temperature))
    return jsonify({"status": "success"})

@app.route('/api/undo', methods=['POST'])
def undo_last_action():
    # revert the most recent sale entry
    with sqlite3.connect(DB_NAME) as conn:
        last_row = conn.execute("SELECT id, detail FROM logs ORDER BY id DESC LIMIT 1").fetchone()
        if last_row:
            conn.execute("DELETE FROM logs WHERE id = ?", (last_row[0],))
            return jsonify({"status": "success", "message": f"{last_row[1]} annulé"})
    return jsonify({"status": "error", "message": "Aucune action"})

@app.route('/api/stats')
def get_stats():
    # retrieve current day statistics and hourly chart data
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_NAME) as conn:
        sales = dict(conn.execute("SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail", (today,)).fetchall())
        peak_hour = conn.execute("SELECT strftime('%H', timestamp) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp) ORDER BY COUNT(*) DESC LIMIT 1", (today,)).fetchone()
        hourly_raw = conn.execute("SELECT strftime('%H', timestamp), detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp), detail", (today,)).fetchall()

    stats = {fmt: sales.get(fmt, 0) for fmt in ["250g", "1kg", "2kg"]}
    hourly_data = {f"{h:02d}": {"250g": 0, "1kg": 0, "2kg": 0} for h in range(10, 19)}
    for h, fmt, count in hourly_raw:
        if h in hourly_data and fmt in hourly_data[h]: hourly_data[h][fmt] = count

    return jsonify({"c250": stats["250g"], "c1kg": stats["1kg"], "c2kg": stats["2kg"], "peak_hour": f"{peak_hour[0]}h00" if peak_hour else "--", "top_format": max(stats, key=stats.get) if sum(stats.values()) > 0 else "--", "total_mass": f"{(stats['250g'] * 0.25) + stats['1kg'] + (stats['2kg'] * 2):.2f} kg", "hourly_data": hourly_data})

@app.route('/api/history')
def get_history():
    # fetch the last five actions for the history feed
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT action_type, detail, timestamp FROM logs ORDER BY id DESC LIMIT 5").fetchall()
    return jsonify([{"type": r[0], "detail": r[1], "heure": datetime.strptime(r[2].split('.')[0], "%Y-%m-%d %H:%M:%S").strftime("%H:%M")} for r in rows])

@app.route('/api/history_days')
def get_history_days():
    # summary of every day with reported sales (most recent first), for the Historique page
    with sqlite3.connect(DB_NAME) as conn:
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
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT strftime('%H', timestamp), detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY strftime('%H', timestamp), detail", (date_str,)).fetchall()
    hourly_data = {f"{h:02d}": {"250g": 0, "1kg": 0, "2kg": 0} for h in range(10, 19)}
    for h, fmt, count in rows:
        if h in hourly_data and fmt in hourly_data[h]: hourly_data[h][fmt] = count
    return jsonify(hourly_data)

@app.route('/api/prediction')
def get_prediction():
    # calculate predictions for the remainder of the day
    now = datetime.now()
    weather_cond, weather_factor, temperature = get_current_weather()
    temp_for_model = temperature if temperature is not None else FALLBACK_TEMPERATURE
    if now.weekday() == 0: return jsonify({"heures_restantes": 0, "meteo": weather_cond, "previsions": {"250g": 0, "1kg": 0, "2kg": 0}, "message": "Fermé"})

    open_hour, close_hour = 10, 18 if now.weekday() in [3, 4] else 17
    start_day, end_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0), now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    mode, time_left = ("PLANNING", (end_day - start_day).total_seconds() / 3600) if now < start_day else ("LIVE", max(0, (end_day - now).total_seconds() / 3600))
    elapsed_hours = 0 if mode == "PLANNING" else (now - start_day).total_seconds() / 3600
    business_hours = max(close_hour - open_hour, 1)

    with sqlite3.connect(DB_NAME) as conn:
        real_sales = dict(conn.execute("SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail", (now.strftime("%Y-%m-%d"),)).fetchall())

    event_name, base_event_factor, nhl_factor = get_special_event(now.date(), db_path=DB_NAME)
    is_game, is_playoff = get_game_info(now.date())
    is_special = 1 if get_event_key(now.date()) else 0
    models, formats = get_models(), ['250g', '1kg', '2kg']
    predictions, predictions_low, predictions_high, debug_msg = {}, {}, {}, "Prêt"

    if models:
        try:
            # Events and holidays are explicitly processed by the ML features now. 
            # We initialize multiplier adjustments internally for scaling via empirical live adjustments.
            ai_day, fmt_multipliers = int(now.strftime('%w')), {fmt: 1.0 for fmt in formats}
            if mode == "LIVE":
                # Fraction of the business day elapsed so far. A popular
                # format's predicted volume for just the first hour can
                # already be large, which used to make confidence (below)
                # jump to a high value almost immediately and let one busy
                # or slow opening stretch swing the whole remaining day's
                # forecast. Multiplying by time_fraction forces confidence
                # to build up gradually over the day regardless of format.
                time_fraction = min(elapsed_hours / business_hours, 1.0)
                for fmt in formats:
                    if fmt not in models: continue
                    past_rows = [{'weekday': ai_day, 'hour': h, 'weather_score': weather_to_score(weather_factor), 'is_game_day': is_game, 'is_playoff_game': is_playoff, 'temperature': temp_for_model, 'is_special_event': is_special} for h in range(open_hour, now.hour + 1)]
                    past_weights = [(now.minute / 60 if h == now.hour else 1) for h in range(open_hour, now.hour + 1)]
                    past_pred, _, _ = _predict_with_range(models[fmt], past_rows, past_weights)
                    if past_pred > 0.1:
                        # Blend the empirical ratio into the prior instead of replacing it outright.
                        empirical_ratio = max(0.3, min(real_sales.get(fmt, 0) / past_pred, 3.0))
                        confidence = time_fraction * (past_pred / (past_pred + CORRECTION_SMOOTHING))
                        blended_ratio = confidence * empirical_ratio + (1 - confidence) * 1.0
                        fmt_multipliers[fmt] = max(0.5, min(fmt_multipliers[fmt] * blended_ratio, 3.0))

            start_h = open_hour if mode == "PLANNING" else now.hour
            for fmt in formats:
                sold = real_sales.get(fmt, 0)
                if fmt not in models:
                    predictions[fmt] = predictions_low[fmt] = predictions_high[fmt] = sold
                    continue
                future_rows = [{'weekday': ai_day, 'hour': h, 'weather_score': weather_to_score(weather_factor), 'is_game_day': is_game, 'is_playoff_game': is_playoff, 'temperature': temp_for_model, 'is_special_event': is_special} for h in range(start_h, close_hour)]
                future_weights = [(max(0, 60 - now.minute) / 60 if mode == "LIVE" and h == now.hour else 1) for h in range(start_h, close_hour)]
                point, low, high = _predict_with_range(models[fmt], future_rows, future_weights)
                predictions[fmt] = int(round(sold + point * fmt_multipliers[fmt]))
                predictions_low[fmt] = int(round(sold + low * fmt_multipliers[fmt]))
                predictions_high[fmt] = int(round(sold + high * fmt_multipliers[fmt]))
            debug_msg = f"{int((sum(fmt_multipliers.values()) / len(fmt_multipliers)) * 100)}%"
        except Exception:
            models, debug_msg = None, "Erreur"

    if not models:
        # No trained model available: fall back to naive linear extrapolation,
        # which has no is_game_day or event feature of its own, so the FULL combined
        # factor (holiday * NHL) belongs here. No tree ensemble to draw a
        # range from, so min/max just mirror the point estimate.
        full_event_factor = base_event_factor * nhl_factor
        for fmt in formats:
            sold = real_sales.get(fmt, 0)
            predictions[fmt] = int(round(sold + ((sold / elapsed_hours) * time_left * weather_factor * full_event_factor))) if mode == "LIVE" and elapsed_hours > 0.1 else sold
            predictions_low[fmt] = predictions_high[fmt] = predictions[fmt]

    return jsonify({
        "heures_restantes": round(time_left, 1),
        "meteo": weather_cond,
        "temperature": temperature,
        "previsions": predictions,
        "previsions_min": predictions_low,
        "previsions_max": predictions_high,
        "evenement": event_name,
        "debug_info": debug_msg
    })

@app.route('/api/forecast_week')
def forecast_week_endpoint():
    # forecast volumes for the next 7 days based on weather and events
    models = get_models()
    if not models: return jsonify({"error": "Modèle manquant"})
    try:
        weekly_results = []
        for day_data in get_weekly_forecast()[1:8]:
            dt = datetime.strptime(day_data['date'], "%Y-%m-%d")
            # Event name needed for return, but model inherently covers the multiplier math now via internal features.
            event_name, _, _ = get_special_event(dt.date(), db_path=DB_NAME)
            is_game, is_playoff = get_game_info(dt.date())
            is_special = 1 if get_event_key(dt.date()) else 0
            # day_data['temperature'] is the forecasted daily mean (see
            # meteo.py), matching the day-level mean the model is trained on.
            day_temp = day_data.get('temperature')
            day_temp = day_temp if day_temp is not None else FALLBACK_TEMPERATURE
            day_stats = {"date_affichee": f"{['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][dt.weekday()]} {dt.day}", "meteo": day_data['description'], "temperature": day_data.get('temperature'), "totals": {}, "ferme": dt.weekday() == 0, "event": event_name}
            if not day_stats["ferme"]:
                try: score_ai = weather_to_score(interpret_weather_code(day_data.get('code', 2))[1])
                except Exception: score_ai = 1
                close_day = 18 if dt.weekday() in [3, 4] else 17
                hours = range(10, close_day)  # close_day itself is not a selling hour
                df_input = pd.DataFrame({'weekday': [int(dt.strftime('%w'))]*len(hours), 'hour': list(hours), 'weather_score': [score_ai]*len(hours), 'is_game_day': [is_game]*len(hours), 'is_playoff_game': [is_playoff]*len(hours), 'temperature': [day_temp]*len(hours), 'is_special_event': [is_special]*len(hours)})[FEATURE_COLUMNS]
                for fmt in ['250g', '1kg', '2kg']: day_stats["totals"][fmt] = int(sum(models[fmt].predict(df_input))) if fmt in models else 0
            weekly_results.append(day_stats)
        return jsonify(weekly_results)
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    # Manual on-demand recalibration — e.g. right after an unusual event you
    # want reflected immediately. A full retrain also now runs automatically
    # every night (see start_scheduler below), so this is a supplement to
    # that, not the only way it happens.
    try:
        snapshot_completed_days()
        train_model()
        return jsonify({"status": "success", "message": "Calibrage terminé"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

def _seconds_until(hour, minute=0):
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()

def _run_retrain_safely(label):
    try:
        print(f"[{datetime.now()}] {label} retrain starting...")
        snapshot_completed_days()
        train_model()
        print(f"[{datetime.now()}] {label} retrain complete.")
    except Exception as e:
        print(f"[{datetime.now()}] {label} retrain failed: {e}")

def _nightly_retrain_loop():
    # Catch up once immediately (covers downtime, or data that piled up since
    # the last run), then settle into a fixed nightly schedule.
    _run_retrain_safely("Startup catch-up")
    while True:
        time_module.sleep(_seconds_until(NIGHTLY_RETRAIN_HOUR))
        _run_retrain_safely("Nightly")

def start_scheduler():
    # Flask's debug reloader runs this module in two processes (a watcher and
    # a worker); only the worker (WERKZEUG_RUN_MAIN=true) should own the
    # background thread, or the retrain job would fire twice in parallel.
    if app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return
    threading.Thread(target=_nightly_retrain_loop, daemon=True).start()
    print(f"Scheduled nightly retrain thread started ({NIGHTLY_RETRAIN_HOUR:02d}:00 daily).")

if __name__ == '__main__':
    init_db()
    start_scheduler()
    app.run(host='0.0.0.0', port=5000, debug=True)
