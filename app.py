from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import os
import joblib
import pandas as pd

from meteo import get_current_weather, get_weekly_forecast, interpret_weather_code
from event_service import get_special_event, get_game_info, get_event_key
from train_model import train_model

app = Flask(__name__)
DB_NAME = "data.db"


# Database

def init_db():
    """
    Creates all required tables if they don't already exist.

    Tables:
      logs             — individual sales / conversion events logged in real time
      daily_snapshots  — one row per completed past day; used to compute learned
                         event multipliers in event_service.py
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id          INTEGER PRIMARY KEY,
            timestamp   TEXT,
            action_type TEXT,
            detail      TEXT,
            meteo_summary TEXT
        )
    ''')

    # Each row summarises a single completed trading day.
    # event_key mirrors the canonical key returned by event_service.get_event_key(),
    # allowing _get_learned_multiplier() to join on it directly.
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date           TEXT PRIMARY KEY,   -- "YYYY-MM-DD"
            weekday        INTEGER NOT NULL,   -- Python convention: Mon=0 … Sun=6
            event_key      TEXT,               -- Non-NHL event key, or NULL
            event_name     TEXT,               -- Human-readable display name
            is_nhl_game    INTEGER DEFAULT 0,  -- 1 if Canadiens played
            is_nhl_playoff INTEGER DEFAULT 0,  -- 1 if playoff game
            total_250g     INTEGER DEFAULT 0,
            total_1kg      INTEGER DEFAULT 0,
            total_2kg      INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()


def snapshot_completed_days():
    """
    Scans the logs table for every completed past day (excluding today) and
    upserts a summary row into daily_snapshots.

    This is the data pipeline that feeds the learned-multiplier system:
    event_service._get_learned_multiplier() reads from daily_snapshots to
    derive per-event sales ratios versus normal-day baselines.

    Called automatically whenever the model is retrained via /api/retrain,
    so the snapshot table stays in sync with fresh sales history.

    Note: get_special_event() is called here *without* db_path intentionally —
    we only need the display name for the record, not a learned multiplier,
    and we avoid any risk of circular reads on a partially committed dataset.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Aggregate sales per past day in a single query
    c.execute("""
        SELECT
            date(timestamp),
            SUM(CASE WHEN detail = '250g' THEN 1 ELSE 0 END),
            SUM(CASE WHEN detail = '1kg'  THEN 1 ELSE 0 END),
            SUM(CASE WHEN detail = '2kg'  THEN 1 ELSE 0 END)
        FROM logs
        WHERE action_type = 'VENTE'
          AND date(timestamp) != ?
        GROUP BY date(timestamp)
    """, (today_str,))
    rows = c.fetchall()

    for date_str, total_250g, total_1kg, total_2kg in rows:
        date_obj   = datetime.strptime(date_str, "%Y-%m-%d").date()
        weekday    = date_obj.weekday()
        event_key  = get_event_key(date_obj)

        # Display name only — no db_path to keep this read-free
        event_name, _ = get_special_event(date_obj)

        is_game, is_playoff = get_game_info(date_obj)

        # UPSERT: re-running retrain refreshes counts if past logs were corrected
        c.execute("""
            INSERT INTO daily_snapshots
                (date, weekday, event_key, event_name,
                 is_nhl_game, is_nhl_playoff,
                 total_250g, total_1kg, total_2kg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_250g     = excluded.total_250g,
                total_1kg      = excluded.total_1kg,
                total_2kg      = excluded.total_2kg,
                event_key      = excluded.event_key,
                event_name     = excluded.event_name,
                is_nhl_game    = excluded.is_nhl_game,
                is_nhl_playoff = excluded.is_nhl_playoff
        """, (
            date_str, weekday, event_key, event_name,
            is_game, is_playoff,
            total_250g or 0, total_1kg or 0, total_2kg or 0
        ))

    conn.commit()
    conn.close()


# Utilities

def weather_to_score(factor):
    if factor < 0.85: return 0
    if factor >= 1.1:  return 2
    return 1


def is_shop_open(dt):
    """Returns True if the shop is open at the given datetime."""
    day  = dt.weekday()  # Mon=0 … Sun=6
    hour = dt.hour

    if day == 0:
        return False  # Closed on Mondays

    close_hour = 18 if day in [3, 4] else 17  # Thu/Fri close at 18h
    return 10 <= hour < close_hour


# Routes — Shop status

@app.route('/')
def index():
    now = datetime.now()
    return render_template('index.html', est_ouvert=is_shop_open(now))


@app.route('/api/status')
def get_status():
    now         = datetime.now()
    open_status = is_shop_open(now)
    return jsonify({
        "ouvert":  open_status,
        "message": "Ouvert" if open_status else "Fermé actuellement"
    })


# Routes — Logging
@app.route('/api/log', methods=['POST'])
def log_action():
    data        = request.json
    action_type = data.get('type')
    detail      = data.get('detail')

    condition, _ = get_current_weather()

    conn    = sqlite3.connect(DB_NAME)
    c       = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO logs (timestamp, action_type, detail, meteo_summary) VALUES (?, ?, ?, ?)",
        (now_str, action_type, detail, condition)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "success"})


@app.route('/api/undo', methods=['POST'])
def undo_last_action():
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("SELECT id, action_type, detail FROM logs ORDER BY id DESC LIMIT 1")
    last_row = c.fetchone()

    if last_row:
        log_id, _, detail = last_row
        c.execute("DELETE FROM logs WHERE id = ?", (log_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Annulé : {detail}"})

    conn.close()
    return jsonify({"status": "error", "message": "Rien à annuler"})


# Routes — Statistics
@app.route('/api/stats')
def get_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_NAME)
    c     = conn.cursor()

    # Sales breakdown by format
    c.execute("""
        SELECT detail, COUNT(*)
        FROM logs
        WHERE action_type = 'VENTE' AND date(timestamp) = ?
        GROUP BY detail
    """, (today,))
    sales_results = c.fetchall()

    stats = {"250g": 0, "1kg": 0, "2kg": 0}
    for row in sales_results:
        if row[0] in stats:
            stats[row[0]] = row[1]

    # Peak hour
    c.execute("""
        SELECT strftime('%H', timestamp), COUNT(*)
        FROM logs
        WHERE action_type = 'VENTE' AND date(timestamp) = ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """, (today,))
    res_hour  = c.fetchone()
    peak_hour = f"{res_hour[0]}h00" if res_hour else "--"

    # Conversion count
    c.execute(
        "SELECT COUNT(*) FROM logs WHERE action_type = 'CONVERSION' AND date(timestamp) = ?",
        (today,)
    )
    nb_conversions = c.fetchone()[0]
    conn.close()

    top_format = max(stats, key=stats.get) if sum(stats.values()) > 0 else "--"
    total_kg   = (stats["250g"] * 0.25) + (stats["1kg"] * 1) + (stats["2kg"] * 2)

    return jsonify({
        "c250":       stats["250g"],
        "c1kg":       stats["1kg"],
        "c2kg":       stats["2kg"],
        "peak_hour":  peak_hour,
        "top_format": top_format,
        "total_mass": f"{total_kg:.2f} kg",
        "total_conv": nb_conversions
    })


@app.route('/api/history')
def get_history():
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "SELECT action_type, detail, timestamp FROM logs ORDER BY id DESC LIMIT 3"
    )
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        dt = datetime.strptime(row[2].split('.')[0], "%Y-%m-%d %H:%M:%S")
        history.append({"type": row[0], "detail": row[1], "heure": dt.strftime("%H:%M")})
    return jsonify(history)


# Routes — Prediction
@app.route('/api/prediction')
def get_prediction():
    now                        = datetime.now()
    weather_cond, weather_factor = get_current_weather()
    weather_score              = weather_to_score(weather_factor)
    weekday                    = now.weekday()

    if weekday == 0:
        return jsonify({
            "heures_restantes": 0,
            "meteo":     weather_cond,
            "previsions": {"250g": 0, "1kg": 0, "2kg": 0},
            "message":   "Fermé"
        })

    open_hour  = 10
    close_hour = 18 if weekday in [3, 4] else 17
    start_day  = now.replace(hour=open_hour,  minute=0, second=0, microsecond=0)
    end_day    = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)

    if now < start_day:
        time_left      = (end_day - start_day).total_seconds() / 3600
        elapsed_hours  = 0
        mode           = "PLANNING"
    else:
        time_left      = max(0, (end_day - now).total_seconds() / 3600)
        elapsed_hours  = (now - start_day).total_seconds() / 3600
        mode           = "LIVE"

    # Current day's sales from logs
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("""
        SELECT detail, COUNT(*)
        FROM logs
        WHERE action_type = 'VENTE' AND date(timestamp) = ?
        GROUP BY detail
    """, (now.strftime("%Y-%m-%d"),))
    real_sales = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    # Event and game context — pass db_path so learned multipliers are used
    event_name, event_factor = get_special_event(now.date(), db_path=DB_NAME)
    is_game, is_playoff      = get_game_info(now.date())

    formats    = ['250g', '1kg', '2kg']
    predictions = {}
    use_ai      = os.path.exists('model.pkl')
    debug_msg   = "Mode Simple (IA inactive)"

    if use_ai:
        try:
            models = joblib.load('model.pkl')
            ai_day = int(now.strftime('%w'))  # 0=Sun … 6=Sat

            # Per-format live-performance multipliers.
            # Initialised with the learned event factor; overridden by real-time
            # sales ratio once enough of the day has elapsed (LIVE mode only).
            fmt_multipliers = {fmt: event_factor for fmt in formats}

            if mode == "LIVE" and elapsed_hours > 0.5:
                for fmt in formats:
                    if fmt not in models:
                        continue

                    # Sum model predictions for elapsed hours to get a theoretical total
                    past_pred = 0
                    for h in range(open_hour, now.hour + 1):
                        df_h = pd.DataFrame([{
                            'weekday':         ai_day,
                            'hour':            h,
                            'weather_score':   weather_score,
                            'is_game_day':     is_game,
                            'is_playoff_game': is_playoff
                        }])
                        val = models[fmt].predict(df_h)[0]
                        # Prorate the current hour by elapsed minutes
                        if h == now.hour:
                            past_pred += val * (now.minute / 60)
                        else:
                            past_pred += val

                    if past_pred > 1:
                        ratio = real_sales.get(fmt, 0) / past_pred
                        # Cap the ratio to prevent outlier spikes from distorting the day
                        fmt_multipliers[fmt] = max(0.5, min(ratio, 2.0))

            # Project remaining hours
            # PLANNING: always start from open_hour, not now.hour.
            # Predicting from pre-opening hours produces phantom peak-hour values
            # because the model was never trained on those out-of-bounds inputs.
            start_h = open_hour if mode == "PLANNING" else now.hour

            for fmt in formats:
                if fmt not in models:
                    predictions[fmt] = real_sales.get(fmt, 0)
                    continue

                pred_future = 0
                for h in range(start_h, close_hour + 1):
                    df_h = pd.DataFrame([{
                        'weekday':         ai_day,
                        'hour':            h,
                        'weather_score':   weather_score,
                        'is_game_day':     is_game,
                        'is_playoff_game': is_playoff
                    }])
                    val = models[fmt].predict(df_h)[0]

                    # Prorate the current (partially elapsed) hour
                    if mode == "LIVE" and h == now.hour:
                        val = val * (max(0, 60 - now.minute) / 60)

                    pred_future += val

                predictions[fmt] = int(round(
                    real_sales.get(fmt, 0) + pred_future * fmt_multipliers[fmt]
                ))

            avg_mult  = sum(fmt_multipliers.values()) / len(fmt_multipliers)
            debug_msg = f"IA active (Tendance: {int(avg_mult * 100)}%)"

        except Exception as e:
            use_ai    = False
            debug_msg = f"Erreur IA (Fallback Simple): {str(e)}"

    if not use_ai:
        # Simple linear extrapolation fallback
        for fmt in formats:
            sold = real_sales.get(fmt, 0)
            if mode == "LIVE" and elapsed_hours > 0.1:
                speed      = sold / elapsed_hours
                remaining  = speed * time_left * weather_factor * event_factor
                predictions[fmt] = int(round(sold + remaining))
            else:
                predictions[fmt] = sold

    return jsonify({
        "heures_restantes": round(time_left, 1),
        "meteo":      weather_cond,
        "previsions": predictions,
        "evenement":  event_name,
        "debug_info": debug_msg
    })


# Routes — Weekly forecast
@app.route('/api/forecast_week')
def forecast_week_endpoint():
    if not os.path.exists('model.pkl'):
        return jsonify({"error": "IA non entraînée"})

    try:
        models           = joblib.load('model.pkl')
        weather_forecast = get_weekly_forecast()
        weekly_results   = []
        formats          = ['250g', '1kg', '2kg']
        DAYS_FR          = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

        # Skip index 0 (today) and process the next 7 days
        days_to_process = weather_forecast[1:8] if len(weather_forecast) > 1 else []

        for day_data in days_to_process:
            dt         = datetime.strptime(day_data['date'], "%Y-%m-%d")
            py_weekday = dt.weekday()
            ai_weekday = int(dt.strftime('%w'))

            # Use learned multipliers (db_path provided) for all forward-looking days
            event_name, multiplier = get_special_event(dt.date(), db_path=DB_NAME)
            is_game, is_playoff    = get_game_info(dt.date())

            day_stats = {
                "date_affichee": f"{DAYS_FR[py_weekday]} {dt.day}",
                "meteo":  day_data['description'],
                "totals": {},
                "ferme":  py_weekday == 0,
                "event":  event_name
            }

            if not day_stats["ferme"]:
                close_h = 18 if py_weekday in [3, 4] else 17

                try:
                    _, weather_factor = interpret_weather_code(day_data.get('code', 2))
                    score_ai          = weather_to_score(weather_factor)
                except Exception:
                    score_ai = 1

                for fmt in formats:
                    total_fmt = 0
                    if fmt in models:
                        hours    = range(10, close_h + 1)
                        df_input = pd.DataFrame({
                            'weekday':         [ai_weekday]  * len(hours),
                            'hour':            list(hours),
                            'weather_score':   [score_ai]    * len(hours),
                            'is_game_day':     [is_game]     * len(hours),
                            'is_playoff_game': [is_playoff]  * len(hours)
                        })
                        # Scale AI output by the (potentially learned) event multiplier
                        total_fmt = int(sum(models[fmt].predict(df_input)) * multiplier)

                    day_stats["totals"][fmt] = total_fmt

            weekly_results.append(day_stats)

        return jsonify(weekly_results)

    except Exception as e:
        return jsonify({"error": str(e)})


# Routes — Model training
@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    """
    Retrains the prediction model and refreshes the daily_snapshots table.
    Snapshotting is done first so the model trains on the most up-to-date data.
    """
    try:
        snapshot_completed_days()  # Rebuild event history before training
        train_model()
        return jsonify({"status": "success", "message": "Modèle réentraîné"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Entry point
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
