import os
import sqlite3
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from meteo import get_current_weather, get_weekly_forecast, interpret_weather_code
from event_service import get_special_event, get_game_info, get_event_key
from train_model import train_model

app = Flask(__name__)
DB_NAME = "data.db"

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

def init_db():
    # initialize tables for logs and daily snapshots
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, timestamp TEXT, action_type TEXT, detail TEXT, meteo_summary TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS daily_snapshots (date TEXT PRIMARY KEY, weekday INTEGER NOT NULL, event_key TEXT, event_name TEXT, is_nhl_game INTEGER DEFAULT 0, is_nhl_playoff INTEGER DEFAULT 0, total_250g INTEGER DEFAULT 0, total_1kg INTEGER DEFAULT 0, total_2kg INTEGER DEFAULT 0)")

def snapshot_completed_days():
    # aggregate past sales into snapshots before retraining
    today_str = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT date(timestamp), SUM(CASE WHEN detail = '250g' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '1kg' THEN 1 ELSE 0 END), SUM(CASE WHEN detail = '2kg' THEN 1 ELSE 0 END) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) != ? GROUP BY date(timestamp)", (today_str,)).fetchall()
        for date_str, t_250g, t_1kg, t_2kg in rows:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            event_name, _ = get_special_event(date_obj)
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

@app.route('/')
def index():
    return render_template('index.html', est_ouvert=is_shop_open(datetime.now()))

@app.route('/api/status')
def get_status():
    open_status = is_shop_open(datetime.now())
    return jsonify({"ouvert": open_status, "message": "Ouvert" if open_status else "Fermé"})

@app.route('/api/log', methods=['POST'])
def log_action():
    # log a new sale entry with current weather context
    data = request.json
    condition, _ = get_current_weather()
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO logs (timestamp, action_type, detail, meteo_summary) VALUES (?, ?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data.get('type'), data.get('detail'), condition))
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

@app.route('/api/prediction')
def get_prediction():
    # calculate predictions for the remainder of the day
    now = datetime.now()
    weather_cond, weather_factor = get_current_weather()
    if now.weekday() == 0: return jsonify({"heures_restantes": 0, "meteo": weather_cond, "previsions": {"250g": 0, "1kg": 0, "2kg": 0}, "message": "Fermé"})

    open_hour, close_hour = 10, 18 if now.weekday() in [3, 4] else 17
    start_day, end_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0), now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    mode, time_left = ("PLANNING", (end_day - start_day).total_seconds() / 3600) if now < start_day else ("LIVE", max(0, (end_day - now).total_seconds() / 3600))
    elapsed_hours = 0 if mode == "PLANNING" else (now - start_day).total_seconds() / 3600

    with sqlite3.connect(DB_NAME) as conn:
        real_sales = dict(conn.execute("SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail", (now.strftime("%Y-%m-%d"),)).fetchall())

    event_name, event_factor = get_special_event(now.date(), db_path=DB_NAME)
    is_game, is_playoff = get_game_info(now.date())
    models, formats, predictions, debug_msg = get_models(), ['250g', '1kg', '2kg'], {}, "Prêt"

    if models:
        try:
            ai_day, fmt_multipliers = int(now.strftime('%w')), {fmt: event_factor for fmt in formats}
            if mode == "LIVE" and elapsed_hours > 0.5:
                for fmt in formats:
                    if fmt not in models: continue
                    past_pred = sum(models[fmt].predict(pd.DataFrame([{'weekday': ai_day, 'hour': h, 'weather_score': weather_to_score(weather_factor), 'is_game_day': is_game, 'is_playoff_game': is_playoff}]))[0] * (now.minute / 60 if h == now.hour else 1) for h in range(open_hour, now.hour + 1))
                    if past_pred > 1: fmt_multipliers[fmt] = max(0.5, min(real_sales.get(fmt, 0) / past_pred, 2.0))
            
            start_h = open_hour if mode == "PLANNING" else now.hour
            for fmt in formats:
                if fmt not in models: predictions[fmt] = real_sales.get(fmt, 0); continue
                pred_future = sum(models[fmt].predict(pd.DataFrame([{'weekday': ai_day, 'hour': h, 'weather_score': weather_to_score(weather_factor), 'is_game_day': is_game, 'is_playoff_game': is_playoff}]))[0] * (max(0, 60 - now.minute) / 60 if mode == "LIVE" and h == now.hour else 1) for h in range(start_h, close_hour + 1))
                predictions[fmt] = int(round(real_sales.get(fmt, 0) + pred_future * fmt_multipliers[fmt]))
            debug_msg = f"{int((sum(fmt_multipliers.values()) / len(fmt_multipliers)) * 100)}%"
        except Exception:
            models, debug_msg = None, "Erreur"

    if not models:
        for fmt in formats:
            sold = real_sales.get(fmt, 0)
            predictions[fmt] = int(round(sold + ((sold / elapsed_hours) * time_left * weather_factor * event_factor))) if mode == "LIVE" and elapsed_hours > 0.1 else sold

    return jsonify({"heures_restantes": round(time_left, 1), "meteo": weather_cond, "previsions": predictions, "evenement": event_name, "debug_info": debug_msg})

@app.route('/api/forecast_week')
def forecast_week_endpoint():
    # forecast volumes for the next 7 days based on weather and events
    models = get_models()
    if not models: return jsonify({"error": "Modèle manquant"})
    try:
        weekly_results = []
        for day_data in get_weekly_forecast()[1:8]:
            dt = datetime.strptime(day_data['date'], "%Y-%m-%d")
            event_name, multiplier = get_special_event(dt.date(), db_path=DB_NAME)
            is_game, is_playoff = get_game_info(dt.date())
            day_stats = {"date_affichee": f"{['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][dt.weekday()]} {dt.day}", "meteo": day_data['description'], "totals": {}, "ferme": dt.weekday() == 0, "event": event_name}
            if not day_stats["ferme"]:
                try: score_ai = weather_to_score(interpret_weather_code(day_data.get('code', 2))[1])
                except Exception: score_ai = 1
                hours = range(10, (18 if dt.weekday() in [3, 4] else 17) + 1)
                df_input = pd.DataFrame({'weekday': [int(dt.strftime('%w'))]*len(hours), 'hour': list(hours), 'weather_score': [score_ai]*len(hours), 'is_game_day': [is_game]*len(hours), 'is_playoff_game': [is_playoff]*len(hours)})
                for fmt in ['250g', '1kg', '2kg']: day_stats["totals"][fmt] = int(sum(models[fmt].predict(df_input)) * multiplier) if fmt in models else 0
            weekly_results.append(day_stats)
        return jsonify(weekly_results)
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    # force model recalculation with latest database snapshots
    try:
        snapshot_completed_days()
        train_model()
        return jsonify({"status": "success", "message": "Calibrage terminé"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
