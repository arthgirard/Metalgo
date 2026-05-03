from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import os
import joblib
import pandas as pd

from meteo import get_current_weather, get_weekly_forecast, interpret_weather_code
from event_service import get_special_event, get_game_info
from train_model import train_model

app = Flask(__name__)
DB_NAME = "data.db"

# Utils
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY, 
                  timestamp TEXT, 
                  action_type TEXT, 
                  detail TEXT, 
                  meteo_summary TEXT)''')
    conn.commit()
    conn.close()

def weather_to_score(factor):
    if factor < 0.85: return 0
    if factor >= 1.1: return 2
    return 1

def is_shop_open(dt):
    day = dt.weekday() # Mon=0 ... Sun=6
    hour = dt.hour
    
    if day == 0: return False # Closed Monday
    
    open_hour = 10
    close_hour = 17
    
    if day in [1, 2]: close_hour = 17 
    elif day in [3, 4]: close_hour = 18 
    elif day in [5, 6]: close_hour = 17 
    
    return open_hour <= hour < close_hour

# Routes
@app.route('/')
def index():
    now = datetime.now()
    return render_template('index.html', est_ouvert=is_shop_open(now))

@app.route('/api/status')
def get_status():
    now = datetime.now()
    open_status = is_shop_open(now)
    return jsonify({
        "ouvert": open_status,
        "message": "Fermé actuellement" if not open_status else "Ouvert"
    })
    
@app.route('/api/log', methods=['POST'])
def log_action():
    data = request.json
    action_type = data.get('type')
    detail = data.get('detail')
    
    condition, _ = get_current_weather()
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO logs (timestamp, action_type, detail, meteo_summary) VALUES (?, ?, ?, ?)", 
              (now_str, action_type, detail, condition))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})

@app.route('/api/undo', methods=['POST'])
def undo_last_action():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, action_type, detail FROM logs ORDER BY id DESC LIMIT 1")
    last_row = c.fetchone()
    
    if last_row:
        log_id, _, detail = last_row
        c.execute("DELETE FROM logs WHERE id = ?", (log_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Annulé : {detail}"})
    else:
        conn.close()
        return jsonify({"status": "error", "message": "Rien à annuler"})

@app.route('/api/stats')
def get_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    query_sales = "SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail"
    c.execute(query_sales, (today,))
    sales_results = c.fetchall()
    
    stats = {"250g": 0, "1kg": 0, "2kg": 0}
    for row in sales_results:
        if row[0] in stats: stats[row[0]] = row[1]

    query_hour = """
        SELECT strftime('%H', timestamp), COUNT(*) 
        FROM logs 
        WHERE action_type = 'VENTE' AND date(timestamp) = ? 
        GROUP BY strftime('%H', timestamp) 
        ORDER BY COUNT(*) DESC 
        LIMIT 1
    """
    c.execute(query_hour, (today,))
    res_hour = c.fetchone()
    peak_hour = f"{res_hour[0]}h00" if res_hour else "--"

    query_conv = "SELECT COUNT(*) FROM logs WHERE action_type = 'CONVERSION' AND date(timestamp) = ?"
    c.execute(query_conv, (today,))
    nb_conversions = c.fetchone()[0]

    conn.close()

    top_format = max(stats, key=stats.get) if sum(stats.values()) > 0 else "--"
    total_kg = (stats["250g"] * 0.25) + (stats["1kg"] * 1) + (stats["2kg"] * 2)

    return jsonify({
        "c250": stats["250g"],
        "c1kg": stats["1kg"],
        "c2kg": stats["2kg"],
        "peak_hour": peak_hour,
        "top_format": top_format,
        "total_mass": f"{total_kg:.2f} kg",
        "total_conv": nb_conversions
    })

@app.route('/api/prediction')
def get_prediction():
    now = datetime.now()
    weather_cond, weather_factor = get_current_weather()
    weather_score = weather_to_score(weather_factor)

    weekday = now.weekday() 
    open_hour = 10
    close_hour = 17
    
    if weekday == 0:
        return jsonify({"heures_restantes": 0, "meteo": weather_cond, "previsions": {"250g":0, "1kg":0, "2kg":0}, "message": "Fermé"})
    
    if weekday in [3, 4]: close_hour = 18

    start_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    end_day = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)

    # Time calculations
    if now < start_day:
        time_left = (end_day - start_day).total_seconds() / 3600
        elapsed_hours = 0
        mode = "PLANNING"
    else:
        time_left = max(0, (end_day - now).total_seconds() / 3600)
        elapsed_hours = (now - start_day).total_seconds() / 3600
        mode = "LIVE"

    # Current Sales
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail", 
              (now.strftime("%Y-%m-%d"),))
    real_sales = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    predictions = {}
    formats = ['250g', '1kg', '2kg']
    
    # Event factors and NHL game status
    event_name, event_factor = get_special_event(now.date())
    is_game, is_playoff = get_game_info(now.date())
    
    use_ai = os.path.exists('model.pkl')
    # Initialize debug_msg so fallback path never hits a NameError
    debug_msg = "Mode Simple (IA inactive)"
    
    if use_ai:
        try:
            models = joblib.load('model.pkl')
            ai_day = int(now.strftime('%w')) # 0=Sun, 6=Sat

            # Per-format performance ratios; default to event_factor until LIVE data is reliable
            fmt_multipliers = {fmt: event_factor for fmt in formats}

            # Live performance ratio (reality vs theory), computed per format
            if mode == "LIVE" and elapsed_hours > 0.5:
                for fmt in formats:
                    if fmt not in models:
                        continue
                    past_pred_fmt = 0
                    for h in range(open_hour, now.hour + 1):
                        df_h = pd.DataFrame([{
                            'weekday': ai_day,
                            'hour': h,
                            'weather_score': weather_score,
                            'is_game_day': is_game,
                            'is_playoff_game': is_playoff
                        }])
                        val = models[fmt].predict(df_h)[0]
                        if h == now.hour:
                            past_pred_fmt += val * (now.minute / 60)
                        else:
                            past_pred_fmt += val

                    if past_pred_fmt > 1:
                        m = real_sales.get(fmt, 0) / past_pred_fmt
                        # Cap at 2.0: avoids runaway inflation from early-morning spikes
                        fmt_multipliers[fmt] = max(0.5, min(m, 2.0))

            # Future Logic
            # PLANNING: always predict the full day from open_hour, not from now.hour.
            # Using now.hour before open causes the model to extrapolate on hours it
            # was never trained on, inflating predictions by 1-4 phantom peak-hour values.
            start_h = open_hour if mode == "PLANNING" else now.hour
            for fmt in formats:
                if fmt not in models:
                    predictions[fmt] = real_sales.get(fmt, 0)
                    continue
                
                pred_future = 0
                for h in range(start_h, close_hour + 1):
                    df_h = pd.DataFrame([{
                        'weekday': ai_day, 
                        'hour': h, 
                        'weather_score': weather_score,
                        'is_game_day': is_game,
                        'is_playoff_game': is_playoff
                    }])
                    val = models[fmt].predict(df_h)[0]
                    
                    if mode == "LIVE" and h == now.hour:
                        val = val * (max(0, 60 - now.minute) / 60)
                    
                    pred_future += val
                
                predictions[fmt] = int(round(real_sales.get(fmt, 0) + (pred_future * fmt_multipliers[fmt])))

            avg_multiplier = sum(fmt_multipliers.values()) / len(fmt_multipliers)
            debug_msg = f"IA active (Tendance: {int(avg_multiplier*100)}%)"
        except Exception as e:
            use_ai = False
            debug_msg = f"Erreur IA (Fallback Simple): {str(e)}"

    if not use_ai:
        # Fallback math mode
        for fmt in formats:
            sold = real_sales.get(fmt, 0)
            if mode == "LIVE" and elapsed_hours > 0.1:
                speed = sold / elapsed_hours
                remaining = (speed * time_left) * weather_factor * event_factor
                predictions[fmt] = int(round(sold + remaining))
            else:
                predictions[fmt] = sold

    return jsonify({
        "heures_restantes": round(time_left, 1),
        "meteo": weather_cond,
        "previsions": predictions,
        "evenement": event_name,
        "debug_info": debug_msg
    })

@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    try:
        train_model()
        return jsonify({"status": "success", "message": "Modèle réentraîné"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/forecast_week')
def forecast_week_endpoint():
    if not os.path.exists('model.pkl'):
        return jsonify({"error": "IA non entraînée"})

    try:
        models = joblib.load('model.pkl')
        weather_forecast = get_weekly_forecast()
        weekly_results = []
        formats = ['250g', '1kg', '2kg']
        DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

        days_to_process = weather_forecast[1:8] if len(weather_forecast) > 1 else []

        for day_data in days_to_process: 
            dt = datetime.strptime(day_data['date'], "%Y-%m-%d")
            
            # Fetch events and game info
            event_name, multiplier = get_special_event(dt.date())
            is_game, is_playoff = get_game_info(dt.date())
            
            ai_weekday = int(dt.strftime('%w'))
            py_weekday = dt.weekday()
            
            day_stats = {
                "date_affichee": f"{DAYS_FR[py_weekday]} {dt.day}",
                "meteo": day_data['description'], 
                "totals": {},
                "ferme": py_weekday == 0,
                "event": event_name
            }
            
            if not day_stats["ferme"]:
                close_h = 18 if py_weekday in [3, 4] else 17
                
                try:
                    _, weather_factor = interpret_weather_code(day_data.get('code', 2))
                    score_ai = weather_to_score(weather_factor)
                except Exception:
                    score_ai = 1 
                
                for fmt in formats:
                    total_fmt = 0
                    if fmt in models:
                        hours = range(10, close_h + 1)
                        df_input = pd.DataFrame({
                            'weekday': [ai_weekday] * len(hours),
                            'hour': list(hours),
                            'weather_score': [score_ai] * len(hours),
                            'is_game_day': [is_game] * len(hours),
                            'is_playoff_game': [is_playoff] * len(hours)
                        })
                        # Apply generic multiplier scaling to the AI prediction
                        total_fmt = int(sum(models[fmt].predict(df_input)) * multiplier)
                    day_stats["totals"][fmt] = total_fmt

            weekly_results.append(day_stats)
        return jsonify(weekly_results)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT action_type, detail, timestamp FROM logs ORDER BY id DESC LIMIT 3")
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        dt = datetime.strptime(row[2].split('.')[0], "%Y-%m-%d %H:%M:%S")
        history.append({"type": row[0], "detail": row[1], "heure": dt.strftime("%H:%M")})
    return jsonify(history)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
