from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import os
import joblib
import pandas as pd

from meteo import get_current_weather, get_weekly_forecast, interpret_weather_code
from event_service import get_special_event
from train_model import train_model

app = Flask(__name__)
DB_NAME = "data.db"

# --- UTILS ---

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

def weather_to_score(condition_text):
    # Converts weather text to AI score (0=Bad, 1=Avg, 2=Good)
    mapping = {
        'Ensoleillé': 2, 'Nuageux': 1, 'Brouillard': 1,
        'Pluie': 0, 'Neige': 0, 'Orages': 0
    }
    return mapping.get(condition_text, 1)

def is_shop_open(dt):
    day = dt.weekday() # Mon=0 ... Sun=6
    hour = dt.hour
    
    if day == 0: return False # Closed Monday
    
    open_hour = 10
    close_hour = 17
    
    if day in [1, 2]: close_hour = 17 # Tue-Wed
    elif day in [3, 4]: close_hour = 18 # Thu-Fri
    elif day in [5, 6]: close_hour = 17 # Sat-Sun
    
    return open_hour <= hour < close_hour

# --- ROUTES ---

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
    
    # CORRECTION ICI : On convertit datetime.now() en string explicitement
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
    
    # 1. Sales Volumes
    query_sales = "SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) LIKE ? GROUP BY detail"
    c.execute(query_sales, (today + '%',))
    sales_results = c.fetchall()
    
    stats = {"250g": 0, "1kg": 0, "2kg": 0}
    for row in sales_results:
        if row[0] in stats: stats[row[0]] = row[1]

    # 2. Peak Hour
    query_hour = """
        SELECT strftime('%H', timestamp), COUNT(*) 
        FROM logs 
        WHERE action_type = 'VENTE' AND date(timestamp) LIKE ? 
        GROUP BY strftime('%H', timestamp) 
        ORDER BY COUNT(*) DESC 
        LIMIT 1
    """
    c.execute(query_hour, (today + '%',))
    res_hour = c.fetchone()
    peak_hour = f"{res_hour[0]}h00" if res_hour else "--"

    # 3. Conversions
    query_conv = "SELECT COUNT(*) FROM logs WHERE action_type = 'CONVERSION' AND date(timestamp) LIKE ?"
    c.execute(query_conv, (today + '%',))
    nb_conversions = c.fetchone()[0]

    conn.close()

    # 4. Top Format & Total Mass
    if sum(stats.values()) > 0:
        top_format = max(stats, key=stats.get)
    else:
        top_format = "--"

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
    
    # --- 1. CONFIG ---
    weather_cond, weather_factor = get_current_weather()
    weather_score = weather_to_score(weather_cond)

    weekday = now.weekday() # 0=Mon
    open_hour = 10
    close_hour = 17
    
    is_closed = False
    if weekday == 0: is_closed = True
    elif weekday in [3, 4]: close_hour = 18

    if is_closed:
         return jsonify({
            "heures_restantes": 0,
            "meteo": weather_cond,
            "previsions": {"250g":0, "1kg":0, "2kg":0},
            "debug_info": "Fermé le lundi",
            "message": "Fermé"
        })

    start_day = now.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    end_day = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)

    if now < start_day:
        # Before Open
        time_left = (end_day - start_day).total_seconds() / 3600
        elapsed_hours = 0
        mode = "PLANNING"
    else:
        # Live
        time_left = (end_day - now).total_seconds() / 3600
        if time_left < 0: time_left = 0
        elapsed_hours = (now - start_day).total_seconds() / 3600
        mode = "LIVE"

    # Real Sales
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT detail, COUNT(*) FROM logs WHERE action_type = 'VENTE' AND date(timestamp) = ? GROUP BY detail", 
              (now.strftime("%Y-%m-%d"),))
    real_sales = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    predictions = {}
    formats = ['250g', '1kg', '2kg']
    
    # --- 2. PREDICTION LOGIC ---
    use_ai = os.path.exists('model.pkl')
    trend_ratio = 1.0 
    
    if use_ai:
        # === AI MODE ===
        try:
            models = joblib.load('model.pkl')
            ai_day = int(now.strftime('%w')) # Sun=0
            
            # Trend Calculation
            if mode == "LIVE" and elapsed_hours > 0.5:
                past_pred = 0
                past_hours = range(open_hour, now.hour)
                if past_hours:
                    df_past = pd.DataFrame({
                        'weekday': [ai_day] * len(past_hours),
                        'hour': list(past_hours),
                        'weather_score': [weather_score] * len(past_hours)
                    })
                    
                    total_p = 0
                    for fmt in formats:
                        if fmt in models:
                            total_p += sum(models[fmt].predict(df_past))
                    
                    past_pred = total_p

                # Global Ratio
                total_current_sales = sum(real_sales.values())
                if past_pred > 5:
                    trend_ratio = total_current_sales / past_pred
                    trend_ratio = max(0.5, min(trend_ratio, 2.0))

            # Future Prediction
            hours_to_predict = range(now.hour, close_hour) if mode == "LIVE" else range(open_hour, close_hour)
            current_min = now.minute

            for fmt in formats:
                if fmt not in models:
                    predictions[fmt] = 0
                    continue
                
                pred_future = 0
                for h in hours_to_predict:
                    val = models[fmt].predict(pd.DataFrame([{
                        'weekday': ai_day, 'hour': h, 'weather_score': weather_score
                    }]))[0]
                    
                    # Adjust current hour
                    if mode == "LIVE" and h == now.hour:
                        val = val * ((60 - current_min) / 60)
                    
                    pred_future += val
                
                # Apply Trend
                final_total = real_sales.get(fmt, 0) + (pred_future * trend_ratio)
                to_produce = final_total - real_sales.get(fmt, 0)
                predictions[fmt] = int(to_produce + 0.99)
                
            debug_msg = f"IA active (Tendance: {int(trend_ratio*100)}%)"

        except Exception as e:
            print(f"AI Error: {e}")
            use_ai = False

    # Event Detection
    event_name, event_factor = get_special_event(now.date())

    if not use_ai:
        # === MATH MODE ===
        for fmt in formats:
            sold = real_sales.get(fmt, 0)
            
            if mode == "LIVE" and elapsed_hours > 0.1:
                speed = sold / elapsed_hours
                remaining_work = (speed * time_left) * weather_factor * event_factor
                predictions[fmt] = int(remaining_work + 0.99)
            else:
                predictions[fmt] = 0 
        
        debug_msg = "Mode Simple (Pas d'IA)"

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
        return jsonify({"status": "success", "message": "Modèle réentraîné avec succès !"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/forecast_week')
def forecast_week_endpoint():
    if not os.path.exists('model.pkl'):
        return jsonify({"error": "IA non entraînée"})

    try:
        models = joblib.load('model.pkl')
    except Exception as e:
        return jsonify({"error": f"Model load error: {str(e)}"})

    weather_forecast = get_weekly_forecast()
    
    weekly_results = []
    formats = ['250g', '1kg', '2kg']
    
    DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    # Days 1 to 7
    days_to_process = weather_forecast[1:8] if len(weather_forecast) > 1 else []

    for day_data in days_to_process: 
        date_str = day_data['date']
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        event_name, multiplier = get_special_event(dt.date())

        ai_weekday = int(dt.strftime('%w'))
        py_weekday = dt.weekday()
        
        day_name = DAYS_FR[py_weekday]
        formatted_date = f"{day_name} {dt.day}"

        if py_weekday == 0:
            open_h, close_h = 0, 0
            is_closed = True
        elif py_weekday in [1, 2]:
            open_h, close_h = 10, 17
            is_closed = False
        elif py_weekday in [3, 4]:
            open_h, close_h = 10, 18
            is_closed = False
        else:
            open_h, close_h = 10, 17
            is_closed = False

        day_stats = {
            "date": date_str, 
            "date_affichee": formatted_date,
            "meteo": day_data['description'], 
            "totals": {},
            "ferme": is_closed,
            "event": event_name,
            "multiplier": multiplier
        }
        
        if not is_closed:
            score_ai = 1
            try:
                _, weather_factor = interpret_weather_code(day_data['code'])
                if weather_factor < 0.85: score_ai = 0
                if weather_factor > 1.1: score_ai = 2
            except NameError:
                pass
            
            for fmt in formats:
                total_fmt = 0
                if fmt in models:
                    hour_range = range(open_h, close_h)
                    if hour_range:
                        df_input = pd.DataFrame({
                            'weekday': [ai_weekday] * len(hour_range),
                            'hour': list(hour_range),
                            'weather_score': [score_ai] * len(hour_range)
                        })
                        try:
                            preds = models[fmt].predict(df_input)
                            total_fmt = int(sum(preds) * multiplier)
                        except:
                            total_fmt = 0
                            
                day_stats["totals"][fmt] = total_fmt

        weekly_results.append(day_stats)
        
    return jsonify(weekly_results)

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
        time_fmt = dt.strftime("%H:%M")
        
        history.append({
            "type": row[0],
            "detail": row[1],
            "heure": time_fmt
        })
    
    return jsonify(history)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)