import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from event_service import get_game_info, get_event_key

DB_NAME = "data.db"
FORMATS = ['250g', '1kg', '2kg']
MIN_LOGS_THRESHOLD = 10

# Single source of truth for the model's feature set/order. Imported by
# app.py so every predict() call builds its DataFrame with exactly these
# columns in exactly this order — sklearn doesn't reliably realign columns
# by name for you, so training and inference must agree explicitly.
FEATURE_COLUMNS = ['weekday', 'hour', 'weather_score', 'is_game_day', 'is_playoff_game', 'temperature', 'is_special_event']

def train_model():
    print(">>> Training model...")
    conn = sqlite3.connect(DB_NAME)

    query = """
        SELECT 
            date(timestamp) as date_val,
            strftime('%w', timestamp) as weekday,
            strftime('%H', timestamp) as hour,
            meteo_summary,
            temperature,
            detail as bag_format
        FROM logs 
        WHERE action_type = 'VENTE'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty: return

    # EXCLUDE TODAY: Prevents poisoning model with incomplete afternoon hours
    today_str = datetime.now().strftime('%Y-%m-%d')
    df = df[df['date_val'] < today_str]

    if df.empty: 
        print("No past valid data for training.")
        return

    # Filter by daily volume to ensure quality data
    counts_per_day = df.groupby('date_val').size()
    valid_days = counts_per_day[counts_per_day >= MIN_LOGS_THRESHOLD].index
    df = df[df['date_val'].isin(valid_days)]

    if df.empty: return

    df['weekday'] = df['weekday'].astype(int)
    df['hour'] = df['hour'].astype(int)

    weather_map = {
        'Ensoleillé': 2, 'Variable': 1, 'Nuageux': 1, 'Brouillard': 1,
        'Pluie': 0, 'Averses': 0, 'Neige': 0, 'Orage': 0, 'Orages': 0, 'Inconnu': 1
    }
    df['weather_score'] = df['meteo_summary'].map(weather_map).fillna(1)

    # `temperature` is a newer column — rows logged before it existed have
    # NULL here. Rather than dropping that history (or crashing on NaN),
    # impute with the median of whatever real readings exist so far, so old
    # days stay usable and just look "temperature unknown/average" to the
    # model instead of being discarded. As more real readings accumulate,
    # the model gains genuine signal on temperature's effect.
    median_temp = df['temperature'].median()
    fallback_temp = median_temp if pd.notna(median_temp) else 15.0
    df['temperature'] = df['temperature'].fillna(fallback_temp)

    # Local cache to avoid repeated NHL & Event API calls for the same day
    game_info_cache = {}
    event_cache = {}
    def fetch_day_info(date_val_str):
        if date_val_str not in game_info_cache:
            dt = datetime.strptime(date_val_str, "%Y-%m-%d").date()
            game_info_cache[date_val_str] = get_game_info(dt)
            event_cache[date_val_str] = 1 if get_event_key(dt) else 0
        return game_info_cache[date_val_str], event_cache[date_val_str]

    # Build feature grid for all hours to handle zero-sales hours.
    # weather_score/temperature are averaged over every sale logged that
    # day (mean for temperature, mode for the weather score) instead of
    # just keeping whichever sale was logged first. A single early sale is
    # not representative of the whole day, and training on it was the
    # reason the weekly forecast (which uses a real day-level forecast
    # value) disagreed so much with same-day predictions.
    daily_context = df.groupby('date_val').agg(
        weekday=('weekday', 'first'),
        temperature=('temperature', 'mean'),
        weather_score=('weather_score', lambda s: s.mode().iloc[0]),
    ).reset_index()

    rows = []
    for _, day in daily_context.iterrows():
        # strftime %w: 0=Sun, 1=Mon...
        close_h = 18 if day['weekday'] in [4, 5] else 17

        # Fetch NHL game info and special event status for this specific day
        (is_game, is_playoff), is_special = fetch_day_info(day['date_val'])

        # close_h is the closing hour, not itself a selling hour (e.g.
        # close_h=17 means the last selling hour is 16h-17h), so the range
        # stops before it. Must match is_shop_open/get_prediction in app.py.
        for h in range(10, close_h):
            for fmt in FORMATS:
                rows.append({
                    'date_val': day['date_val'], 
                    'hour': h, 
                    'bag_format': fmt,
                    'weekday': day['weekday'],
                    'weather_score': day['weather_score'],
                    'temperature': day['temperature'],
                    'is_game_day': is_game,
                    'is_playoff_game': is_playoff,
                    'is_special_event': is_special
                })

    df_grid = pd.DataFrame(rows)
    actual_sales = df.groupby(['date_val', 'hour', 'bag_format']).size().reset_index(name='sales')
    df_final = pd.merge(df_grid, actual_sales, on=['date_val', 'hour', 'bag_format'], how='left')
    df_final['sales'] = df_final['sales'].fillna(0)

    models = {}
    for fmt in FORMATS:
        data_fmt = df_final[df_final['bag_format'] == fmt]
        if data_fmt.empty: continue

        X = data_fmt[FEATURE_COLUMNS]
        y = data_fmt['sales']

        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X, y)
        models[fmt] = regr

    joblib.dump(models, 'model.pkl')
    print("Training complete.")

if __name__ == "__main__":
    train_model()
