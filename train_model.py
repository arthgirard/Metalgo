import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

DB_NAME = "data.db"
FORMATS = ['250g', '1kg', '2kg']
MIN_LOGS_THRESHOLD = 10 

def train_model():
    print(">>> Starting model training...")
    
    conn = sqlite3.connect(DB_NAME)
    
    query = """
        SELECT 
            date(timestamp) as date_val,
            strftime('%w', timestamp) as weekday,
            strftime('%H', timestamp) as hour,
            meteo_summary,
            detail as bag_format
        FROM logs 
        WHERE action_type = 'VENTE'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("Database is empty.")
        return

    # Filter incomplete days
    counts_per_day = df.groupby('date_val').size()
    valid_days = counts_per_day[counts_per_day >= MIN_LOGS_THRESHOLD].index
    
    print(f"Total days: {len(counts_per_day)}")
    print(f"Valid days (>{MIN_LOGS_THRESHOLD} sales): {len(valid_days)}")
    
    df = df[df['date_val'].isin(valid_days)]

    if df.empty:
        print("No valid data for training.")
        return

    # Preprocessing
    df['weekday'] = df['weekday'].astype(int)
    df['hour'] = df['hour'].astype(int)
    
    # Mapping must match strings from meteo.py
    weather_map = {
        'Ensoleillé': 2, 'Nuageux': 1, 'Brouillard': 1,
        'Pluie': 0, 'Neige': 0, 'Orages': 0, 'Inconnu': 1
    }
    df['weather_score'] = df['meteo_summary'].map(weather_map).fillna(1)

    df_grouped = df.groupby(['weekday', 'hour', 'weather_score', 'bag_format']).size().reset_index(name='sales')

    models = {}

    for fmt in FORMATS:
        print(f"Training for {fmt}...")
        data_fmt = df_grouped[df_grouped['bag_format'] == fmt]
        
        if data_fmt.empty: continue

        X = data_fmt[['weekday', 'hour', 'weather_score']]
        y = data_fmt['sales']

        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X, y)
        models[fmt] = regr

    joblib.dump(models, 'model.pkl')
    print("Training complete.")

if __name__ == "__main__":
    train_model()