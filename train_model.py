import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

DB_NAME = "data.db"
FORMATS = ['250g', '1kg', '2kg']
MIN_LOGS_THRESHOLD = 10 

def train_model():
    print(">>> Training model...")
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

    # Build feature grid for all hours to handle zero-sales hours
    # Removed day_of_year to allow extrapolation for future weeks
    daily_context = df[['date_val', 'weekday', 'weather_score']].drop_duplicates('date_val')
    
    rows = []
    for _, day in daily_context.iterrows():
        # strftime %w: 0=Sun, 1=Mon...
        close_h = 18 if day['weekday'] in [4, 5] else 17
        for h in range(10, close_h + 1):
            for fmt in FORMATS:
                rows.append({
                    'date_val': day['date_val'], 
                    'hour': h, 
                    'bag_format': fmt,
                    'weekday': day['weekday'],
                    'weather_score': day['weather_score']
                })
    
    df_grid = pd.DataFrame(rows)
    actual_sales = df.groupby(['date_val', 'hour', 'bag_format']).size().reset_index(name='sales')
    df_final = pd.merge(df_grid, actual_sales, on=['date_val', 'hour', 'bag_format'], how='left')
    df_final['sales'] = df_final['sales'].fillna(0)

    models = {}
    for fmt in FORMATS:
        data_fmt = df_final[df_final['bag_format'] == fmt]
        if data_fmt.empty: continue
        
        # Reduced features to fundamental signals
        X = data_fmt[['weekday', 'hour', 'weather_score']]
        y = data_fmt['sales']
        
        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X, y)
        models[fmt] = regr

    joblib.dump(models, 'model.pkl')
    print("Training complete.")

if __name__ == "__main__":
    train_model()
