import os
import sqlite3
import tempfile
from datetime import datetime

import joblib

import shop_schedule as sched
from history import MIN_LOGS_THRESHOLD
from legacy_forest import (FEATURE_COLUMNS, RF_PARAMS, _baseline_prediction,  # noqa: F401
                           _build_grid, _daily_context, _evaluate, _fit,
                           _load_sales, _load_weather_log, _mode_or, _prepare_sales)
from meteo import now_local

DB_NAME = "data.db"
MODEL_PATH = "model.pkl"
FORMATS = list(sched.FORMATS)

# A weekday with fewer observed days than this is flagged to the UI as thin.
# Live data has 21 Sundays and 14 Saturdays but only 7 of each weekday from
# Tuesday to Friday, and the `weekday` feature is one the forest leans on.
THIN_COVERAGE_DAYS = 8


def _weekday_coverage(dataset):
    """
    How many distinct days of history exist per weekday, and which of those
    are too thin to trust. Surfaced in the UI so nobody reads a Wednesday
    forecast built on seven Wednesdays as though it were a Sunday forecast
    built on twenty-one.
    """
    days = dataset[['date_val', 'weekday']].drop_duplicates()
    counts = {int(w): int(n) for w, n in days.groupby('weekday').size().items()}
    return {
        "days_per_weekday": counts,
        "thin_weekdays": sorted(w for w, n in counts.items() if n < THIN_COVERAGE_DAYS),
        "threshold": THIN_COVERAGE_DAYS,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model_payload(payload, path=MODEL_PATH):
    """
    Write the model atomically.

    joblib.dump truncates the target in place, and app.py reloads whenever
    model.pkl's mtime moves. Hitting "Recalibrer" during business hours could
    therefore hand a live request a half-written pickle, which surfaced as a
    silent fallback to naive extrapolation. Writing to a temporary file in
    the same directory and renaming it makes readers see either the whole old
    file or the whole new one.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".model-", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(payload, tmp_path)
        # mkstemp creates 0600; keep the readable mode the old in-place dump
        # produced so a WSGI worker running as another user can still load it.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def train_model(db_name=DB_NAME, model_path=MODEL_PATH):
    """
    Retrain from scratch and persist. Returns the metrics dict, or None when
    there wasn't enough data to train at all.
    """
    print(">>> Training model...")
    conn = sqlite3.connect(db_name, timeout=30.0)
    try:
        sales = _load_sales(conn)
        weather_log = _load_weather_log(conn)
    finally:
        conn.close()

    if sales.empty:
        print("No sales logged yet.")
        return None

    # EXCLUDE TODAY: prevents poisoning the model with incomplete afternoon
    # hours, which would look like a collapse in demand after lunch.
    today_str = now_local().strftime('%Y-%m-%d')
    sales = sales[sales['date_val'] < today_str]
    if sales.empty:
        print("No past valid data for training.")
        return None

    # Drop days that look like a partial log rather than a quiet day.
    counts_per_day = sales.groupby('date_val').size()
    sales = sales[sales['date_val'].isin(counts_per_day[counts_per_day >= MIN_LOGS_THRESHOLD].index)]
    if sales.empty:
        print(f"No day reached {MIN_LOGS_THRESHOLD} logged sales.")
        return None

    sales = _prepare_sales(sales)
    if not weather_log.empty:
        weather_log = weather_log[weather_log['date_val'] < today_str]

    dataset = _build_grid(sales, weather_log)

    metrics = _evaluate(dataset)
    metrics.update(_weekday_coverage(dataset))

    models = _fit(dataset)
    if not models:
        print("Nothing to fit.")
        return None

    save_model_payload({
        "models": models,
        "feature_columns": list(FEATURE_COLUMNS),
        "formats": list(FORMATS),
        "metrics": metrics,
        "trained_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
        "n_training_rows": int(len(dataset)),
    }, model_path)

    # The server log is where model quality lives: the counter UI deliberately
    # shows staff a forecast and its range, not diagnostics.
    if metrics.get("status") == "ok":
        verdict = ("beats" if metrics["baseline_mae"] > metrics["mae"] else "LOSES TO")
        print(f"Training complete. Holdout MAE {metrics['mae']} {verdict} the "
              f"same-weekday-hour baseline {metrics['baseline_mae']} "
              f"(ratio {metrics['improvement_vs_baseline']}) over "
              f"{metrics['n_holdout_days']} of {metrics['n_days']} days.")
    else:
        print(f"Training complete. Not enough history for holdout metrics "
              f"({metrics.get('n_days', 0)} days).")

    thin = metrics.get("thin_weekdays") or []
    if thin:
        counts = metrics.get("days_per_weekday", {})
        detail = ", ".join(f"{sched.WEEKDAY_NAMES_FR[sched.py_weekday_from_sql(w)]}={counts.get(w, 0)}"
                           for w in thin)
        print(f"  Thin weekday coverage (under {metrics['threshold']} days): {detail}")
    return metrics


if __name__ == "__main__":
    train_model()
