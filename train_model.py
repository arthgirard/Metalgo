"""
Nightly fit of the day-level demand model (forecast.py), with a safety check
before it replaces the model in service.

Runs at 03:00, once at startup, and from the "Recalibrer" button:

    python train_model.py
"""

import os
import sqlite3
import tempfile
from datetime import timedelta

import joblib
import numpy as np

import forecast
import history
import shop_schedule as sched
import stock
from event_service import get_event_key, get_special_event
from meteo import get_weekly_forecast, now_local

DB_NAME = "data.db"
MODEL_PATH = "model.pkl"
FORMATS = list(sched.FORMATS)

PAYLOAD_VERSION = 2

# The light gate: one-day-ahead error over this many recent days, against a
# plain same-weekday mean on the same days. The full walk-forward lives in
# backtest.py; this only has to catch a fit gone wrong overnight.
GATE_DAYS = 14

# How much worse than the same-weekday mean a new fit may score there and
# still replace the old one. Over 14 days the gap between two reasonable
# forecasts is mostly noise (live data: 10.85 vs 10.37 for a model that wins
# the 44-day walk-forward), and rejecting on noise would freeze the model
# with a stale level night after night. The check exists to catch a fit
# that has gone clearly wrong.
GATE_TOLERANCE = 1.25


# ---------------------------------------------------------------------------
# Inputs for one target day
# ---------------------------------------------------------------------------

def day_inputs(day, db_path, weekly_forecast=None):
    """
    The keyword arguments forecast.mean/recommend take for `day`, following
    the measured switches in forecast.py. `weekly_forecast` is the list from
    meteo.get_weekly_forecast(); it is only read when a weather switch is on.
    """
    _, base, nhl = get_special_event(day, db_path=db_path)
    factor = 1.0
    if forecast.USE_EVENTS:
        factor *= base
    if forecast.USE_NHL:
        factor *= nhl
    kwargs = {"event_factor": factor}

    if forecast.USE_WEATHER or forecast.USE_TEMPERATURE:
        entry = next((d for d in weekly_forecast or [] if d["date"] == day.strftime("%Y-%m-%d")), {})
        if forecast.USE_WEATHER:
            kwargs["weather_score"] = entry.get("score")
        if forecast.USE_TEMPERATURE:
            kwargs["temperature"] = entry.get("temperature")
    return kwargs


def _fit_inputs(conn, before):
    """Everything forecast.fit needs, read once."""
    daily = history.load_daily_sales(conn, before)
    inputs = {
        "daily": daily,
        "baskets": history.load_baskets(conn, before),
        "profile": history.hourly_profile(conn, before),
        "fit_kwargs": {"censored": stock.confirmed_sellouts(conn, before)},
    }
    if forecast.USE_EVENTS:
        inputs["fit_kwargs"]["event_days"] = {d for d in daily["date"] if get_event_key(d)}
    if forecast.USE_WEATHER or forecast.USE_TEMPERATURE:
        weather = history.daily_weather(conn, before)
        if forecast.USE_WEATHER:
            inputs["fit_kwargs"]["weather_by_day"] = {
                d: w["score"] for d, w in weather.items() if w["score"] is not None}
        if forecast.USE_TEMPERATURE:
            inputs["fit_kwargs"]["temperature_by_day"] = {
                d: w["temperature"] for d, w in weather.items() if w["temperature"] is not None}
    return inputs


def _fit(inputs, as_of):
    return forecast.fit(inputs["daily"], inputs["baskets"], inputs["profile"], as_of,
                        **inputs["fit_kwargs"])


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------

def _light_gate(inputs):
    """
    D+1 error on the day's total over the last GATE_DAYS trading days,
    model vs same-weekday mean, each fitted only on days before the eve.
    """
    daily = inputs["daily"]
    model_errors, baseline_errors = [], []
    for row in daily.tail(GATE_DAYS).itertuples(index=False):
        cutoff = row.date - timedelta(days=1)
        model = _fit(inputs, cutoff)
        past = daily[daily["date"] < cutoff]
        same = past[past["weekday"] == row.weekday]
        if model is None or same.empty:
            continue
        predicted = sum(forecast.mean(model, row.date, fmt) for fmt in FORMATS)
        model_errors.append(abs(predicted - row.total))
        baseline_errors.append(abs(same["total"].mean() - row.total))
    if not model_errors:
        return {"mae": None, "baseline_mae": None, "n_days": 0}
    return {"mae": round(float(np.mean(model_errors)), 2),
            "baseline_mae": round(float(np.mean(baseline_errors)), 2),
            "n_days": len(model_errors)}


def _has_current_payload(path):
    try:
        payload = joblib.load(path)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("version") == PAYLOAD_VERSION


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
    Fit as of today, check it, persist it, and record the recommendations it
    makes for the upcoming open days. Returns the metrics dict, or None when
    there is not enough history to fit at all.
    """
    print(">>> Training model...")
    now = now_local()
    today = now.date()
    conn = sqlite3.connect(db_name, timeout=30.0)
    try:
        # The tables this writes to may not exist yet on a database the app
        # never opened; creating them is additive and idempotent.
        stock.init_tables(conn)
        inputs = _fit_inputs(conn, today)
    finally:
        conn.close()

    model = _fit(inputs, today)
    if model is None:
        print(f"Not enough history to fit ({len(inputs['daily'])} trading days, "
              f"need {forecast.MIN_FIT_DAYS}).")
        return None

    gate = _light_gate(inputs)
    loses = gate["mae"] is not None and gate["mae"] > gate["baseline_mae"]
    broken = gate["mae"] is not None and gate["mae"] > gate["baseline_mae"] * GATE_TOLERANCE
    metrics = {"status": "ok", "gate": gate, "n_days": len(inputs["daily"]),
               "censored_days": len(inputs["fit_kwargs"]["censored"])}

    if broken and _has_current_payload(model_path):
        print(f"Nightly fit rejected: MAE {gate['mae']} vs baseline {gate['baseline_mae']}; "
              f"keeping previous model.")
        metrics["status"] = "rejected"
        return metrics

    save_model_payload({
        "version": PAYLOAD_VERSION,
        "demand_model": model,
        "metrics": metrics,
        "trained_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "switches": {"events": forecast.USE_EVENTS, "nhl": forecast.USE_NHL,
                     "weather": forecast.USE_WEATHER, "temperature": forecast.USE_TEMPERATURE},
    }, model_path)

    _record_forecasts(db_name, model, now)

    # The server log is where model quality lives: the counter UI shows staff
    # a recommendation and its margin, not diagnostics.
    if gate["mae"] is None:
        print(f"Training complete. Not enough history for the check ({metrics['n_days']} days).")
    else:
        verdict = "LOSES TO" if loses else "beats"
        print(f"Training complete. D+1 MAE {gate['mae']} {verdict} the same-weekday mean "
              f"{gate['baseline_mae']} over the last {gate['n_days']} days.")
    return metrics


def _record_forecasts(db_name, model, now):
    """What this model recommends for each upcoming open day, as of now."""
    weekly = None
    if forecast.USE_WEATHER or forecast.USE_TEMPERATURE:
        weekly = get_weekly_forecast()
    rows = []
    for day in sched.upcoming_open_days(now.date()):
        kwargs = day_inputs(day, db_name, weekly)
        for fmt in FORMATS:
            rows.append({"target_date": day.strftime("%Y-%m-%d"), "format": fmt,
                         "horizon_days": (day - now.date()).days,
                         "expected": forecast.mean(model, day, fmt, **kwargs),
                         "recommended": forecast.recommend(model, day, fmt, **kwargs)})
    conn = sqlite3.connect(db_name, timeout=30.0)
    try:
        stock.record_forecast(conn, rows, now)
        conn.commit()
    finally:
        conn.close()


def main():
    """Command-line run, pointed at the same files as the app (METALGO_DB / METALGO_MODEL)."""
    return train_model(os.environ.get("METALGO_DB", DB_NAME),
                       os.environ.get("METALGO_MODEL", MODEL_PATH))


if __name__ == "__main__":
    main()
