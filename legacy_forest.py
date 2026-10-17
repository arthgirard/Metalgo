"""
The per-hour random forest that served live predictions until the day-level
demand model (forecast.py) replaced it.

Kept, unchanged, for one reason: backtest.py compares every candidate model
against it, and the ship gate is "beat the forest". Nothing in the live app
imports this module.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import shop_schedule as sched
from event_service import get_game_info
from meteo import FALLBACK_TEMPERATURE, NEUTRAL_WEATHER_SCORE, weather_score_from_label

FORMATS = list(sched.FORMATS)

# Holdout evaluation only runs once there is enough history for the split to
# mean anything; below this the metrics would be noise.
MIN_DAYS_FOR_METRICS = 8
HOLDOUT_FRACTION = 0.2

# Single source of truth for the model's feature set/order. Imported by
# app.py so every predict() call builds its DataFrame with exactly these
# columns in exactly this order — sklearn doesn't reliably realign columns
# by name for you, so training and inference must agree explicitly.
#
# `weekday` here is the SQLite `strftime('%w')` convention (Sunday=0), not
# Python's. See shop_schedule for the converters.
#
# `is_special_event` was removed deliberately: as a single binary it made
# Saint-Jean (roughly 2x) and Halloween (roughly 1.3x) identical to the
# model, and with only a couple of event days ever recorded it could never
# learn to tell them apart. Calendar events are applied at prediction time
# by event_service's Bayesian blend instead, which needs far less data.
#
# `temperature` was removed for a different reason: it is stored as the
# day's MEAN, which across 61 days of history produced 61 distinct values
# with a minimum gap of 0.046 °C. That makes it a unique fingerprint for
# each day rather than a weather signal, and the forest duly used it to
# memorise individual days — it took 37% of the feature importance while
# training error (1.04) ran at less than half the holdout error (2.74).
# Dropping it, the holdout MAE improved from 1.839 to 1.672.
#
# The dataset still carries a `temperature` column and the weather sampler
# still records it, so once enough genuine per-HOUR readings accumulate it
# can be reinstated here and re-measured against the baseline. Hour-level
# temperature varies within a day and so cannot fingerprint one.
FEATURE_COLUMNS = ['weekday', 'hour', 'weather_score', 'is_game_day',
                   'is_playoff_game']

# Forest hyperparameters, shared by the fitted model and the holdout
# evaluation so the reported metrics describe the model actually shipped.
#
# max_depth is the load-bearing one. Left unconstrained (the sklearn
# default, with min_samples_leaf=1) the forest memorised individual days
# and scored WORSE than a plain same-weekday-and-hour average — a ratio of
# 0.91. Capping the depth costs nothing in fit time and brought it back to
# parity. Measured alternatives: max_depth=6 -> 0.98, min_samples_leaf=20
# -> 0.99, max_depth=4 -> 1.00.
RF_PARAMS = {"n_estimators": 100, "random_state": 42, "max_depth": 4}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_sales(conn):
    """One row per logged sale, with its raw context."""
    return pd.read_sql_query("""
        SELECT date(timestamp)          AS date_val,
               strftime('%w', timestamp) AS weekday,
               strftime('%H', timestamp) AS hour,
               meteo_summary,
               temperature,
               detail                   AS bag_format
        FROM logs
        WHERE action_type = 'VENTE'
    """, conn)


def _load_weather_log(conn):
    """
    Hourly weather samples, or an empty frame if none have been recorded.

    These are sampled on a timer regardless of whether anything sold, which
    is the point: `logs.meteo_summary` is only ever written when a sale
    happens, so a rainy dead morning contributed zero rain observations and
    the weather feature was effectively sampled proportional to sales.
    """
    empty = pd.DataFrame(columns=['date_val', 'hour', 'weather_score', 'temperature'])
    try:
        df = pd.read_sql_query("""
            SELECT date(timestamp)           AS date_val,
                   strftime('%H', timestamp) AS hour,
                   weather_score,
                   temperature
            FROM weather_log
        """, conn)
    except Exception:
        return empty          # table doesn't exist yet on an older DB
    if df.empty:
        return empty
    df['hour'] = df['hour'].astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _prepare_sales(df):
    """Type-cast, score the weather, and impute missing temperatures."""
    df = df.copy()
    df['weekday'] = df['weekday'].astype(int)
    df['hour'] = df['hour'].astype(int)

    # None (not a neutral score) for "Indisponible" and friends, so the
    # imputation below can tell "we don't know" apart from "it was average".
    df['weather_score'] = df['meteo_summary'].map(weather_score_from_label)

    # `temperature` is a newer column — rows logged before it existed have
    # NULL here. Rather than dropping that history (or crashing on NaN),
    # impute with the median of whatever real readings exist so far, so old
    # days stay usable and just look "temperature unknown/average" to the
    # model instead of being discarded.
    median_temp = df['temperature'].median()
    df['temperature'] = df['temperature'].fillna(
        median_temp if pd.notna(median_temp) else FALLBACK_TEMPERATURE)
    return df


def _mode_or(series, default):
    """Most common non-null value, or `default` when there isn't one."""
    values = series.dropna()
    if values.empty:
        return default
    return values.mode().iloc[0]


def _daily_context(sales, weather_log):
    """
    One row per day: weekday, and the day's representative weather.

    Weather is aggregated across the whole day (mean temperature, modal
    score) rather than taken from whichever sale happened to be logged
    first. A single early sale is not representative of the day, and
    training on it was why the weekly forecast — which uses a real day-level
    forecast value — disagreed so much with same-day predictions.
    """
    context = sales.groupby('date_val').agg(
        weekday=('weekday', 'first'),
        temperature=('temperature', 'mean'),
        weather_score=('weather_score', lambda s: _mode_or(s, np.nan)),
    ).reset_index()

    if not weather_log.empty:
        # Unbiased samples win over the sales-sampled ones wherever they exist.
        sampled = weather_log.groupby('date_val').agg(
            sampled_temperature=('temperature', 'mean'),
            sampled_score=('weather_score', lambda s: _mode_or(s, np.nan)),
        ).reset_index()
        context = context.merge(sampled, on='date_val', how='left')
        context['temperature'] = context['sampled_temperature'].fillna(context['temperature'])
        context['weather_score'] = context['sampled_score'].fillna(context['weather_score'])
        context = context.drop(columns=['sampled_temperature', 'sampled_score'])

    context['weather_score'] = context['weather_score'].fillna(NEUTRAL_WEATHER_SCORE)
    context['temperature'] = context['temperature'].fillna(FALLBACK_TEMPERATURE)
    return context


def _build_grid(sales, weather_log):
    """
    The (day, hour, format) grid the model trains on, with zero-sale hours
    present so the forest learns when *not* to expect sales.

    Hours come from `shop_schedule.hour_buckets`, which is the scheduled
    selling window UNION the hours that actually have sales. The old grid
    used the scheduled window alone and merged sales into it with a left
    join, so every sale outside it vanished without a warning — 26 of them
    in the live database, at 9h, at 17h on regular days, and at 18h.
    """
    context = _daily_context(sales, weather_log)
    observed_hours = sales.groupby('date_val')['hour'].apply(set).to_dict()

    game_info_cache = {}

    def game_info(date_val):
        if date_val not in game_info_cache:
            day = datetime.strptime(date_val, "%Y-%m-%d").date()
            game_info_cache[date_val] = get_game_info(day)
        return game_info_cache[date_val]

    # Hour-level weather, where the sampler has recorded it. Days that
    # predate the sampler fall back to their day-level aggregate.
    hourly_weather = {}
    if not weather_log.empty:
        for row in weather_log.groupby(['date_val', 'hour']).agg(
                weather_score=('weather_score', lambda s: _mode_or(s, np.nan)),
                temperature=('temperature', 'mean')).reset_index().itertuples():
            hourly_weather[(row.date_val, row.hour)] = (row.weather_score, row.temperature)

    rows = []
    for day in context.itertuples():
        is_game, is_playoff = game_info(day.date_val)
        hours = sched.hour_buckets_sql(day.weekday, observed_hours.get(day.date_val, ()))
        for hour in hours:
            score, temperature = hourly_weather.get((day.date_val, hour),
                                                    (day.weather_score, day.temperature))
            if pd.isna(score):
                score = day.weather_score
            if pd.isna(temperature):
                temperature = day.temperature
            for fmt in FORMATS:
                rows.append({
                    'date_val': day.date_val,
                    'weekday': day.weekday,
                    'hour': hour,
                    'weather_score': score,
                    'temperature': temperature,
                    'is_game_day': is_game,
                    'is_playoff_game': is_playoff,
                    'bag_format': fmt,
                })

    grid = pd.DataFrame(rows)
    actual = sales.groupby(['date_val', 'hour', 'bag_format']).size().reset_index(name='sales')
    final = pd.merge(grid, actual, on=['date_val', 'hour', 'bag_format'], how='left')
    final['sales'] = final['sales'].fillna(0)
    return final


# ---------------------------------------------------------------------------
# Fitting and evaluation
# ---------------------------------------------------------------------------

def _fit(dataset):
    """One RandomForest per bag format."""
    models = {}
    for fmt in FORMATS:
        rows = dataset[dataset['bag_format'] == fmt]
        if rows.empty:
            continue
        regr = RandomForestRegressor(**RF_PARAMS)
        regr.fit(rows[FEATURE_COLUMNS], rows['sales'])
        models[fmt] = regr
    return models


def _baseline_prediction(train_rows, test_rows):
    """
    The forecast to beat: the mean sales for that weekday and hour, learned
    from the training days alone.

    Without this there is no way to tell whether the forest is adding
    anything over "the same weekday usually sells about this much", which is
    what a member of staff would guess unaided.
    """
    means = train_rows.groupby(['weekday', 'hour'])['sales'].mean()
    overall = train_rows['sales'].mean()
    keys = list(zip(test_rows['weekday'], test_rows['hour']))
    return np.array([means.get(key, overall) for key in keys], dtype=float)


def _evaluate(dataset):
    """
    Holdout metrics on the most recent days.

    The split is chronological, never random: sales are a time series, and a
    random split would leak the same day's other hours into training and
    report a flatteringly low error.
    """
    days = sorted(dataset['date_val'].unique())
    if len(days) < MIN_DAYS_FOR_METRICS:
        return {"status": "insufficient_data", "n_days": len(days)}

    n_holdout = max(2, int(round(len(days) * HOLDOUT_FRACTION)))
    holdout_days = set(days[-n_holdout:])
    train = dataset[~dataset['date_val'].isin(holdout_days)]
    test = dataset[dataset['date_val'].isin(holdout_days)]
    if train.empty or test.empty:
        return {"status": "insufficient_data", "n_days": len(days)}

    per_format = {}
    model_errors, baseline_errors = [], []
    for fmt in FORMATS:
        train_rows = train[train['bag_format'] == fmt]
        test_rows = test[test['bag_format'] == fmt]
        if train_rows.empty or test_rows.empty:
            continue

        regr = RandomForestRegressor(**RF_PARAMS)
        regr.fit(train_rows[FEATURE_COLUMNS], train_rows['sales'])

        actual = test_rows['sales'].to_numpy(dtype=float)
        predicted = regr.predict(test_rows[FEATURE_COLUMNS])
        baseline = _baseline_prediction(train_rows, test_rows)

        model_abs = np.abs(predicted - actual)
        baseline_abs = np.abs(baseline - actual)
        model_errors.append(model_abs)
        baseline_errors.append(baseline_abs)

        # Error on the day's total, which is what staff actually act on,
        # rather than on individual hours.
        daily = (test_rows.assign(pred=predicted)
                 .groupby('date_val')
                 .agg(actual_total=('sales', 'sum'), pred_total=('pred', 'sum')))
        daily_total_mae = float((daily['pred_total'] - daily['actual_total']).abs().mean())

        per_format[fmt] = {
            "mae": round(float(model_abs.mean()), 3),
            "baseline_mae": round(float(baseline_abs.mean()), 3),
            "daily_total_mae": round(daily_total_mae, 2),
        }

    if not per_format:
        return {"status": "insufficient_data", "n_days": len(days)}

    mae = float(np.concatenate(model_errors).mean())
    baseline_mae = float(np.concatenate(baseline_errors).mean())
    return {
        "status": "ok",
        "n_days": len(days),
        "n_holdout_days": len(holdout_days),
        "mae": round(mae, 3),
        "baseline_mae": round(baseline_mae, 3),
        # >1 means the model beats the same-weekday average; <=1 means it
        # is not earning its complexity yet.
        "improvement_vs_baseline": round(baseline_mae / mae, 2) if mae > 0 else None,
        "per_format": per_format,
    }
