"""
Walk-forward evaluation: the judge every forecasting change must pass.

For each of the last `test_days` trading days, and for each horizon h, every
candidate is fitted on the days strictly before (day - h) and asked for that
day. h = 0 is the pre-opening forecast, h = 2 is a weekend planned on
Thursday. Nothing a candidate sees was logged after the moment its forecast
would really have been made.

Reported per candidate and horizon:

  mae           daily-total absolute error per format, and on the day's total
  exceed_q90    share of format-days whose sales beat the 90th percentile
                (should sit near 0.10 for a calibrated spread)
  sellout_rate  share of format-days that would have sold out had the
                recommendation been bagged (sales >= recommendation)
  leftover      mean bags left over when bagging the recommendation

The sell-out rate is optimistic: history is itself censored by the sell-outs
that really happened, so true demand on those days was higher than the
sales the harness compares against.

    python backtest.py [--db data.db] [--days 44] [--horizons 0 1 2] [--trials]
    python backtest.py --intraday 12 14

`--intraday` scores the same-day forecast instead: at each cut hour, the
model fitted before the day is given the sales and customers logged before
the cut, with and without the customer-pace correction.
"""

import argparse
import sqlite3
from datetime import datetime, time, timedelta

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import event_service
import forecast
import history
import legacy_forest
import shop_schedule as sched
from meteo import today_local

FORMATS = list(sched.FORMATS)
CANDIDATES = ("forest", "weekday_mean", "blend")
TRIALS = ("blend+events", "blend+events+nhl", "blend+weather", "blend+temperature")
CENSORED_NOTE = ("Note: sell-out rate is measured on censored history and is optimistic "
                 "(days that really sold out understate demand).")
TRIAL_NOTE = ("Trials: events use the hardcoded priors (no learned multipliers, which would "
              "see the future); weather and temperature use each test day's observed values "
              "as a stand-in for its forecast; blend+temperature is scored only on days with "
              "sampler readings.")


def _open_ro(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


class _Forest:
    """The legacy per-hour forest, refitted per cut-off exactly as it shipped."""

    def __init__(self, conn, kept_days):
        sales = legacy_forest._load_sales(conn)
        sales = sales[sales["date_val"].isin({d.strftime("%Y-%m-%d") for d in kept_days})]
        weather = legacy_forest._load_weather_log(conn)
        self.grid = legacy_forest._build_grid(legacy_forest._prepare_sales(sales), weather)
        self._cache = {}

    def predict(self, cutoff, target):
        cut, day = cutoff.strftime("%Y-%m-%d"), target.strftime("%Y-%m-%d")
        if cut not in self._cache:
            train = self.grid[self.grid["date_val"] < cut]
            models = {}
            for fmt in FORMATS:
                rows = train[train["bag_format"] == fmt]
                if not rows.empty:
                    models[fmt] = RandomForestRegressor(**legacy_forest.RF_PARAMS).fit(
                        rows[legacy_forest.FEATURE_COLUMNS], rows["sales"])
            self._cache[cut] = models
        models = self._cache[cut]
        test = self.grid[self.grid["date_val"] == day]
        out = {}
        for fmt in FORMATS:
            rows = test[test["bag_format"] == fmt]
            out[fmt] = (float(models[fmt].predict(rows[legacy_forest.FEATURE_COLUMNS]).sum())
                        if fmt in models and not rows.empty else 0.0)
        return out


def _weekday_mean(daily, cutoff, target):
    past = daily[daily["date"] < cutoff]
    if past.empty:
        return None
    same = past[past["weekday"] == target.weekday()]
    source = same if not same.empty else past
    return {fmt: float(source[fmt].mean()) for fmt in FORMATS}


def _blend(daily, baskets, profile, cutoff, target, variant="blend", context=None):
    """The day-level model, optionally with one of the trial adjustments."""
    context = context or {}
    fit_kwargs, mean_kwargs, event_factor = {}, {}, 1.0
    if variant.startswith("blend+events"):
        fit_kwargs["event_days"] = context["event_days"]
        _, base, nhl = event_service.get_special_event(target)
        event_factor = base * (nhl if variant == "blend+events+nhl" else 1.0)
    elif variant == "blend+weather":
        fit_kwargs["weather_by_day"] = {d: w["score"] for d, w in context["weather"].items()
                                        if w["score"] is not None}
        mean_kwargs["weather_score"] = (context["weather"].get(target) or {}).get("score")
    elif variant == "blend+temperature":
        temperature = (context["weather"].get(target) or {}).get("temperature")
        if temperature is None:
            return None
        fit_kwargs["temperature_by_day"] = {d: w["temperature"] for d, w in context["weather"].items()
                                            if w["temperature"] is not None}
        mean_kwargs["temperature"] = temperature

    model = forecast.fit(daily, baskets, profile, cutoff, **fit_kwargs)
    if model is None:
        return None
    out = {}
    for fmt in FORMATS:
        mu = forecast.mean(model, target, fmt, event_factor, **mean_kwargs)
        var = forecast.variance(model, fmt, mu)
        out[fmt] = {"mean": mu,
                    "q90": forecast.quantile(mu, var, 0.90),
                    "recommended": forecast.recommend(model, target, fmt, event_factor,
                                                      **mean_kwargs)}
    return out


def _summarise(records, with_spread):
    """records: list of (actual: dict, prediction: dict) for one candidate/horizon."""
    if not records:
        return None
    mae = {fmt: float(np.mean([abs(p[fmt]["mean"] - a[fmt]) for a, p in records]))
           for fmt in FORMATS}
    mae["all"] = float(np.mean([abs(sum(p[f]["mean"] for f in FORMATS) - sum(a[f] for f in FORMATS))
                                for a, p in records]))
    pairs = [(a[f], p[f]) for a, p in records for f in FORMATS]
    return {
        "mae": mae,
        "exceed_q90": (float(np.mean([y > p["q90"] for y, p in pairs])) if with_spread else None),
        "sellout_rate": float(np.mean([y >= p["recommended"] for y, p in pairs])),
        "leftover": float(np.mean([max(0, p["recommended"] - y) for y, p in pairs])),
        "n_days": len(records),
    }


def _as_point(means):
    return {fmt: {"mean": m, "recommended": int(round(m))} for fmt, m in means.items()}


def run(db_path, test_days=44, horizons=(0, 1, 2), candidates=CANDIDATES):
    conn = _open_ro(db_path)
    try:
        before = today_local()
        daily = history.load_daily_sales(conn, before)
        baskets = history.load_baskets(conn, before)
        profile = history.hourly_profile(conn, before)
        forest = _Forest(conn, set(daily["date"])) if "forest" in candidates else None
        context = {"weather": history.daily_weather(conn, before),
                   "event_days": {d for d in daily["date"] if event_service.get_event_key(d)}}
    finally:
        conn.close()

    tests = daily.tail(test_days)
    results = {}
    for name in candidates:
        results[name] = {}
        for h in horizons:
            records = []
            for row in tests.itertuples(index=False):
                target = row.date
                cutoff = target - timedelta(days=h)
                actual = {fmt: int(getattr(row, f"_{FORMATS.index(fmt) + 2}")) for fmt in FORMATS}
                if name.startswith("blend"):
                    pred = _blend(daily, baskets, profile, cutoff, target, name, context)
                elif name == "weekday_mean":
                    means = _weekday_mean(daily, cutoff, target)
                    pred = _as_point(means) if means else None
                else:
                    pred = _as_point(forest.predict(cutoff, target))
                if pred is not None:
                    records.append((actual, pred))
            results[name][h] = _summarise(records, with_spread=name.startswith("blend"))
    return results


INTRADAY_CORRECTIONS = {"none": float("inf"), "customers": forecast.CUSTOMER_SMOOTHING}


def run_intraday(db_path, test_days=44, cut_hours=(12, 14)):
    """{correction: {cut_hour: {"mae", "n_days"}}}; "none" trusts the day's pace not at all."""
    conn = _open_ro(db_path)
    try:
        before = today_local()
        daily = history.load_daily_sales(conn, before)
        baskets = history.load_baskets(conn, before)
        profile = history.hourly_profile(conn, before)
        rows = history._load_rows(conn, before)
    finally:
        conn.close()
    rows = rows.assign(basket=history._group_baskets(rows))

    records = {name: {cut: [] for cut in cut_hours} for name in INTRADAY_CORRECTIONS}
    for row in daily.tail(test_days).itertuples(index=False):
        target = row.date
        model = forecast.fit(daily, baskets, profile, target)
        if model is None:
            continue
        actual = {fmt: int(getattr(row, f"_{FORMATS.index(fmt) + 2}")) for fmt in FORMATS}
        day_rows = rows[rows["date"] == target]
        for cut in cut_hours:
            now = datetime.combine(target, time(cut))
            seen = day_rows[day_rows["ts"] < now]
            sold = {fmt: int((seen["detail"] == fmt).sum()) for fmt in FORMATS}
            for name, smoothing in INTRADAY_CORRECTIONS.items():
                out = forecast.intraday(model, target, now, sold, int(seen["basket"].nunique()),
                                        smoothing=smoothing)
                pred = {fmt: {"mean": f["point"], "recommended": f["point"]}
                        for fmt, f in out["formats"].items()}
                records[name][cut].append((actual, pred))
    return {name: {cut: _summarise(recs, with_spread=False) for cut, recs in by_cut.items()}
            for name, by_cut in records.items()}


def _print_intraday(results):
    print(f"{'correction':12s} {'cut':>4s} {'250g':>6s} {'1kg':>6s} {'2kg':>6s} {'all':>6s} {'days':>5s}")
    for name, by_cut in results.items():
        for cut, r in by_cut.items():
            if r is None:
                print(f"{name:12s} {cut:3d}h (not enough history)")
                continue
            print(f"{name:12s} {cut:3d}h {r['mae']['250g']:6.2f} {r['mae']['1kg']:6.2f} "
                  f"{r['mae']['2kg']:6.2f} {r['mae']['all']:6.2f} {r['n_days']:5d}")


def _print(results):
    horizons = sorted({h for by_h in results.values() for h in by_h})
    for h in horizons:
        print(f"\nHorizon D+{h}")
        print(f"{'candidate':22s} {'250g':>6s} {'1kg':>6s} {'2kg':>6s} {'all':>6s} "
              f"{'>q90':>6s} {'sellout':>8s} {'leftover':>9s}")
        for name, by_h in results.items():
            r = by_h.get(h)
            if r is None:
                print(f"{name:22s} (not enough history)")
                continue
            exceed = "-" if r["exceed_q90"] is None else f"{r['exceed_q90']:.2f}"
            print(f"{name:22s} {r['mae']['250g']:6.2f} {r['mae']['1kg']:6.2f} {r['mae']['2kg']:6.2f} "
                  f"{r['mae']['all']:6.2f} {exceed:>6s} {r['sellout_rate']:8.2f} {r['leftover']:9.2f}")
    print("\n" + CENSORED_NOTE)
    if any(name in TRIALS for name in results):
        print(TRIAL_NOTE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default="data.db")
    parser.add_argument("--days", type=int, default=44)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--trials", action="store_true",
                        help="also score the event, NHL, weather and temperature trials")
    parser.add_argument("--intraday", type=int, nargs="+", metavar="HOUR",
                        help="score the same-day forecast at these cut hours instead")
    args = parser.parse_args(argv)
    if args.intraday:
        _print_intraday(run_intraday(args.db, test_days=args.days, cut_hours=tuple(args.intraday)))
        return
    candidates = CANDIDATES + (TRIALS if args.trials else ())
    _print(run(args.db, test_days=args.days, horizons=tuple(args.horizons),
               candidates=candidates))


if __name__ == "__main__":
    main()
