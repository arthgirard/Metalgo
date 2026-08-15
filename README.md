# Metalgo 🧀

**Metalgo** is a bespoke inventory management and sales prediction system developed for **Fromagerie Le Métayer (https://fromagerielemetayer.com)**.

It is designed to solve a specific challenge in fresh cheese curd production: predicting the exact quantity of cheese to bag in different formats (250g, 1kg, 2kg) throughout the day to minimize waste and maximize freshness.

The application combines real-time sales logging with machine learning to provide staff with live production targets based on current trends, weather conditions, and special events.

## 🚀 Key Features

* **Real-Time Dashboard:** Live tracking of sales volume, peak hours, and total mass.
* **AI-Driven Forecasting:** A **Random Forest Regressor** per bag format predicts the day's demand, shown with an error band derived from the model's measured accuracy on held-out days — which narrows through the day as the remaining hours shrink.
* **Context Awareness:**
    * **Weather Integration:** Fetches current conditions, an hour-by-hour curve for today, and a 7-day outlook (via Open-Meteo). Weather is also sampled on a timer while the shop is open, independently of whether anything sells.
    * **Event Detection:** Detects Québec holidays, fixed occasions (St-Jean, Halloween), moving ones (Easter, Super Bowl) and Canadiens game days, and learns each one's real effect from history.
* **Offline-Tolerant Logging:** A tap that can't reach the server is queued on the device, shown as pending, and replayed with its original timestamp once the connection returns.
* **Mobile-First UI:** A modern, high-contrast interface designed for quick usage on tablets and smartphones behind the counter.

## 🛠 Tech Stack

* **Backend:** Python 3.14, Flask
* **Database:** SQLite (lightweight, serverless; runs in WAL mode)
* **Machine Learning:** scikit-learn (RandomForestRegressor), Pandas, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js
* **APIs:** Open-Meteo (weather), NHL public API (Canadiens schedule)

## 🏃 Running it

```bash
pip install -r requirements.txt
python app.py
```

Configuration is entirely through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `METALGO_DB` | `data.db` | SQLite database path |
| `METALGO_MODEL` | `model.pkl` | Trained model path |
| `METALGO_HOST` | `0.0.0.0` | Bind address |
| `METALGO_PORT` | `5000` | Bind port |
| `METALGO_DEBUG` | *(off)* | Set to `1` for the Werkzeug debugger. **Never in production** — it grants remote code execution to anyone who can reach the port. |
| `METALGO_NO_SCHEDULER` | *(off)* | Set to `1` to suppress the background retrain and weather threads |

Schema creation, migrations and the background jobs run at import, so a WSGI
server works too:

```bash
gunicorn app:app
```

## 🧠 Machine Learning Model

The system learns from its own history. As sales are logged, the dataset grows.

* **Training:** a full retrain runs automatically every night at 03:00, once at
  startup, and on demand from the **Recalibrer** button. To run it by hand:

      python train_model.py

* **Data requirements:** a day needs at least **10 logged sales**
  (`MIN_LOGS_THRESHOLD`) to be considered a real trading day rather than a
  partial log, and at least **8 distinct days** of history before holdout
  metrics are reported at all.

* **Features used for training:**
    * Day of week (SQLite `strftime('%w')` convention, Sunday = 0)
    * Hour of the day
    * Weather condition score (0 = bad, 1 = neutral, 2 = good)
    * Canadiens game day / playoff game day

  One model is trained **per bag format**, so the format is not a feature.
  Trees are capped at `max_depth=4` (`RF_PARAMS`) — see below.

* **Not a feature — temperature.** It is recorded as a *daily mean*, which
  over 61 days of history produced 61 distinct values with a smallest gap of
  0.046 °C: a unique fingerprint per day rather than a weather signal. An
  unconstrained forest used it to memorise individual days, taking 37% of the
  feature importance while its training error (1.04) sat at well under half
  its holdout error (2.74). It is still collected, and can be reinstated once
  the hourly weather sampler has gathered enough per-hour readings — those
  vary within a day and so cannot fingerprint one.

* **Not a feature — calendar events.** As a single binary flag, `is_special_event`
  made St-Jean (≈2x) and Halloween (≈1.3x) identical to the model, and with only a
  handful of event days on record it could never learn to separate them. Events are
  applied at prediction time instead, through a Bayesian blend of each event's
  observed effect against a hardcoded prior (`event_service.py`). Canadiens games
  stay model features, because there are enough recorded game days to learn from.

* **Honest accuracy reporting.** Every retrain evaluates itself on a
  chronological holdout of the most recent ~20% of days and compares its MAE
  against a naive "average for this weekday and hour" baseline. The result is
  printed to the server log on every retrain. **A ratio at or below 1.0 means
  the forest is not beating that simple average** — worth knowing before
  trusting it.

  That baseline is a genuine rival, not a straw man: it *is* the mean sales
  per weekday and hour, and a tree given only `weekday` and `hour` converges
  to exactly the same function. The forest can only win on the strength of
  the remaining features, so a ratio near 1.0 means weekday × hour is
  currently carrying nearly all the exploitable signal. Capping tree depth
  took the ratio from 0.91 to 1.00; getting meaningfully above that needs
  more history spanning more weather, not more model.

* **The range shown to staff comes from these metrics.** `daily_total_mae` —
  the mean absolute error on a whole day's total for that format — is the
  half-width of the band under each number, scaled by the square root of the
  fraction of the day still ahead. So it is widest at opening and reaches
  zero at closing, and it reports how wrong the model *actually tends to be*
  rather than how much its trees happen to agree.

## 🧪 Tests

    pytest

Covers the opening-hours and weekday-convention helpers, event detection
(including Easter and Super Bowl math and the years where two events collide),
the learned-multiplier blend, input validation, undo scoping, snapshot pruning
and the prediction payload.

## 🌍 Localization

* **Codebase:** English (variables, comments, logs).
* **Frontend:** French (tailored for the local staff in Quebec).
* **Location:** Hardcoded coordinates for the cheese shop (Lat: 45.183, Lon: -73.417).
* **Time:** All timestamps are on the shop's wall clock (`America/Toronto`),
  independent of the host's own timezone.

## 🗄 Schema notes

`logs.action_type` also contains historical `CONVERSION` rows from a retired
re-bagging feature. They are preserved, excluded from every sales figure, and
displayed as-is rather than as sales.

Two weekday conventions coexist deliberately: `daily_snapshots.weekday` uses
Python's (Monday = 0) while the model's `weekday` feature uses SQLite's
(Sunday = 0). All conversions go through `shop_schedule.py`.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
