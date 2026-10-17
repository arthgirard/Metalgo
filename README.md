# Metalgo 🧀

**Metalgo** is a bespoke inventory management and sales prediction system developed for **Fromagerie Le Métayer (https://fromagerielemetayer.com)**.

It is designed to solve a specific challenge in fresh cheese curd production: curds are bagged once, in the morning, in three formats (250g, 1kg, 2kg), and nothing is bagged later in the day. Metalgo predicts the day's sales and recommends how many bags of each format to prepare.

Staff log each bag sold with one press of a USB button by the register. Those presses are all the data the forecast needs.

## 🚀 Key Features

* **Real-Time Dashboard:** Live tracking of sales volume, peak hours, and total mass.
* **7-day forecast:** For the next seven calendar days after today (closed days skipped), the expected sales per format. A whole weekend can be prepared two days ahead.
* **Same-day forecast:** The Aujourd'hui view shows the day's expected total per format and its likely range, adjusted to how many customers have come in so far.
* **Context Awareness:**
    * **Weather Integration:** Fetches current conditions, an hour-by-hour curve for today, and a 7-day outlook (via Open-Meteo). Weather is also sampled on a timer while the shop is open, independently of whether anything sells.
    * **Event Detection:** Detects Québec holidays, fixed occasions (St-Jean, Halloween), moving ones (Easter, Super Bowl) and Canadiens game days, and learns each one's real effect from history.
* **Offline-Tolerant Logging:** A tap that can't reach the server is queued on the device, shown as pending, and replayed with its original timestamp once the connection returns.
* **Mobile-First UI:** A modern, high-contrast interface designed for quick usage on tablets and smartphones behind the counter.

## 🛠 Tech Stack

* **Backend:** Python 3.14, Flask
* **Database:** SQLite (lightweight, serverless; runs in WAL mode)
* **Forecasting:** Pandas, NumPy, SciPy (negative binomial), Joblib; scikit-learn only for the retired forest in the evaluation harness
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

## 🧠 Demand model

The forecast is a day-level demand model (`forecast.py`), not a per-hour
model: what matters is the number of bags to prepare each morning.

* **Expected sales** for a day and format blend two estimates equally: the
  recent level (last 10 trading days) × that weekday's index × the format's
  share, and the mean of the last three same weekdays. Calendar events
  (`event_service.py`) multiply the result; each event's effect is learned
  against ordinary same-weekday days within three weeks of it and blended
  with a hardcoded prior.
* **Spread.** Sales are negative binomial. Customers buy several bags at
  once (presses at most 2 s apart count as one customer), which sets a floor
  on the noise; day-to-day variation on top of that is measured from the
  model's own one-step-ahead errors over the last 30 days.
* **Recommendation** = 90th percentile + 1, so the chance that demand
  *reaches* the stock is at most 10% (`SELLOUT_TARGET`). Leftovers are
  sold the next morning or reground, so the target leans towards having
  enough.
* **Sell-outs are not corrected for.** A day a format ran out understates
  its demand. On the history to 2026-10-17, only 8 of 354 format-days show a
  format going silent while the others kept selling, all after 15h, so the
  effect is a few bags late in the day and the model does not try to
  correct it.
* **Switched off on measurement.** NHL games, weather and temperature are
  available as switches in `forecast.py`, but each made the walk-forward
  error worse on the current history and is off.
* **Same-day correction.** During the day, the hours left are scaled by the
  number of customers so far against the number expected by now, trusted
  more as the day goes on (`CUSTOMER_SMOOTHING`).

### Training

A fit runs every night at 03:00, once at startup, and from the
**Recalibrer** button. By hand (it reads `METALGO_DB` / `METALGO_MODEL`):

    python train_model.py

A day needs at least **10 logged sales** (`MIN_LOGS_THRESHOLD`) to count as
a trading day; Mondays, Christmas Day and New Year's Day are closed and
never used. Sales a little before opening or after closing count. The fit
needs at least 8 trading days.

Each night the new fit is checked over the last 14 days at D+1 against the
same-weekday mean. It is rejected, and the previous model kept, only if its
error is more than 1.25× the baseline's (`GATE_TOLERANCE`): on two weeks of
data a good model can lose narrowly by chance. Each fit also writes its
expected sales for the coming open days into `forecast_log`, so the
forecasts can later be compared with what happened.

### Evaluation

`backtest.py` is the judge for any forecasting change. For each of the last
44 trading days it refits on the days before the forecast would really have
been made (D+0, D+1, D+2) and reports per format the error, the share of
days above the 90th percentile, the sell-out rate and the leftover:

    python backtest.py --db data.db --days 44 --horizons 0 1 2
    python backtest.py --trials            # also score events, NHL, weather, temperature
    python backtest.py --intraday 12 14    # same-day forecast at 12h and 14h

On the history as of 2026-10-17, the day-total error at D+0 / D+1 / D+2 is
11.27 / 11.44 / 11.60 bags, against 12.27 / 12.27 / 12.20 for the retired
random forest and 12.21 for the same-weekday mean. Sell-outs at the
recommendation run at 10–11%. That rate is measured on history that is
itself cut short by the few real sell-outs, so it is slightly optimistic.

## 🧪 Tests

    pytest

Covers the opening-hours, closed-date and weekday-convention helpers, event
detection (including Easter and Super Bowl math and the years where two events
collide), the learned event multipliers, the demand model and its spread, the
same-day correction, the nightly fit and its gate, the evaluation harness,
the 7-day forecast endpoint, input validation, undo scoping, snapshot
pruning and the prediction payload.

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

Two weekday conventions coexist: `daily_snapshots.weekday` and the demand
model use Python's (Monday = 0), while SQL queries read SQLite's
`strftime('%w')` (Sunday = 0). All conversions go through `shop_schedule.py`.

`forecast_log` (`forecast_log.py`) keeps every nightly forecast of expected
sales, so forecasts can later be compared with what sold. Its first build
also logged the bagging recommendation; that column is dropped on startup. Databases that ran an
earlier build may also hold empty `bagging_plan`, `conversions` and
`closing_record` tables; nothing reads them.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
