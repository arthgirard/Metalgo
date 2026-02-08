# Metalgo 🧀

**Metalgo** is a bespoke inventory management and sales prediction system developed for **Fromagerie Le Métayer (https://fromagerielemetayer.com)**.

It is designed to solve a specific challenge in fresh cheese curd production: predicting the exact quantity of cheese to bag in different formats (250g, 1kg, 2kg) throughout the day to minimize waste and maximize freshness.

The application combines real-time sales logging with machine learning to provide staff with live production targets based on current trends, weather conditions, and special events.

## 🚀 Key Features

* **Real-Time Dashboard:** Live tracking of sales volume, peak hours, and top-selling formats.
* **AI-Driven Forecasting:** Uses a **Random Forest Regressor** to predict remaining demand for the day based on historical data.
* **Context Awareness:**
    * **Weather Integration:** Automatically fetches real-time weather and forecasts (via Open-Meteo) to adjust sales predictions (e.g., rain reduces foot traffic).
    * **Event Detection:** Automatically detects holidays (St-Jean, Christmas) and special events (Super Bowl) to apply sales multipliers.
* **Smart Inventory:** Tracks "conversions" (re-bagging larger formats into smaller ones) to keep inventory counts accurate.
* **Mobile-First UI:** A modern, high-contrast interface designed for quick usage on tablets and smartphones behind the counter.

## 🛠 Tech Stack

* **Backend:** Python 3.12+, Flask
* **Database:** SQLite (Lightweight, serverless)
* **Machine Learning:** Scikit-learn (RandomForestRegressor), Pandas, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **APIs:** Open-Meteo (Weather data)

## ⚙️ Installation & Setup

1.  **Clone the repository**
    git clone https://github.com/yourusername/metalgo.git
    cd metalgo

2.  **Install dependencies**
    pip install -r requirements.txt

3.  **Run the Application**
    python app.py
    ### The database is initialized automatically on first run.
    ### Access at http://localhost:5000

## 🧠 Machine Learning Model

The system learns from its own history. As sales are logged, the dataset grows.

* **Training:** To train (or retrain) the model, run:
    python train_model.py

    *Note: The model requires a minimum of 10 days of logged data to begin making accurate predictions.*

* **Features used for training:**
    * Day of the week
    * Hour of the day
    * Weather condition score (0=Bad to 2=Good)
    * Specific bag format (250g, 1kg, 2kg)

## 🌍 Localization

* **Codebase:** English (Variables, Comments, Logs).
* **Frontend:** French (Tailored for the local staff in Quebec).
* **Location:** Hardcoded coordinates for the cheese shop (Lat: 45.183, Lon: -73.417).

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
