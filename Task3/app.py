from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime
import math
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_regression_pipeline.joblib"

# Load the exact Task 2 pipeline once at startup.
model = joblib.load(MODEL_PATH)

CAR_CONDITIONS = ["Bad", "Good", "Very Good", "Excellent"]
WEATHER_OPTIONS = ["sunny", "windy", "cloudy", "stormy", "rainy"]
TRAFFIC_OPTIONS = ["Flow Traffic", "Dense Traffic", "Congested Traffic"]


def form_value(form, *names):
    for name in names:
        value = form.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_float(form, names, label, minimum=None, maximum=None):
    if isinstance(names, str):
        names = (names,)

    raw = form_value(form, *names)
    if raw == "":
        raise ValueError(f"{label} is required.")

    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{label} must be a valid number.")

    if not math.isfinite(value):
        raise ValueError(f"{label} must be a valid number.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{label} cannot be less than {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{label} cannot be greater than {maximum}.")

    return value


def get_int(form, names, label, minimum=None, maximum=None):
    value = get_float(form, names, label, minimum, maximum)
    if int(value) != value:
        raise ValueError(f"{label} must be a whole number.")
    return int(value)


def parse_datetime(form):
    date_value = form_value(form, "pickup_date")
    time_value = form_value(form, "pickup_time")

    if not date_value:
        raise ValueError("Pickup date is required.")
    if not time_value:
        raise ValueError("Pickup time is required.")

    try:
        date_obj = datetime.strptime(date_value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Pickup date is invalid.")

    # The Task 2 training data covers 2009–2015. Keep UI inputs inside
    # the model's observed time domain instead of extrapolating to 2026+.
    if date_obj.year < 2009 or date_obj.year > 2015:
        raise ValueError("Pickup date must be between 2009 and 2015 for this trained model.")

    try:
        time_obj = datetime.strptime(time_value, "%H:%M").time()
    except ValueError:
        raise ValueError("Pickup time is invalid.")

    return datetime.combine(date_obj, time_obj)


def build_prediction_row(form):
    """
    Build the exact 18 columns expected by the saved Task 2 pipeline.

    Task 2's saved pipeline expects hour_sin/hour_cos as input columns.
    Bearing is displayed as degrees in the UI but converted to radians,
    matching the cleaned dataset representation.
    """

    car = form_value(form, "car_condition")
    weather = form_value(form, "weather")
    traffic = form_value(form, "traffic_condition")

    if car not in CAR_CONDITIONS:
        raise ValueError("Please select a valid car condition.")
    if weather not in WEATHER_OPTIONS:
        raise ValueError("Please select a valid weather option.")
    if traffic not in TRAFFIC_OPTIONS:
        raise ValueError("Please select a valid traffic condition.")

    passengers = get_int(
        form, "passenger_count", "Passenger count", 1, 6
    )

    dt = parse_datetime(form)

    distance = get_float(
        form, "distance", "Trip distance", 0, 50
    )

    bearing_degrees = get_float(
        form, "bearing", "Bearing", -180, 180
    )

    # Dataset stores bearing in radians.
    bearing = math.radians(bearing_degrees)

    jfk = get_float(form, "jfk_dist", "JFK distance", 0)
    ewr = get_float(form, "ewr_dist", "EWR distance", 0)
    lga = get_float(form, "lga_dist", "LGA distance", 0)
    sol = get_float(form, "sol_dist", "Statue of Liberty distance", 0)
    nyc = get_float(form, "nyc_dist", "NYC distance", 0)

    hour = dt.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    row = {
        "Car Condition": car,
        "Weather": weather,
        "Traffic Condition": traffic,
        "passenger_count": passengers,
        "hour": hour,
        "day": dt.day,
        "month": dt.month,
        "weekday": dt.weekday(),
        "year": dt.year,
        "jfk_dist": jfk,
        "ewr_dist": ewr,
        "lga_dist": lga,
        "sol_dist": sol,
        "nyc_dist": nyc,
        "distance": distance,
        "bearing": bearing,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }

    columns = [
        "Car Condition",
        "Weather",
        "Traffic Condition",
        "passenger_count",
        "hour",
        "day",
        "month",
        "weekday",
        "year",
        "jfk_dist",
        "ewr_dist",
        "lga_dist",
        "sol_dist",
        "nyc_dist",
        "distance",
        "bearing",
        "hour_sin",
        "hour_cos",
    ]

    return pd.DataFrame([row], columns=columns), dt


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            X, dt = build_prediction_row(request.form)

            prediction = float(model.predict(X)[0])

            if not math.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            prediction = max(0.0, prediction)
            form_data["prediction_datetime"] = dt.strftime("%b %d, %Y · %I:%M %p")

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        car_conditions=CAR_CONDITIONS,
        weather_options=WEATHER_OPTIONS,
        traffic_options=TRAFFIC_OPTIONS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        X, dt = build_prediction_row(request.form)
        prediction = float(model.predict(X)[0])

        if not math.isfinite(prediction):
            raise ValueError("The model returned an invalid prediction.")

        prediction = max(0.0, prediction)

        return {
            "ok": True,
            "prediction": round(prediction, 2),
            "pickup": dt.strftime("%b %d, %Y · %I:%M %p"),
            "distance": float(request.form.get("distance", 0)),
            "passengers": int(float(request.form.get("passenger_count", 0))),
            "request_id": datetime.now().strftime("%H%M%S%f"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 400


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
