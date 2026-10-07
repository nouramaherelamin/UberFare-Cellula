# Uber Fare Intelligence — Task 3 REAL FINAL

This deployment is built against the actual Task 1 / Task 2 files supplied for the Uber Fare project.

## Source alignment
The Task 2 notebook defines these categorical features:
- Car Condition
- Weather
- Traffic Condition

And these numeric features:
- passenger_count
- hour
- day
- month
- weekday
- year
- jfk_dist
- ewr_dist
- lga_dist
- sol_dist
- nyc_dist
- distance
- bearing
- hour_sin
- hour_cos

The app creates `hour_sin` and `hour_cos` exactly as Task 2 does. Bearing is entered in degrees in the UI and converted to radians because the cleaned CSV stores bearing in radians.

The saved `final_regression_pipeline.joblib` is loaded once when Flask starts. No retraining occurs in the web app.

## Run

Use Python with the versions in `requirements.txt` (especially scikit-learn 1.9.0 because the saved pipeline was serialized with that version).

```powershell
py -m pip install -r requirements.txt
py app.py
```

Open:

`http://127.0.0.1:5000`

## Important
Stop any older Flask process before starting this package. Do not mix its `app.py`, `templates`, or model with files from older Task 3 folders.

## Task 3 requirements covered
- Same saved Task 2 pipeline
- Clear web form
- Current values sent on every prediction
- Result updates without stale page values
- Missing/invalid input handling
- Model loaded once at startup
- `/health` endpoint
- `/api/predict` endpoint

The official Task 3 requires at least three successful UI predictions and one invalid-input screenshot; capture those from the browser after running the app.


### Input-domain safeguards
The UI intentionally keeps prediction inputs inside the observed Task 2 domain: training years 2009–2015, trip distance 0–50 km, passenger count 1–6, and bearing −180° to 180°. The Task 2 notebook explicitly reports the long-tailed distance distribution and identifies values above 50 km as extreme; the dataset stores bearing in radians with an observed range of approximately −π to +π.

## Premium UI Motion
- Hero entrance animation
- Blue car floating/glow animation
- Animated orbit rings and light sweep
- Staggered hero badges
- Smooth hover/focus micro-interactions
- Animated prediction/result card
- Scroll reveal for dashboard sections
- Reduced-motion accessibility support
