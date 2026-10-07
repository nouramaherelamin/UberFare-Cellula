"""Helper used by the saved Uber fare pipeline.

Keep this file in the same folder as final_regression_pipeline.joblib
(or anywhere on PYTHONPATH). joblib stores a reference to
`uber_features.add_cyclical_hour`, so it must be importable when loading.
"""
import numpy as np


def add_cyclical_hour(X):
    """Add hour_sin / hour_cos so 23:00 and 00:00 are close together.

    Uses only each row's own `hour` value -> no target info, no leakage.
    """
    X = X.copy()
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    return X
