"""Shared, fully-vectorized feature building utilities.

Used by both trainer.py and outlier_detection.py to ensure consistent,
high-performance feature construction.

Features per point (4 total):
    0 - raw quality score
    1 - rolling mean  (window)
    2 - rolling std   (window)
    3 - delta from previous score
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def build_feature_matrix(scores: np.ndarray, window: int = 10) -> np.ndarray:
    """Build a (n, 4) feature matrix from raw quality scores.

    Fully vectorized — no Python loops. Uses scipy's uniform_filter1d
    for O(n) rolling statistics with C-level speed.

    Parameters
    ----------
    scores:
        1-D array of quality scores.
    window:
        Rolling window size for mean/std.

    Returns
    -------
    np.ndarray of shape (n, 4).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    features = np.empty((n, 4), dtype=np.float64)

    # Feature 0: raw score
    features[:, 0] = scores

    # Feature 1 & 2: rolling mean and std via uniform_filter1d
    # mode='nearest' replicates edge values (equivalent to min(i, window) behaviour)
    rolling_mean = uniform_filter1d(scores, size=window, mode="nearest", origin=0)
    rolling_sq_mean = uniform_filter1d(scores ** 2, size=window, mode="nearest", origin=0)
    # Variance = E[X^2] - (E[X])^2, clamp to 0 for numerical safety
    variance = np.maximum(rolling_sq_mean - rolling_mean ** 2, 0.0)

    features[:, 1] = rolling_mean
    features[:, 2] = np.sqrt(variance)

    # Feature 3: delta from previous score
    features[0, 3] = 0.0
    features[1:, 3] = np.diff(scores)

    return features


def build_single_point_features(
    score: float, history: list[float], window: int = 10
) -> np.ndarray:
    """Build a 1×4 feature vector for a single new score.

    Used for real-time single-event detection when batch mode is not available.
    """
    recent = history[-window:] if len(history) >= window else history[:]
    recent_with = recent + [score]
    arr = np.array(recent_with, dtype=np.float64)
    rolling_mean = float(arr.mean())
    rolling_std = float(arr.std()) if len(arr) > 1 else 0.0
    delta = score - history[-1] if history else 0.0
    return np.array([[score, rolling_mean, rolling_std, delta]], dtype=np.float64)
