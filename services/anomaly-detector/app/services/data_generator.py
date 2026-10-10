"""Synthetic data generator for pre-training anomaly detection models.

Generates realistic data quality and volume patterns so that the anomaly
detectors have a meaningful baseline before real events arrive.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_quality_scores(
    n_points: int = 1000,
    base_score: float = 0.92,
    noise_std: float = 0.02,
    anomaly_ratio: float = 0.03,
    random_state: int = 42,
) -> list[float]:
    """Generate realistic quality score history.

    Creates a stream of quality scores with configurable baseline, normal
    noise, and injected anomalies (sudden drops).
    """
    rng = np.random.RandomState(random_state)
    scores = rng.normal(loc=base_score, scale=noise_std, size=n_points)

    # Inject anomalous drops
    n_anomalies = max(1, int(n_points * anomaly_ratio))
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)
    for idx in anomaly_indices:
        drop_magnitude = rng.uniform(0.15, 0.40)
        scores[idx] = base_score - drop_magnitude

    # Clamp to [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    return scores.tolist()


def generate_volume_series(
    n_points: int = 1000,
    base_volume: int = 10000,
    daily_amplitude: float = 0.3,
    weekly_amplitude: float = 0.15,
    noise_std: float = 0.05,
    anomaly_ratio: float = 0.03,
    start_time: datetime | None = None,
    interval_hours: int = 1,
    random_state: int = 42,
) -> list[tuple[datetime, int]]:
    """Generate realistic volume time-series with seasonal patterns.

    Simulates hourly data volume with daily and weekly seasonality,
    plus injected anomalous spikes/drops.
    """
    rng = np.random.RandomState(random_state)
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=n_points * interval_hours)

    timestamps = [start_time + timedelta(hours=i * interval_hours) for i in range(n_points)]

    volumes = np.zeros(n_points)
    for i, ts in enumerate(timestamps):
        hour_of_day = ts.hour
        day_of_week = ts.weekday()

        # Daily cycle: peak at 14:00, trough at 03:00
        daily_factor = 1.0 + daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 3) / 24)
        # Weekly cycle: lower on weekends
        weekly_factor = 1.0 - weekly_amplitude * (1.0 if day_of_week >= 5 else 0.0)

        noise = rng.normal(0, noise_std)
        volumes[i] = base_volume * daily_factor * weekly_factor * (1 + noise)

    # Inject anomalies (spikes and drops)
    n_anomalies = max(1, int(n_points * anomaly_ratio))
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)
    for idx in anomaly_indices:
        if rng.random() > 0.5:
            volumes[idx] *= rng.uniform(2.5, 5.0)  # spike
        else:
            volumes[idx] *= rng.uniform(0.05, 0.3)  # drop

    volumes = np.clip(volumes, 0, None).astype(int)
    return list(zip(timestamps, volumes.tolist()))


_PROFILE_CONFIGS: dict[str, dict] = {
    "stable": {
        "base_score": 0.93,
        "noise_std": 0.015,
        "anomaly_ratio": 0.02,
        "base_volume": 12000,
        "daily_amplitude": 0.25,
        "volume_noise_std": 0.04,
    },
    "volatile": {
        "base_score": 0.85,
        "noise_std": 0.06,
        "anomaly_ratio": 0.08,
        "base_volume": 5000,
        "daily_amplitude": 0.5,
        "volume_noise_std": 0.12,
    },
    "degrading": {
        "base_score": 0.88,
        "noise_std": 0.03,
        "anomaly_ratio": 0.05,
        "base_volume": 8000,
        "daily_amplitude": 0.35,
        "volume_noise_std": 0.07,
    },
}


def generate_training_dataframe(
    profile: str = "stable",
    n_points: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a complete training DataFrame for a given profile.

    Returns a DataFrame with columns: timestamp, record_count, overall_score,
    schema_changed.
    """
    cfg = _PROFILE_CONFIGS.get(profile, _PROFILE_CONFIGS["stable"])
    rng = np.random.RandomState(random_state)

    quality_scores = generate_quality_scores(
        n_points=n_points,
        base_score=cfg["base_score"],
        noise_std=cfg["noise_std"],
        anomaly_ratio=cfg["anomaly_ratio"],
        random_state=random_state,
    )

    volume_series = generate_volume_series(
        n_points=n_points,
        base_volume=cfg["base_volume"],
        daily_amplitude=cfg["daily_amplitude"],
        noise_std=cfg["volume_noise_std"],
        anomaly_ratio=cfg["anomaly_ratio"],
        random_state=random_state + 1,
    )

    # For degrading profile, add a downward trend to quality scores
    if profile == "degrading":
        drift = np.linspace(0, -0.12, n_points)
        quality_scores = np.clip(np.array(quality_scores) + drift, 0.0, 1.0).tolist()

    schema_changes = rng.random(n_points) < 0.01  # ~1% schema changes

    df = pd.DataFrame({
        "timestamp": [ts for ts, _ in volume_series],
        "record_count": [vol for _, vol in volume_series],
        "overall_score": quality_scores,
        "schema_changed": schema_changes.tolist(),
    })

    return df
