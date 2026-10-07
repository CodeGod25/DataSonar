"""Centralized model training pipeline — optimized for terminal use.

Trains multi-feature IsolationForest models from DataFrames.
Prophet removed entirely; time-series handled by EWMA detector at runtime.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.config import get_settings
from app.services.feature_utils import build_feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class TrainedModelBundle:
    """Container for all trained artifacts for a single source."""

    source_id: str
    isolation_forest: IsolationForest | None = None
    quality_history: list[float] = field(default_factory=list)
    volume_history: list[tuple[datetime, int]] = field(default_factory=list)
    baseline_stats: dict = field(default_factory=dict)
    trained_at: datetime | None = None

    def serialize_isolation_forest(self) -> bytes | None:
        if self.isolation_forest is None:
            return None
        buf = io.BytesIO()
        joblib.dump(self.isolation_forest, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_isolation_forest(data: bytes) -> IsolationForest:
        buf = io.BytesIO(data)
        return joblib.load(buf)

    def serialize_history(self) -> bytes:
        payload = {
            "quality_history": self.quality_history,
            "volume_history": self.volume_history,
            "baseline_stats": self.baseline_stats,
        }
        buf = io.BytesIO()
        joblib.dump(payload, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_history(data: bytes) -> dict:
        buf = io.BytesIO(data)
        return joblib.load(buf)


def validate_dataframe(df: pd.DataFrame, min_rows: int = 50) -> list[str]:
    """Validate that a DataFrame has the required columns and enough data."""
    errors = []

    required_columns = {"timestamp", "record_count", "overall_score"}
    missing = required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")

    if len(df) < min_rows:
        errors.append(
            f"Insufficient data: got {len(df)} rows, need at least {min_rows}"
        )

    if not errors:
        if not pd.api.types.is_numeric_dtype(df["record_count"]):
            errors.append("Column 'record_count' must be numeric")
        if not pd.api.types.is_numeric_dtype(df["overall_score"]):
            errors.append("Column 'overall_score' must be numeric")
        if df["overall_score"].min() < 0 or df["overall_score"].max() > 1:
            errors.append("Column 'overall_score' values must be between 0 and 1")

    return errors


def train_from_dataframe(
    source_id: str,
    df: pd.DataFrame,
    contamination: float | None = None,
    random_state: int | None = None,
) -> TrainedModelBundle:
    """Train anomaly detection models from a DataFrame.

    Uses multi-feature IsolationForest with optimized hyperparameters.

    Parameters
    ----------
    source_id:
        Identifier for the data source.
    df:
        DataFrame with columns: timestamp, record_count, overall_score.
    contamination:
        IsolationForest contamination parameter.
    random_state:
        Random state for reproducibility.

    Returns
    -------
    TrainedModelBundle with fitted IsolationForest, historical data,
    and computed baseline statistics.
    """
    settings = get_settings()
    if contamination is None:
        contamination = settings.ISO_FOREST_CONTAMINATION
    if random_state is None:
        random_state = settings.ISO_FOREST_RANDOM_STATE

    quality_scores = df["overall_score"].values.astype(np.float64)

    # Build multi-feature matrix (fully vectorized — no Python loops)
    feature_matrix = build_feature_matrix(quality_scores)

    # Train optimized IsolationForest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=settings.ISO_FOREST_ESTIMATORS,
        max_samples="auto",
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,  # parallel tree building
    )
    iso_forest.fit(feature_matrix)
    logger.info(
        "Trained IsolationForest for '%s': %d points × 4 features, %d trees",
        source_id,
        len(quality_scores),
        settings.ISO_FOREST_ESTIMATORS,
    )

    # Build volume history
    timestamps = pd.to_datetime(df["timestamp"])
    record_counts = df["record_count"].astype(int).tolist()
    volume_history = list(zip(timestamps.tolist(), record_counts))

    # Compute baseline statistics (fully vectorized)
    volume_arr = np.array(record_counts, dtype=np.float64)
    baseline_stats = {
        "quality_mean": float(quality_scores.mean()),
        "quality_std": float(quality_scores.std()),
        "quality_median": float(np.median(quality_scores)),
        "quality_p5": float(np.percentile(quality_scores, 5)),
        "quality_p95": float(np.percentile(quality_scores, 95)),
        "volume_mean": float(volume_arr.mean()),
        "volume_std": float(volume_arr.std()),
        "volume_median": float(np.median(volume_arr)),
        "data_points": len(df),
        "n_features": 4,
        "n_estimators": settings.ISO_FOREST_ESTIMATORS,
    }

    # Score training data for anomaly threshold calibration
    train_scores = -iso_forest.score_samples(feature_matrix)
    baseline_stats["anomaly_score_mean"] = float(train_scores.mean())
    baseline_stats["anomaly_score_std"] = float(train_scores.std())

    bundle = TrainedModelBundle(
        source_id=source_id,
        isolation_forest=iso_forest,
        quality_history=quality_scores.tolist(),
        volume_history=volume_history,
        baseline_stats=baseline_stats,
        trained_at=datetime.utcnow(),
    )

    logger.info(
        "Bundle for '%s': quality_mean=%.4f, volume_mean=%.1f",
        source_id,
        baseline_stats["quality_mean"],
        baseline_stats["volume_mean"],
    )

    return bundle
