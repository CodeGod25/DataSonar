"""Model management — training, persistence, and drift detection.

Fully synchronous, using local filesystem storage instead of MinIO.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.services.data_generator import generate_training_dataframe
from app.services.local_storage import LocalModelStorage
from app.services.outlier_detection import QualityOutlierDetector
from app.services.time_series import TimeSeriesAnomalyDetector
from app.services.trainer import TrainedModelBundle, train_from_dataframe, validate_dataframe

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model training, persistence, loading, and drift detection.

    All operations are synchronous — no external services required.
    """

    def __init__(
        self,
        storage: LocalModelStorage,
        outlier_detector: QualityOutlierDetector,
        time_series_detector: TimeSeriesAnomalyDetector | None = None,
    ) -> None:
        self._settings = get_settings()
        self._storage = storage
        self._outlier_detector = outlier_detector
        self._time_series_detector = time_series_detector
        self._loaded_sources: set[str] = set()

    def initialize(self) -> None:
        """Load any previously persisted models from local storage."""
        existing = self._storage.list_persisted_sources()
        for source_id in existing:
            try:
                self._load_persisted_model(source_id)
                logger.info("Loaded persisted model for '%s'", source_id)
            except Exception:
                logger.warning(
                    "Failed to load model for '%s', will retrain",
                    source_id,
                    exc_info=True,
                )

    def pretrain(self, profiles: list[str] | None = None) -> list[str]:
        """Pre-train models on synthetic data. Returns list of source IDs."""
        if profiles is None:
            profiles = self._settings.PRETRAIN_PROFILES

        trained = []
        for profile in profiles:
            source_id = f"pretrained-{profile}"
            df = generate_training_dataframe(profile=profile, n_points=1000)
            self.train_from_dataframe(source_id=source_id, df=df)
            trained.append(source_id)
            logger.info("Pre-trained model for profile '%s'", profile)

        return trained

    def _load_persisted_model(self, source_id: str) -> bool:
        """Load a model bundle from local storage into the detectors."""
        model = self._storage.load_model(source_id, "isolation_forest.joblib")
        if model is None:
            return False

        history_data = self._storage.load_model(source_id, "history.joblib")
        quality_history = []
        volume_history = []
        if history_data is not None:
            quality_history = history_data.get("quality_history", [])
            volume_history = history_data.get("volume_history", [])

        self._outlier_detector.load_model(
            source_id=source_id,
            model=model,
            history=quality_history,
        )

        if self._time_series_detector and volume_history:
            self._time_series_detector.load_history(
                source_id=source_id,
                history=volume_history,
            )

        self._loaded_sources.add(source_id)
        return True

    def _persist_model(self, source_id: str, bundle: TrainedModelBundle) -> None:
        """Persist a trained model bundle to local storage."""
        if bundle.isolation_forest is not None:
            self._storage.save_model(
                source_id, bundle.isolation_forest, "isolation_forest.joblib"
            )

        history_payload = {
            "quality_history": bundle.quality_history,
            "volume_history": bundle.volume_history,
            "baseline_stats": bundle.baseline_stats,
        }
        self._storage.save_model(source_id, history_payload, "history.joblib")

        baseline = {
            "source_id": source_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "quality_mean": bundle.baseline_stats.get("quality_mean"),
            "quality_std": bundle.baseline_stats.get("quality_std"),
            "volume_mean": bundle.baseline_stats.get("volume_mean"),
            "data_points": bundle.baseline_stats.get("data_points"),
            "trained_at": bundle.trained_at.isoformat() if bundle.trained_at else None,
        }
        self._storage.save_baseline(source_id, baseline)
        logger.info("Persisted model bundle for '%s'", source_id)

    def train_from_dataframe(
        self, source_id: str, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Train models from a DataFrame and persist them."""
        bundle = train_from_dataframe(source_id=source_id, df=df)

        if bundle.isolation_forest is not None:
            self._outlier_detector.load_model(
                source_id=source_id,
                model=bundle.isolation_forest,
                history=bundle.quality_history,
            )

        if self._time_series_detector and bundle.volume_history:
            self._time_series_detector.load_history(
                source_id=source_id,
                history=bundle.volume_history,
            )

        self._persist_model(source_id, bundle)
        self._loaded_sources.add(source_id)
        return bundle.baseline_stats

    def train_from_csv(
        self, source_id: str, csv_path: str
    ) -> dict[str, Any]:
        """Train models from a CSV file path."""
        df = pd.read_csv(csv_path)
        errors = validate_dataframe(df, min_rows=self._settings.MIN_TRAINING_ROWS)
        if errors:
            raise ValueError("; ".join(errors))
        return self.train_from_dataframe(source_id=source_id, df=df)

    def ensure_source_baseline(self, source_id: str) -> dict[str, Any]:
        baseline = self._storage.load_baseline(source_id)
        if baseline is None:
            baseline = {
                "source_id": source_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "quality_mean": None,
            }
            self._storage.save_baseline(source_id, baseline)
        self._loaded_sources.add(source_id)
        return baseline

    def retrain_source(self, source_id: str) -> dict[str, Any]:
        """Retrain models for a source from accumulated history."""
        self._outlier_detector.retrain(source_id)

        quality_mean = self._outlier_detector.baseline_mean(source_id)
        quality_history = self._outlier_detector.get_history(source_id)
        volume_history = []
        if self._time_series_detector:
            volume_history = self._time_series_detector.get_history(source_id)

        baseline = {
            "source_id": source_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "quality_mean": quality_mean,
        }
        self._storage.save_baseline(source_id, baseline)

        model = self._outlier_detector.get_model(source_id)
        if model is not None:
            bundle = TrainedModelBundle(
                source_id=source_id,
                isolation_forest=model,
                quality_history=quality_history,
                volume_history=volume_history,
                baseline_stats={"quality_mean": quality_mean},
                trained_at=datetime.utcnow(),
            )
            self._persist_model(source_id, bundle)

        self._loaded_sources.add(source_id)
        return baseline

    def detect_drift(self, source_id: str) -> bool:
        baseline = self._storage.load_baseline(source_id)
        if not baseline:
            return False
        baseline_mean = baseline.get("quality_mean")
        current_mean = self._outlier_detector.baseline_mean(source_id)
        if baseline_mean is None or current_mean is None:
            return False
        return (
            abs(float(current_mean) - float(baseline_mean))
            >= self._settings.DRIFT_MEAN_DELTA_THRESHOLD
        )

    def is_loaded(self, source_id: str) -> bool:
        return source_id in self._loaded_sources
