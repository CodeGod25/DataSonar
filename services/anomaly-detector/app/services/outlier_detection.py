"""Quality score anomaly detection using optimized IsolationForest.

Optimizations:
- Multi-feature detection: [score, rolling_mean, rolling_std, delta]
- Fully vectorized feature building (no Python loops)
- Batch detection: predict/score_samples on entire arrays at once
- Lazy retraining: only retrain after significant history growth
- n_jobs=-1 for parallel tree building
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest

from app.core.config import get_settings
from app.services.feature_utils import build_feature_matrix, build_single_point_features

logger = logging.getLogger(__name__)


class QualityOutlierDetector:
    """Detect quality score degradations per source.

    Uses a multi-feature IsolationForest for higher accuracy.
    Supports both single-event and batch detection modes.
    """

    def __init__(self, history_limit: int | None = None) -> None:
        settings = get_settings()
        self._history_limit = history_limit or settings.QUALITY_HISTORY_LIMIT
        self._n_estimators = settings.ISO_FOREST_ESTIMATORS
        self._contamination = settings.ISO_FOREST_CONTAMINATION
        self._random_state = settings.ISO_FOREST_RANDOM_STATE

        self._history: dict[str, list[float]] = defaultdict(list)
        self._models: dict[str, IsolationForest] = {}
        # Track how many points the model was trained on to avoid excessive retraining
        self._model_train_size: dict[str, int] = defaultdict(int)
        # Minimum new points before retraining (avoids retraining on every append)
        self._retrain_interval = 200

    # ── Model injection ──────────────────────────────────────────────

    def load_model(
        self,
        source_id: str,
        model: IsolationForest,
        history: list[float] | None = None,
    ) -> None:
        """Inject a pre-trained IsolationForest model for a source."""
        self._models[source_id] = model
        if history:
            self._history[source_id] = history[-self._history_limit :]
            self._model_train_size[source_id] = len(self._history[source_id])
        logger.info(
            "Loaded IsolationForest for source '%s' (history=%d)",
            source_id,
            len(self._history.get(source_id, [])),
        )

    # ── Training ─────────────────────────────────────────────────────

    def _create_forest(self) -> IsolationForest:
        return IsolationForest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            max_samples="auto",
            max_features=1.0,
            bootstrap=True,
            random_state=self._random_state,
            n_jobs=-1,
        )

    def retrain(self, source_id: str) -> bool:
        """Retrain the IsolationForest from accumulated history."""
        history = self._history.get(source_id, [])
        if len(history) < 25:
            logger.warning(
                "Cannot retrain '%s': only %d points (need 25+)",
                source_id,
                len(history),
            )
            return False

        features = build_feature_matrix(np.array(history, dtype=np.float64))
        model = self._create_forest()
        model.fit(features)
        self._models[source_id] = model
        self._model_train_size[source_id] = len(history)
        logger.info(
            "Retrained IsolationForest for '%s' on %d points",
            source_id,
            len(history),
        )
        return True

    def _should_retrain(self, source_id: str) -> bool:
        """Check if enough new data has accumulated to justify retraining."""
        current_size = len(self._history.get(source_id, []))
        last_train_size = self._model_train_size.get(source_id, 0)
        return (current_size - last_train_size) >= self._retrain_interval

    # ── Batch Detection (NEW — primary hot path) ─────────────────────

    def detect_batch(
        self, source_id: str, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect anomalies for an entire array of quality scores at once.

        This is the fast path — builds features once, calls predict/score_samples
        once on the full matrix instead of per-row.

        Parameters
        ----------
        source_id:
            Data source identifier.
        scores:
            1-D numpy array of quality scores.

        Returns
        -------
        Tuple of (is_anomaly_array, anomaly_score_array), both shape (n,).
        """
        n = len(scores)
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_scores = np.zeros(n, dtype=np.float64)

        history = self._history[source_id]
        model = self._models.get(source_id)

        if model is not None:
            # Prepend history to scores for accurate rolling features at the start
            combined = np.concatenate(
                [np.array(history[-10:], dtype=np.float64), scores]
            ) if history else scores
            offset = len(combined) - n

            feature_matrix = build_feature_matrix(combined)
            # Only take the rows corresponding to the new scores
            new_features = feature_matrix[offset:]

            # Single batch call — this is the big speedup
            predictions = model.predict(new_features)
            raw_scores = -model.score_samples(new_features)

            is_anomaly = predictions == -1
            anomaly_scores = np.clip(raw_scores, 0.0, 1.0)

        elif len(history) >= 25:
            # No model yet but enough history — train one now
            features_train = build_feature_matrix(
                np.array(history, dtype=np.float64)
            )
            model = self._create_forest()
            model.fit(features_train)
            self._models[source_id] = model
            self._model_train_size[source_id] = len(history)

            # Now predict on new scores
            combined = np.concatenate(
                [np.array(history[-10:], dtype=np.float64), scores]
            )
            offset = len(combined) - n
            feature_matrix = build_feature_matrix(combined)
            new_features = feature_matrix[offset:]

            predictions = model.predict(new_features)
            raw_scores = -model.score_samples(new_features)
            is_anomaly = predictions == -1
            anomaly_scores = np.clip(raw_scores, 0.0, 1.0)

        elif len(history) >= 8:
            # Fallback: vectorized z-score detection
            mean = float(np.mean(history))
            std = float(np.std(history))
            z = np.abs((scores - mean) / (std or 1.0))
            is_anomaly = (z >= 3.0) & (scores < mean)
            anomaly_scores = np.minimum(z / 6.0, 1.0)

        # Bulk-append to history and trim
        history.extend(scores.tolist())
        if len(history) > self._history_limit:
            self._history[source_id] = history[-self._history_limit :]

        # Periodic retraining based on new data accumulation
        if self._should_retrain(source_id):
            self.retrain(source_id)

        return is_anomaly, anomaly_scores

    # ── Single-event Detection (kept for compatibility) ──────────────

    def detect(self, source_id: str, overall_score: float) -> tuple[bool, float]:
        """Detect whether a new quality score is anomalous.

        Returns (is_anomaly, anomaly_score) where score is in [0, 1].
        """
        history = self._history[source_id]
        is_anomaly = False
        score = 0.0

        model = self._models.get(source_id)

        if model is not None:
            point = build_single_point_features(overall_score, history)
            prediction = model.predict(point)[0]
            raw_score = -float(model.score_samples(point)[0])
            is_anomaly = bool(prediction == -1)
            score = min(max(raw_score, 0.0), 1.0)

        elif len(history) >= 25:
            # No model yet but enough history — train one now
            features = build_feature_matrix(np.array(history, dtype=np.float64))
            model = self._create_forest()
            model.fit(features)
            self._models[source_id] = model
            self._model_train_size[source_id] = len(history)

            point = build_single_point_features(overall_score, history)
            prediction = model.predict(point)[0]
            raw_score = -float(model.score_samples(point)[0])
            is_anomaly = bool(prediction == -1)
            score = min(max(raw_score, 0.0), 1.0)

        elif len(history) >= 8:
            # Fallback: z-score detection
            mean = float(np.mean(history))
            std = float(np.std(history))
            z = abs((overall_score - mean) / (std or 1.0))
            is_anomaly = z >= 3.0 and overall_score < mean
            score = min(z / 6.0, 1.0)

        # Append after scoring
        history.append(overall_score)

        # Trim history if needed, but only retrain periodically
        if len(history) > self._history_limit:
            self._history[source_id] = history[-self._history_limit :]

        # Periodic retraining based on new data accumulation
        if self._should_retrain(source_id):
            self.retrain(source_id)

        return is_anomaly, score

    # ── Accessors ────────────────────────────────────────────────────

    def baseline_mean(self, source_id: str) -> float | None:
        h = self._history.get(source_id, [])
        return float(np.mean(h)) if h else None

    def get_model(self, source_id: str) -> IsolationForest | None:
        return self._models.get(source_id)

    def get_history(self, source_id: str) -> list[float]:
        return list(self._history.get(source_id, []))

    def has_model(self, source_id: str) -> bool:
        return source_id in self._models
