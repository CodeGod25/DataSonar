"""Tests for QualityOutlierDetector including pre-loaded model support."""

import numpy as np
from sklearn.ensemble import IsolationForest

from app.services.outlier_detection import QualityOutlierDetector


def test_quality_detector_flags_significant_drop_after_history():
    detector = QualityOutlierDetector()

    for value in [0.92, 0.93, 0.91, 0.94, 0.95, 0.93, 0.92, 0.94, 0.93, 0.92]:
        detector.detect("source-a", value)

    is_anomaly, score = detector.detect("source-a", 0.45)

    assert is_anomaly is True
    assert score > 0


def test_quality_detector_no_anomaly_with_stable_scores():
    detector = QualityOutlierDetector()

    for value in [0.9, 0.91, 0.89, 0.9, 0.92, 0.91, 0.9, 0.89, 0.9]:
        detector.detect("source-b", value)

    is_anomaly, score = detector.detect("source-b", 0.9)

    assert is_anomaly is False
    assert score >= 0


# ---------- Pre-loaded Model Tests ----------


def test_load_model_eliminates_cold_start():
    """With a pre-loaded model, detection works on the very first event."""
    from app.services.feature_utils import build_feature_matrix

    detector = QualityOutlierDetector()

    # Train a 4-feature model externally (matching the detector's feature space)
    raw_scores = np.random.RandomState(42).normal(0.9, 0.02, 100)
    training_features = build_feature_matrix(raw_scores)
    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(training_features)

    # Load it into the detector
    detector.load_model("pre-src", model=model, history=raw_scores.tolist())

    # First event — should detect anomaly immediately (no cold-start)
    is_anomaly, score = detector.detect("pre-src", 0.40)
    assert is_anomaly is True
    assert score > 0


def test_load_model_normal_score_not_flagged():
    """Pre-loaded model should not flag normal scores."""
    from app.services.feature_utils import build_feature_matrix

    detector = QualityOutlierDetector()

    raw_scores = np.array([0.90, 0.91, 0.92, 0.89, 0.93, 0.91] * 10)
    training_features = build_feature_matrix(raw_scores)
    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(training_features)

    detector.load_model("pre-src", model=model, history=raw_scores.tolist())

    is_anomaly, score = detector.detect("pre-src", 0.91)
    assert is_anomaly is False


def test_retrain_updates_model():
    """After retraining, the model should reflect new data patterns."""
    detector = QualityOutlierDetector()

    # Build enough history for retraining
    for v in [0.90, 0.91, 0.92, 0.89, 0.93, 0.91] * 5:
        detector.detect("retrain-src", v)

    retrained = detector.retrain("retrain-src")
    assert retrained is True
    assert detector.has_model("retrain-src")


def test_retrain_fails_with_insufficient_history():
    """Retraining should fail gracefully with too few data points."""
    detector = QualityOutlierDetector()

    for v in [0.90, 0.91, 0.92]:
        detector.detect("small-src", v)

    retrained = detector.retrain("small-src")
    assert retrained is False


def test_get_model_and_history():
    """Accessor methods should return the model and history."""
    from app.services.feature_utils import build_feature_matrix

    detector = QualityOutlierDetector()

    training_data = [0.90, 0.91, 0.92] * 10
    training_features = build_feature_matrix(np.array(training_data))
    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(training_features)

    detector.load_model("acc-src", model=model, history=training_data)

    assert detector.get_model("acc-src") is model
    assert len(detector.get_history("acc-src")) == 30
    assert detector.has_model("acc-src") is True
    assert detector.has_model("unknown-src") is False
