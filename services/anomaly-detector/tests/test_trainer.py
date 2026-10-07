"""Tests for the training pipeline and data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data_generator import (
    generate_quality_scores,
    generate_training_dataframe,
    generate_volume_series,
)
from app.services.trainer import (
    TrainedModelBundle,
    train_from_dataframe,
    validate_dataframe,
)


# ---------- Data Generator Tests ----------


class TestGenerateQualityScores:
    def test_returns_correct_length(self):
        scores = generate_quality_scores(n_points=200)
        assert len(scores) == 200

    def test_scores_are_in_valid_range(self):
        scores = generate_quality_scores(n_points=500)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_contains_anomalies(self):
        scores = generate_quality_scores(
            n_points=500, base_score=0.92, anomaly_ratio=0.05
        )
        # Some scores should be significantly below the base
        low_scores = [s for s in scores if s < 0.70]
        assert len(low_scores) > 0

    def test_deterministic_with_same_seed(self):
        s1 = generate_quality_scores(n_points=100, random_state=99)
        s2 = generate_quality_scores(n_points=100, random_state=99)
        assert s1 == s2


class TestGenerateVolumeSeries:
    def test_returns_correct_length(self):
        series = generate_volume_series(n_points=200)
        assert len(series) == 200

    def test_returns_timestamp_int_tuples(self):
        series = generate_volume_series(n_points=10)
        for ts, vol in series:
            assert hasattr(ts, "hour")  # datetime-like
            assert isinstance(vol, int)

    def test_volumes_are_non_negative(self):
        series = generate_volume_series(n_points=500)
        assert all(vol >= 0 for _, vol in series)


class TestGenerateTrainingDataframe:
    def test_has_required_columns(self):
        df = generate_training_dataframe(profile="stable", n_points=100)
        assert "timestamp" in df.columns
        assert "record_count" in df.columns
        assert "overall_score" in df.columns
        assert "schema_changed" in df.columns

    def test_stable_profile(self):
        df = generate_training_dataframe(profile="stable", n_points=500)
        assert len(df) == 500
        assert df["overall_score"].mean() > 0.85

    def test_volatile_profile(self):
        df = generate_training_dataframe(profile="volatile", n_points=500)
        assert len(df) == 500
        # Volatile should have higher std
        assert df["overall_score"].std() > 0.03

    def test_degrading_profile_has_downward_trend(self):
        df = generate_training_dataframe(profile="degrading", n_points=500)
        first_half = df["overall_score"][:250].mean()
        second_half = df["overall_score"][250:].mean()
        # Second half should have lower quality on average
        assert second_half < first_half


# ---------- Validation Tests ----------


class TestValidateDataframe:
    def test_valid_dataframe_passes(self):
        df = generate_training_dataframe(n_points=100)
        errors = validate_dataframe(df, min_rows=50)
        assert errors == []

    def test_missing_columns(self):
        df = pd.DataFrame({"timestamp": [1, 2, 3], "record_count": [10, 20, 30]})
        errors = validate_dataframe(df, min_rows=1)
        assert any("overall_score" in e for e in errors)

    def test_insufficient_rows(self):
        df = generate_training_dataframe(n_points=10)
        errors = validate_dataframe(df, min_rows=50)
        assert any("Insufficient" in e for e in errors)

    def test_score_out_of_range(self):
        df = generate_training_dataframe(n_points=60)
        df.loc[0, "overall_score"] = 1.5
        errors = validate_dataframe(df, min_rows=50)
        assert any("between 0 and 1" in e for e in errors)


# ---------- Trainer Tests ----------


class TestTrainFromDataframe:
    def test_trains_isolation_forest(self):
        df = generate_training_dataframe(n_points=200)
        bundle = train_from_dataframe(source_id="test-src", df=df)

        assert bundle.source_id == "test-src"
        assert bundle.isolation_forest is not None
        assert len(bundle.quality_history) == 200
        assert len(bundle.volume_history) == 200
        assert bundle.trained_at is not None

    def test_baseline_stats_computed(self):
        df = generate_training_dataframe(n_points=200)
        bundle = train_from_dataframe(source_id="test-src", df=df)

        stats = bundle.baseline_stats
        assert "quality_mean" in stats
        assert "quality_std" in stats
        assert "volume_mean" in stats
        assert "data_points" in stats
        assert stats["data_points"] == 200
        assert 0.0 < stats["quality_mean"] < 1.0

    def test_model_can_predict(self):
        df = generate_training_dataframe(n_points=200)
        bundle = train_from_dataframe(source_id="test-src", df=df)

        # Should detect a very low score as anomaly
        prediction = bundle.isolation_forest.predict(np.array([[0.3, 0.3, 0.0, -0.6]]))[0]
        assert prediction == -1  # anomaly

        # Should not flag a normal score
        m = bundle.baseline_stats["quality_mean"]
        normal_point = np.array([[m, m, 0.0, 0.0]])
        prediction = bundle.isolation_forest.predict(normal_point)[0]
        assert prediction == 1  # normal


# ---------- Serialization Tests ----------


class TestModelBundleSerialization:
    def test_isolation_forest_round_trip(self):
        df = generate_training_dataframe(n_points=200)
        bundle = train_from_dataframe(source_id="test-src", df=df)

        serialized = bundle.serialize_isolation_forest()
        assert serialized is not None
        assert len(serialized) > 0

        restored = TrainedModelBundle.deserialize_isolation_forest(serialized)
        # Verify the restored model produces same predictions
        test_point = np.array([[0.5, 0.5, 0.0, 0.0]])
        original_pred = bundle.isolation_forest.predict(test_point)[0]
        restored_pred = restored.predict(test_point)[0]
        assert original_pred == restored_pred

    def test_history_round_trip(self):
        df = generate_training_dataframe(n_points=100)
        bundle = train_from_dataframe(source_id="test-src", df=df)

        serialized = bundle.serialize_history()
        assert len(serialized) > 0

        restored = TrainedModelBundle.deserialize_history(serialized)
        assert len(restored["quality_history"]) == 100
        assert len(restored["volume_history"]) == 100
        assert "baseline_stats" in restored
