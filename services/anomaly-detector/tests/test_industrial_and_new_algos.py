"""Tests for newly added algorithms (Bollinger, DBSCAN, Prophet/Trend) and Industrial Stream Detector."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.industrial_stream import (
    IndustrialStreamDetector,
    ReservoirSampler,
    WelfordAccumulator,
)
from app.services.universal_detector import (
    _bollinger_bands,
    _dbscan_clustering,
    _prophet_or_trend_forecaster,
    detect_anomalies,
)


def _write_csv(content: str) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write(content)
        return f.name


def test_bollinger_bands():
    normal = np.ones(50) * 100.0
    normal[25] = 999.0  # Spike anomaly
    scores, upper, lower = _bollinger_bands(normal, window=10, num_std=2.0)
    assert len(scores) == 50
    assert scores[25] > 0.45
    assert scores[0] == 0.0


def test_dbscan_clustering():
    # 30 points clustered around (0,0), and 2 distant points
    rng = np.random.RandomState(42)
    cluster = rng.normal(0, 0.2, (30, 2))
    outliers = np.array([[15.0, 15.0], [-15.0, -15.0]])
    X = np.vstack([cluster, outliers])

    preds, scores = _dbscan_clustering(X, eps=0.5, min_samples=3)
    assert len(preds) == 32
    # Outliers should be marked as noise (-1)
    assert preds[30] == -1
    assert preds[31] == -1
    assert scores[30] > 0.3


def test_trend_forecaster():
    dates = pd.date_range("2024-01-01", periods=30, freq="D").astype(str)
    vals = [10.0 + i for i in range(29)] + [500.0]  # Trend with sudden huge break
    df = pd.DataFrame({"timestamp": dates, "value": vals})

    scores, reasons, algos = _prophet_or_trend_forecaster(df, ["timestamp"], ["value"])
    assert len(scores) == 30
    assert scores[29] > 0.4
    assert len(algos) > 0


def test_welford_accumulator():
    acc = WelfordAccumulator()
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    acc.update_batch(arr)
    assert np.isclose(acc.mean, 30.0)
    assert np.isclose(acc.variance, 250.0)
    assert acc.min_val == 10.0
    assert acc.max_val == 50.0
    assert acc.count == 5


def test_reservoir_sampler():
    sampler = ReservoirSampler(sample_size=20, random_state=42)
    stream = np.arange(100).reshape(-1, 1)
    sampler.update_batch(stream)
    sample = sampler.get_sample_matrix()
    assert len(sample) == 20
    assert sample.shape[1] == 1


def test_industrial_stream_detector():
    csv_text = "id,temp,pressure,vibration\n"
    for i in range(200):
        # Injected anomaly at row 150
        if i == 150:
            csv_text += f"{i},9999.0,8888.0,7777.0\n"
        else:
            csv_text += f"{i},{50.0 + (i%5)},{100.0 + (i%3)},{1.2}\n"

    path = _write_csv(csv_text)
    detector = IndustrialStreamDetector(chunk_size=50, reservoir_size=100)

    # Test stream training
    bundle = detector.train_on_stream(path)
    assert "temp" in bundle.numeric_columns
    assert bundle.online_stats["temp"]["count"] == 200

    # Test stream detection
    summary = detector.detect_stream(path, bundle)
    assert summary.total_records == 200
    assert summary.anomaly_count >= 1

    # Injected anomaly row 150 should be detected
    anom_rows = [r.row_index for r in summary.row_anomalies]
    assert 150 in anom_rows

    # Test stream cleaning to file
    out_path = Path(path).with_name("cleaned_industrial.csv")
    res = detector.stream_clean_to_file(path, out_path, bundle)
    assert res["rows_cleaned"] == 200
    assert out_path.exists()

    # Clean up
    Path(path).unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
