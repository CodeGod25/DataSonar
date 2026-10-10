"""Synchronous event processing pipeline for terminal-based anomaly detection.

Processes events from a DataFrame (synthetic or CSV) and returns
structured results for terminal display.

Optimized: uses batch detection (single sklearn call for all rows)
instead of row-by-row processing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.model_manager import ModelManager
from app.services.outlier_detection import QualityOutlierDetector
from app.services.time_series import TimeSeriesAnomalyDetector

logger = logging.getLogger(__name__)


class DetectionResult:
    """Result of anomaly detection on a single event."""

    __slots__ = (
        "event_index",
        "timestamp",
        "quality_score",
        "record_count",
        "schema_changed",
        "is_anomaly",
        "anomaly_types",
        "anomaly_score",
        "severity",
    )

    def __init__(
        self,
        event_index: int,
        timestamp: datetime,
        quality_score: float,
        record_count: int,
        schema_changed: bool,
        is_anomaly: bool,
        anomaly_types: list[str],
        anomaly_score: float,
        severity: str,
    ) -> None:
        self.event_index = event_index
        self.timestamp = timestamp
        self.quality_score = quality_score
        self.record_count = record_count
        self.schema_changed = schema_changed
        self.is_anomaly = is_anomaly
        self.anomaly_types = anomaly_types
        self.anomaly_score = anomaly_score
        self.severity = severity


class EventProcessor:
    """Synchronous event processor for terminal-mode detection.

    Uses batch detection for maximum throughput on DataFrames.
    """

    def __init__(
        self,
        time_series_detector: TimeSeriesAnomalyDetector,
        quality_detector: QualityOutlierDetector,
        model_manager: ModelManager,
    ) -> None:
        self._settings = get_settings()
        self._ts_detector = time_series_detector
        self._quality_detector = quality_detector
        self._model_manager = model_manager

    def process_dataframe(
        self, source_id: str, df: pd.DataFrame
    ) -> list[DetectionResult]:
        """Run anomaly detection on every row of a DataFrame.

        Uses batch detection — builds features once and calls sklearn
        predict/score_samples once on the full matrix instead of per-row.

        Returns a list of DetectionResult objects.
        """
        n = len(df)
        threshold = self._settings.ANOMALY_ALERT_THRESHOLD

        # Extract arrays once
        timestamps = pd.to_datetime(df["timestamp"])
        ts_array = timestamps.values  # numpy datetime64 for batch
        ts_pydatetime = np.array(timestamps.dt.to_pydatetime())  # for DetectionResult objects
        scores = df["overall_score"].values.astype(np.float64)
        counts = df["record_count"].values.astype(np.int64)
        schema_flags = (
            df["schema_changed"].values.astype(bool)
            if "schema_changed" in df.columns
            else np.zeros(n, dtype=bool)
        )

        # ── Batch quality detection ──────────────────────────────────
        qual_anomaly, qual_scores = self._quality_detector.detect_batch(
            source_id=source_id,
            scores=scores,
        )

        # ── Batch volume detection ───────────────────────────────────
        vol_anomaly, vol_scores = self._ts_detector.detect_batch(
            source_id=source_id,
            timestamps=ts_pydatetime,
            record_counts=counts,
        )

        # ── Vectorized result assembly ───────────────────────────────
        # Compute combined anomaly scores (max of volume, quality, schema)
        combined_scores = np.maximum(vol_scores, qual_scores)
        combined_scores = np.where(schema_flags, np.maximum(combined_scores, 0.7), combined_scores)

        # Determine which rows have any anomaly type
        has_vol = vol_anomaly
        has_qual = qual_anomaly
        has_schema = schema_flags
        has_any = has_vol | has_qual | has_schema

        # Final anomaly flag: has anomaly type AND score >= threshold
        is_anomaly_arr = has_any & (combined_scores >= threshold)

        # Severity: HIGH if >= 0.85, else MEDIUM
        severity_arr = np.where(
            is_anomaly_arr & (combined_scores >= 0.85), 2,  # HIGH
            np.where(is_anomaly_arr, 1, 0)  # MEDIUM or NONE
        )

        # ── Build results list ───────────────────────────────────────
        results: list[DetectionResult] = []
        for i in range(n):
            anomaly_types: list[str] = []
            if has_vol[i]:
                anomaly_types.append("volume")
            if has_qual[i]:
                anomaly_types.append("quality")
            if has_schema[i]:
                anomaly_types.append("schema_change")

            sev = ""
            if severity_arr[i] == 2:
                sev = "HIGH"
            elif severity_arr[i] == 1:
                sev = "MEDIUM"

            results.append(
                DetectionResult(
                    event_index=i,
                    timestamp=ts_pydatetime[i],
                    quality_score=float(scores[i]),
                    record_count=int(counts[i]),
                    schema_changed=bool(schema_flags[i]),
                    is_anomaly=bool(is_anomaly_arr[i]),
                    anomaly_types=anomaly_types,
                    anomaly_score=float(combined_scores[i]),
                    severity=sev,
                )
            )

        return results
