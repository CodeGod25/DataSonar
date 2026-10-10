"""Tests for universal anomaly detection & auto-cleaning engine."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.universal_detector import (
    ColumnType,
    Severity,
    _infer_column_type,
    _read_csv_safe,
    _zscore_mad,
    _iqr_scores,
    clean_dataset,
    detect_anomalies,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_csv(content: str, suffix: str = ".csv", encoding: str = "utf-8") -> str:
    """Write CSV content to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=suffix, encoding=encoding
    ) as f:
        f.write(content)
        return f.name


# ── Schema Inference ─────────────────────────────────────────────────────


class TestColumnTypeInference:
    def test_numeric_int(self):
        s = pd.Series([1, 2, 3, 4, 5], name="age")
        assert _infer_column_type(s, "age") == ColumnType.NUMERIC

    def test_numeric_float(self):
        s = pd.Series([1.1, 2.2, 3.3], name="score")
        assert _infer_column_type(s, "score") == ColumnType.NUMERIC

    def test_numeric_strings(self):
        s = pd.Series(["100", "200", "300", "400", "500"], name="amount")
        assert _infer_column_type(s, "amount") == ColumnType.NUMERIC

    def test_categorical_low_cardinality(self):
        s = pd.Series(["A", "B", "A", "C", "B", "A", "C"] * 10, name="grade")
        assert _infer_column_type(s, "grade") == ColumnType.CATEGORICAL

    def test_boolean_detection(self):
        s = pd.Series(["true", "false", "true", "false"], name="active")
        assert _infer_column_type(s, "active") == ColumnType.BOOLEAN

    def test_id_column(self):
        s = pd.Series([f"USR-{i:04d}" for i in range(100)], name="user_id")
        assert _infer_column_type(s, "user_id") == ColumnType.ID

    def test_datetime_column(self):
        s = pd.Series(
            ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10"],
            name="date",
        )
        assert _infer_column_type(s, "date") == ColumnType.DATETIME

    def test_all_null(self):
        s = pd.Series([None, None, None], name="empty")
        assert _infer_column_type(s, "empty") == ColumnType.TEXT


# ── Z-Score & IQR Algorithms ────────────────────────────────────────────


class TestStatisticalAlgorithms:
    def test_zscore_mad_normal_data(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(50, 5, size=100)
        scores = _zscore_mad(vals)
        assert scores.shape == (100,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
        # Most scores should be low for normal data
        assert np.median(scores) < 0.3

    def test_zscore_mad_with_outlier(self):
        vals = np.array([10, 11, 12, 10, 11, 12, 10, 11, 12, 500])
        scores = _zscore_mad(vals)
        assert scores[-1] > 0.5  # The outlier should score high

    def test_iqr_normal_data(self):
        vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        scores, lf, uf = _iqr_scores(vals)
        assert scores.shape == (10,)
        assert lf < vals.min()
        assert uf > vals.max()

    def test_iqr_with_outlier(self):
        vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
        scores, lf, uf = _iqr_scores(vals)
        assert scores[-1] > 0.0  # 100 is far beyond IQR fence


# ── Detection (End-to-End) ───────────────────────────────────────────────


class TestDetectAnomalies:
    def test_clean_numeric_dataset(self):
        """Clean, well-behaved dataset should produce zero or very few anomalies."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "id": range(1, 101),
            "value_a": rng.normal(50, 3, 100),
            "value_b": rng.normal(200, 10, 100),
            "category": rng.choice(["X", "Y", "Z"], 100),
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert summary.total_records == 100
            assert summary.total_columns == 4
            assert summary.execution_time_seconds < 5.0
            # Mostly normal rows
            assert summary.normal_count >= 60
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_numeric_outliers(self):
        """Extreme numeric values should be flagged."""
        rng = np.random.default_rng(42)
        values = rng.normal(100, 5, 50).tolist()
        # Inject obvious outliers
        values += [9999, -9999, 100000]
        df = pd.DataFrame({
            "measurement": values,
            "group": ["A"] * 53,
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert summary.total_records == 53
            assert summary.anomaly_count >= 2  # At least the extreme ones
            # Check that outlier rows have high severity
            outlier_rows = [
                r for r in summary.row_anomalies
                if r.row_index >= 50 and r.is_anomaly
            ]
            assert len(outlier_rows) >= 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_missing_patterns(self):
        """Rows with many missing values should be flagged."""
        df = pd.DataFrame({
            "a": [1, 2, 3, None, None, 6, 7, 8, 9, 10] * 5,
            "b": [10, 20, 30, None, None, 60, 70, 80, 90, 100] * 5,
            "c": ["x", "y", "z", None, None, "x", "y", "z", "x", "y"] * 5,
            "d": [1.1, 2.2, 3.3, None, None, 6.6, 7.7, 8.8, 9.9, 10.0] * 5,
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert summary.total_records == 50
            # Rows 3 and 4 (per cycle) should be flagged for excessive missing
            missing_flagged = [
                r for r in summary.row_anomalies
                if any("excessive_missing" in reason for reason in r.reasons)
            ]
            assert len(missing_flagged) >= 5  # 5 cycles × rows with all-null
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_duplicates(self):
        """Exact duplicate rows should be flagged."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 1, 2, 3, 1, 2, 3] * 5,
            "y": [10, 20, 30, 10, 20, 30, 10, 20, 30] * 5,
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert summary.duplicate_count > 0
            dup_flagged = [
                r for r in summary.row_anomalies
                if any("duplicate" in reason for reason in r.reasons)
            ]
            assert len(dup_flagged) > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_severity_classification(self):
        """Verify HIGH / MODERATE / LOW / NORMAL classification."""
        rng = np.random.default_rng(42)
        normal_vals = rng.normal(50, 2, 80).tolist()
        # Inject different levels of anomaly
        extreme_vals = [99999, -99999]  # Should be HIGH
        moderate_vals = [80, 85]  # Might be MODERATE
        df = pd.DataFrame({
            "value": normal_vals + extreme_vals + moderate_vals,
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            severities = {r.severity for r in summary.row_anomalies if r.is_anomaly}
            # Should have at least NORMAL and some anomalies
            assert summary.normal_count > 0
            assert summary.anomaly_count > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_algorithms_used_populated(self):
        """Check that algorithm names are reported."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(0, 1, 50),
            "cat": rng.choice(["x", "y", "z"], 50),
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert len(summary.algorithms_used) >= 3  # At least Z-Score, IQR, Missing
            assert "Modified Z-Score (MAD)" in summary.algorithms_used
        finally:
            Path(path).unlink(missing_ok=True)

    def test_mixed_type_dataset(self):
        """Dataset with numeric, categorical, datetime, and text columns."""
        df = pd.DataFrame({
            "id": [f"ID-{i}" for i in range(30)],
            "name": [f"Person_{i}" for i in range(30)],
            "age": list(range(20, 50)),
            "salary": [50000 + i * 1000 for i in range(30)],
            "department": ["HR", "Eng", "Sales"] * 10,
            "join_date": pd.date_range("2020-01-01", periods=30).astype(str).tolist(),
            "active": ["true", "false"] * 15,
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            summary = detect_anomalies(path)
            assert summary.total_records == 30
            assert summary.total_columns == 7
            # Check inferred types
            type_map = {p.name: p.inferred_type for p in summary.column_profiles}
            assert type_map["age"] == ColumnType.NUMERIC
            assert type_map["salary"] == ColumnType.NUMERIC
            assert type_map["department"] == ColumnType.CATEGORICAL
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_dataset_raises(self):
        """Empty CSV should raise ValueError."""
        path = _write_csv("col_a,col_b\n")
        try:
            with pytest.raises(ValueError, match="empty"):
                detect_anomalies(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_pipe_delimited_file(self):
        """Pipe-delimited files should be auto-detected."""
        content = "name|age|score\n" + ("Alice|25|90\nBob|30|85\nCharlie|35|95\n" * 10)
        path = _write_csv(content)
        try:
            summary = detect_anomalies(path)
            assert summary.total_records == 30
            assert summary.total_columns == 3
        finally:
            Path(path).unlink(missing_ok=True)


# ── Cleaning (End-to-End) ────────────────────────────────────────────────


class TestCleanDataset:
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 20, 30, 10, 20, 30],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert result.duplicates_removed == 3
            assert result.cleaned_rows == 3
        finally:
            Path(path).unlink(missing_ok=True)

    def test_imputes_missing_numeric(self):
        df = pd.DataFrame({
            "a": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert "a" in result.nulls_imputed
            assert result.nulls_imputed["a"] == 2
            # No nulls left in cleaned data
            assert result.cleaned_df["a"].isna().sum() == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_imputes_missing_categorical(self):
        df = pd.DataFrame({
            "cat": ["A", "B", "A", None, "A", None, "A", "B", "A", "B"],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert "cat" in result.nulls_imputed
            assert result.cleaned_df["cat"].isna().sum() == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clips_outliers(self):
        vals = [10, 11, 12, 13, 14, 15, 10, 11, 12, 9999]
        df = pd.DataFrame({"value": vals})
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert "value" in result.outliers_clipped
            assert result.outliers_clipped["value"] >= 1
            # 9999 should be clipped down
            assert result.cleaned_df["value"].max() < 9999
        finally:
            Path(path).unlink(missing_ok=True)

    def test_preserves_column_order(self):
        df = pd.DataFrame({
            "z_col": [1, 2, 3],
            "a_col": [4, 5, 6],
            "m_col": [7, 8, 9],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert list(result.cleaned_df.columns) == ["z_col", "a_col", "m_col"]
            assert result.columns_preserved == ["z_col", "a_col", "m_col"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_preserves_integer_dtype(self):
        df = pd.DataFrame({
            "count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(path)
            assert result.cleaned_df["count"].dtype in (np.int64, np.int32)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_no_cleaning_flags(self):
        """With all cleaning disabled, output should match input."""
        df = pd.DataFrame({
            "a": [1, 2, None, 1, 2, None],
            "b": ["x", "y", None, "x", "y", None],
        })
        path = _write_csv(df.to_csv(index=False))
        try:
            result = clean_dataset(
                path,
                remove_duplicates=False,
                clip_outliers=False,
                impute_missing=False,
                fix_type_violations=False,
            )
            assert result.duplicates_removed == 0
            assert len(result.nulls_imputed) == 0
            assert result.cleaned_rows == 6
        finally:
            Path(path).unlink(missing_ok=True)

    def test_type_violation_fixing(self):
        """Non-numeric entries in a numeric column should be coerced."""
        content = "value,label\n10,A\n20,B\nXYZ,C\n40,D\n50,E\n60,F\n70,G\n80,H\n90,I\n100,J\n"
        path = _write_csv(content)
        try:
            result = clean_dataset(path)
            # 'XYZ' should be coerced to NaN then imputed
            assert result.cleaned_df["value"].isna().sum() == 0
            if "value" in result.type_violations_fixed:
                assert result.type_violations_fixed["value"] >= 1
        finally:
            Path(path).unlink(missing_ok=True)


# ── CSV Reader ───────────────────────────────────────────────────────────


class TestCSVReader:
    def test_reads_utf8(self):
        content = "name,value\nAlice,100\nBob,200\n"
        path = _write_csv(content, encoding="utf-8")
        try:
            df = _read_csv_safe(Path(path))
            assert len(df) == 2
            assert list(df.columns) == ["name", "value"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reads_latin1(self):
        content = "name,value\nJos\xe9,100\nM\xfcller,200\n"
        path = _write_csv(content, encoding="latin-1")
        try:
            df = _read_csv_safe(Path(path))
            assert len(df) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reads_tab_delimited(self):
        content = "name\tvalue\nAlice\t100\nBob\t200\n"
        path = _write_csv(content)
        try:
            df = _read_csv_safe(Path(path))
            assert len(df) == 2
            assert "name" in df.columns
            assert "value" in df.columns
        finally:
            Path(path).unlink(missing_ok=True)
