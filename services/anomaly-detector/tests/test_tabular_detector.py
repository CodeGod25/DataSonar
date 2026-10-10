"""Tests for tabular real-world dataset anomaly detection."""

import tempfile
from pathlib import Path
import pytest
from app.services.tabular_detector import detect_tabular_dataset


def test_tabular_detector_on_clean_data():
    content = (
        "Student_ID,First_Name,Last_Name,Age,Gender,Course,Enrollment_Date,Total_Payments\n"
        "101,John,Smith,22,M,Data Science,2022-05-15,$1200\n"
        "102,Emily,Johnson,24,F,Machine Learning,2022-03-18,$1400\n"
        "103,Michael,Williams,21,M,Data Science,2022-06-25,$900\n"
        "104,Sarah,Brown,23,F,Data Science,2022-01-10,$1100\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        f_path = f.name

    try:
        summary = detect_tabular_dataset(f_path)
        assert summary.total_records == 4
        assert summary.normal_count == 4
        assert summary.anomaly_count == 0
        assert summary.mean_quality_score == 1.0
        assert summary.execution_time_seconds < 1.0
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_tabular_detector_detects_messy_anomalies():
    content = (
        "Student_ID,First_Name,Last_Name,Age,Gender,Course,Enrollment_Date,Total_Payments\n"
        "101,John,Smith,22,M,Data Science,2022-05-15,$1200\n"
        "Kylian,Kylian,Eden,78*,M,Machine Learning,2024-01-08,£20,000\n"
        "Anita,Anita,West,,F 24,4,2020-01-05,£30,000\n"
        ",,,,,,2021-09-11,\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="latin-1") as f:
        f.write(content)
        f_path = f.name

    try:
        summary = detect_tabular_dataset(f_path)
        assert summary.total_records == 4
        assert summary.normal_count == 1  # John Smith
        assert summary.anomaly_count == 3
        assert summary.high_count >= 2
    finally:
        Path(f_path).unlink(missing_ok=True)
