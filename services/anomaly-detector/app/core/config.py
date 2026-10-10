"""Application configuration for terminal-only anomaly detector."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Anomaly detector configuration — terminal mode."""

    # ── General ──────────────────────────────────────────────────────
    SERVICE_NAME: str = "datasonar-anomaly-detector"
    LOG_LEVEL: str = "INFO"

    # ── Model storage ────────────────────────────────────────────────
    MODEL_DIR: str = ""  # resolved in __post_init__
    BASELINE_DIR: str = ""  # resolved in __post_init__

    # ── ML thresholds ────────────────────────────────────────────────
    ANOMALY_ALERT_THRESHOLD: float = 0.6
    DRIFT_MEAN_DELTA_THRESHOLD: float = 0.1

    # ── Training parameters ──────────────────────────────────────────
    PRETRAIN_PROFILES: list[str] = field(
        default_factory=lambda: ["stable", "volatile", "degrading"]
    )
    MIN_TRAINING_ROWS: int = 50
    ISO_FOREST_ESTIMATORS: int = 100
    ISO_FOREST_CONTAMINATION: float = 0.08
    ISO_FOREST_RANDOM_STATE: int = 42

    # ── Time-series EWMA ─────────────────────────────────────────────
    EWMA_SPAN: int = 24
    EWMA_Z_THRESHOLD: float = 3.0

    # ── History limits ───────────────────────────────────────────────
    QUALITY_HISTORY_LIMIT: int = 1000
    VOLUME_HISTORY_LIMIT: int = 500

    # ── Batch processing ─────────────────────────────────────────────
    BATCH_CHUNK_SIZE: int = 50_000

    def __post_init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        if not self.MODEL_DIR:
            self.MODEL_DIR = str(project_root / "models" / "trained")
        if not self.BASELINE_DIR:
            self.BASELINE_DIR = str(project_root / "models" / "baselines")
        # Ensure directories exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.BASELINE_DIR, exist_ok=True)


# Module-level singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
