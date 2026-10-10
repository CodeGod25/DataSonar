"""Local filesystem storage for model artifacts — replaces MinIO."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import joblib

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LocalModelStorage:
    """Persist and load model artifacts on the local filesystem.

    Directory layout::

        models/
        ├── trained/
        │   └── {source_id}/
        │       ├── isolation_forest.joblib
        │       └── history.joblib
        └── baselines/
            └── {source_id}.json
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_dir = Path(self._settings.MODEL_DIR)
        self._baseline_dir = Path(self._settings.BASELINE_DIR)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._baseline_dir.mkdir(parents=True, exist_ok=True)

    # ── Model persistence ────────────────────────────────────────────

    def save_model(self, source_id: str, obj: Any, filename: str) -> Path:
        """Serialize an object to a joblib file under the source directory."""
        source_dir = self._model_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / filename
        joblib.dump(obj, path)
        logger.debug("Saved model artifact: %s", path)
        return path

    def load_model(self, source_id: str, filename: str) -> Any | None:
        """Load a joblib-serialized object. Returns None if missing."""
        path = self._model_dir / source_id / filename
        if not path.exists():
            return None
        return joblib.load(path)

    # ── Baseline JSON persistence ────────────────────────────────────

    def save_baseline(self, source_id: str, data: dict[str, Any]) -> Path:
        """Write baseline statistics as a JSON file."""
        path = self._baseline_dir / f"{source_id}.json"
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        logger.debug("Saved baseline: %s", path)
        return path

    def load_baseline(self, source_id: str) -> dict[str, Any] | None:
        """Load baseline JSON. Returns None if missing."""
        path = self._baseline_dir / f"{source_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Discovery ────────────────────────────────────────────────────

    def list_persisted_sources(self) -> list[str]:
        """Return source IDs that have persisted model directories."""
        if not self._model_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._model_dir.iterdir()
            if d.is_dir() and (d / "isolation_forest.joblib").exists()
        )
