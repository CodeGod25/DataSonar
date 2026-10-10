"""Universal anomaly detection & auto-cleaning engine.

Works on ANY CSV dataset regardless of schema. Auto-detects column types,
runs a multi-algorithm ensemble for anomaly detection, classifies severity
as HIGH / MODERATE / LOW, and auto-cleans while preserving column characteristics.

Algorithms used (ensemble):
    1. Isolation Forest        — multivariate statistical outliers (numeric)
    2. Local Outlier Factor    — density-based outliers (numeric)
    3. Modified Z-Score (MAD)  — robust univariate extremes (per numeric col)
    4. IQR Fences              — robust univariate outliers (per numeric col)
    5. Categorical Rarity      — rare/unseen categories (per categorical col)
    6. Missing Pattern Analysis— systematic missingness (all columns)
    7. Duplicate Detection     — exact duplicate rows (full row)
    8. Type Violation Detection— wrong-dtype entries (all columns)
    9. Datetime Range Checks   — future/ancient dates (per datetime col)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Enums & Constants ────────────────────────────────────────────────────


class ColumnType(str, Enum):
    """Inferred semantic type of a column."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    BOOLEAN = "boolean"


class Severity(str, Enum):
    """Anomaly severity level."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NORMAL = "NORMAL"


# Severity score thresholds
_THRESH_HIGH = 0.70
_THRESH_MODERATE = 0.40
_THRESH_LOW = 0.15

# Minimum samples for ML models (IF / LOF)
_MIN_SAMPLES_ML = 20

# IQR multipliers
_IQR_MODERATE = 1.5
_IQR_EXTREME = 3.0

# Categorical rarity: categories with frequency below this are "rare"
_RARE_CATEGORY_FRAC = 0.01


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    inferred_type: ColumnType
    original_dtype: str
    null_count: int
    null_rate: float
    unique_count: int
    unique_rate: float
    # Numeric-only stats
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    # Categorical-only stats
    mode_value: str | None = None
    top_categories: dict[str, int] | None = None
    # Type-violation count
    type_violation_count: int = 0
    type_violation_rate: float = 0.0


@dataclass
class RowAnomaly:
    """Anomaly report for a single row."""

    row_index: int
    anomaly_score: float
    severity: Severity
    is_anomaly: bool
    reasons: list[str] = field(default_factory=list)
    column_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectionSummary:
    """Full output of an anomaly-detection run."""

    source_id: str
    total_records: int
    total_columns: int
    column_profiles: list[ColumnProfile]
    # Counts
    normal_count: int
    anomaly_count: int
    high_count: int
    moderate_count: int
    low_count: int
    duplicate_count: int
    # Per-row results
    row_anomalies: list[RowAnomaly]
    # Breakdown by anomaly type
    anomaly_type_counts: dict[str, int]
    # Performance
    execution_time_seconds: float
    throughput_rows_per_sec: float
    algorithms_used: list[str]


@dataclass
class CleaningSummary:
    """Report on auto-cleaning actions, plus the cleaned DataFrame."""

    source_id: str
    original_rows: int
    cleaned_rows: int
    duplicates_removed: int
    nulls_imputed: dict[str, int]
    outliers_clipped: dict[str, int]
    type_violations_fixed: dict[str, int]
    columns_preserved: list[str]
    cleaned_df: pd.DataFrame


# ── CSV Reader ───────────────────────────────────────────────────────────


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV with automatic encoding and delimiter detection."""
    with open(path, "rb") as fh:
        raw = fh.read(16_384)

    enc = "utf-8"
    for candidate in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw.decode(candidate)
            enc = candidate
            break
        except UnicodeDecodeError:
            continue

    sample = raw.decode(enc, errors="replace")
    first_line = sample.split("\n")[0] if "\n" in sample else sample

    # Pick delimiter by highest count in the header line
    counts = {d: first_line.count(d) for d in ("|", "\t", ";", ",")}
    delimiter = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[delimiter] == 0:
        delimiter = ","

    return pd.read_csv(path, encoding=enc, delimiter=delimiter, on_bad_lines="warn")


# ── Schema Inference ─────────────────────────────────────────────────────


def _is_datetime_column(series: pd.Series) -> bool:
    """Heuristic: can ≥70 % of non-null values be parsed as datetimes?"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return False
    sample_str = sample.astype(str).str.strip()
    # Datetimes typically contain separators (-, /, :, T, or spaces)
    has_date_char = sample_str.str.contains(r"[-/:T\s]", regex=True).mean() >= 0.70
    if not has_date_char:
        return False
    try:
        parsed = pd.to_datetime(sample_str, errors="coerce")
        return parsed.notna().sum() / len(sample_str) >= 0.70
    except Exception:
        return False


def _infer_column_type(series: pd.Series, col_name: str) -> ColumnType:
    """Infer the semantic type of a column from its data."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.TEXT

    n_total = len(series)
    n_unique = non_null.nunique()
    unique_ratio = n_unique / len(non_null)

    # Boolean dtype or values
    if pd.api.types.is_numeric_dtype(series):
        if set(non_null.unique()) <= {0, 1, 0.0, 1.0, True, False} and n_unique <= 2:
            return ColumnType.BOOLEAN

    # Boolean strings
    lower_vals = set(non_null.astype(str).str.strip().str.lower().unique())
    if lower_vals <= {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"} and n_unique <= 2:
        return ColumnType.BOOLEAN

    # ID-like columns: check name or string hash/UUID patterns
    col_name_lower = col_name.strip().lower()
    id_keywords = {"id", "index", "key", "code", "number", "no", "num", "#", "uuid", "guid", "token", "hash", "ref"}
    is_id_name = (
        col_name_lower in id_keywords
        or col_name_lower.endswith(("_id", ".id", "-id", "id"))
        or col_name_lower.startswith(("id_", "id-", "id."))
    )
    if is_id_name and unique_ratio > 0.70:
        return ColumnType.ID
    if (
        unique_ratio > 0.95
        and n_total > 20
        and not pd.api.types.is_numeric_dtype(series)
        and non_null.astype(str).str.len().mean() > 4
    ):
        return ColumnType.ID

    # Datetime (strings with date separators)
    if _is_datetime_column(non_null):
        return ColumnType.DATETIME

    # Already numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC

    # Coercible to numeric (≥80 % success)
    numeric_coerced = pd.to_numeric(non_null, errors="coerce")
    if numeric_coerced.notna().sum() / len(non_null) >= 0.80:
        return ColumnType.NUMERIC

    # Low-cardinality ⇒ categorical
    if n_unique <= 50 or unique_ratio < 0.50:
        return ColumnType.CATEGORICAL

    return ColumnType.TEXT


def _profile_column(series: pd.Series, col_type: ColumnType) -> ColumnProfile:
    """Build a statistical profile for one column."""
    n = len(series)
    null_count = int(series.isna().sum())
    non_null = series.dropna()

    prof = ColumnProfile(
        name=str(series.name),
        inferred_type=col_type,
        original_dtype=str(series.dtype),
        null_count=null_count,
        null_rate=null_count / n if n else 0.0,
        unique_count=int(non_null.nunique()),
        unique_rate=non_null.nunique() / len(non_null) if len(non_null) else 0.0,
    )

    if col_type == ColumnType.NUMERIC and len(non_null) > 0:
        nums = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(nums) > 0:
            prof.mean = float(nums.mean())
            prof.std = float(nums.std()) if len(nums) > 1 else 0.0
            prof.median = float(nums.median())
            prof.q1 = float(nums.quantile(0.25))
            prof.q3 = float(nums.quantile(0.75))
            prof.iqr = prof.q3 - prof.q1
            prof.min_val = float(nums.min())
            prof.max_val = float(nums.max())
        if not pd.api.types.is_numeric_dtype(series):
            violations = pd.to_numeric(non_null, errors="coerce").isna().sum()
            prof.type_violation_count = int(violations)
            prof.type_violation_rate = violations / len(non_null) if len(non_null) else 0.0

    if col_type == ColumnType.CATEGORICAL and len(non_null) > 0:
        vc = non_null.value_counts()
        prof.mode_value = str(vc.index[0]) if len(vc) else None
        prof.top_categories = {str(k): int(v) for k, v in vc.head(10).items()}

    return prof


# ── Individual Detection Algorithms ──────────────────────────────────────


def _zscore_mad(values: np.ndarray) -> np.ndarray:
    """Modified Z-Score via Median Absolute Deviation (more robust than μ/σ).

    Returns per-element scores in [0, 1]. Inliers (z < 2.5) score 0.0,
    mild outliers score 0.20-0.50, and severe outliers score up to 1.0.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        z = 0.6745 * np.abs(values - median) / mad
    else:
        std = np.std(values)
        if std == 0:
            return np.zeros(len(values))
        z = np.abs(values - np.mean(values)) / std

    scores = np.zeros(len(values))
    mask = z >= 2.5
    if mask.any():
        scores[mask] = np.clip(0.20 + ((z[mask] - 2.5) / 3.5) * 0.80, 0.20, 1.0)
    return scores


def _iqr_scores(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """IQR-fence scores.  Returns (scores, lower_fence, upper_fence)."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(values)), q1, q3

    lf = q1 - _IQR_MODERATE * iqr
    uf = q3 + _IQR_MODERATE * iqr
    le = q1 - _IQR_EXTREME * iqr
    ue = q3 + _IQR_EXTREME * iqr

    scores = np.zeros(len(values))
    below = values < lf
    above = values > uf
    if below.any():
        scores[below] = np.clip(0.30 + ((lf - values[below]) / (iqr * 2)) * 0.70, 0.30, 1.0)
    if above.any():
        scores[above] = np.clip(0.30 + ((values[above] - uf) / (iqr * 2)) * 0.70, 0.30, 1.0)
    extreme = (values < le) | (values > ue)
    if extreme.any():
        scores[extreme] = np.clip(np.maximum(scores[extreme], 0.75), 0.75, 1.0)
    return scores, lf, uf


def _isolation_forest(X: np.ndarray, contamination: float) -> tuple[np.ndarray, np.ndarray]:
    """Isolation Forest on a scaled numeric feature matrix.

    Returns (predictions ∈ {-1,1}, normalised scores ∈ [0,1]).
    """
    n = X.shape[0]
    if n < _MIN_SAMPLES_ML:
        return np.ones(n, dtype=int), np.zeros(n)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=min(contamination, 0.5 - 1e-6),
        n_estimators=100,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(Xs)
    dfunc = iso.decision_function(Xs)
    scores = np.zeros(n)
    outlier_mask = preds == -1
    if outlier_mask.any():
        neg_vals = -dfunc[outlier_mask]
        max_neg = float(neg_vals.max()) if neg_vals.size > 0 else 1.0
        scores[outlier_mask] = np.clip(0.35 + (neg_vals / (max_neg or 1.0)) * 0.65, 0.35, 1.0)
    return preds, scores


def _local_outlier_factor(X: np.ndarray, contamination: float) -> tuple[np.ndarray, np.ndarray]:
    """LOF on a scaled numeric feature matrix.

    Returns (predictions ∈ {-1,1}, normalised scores ∈ [0,1]).
    """
    n = X.shape[0]
    if n < _MIN_SAMPLES_ML:
        return np.ones(n, dtype=int), np.zeros(n)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lof = LocalOutlierFactor(
        n_neighbors=min(20, n - 1),
        contamination=min(contamination, 0.5 - 1e-6),
        novelty=False,
        n_jobs=-1,
    )
    preds = lof.fit_predict(Xs)
    scores = np.zeros(n)
    outlier_mask = preds == -1
    if outlier_mask.any():
        excess = lof.offset_ - lof.negative_outlier_factor_[outlier_mask]
        max_excess = float(excess.max()) if excess.size > 0 else 1.0
        scores[outlier_mask] = np.clip(0.35 + (excess / (max_excess or 1.0)) * 0.65, 0.35, 1.0)
    return preds, scores


def _bollinger_bands(values: np.ndarray, window: int = 20, num_std: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1: Moving Average with Bollinger Bands.

    Computes moving average and standard deviation (using expanding window for initial points).
    Returns (scores ∈ [0, 1], upper_band, lower_band).
    """
    n = len(values)
    if n < 4:
        return np.zeros(n), values.copy(), values.copy()

    series = pd.Series(values)
    min_p = max(3, min(window // 4, n))
    rolling_mean = series.rolling(window=window, min_periods=min_p).mean()
    rolling_std = series.rolling(window=window, min_periods=min_p).std()

    # Fill initial points with expanding stats
    expanding_mean = series.expanding(min_periods=1).mean()
    expanding_std = series.expanding(min_periods=1).std().fillna(1.0)

    rmean = rolling_mean.fillna(expanding_mean).values
    rstd = rolling_std.fillna(expanding_std).values
    rstd = np.where(rstd <= 1e-6, 1.0, rstd)

    upper_band = rmean + num_std * rstd
    lower_band = rmean - num_std * rstd

    scores = np.zeros(n)
    above = values > upper_band
    below = values < lower_band
    outside = above | below

    if outside.any():
        dev = np.where(above, (values - upper_band) / rstd, (lower_band - values) / rstd)
        scores[outside] = np.clip(0.30 + (dev[outside] / 3.0) * 0.70, 0.30, 1.0)
    return scores, upper_band, lower_band


def _dbscan_clustering(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2: DBSCAN density-based clustering for pattern detection.

    Identifies unclustered noise points (label == -1).
    Returns (predictions ∈ {-1, 1}, normalised scores ∈ [0, 1]).
    """
    n = X.shape[0]
    if n < _MIN_SAMPLES_ML:
        return np.ones(n, dtype=int), np.zeros(n)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Adaptive eps heuristic based on feature dimension
    d = Xs.shape[1]
    adaptive_eps = max(0.3, eps * np.sqrt(d))
    min_s = max(3, min(min_samples, n // 20))

    db = DBSCAN(eps=adaptive_eps, min_samples=min_s, n_jobs=-1)
    labels = db.fit_predict(Xs)

    preds = np.where(labels == -1, -1, 1)
    scores = np.zeros(n)

    noise_mask = labels == -1
    if noise_mask.any() and db.core_sample_indices_.size > 0:
        core_samples = Xs[db.core_sample_indices_]
        if len(core_samples) > 1000:
            sub_indices = np.random.RandomState(42).choice(len(core_samples), 1000, replace=False)
            core_samples = core_samples[sub_indices]

        noise_pts = Xs[noise_mask]
        dists = pairwise_distances(noise_pts, core_samples, metric="euclidean")
        min_dists = dists.min(axis=1)
        max_dist = float(min_dists.max()) if min_dists.size > 0 else 1.0
        scores[noise_mask] = np.clip(0.40 + (min_dists / (max_dist or 1.0)) * 0.60, 0.40, 1.0)
    elif noise_mask.any():
        scores[noise_mask] = 0.50

    return preds, scores


def _prophet_or_trend_forecaster(
    df: pd.DataFrame, dt_cols: list[str], numeric_cols: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stage 2: Prophet or robust time-series forecasting for trend anomalies.

    Detects trend anomalies where observed metrics diverge from forecasted trajectory.
    Returns (scores ∈ [0, 1], reasons, algorithms_used).
    """
    n = len(df)
    if not dt_cols or not numeric_cols or n < 10:
        return np.zeros(n), ["" for _ in range(n)], []

    dt_col = dt_cols[0]
    num_col = numeric_cols[0]

    scores = np.zeros(n)
    reasons = ["" for _ in range(n)]

    has_prophet = False
    try:
        from prophet import Prophet  # type: ignore[import-untyped]
        has_prophet = True
    except ImportError:
        has_prophet = False

    dates = pd.to_datetime(df[dt_col], errors="coerce")
    vals = pd.to_numeric(df[num_col], errors="coerce")
    valid_mask = dates.notna() & vals.notna()

    if valid_mask.sum() < 10:
        return scores, reasons, []

    if has_prophet:
        try:
            prophet_df = pd.DataFrame({"ds": dates[valid_mask], "y": vals[valid_mask]})
            m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
            m.fit(prophet_df)
            forecast = m.predict(prophet_df)
            yhat_lower = forecast["yhat_lower"].values
            yhat_upper = forecast["yhat_upper"].values
            actuals = vals[valid_mask].values

            anom_sub = (actuals < yhat_lower) | (actuals > yhat_upper)
            valid_indices = np.where(valid_mask)[0]
            for idx, is_anom in zip(valid_indices, anom_sub):
                if is_anom:
                    scores[idx] = 0.65
                    reasons[idx] = f"trend_anomaly(Prophet:{num_col})"
            return scores, reasons, ["Prophet (Time-Series Forecasting)"]
        except Exception as e:
            logger.debug("Prophet fit failed, falling back to robust trend forecaster: %s", e)

    # Robust statistical trend forecaster fallback (EWMA trend + residual interval)
    sort_order = np.argsort(dates[valid_mask].values)
    sorted_vals = vals[valid_mask].values[sort_order]

    s = pd.Series(sorted_vals)
    span = max(5, min(30, len(s) // 5))
    trend = s.ewm(span=span).mean().values
    residual = np.abs(sorted_vals - trend)
    res_std = float(np.std(residual)) or 1.0

    anom_sorted = residual > (3.0 * res_std)
    valid_indices = np.where(valid_mask)[0]
    orig_indices = valid_indices[sort_order]

    for idx, is_anom in zip(orig_indices, anom_sorted):
        if is_anom:
            scores[idx] = 0.60
            reasons[idx] = f"trend_anomaly(TrendForecaster:{num_col})"

    algo_name = "Prophet (Time-Series Forecasting)" if has_prophet else "Time-Series Trend Forecaster (Stage 2)"
    return scores, reasons, [algo_name]


def _categorical_rarity_scores(series: pd.Series) -> np.ndarray:
    """Score each element by how rare its category is (0 = common, ~0.6 = very rare)."""
    n = len(series)
    scores = np.zeros(n)
    non_null = series.notna()
    if non_null.sum() == 0:
        return scores

    freq = series[non_null].value_counts(normalize=True)
    rare = set(freq[freq < _RARE_CATEGORY_FRAC].index)
    if not rare:
        return scores

    for cat in rare:
        mask = series == cat
        f = freq.get(cat, 0.0)
        scores[mask] = min(1.0, (_RARE_CATEGORY_FRAC - f) / _RARE_CATEGORY_FRAC) * 0.6
    return scores


def _missing_pattern_scores(df: pd.DataFrame) -> np.ndarray:
    """Per-row score based on the fraction of missing values in that row."""
    n_cols = df.shape[1]
    if n_cols == 0:
        return np.zeros(len(df))

    rates = df.isna().sum(axis=1).values / n_cols
    avg = rates.mean()
    if avg == 0:
        return np.zeros(len(df))

    scores = np.zeros(len(df))
    above = rates > avg
    max_rate = max(rates.max(), 0.5)
    scores[above] = np.clip(rates[above] / max_rate, 0.0, 1.0)
    scores[rates >= 0.90] = 1.0
    return scores


def _duplicate_flags(df: pd.DataFrame) -> np.ndarray:
    """1.0 for duplicate rows (keeps first occurrence as 0)."""
    return df.duplicated(keep="first").values.astype(float)


def _datetime_range_scores(series: pd.Series) -> np.ndarray:
    """Score datetime values that are in the future or unreasonably old."""
    scores = np.zeros(len(series))
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except Exception:
        return scores

    valid = parsed.notna()
    if valid.sum() == 0:
        return scores

    now = pd.Timestamp.now()
    scores[(parsed > now + pd.Timedelta(days=1)) & valid] = 0.70
    scores[(parsed < pd.Timestamp("1970-01-01")) & valid] = 0.50
    scores[(parsed < pd.Timestamp("1900-01-01")) & valid] = 0.80
    return scores


# ── Severity Mapping ─────────────────────────────────────────────────────


def _severity(score: float) -> Severity:
    if score >= _THRESH_HIGH:
        return Severity.HIGH
    if score >= _THRESH_MODERATE:
        return Severity.MODERATE
    if score >= _THRESH_LOW:
        return Severity.LOW
    return Severity.NORMAL


# ── Main Entry Points ────────────────────────────────────────────────────


def detect_anomalies(
    csv_path: str | Path,
    source_id: str | None = None,
    contamination: float = 0.10,
) -> DetectionSummary:
    """Run the full multi-algorithm anomaly detection pipeline on any CSV.

    Parameters
    ----------
    csv_path
        Path to a CSV / TSV / pipe-delimited file.
    source_id
        Human-readable identifier (defaults to the file stem).
    contamination
        Expected fraction of outliers (used by IF & LOF).

    Returns
    -------
    DetectionSummary
        Per-row anomaly scores, severity, reasons; column profiles;
        aggregate statistics; and performance metrics.
    """
    t0 = time.perf_counter()
    p = Path(csv_path)
    source_id = source_id or p.stem

    df = _read_csv_safe(p)
    n_rows, n_cols = df.shape
    if n_rows == 0:
        raise ValueError(f"Dataset '{p.name}' is empty.")

    algos_used: list[str] = []

    # ── 1. Schema inference & profiling ──────────────────────────────
    col_types: dict[str, ColumnType] = {}
    col_profiles: list[ColumnProfile] = []
    for col in df.columns:
        ct = _infer_column_type(df[col], col)
        col_types[col] = ct
        col_profiles.append(_profile_column(df[col], ct))

    numeric_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
    cat_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]
    dt_cols = [c for c, t in col_types.items() if t == ColumnType.DATETIME]

    # Accumulators — one list/array per scoring "channel"
    channel_scores: list[tuple[np.ndarray, float]] = []  # (scores, weight)
    row_reasons: list[list[str]] = [[] for _ in range(n_rows)]
    row_col_scores: list[dict[str, float]] = [{} for _ in range(n_rows)]

    # ── 2. Numeric — univariate (Z-Score + IQR) ─────────────────────
    if numeric_cols:
        # Build a clean numeric DataFrame (coerce strings → NaN)
        num_df = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            num_df[col] = (
                df[col] if pd.api.types.is_numeric_dtype(df[col])
                else pd.to_numeric(df[col], errors="coerce")
            )

        univariate_max = np.zeros(n_rows)
        for col in numeric_cols:
            valid = num_df[col].dropna()
            if len(valid) < 3:
                continue
            vals = num_df[col].fillna(num_df[col].median()).values

            zs = _zscore_mad(vals)
            iq, _, _ = _iqr_scores(vals)
            bb, _, _ = _bollinger_bands(vals)
            col_score = np.maximum.reduce([zs, iq, bb])
            univariate_max = np.maximum(univariate_max, col_score)

            for i in np.where(col_score > 0.10)[0]:
                row_col_scores[i][col] = float(col_score[i])
                if zs[i] >= 0.60 or iq[i] >= 0.60:
                    row_reasons[i].append(f"extreme_value({col}={vals[i]:.4g})")
                elif bb[i] >= 0.40:
                    row_reasons[i].append(f"bollinger_outlier({col}={vals[i]:.4g})")
                elif col_score[i] >= 0.25:
                    row_reasons[i].append(f"outlier({col}={vals[i]:.4g})")

        channel_scores.append((univariate_max, 0.20))
        algos_used += ["Modified Z-Score (MAD)", "IQR Fences", "Moving Average with Bollinger Bands"]

    # ── 3. Numeric — multivariate (Isolation Forest) ─────────────────
    if numeric_cols and n_rows >= _MIN_SAMPLES_ML:
        num_df_filled = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            s = (
                df[col] if pd.api.types.is_numeric_dtype(df[col])
                else pd.to_numeric(df[col], errors="coerce")
            )
            num_df_filled[col] = s.fillna(s.median())
        X = num_df_filled.values

        if_preds, if_scores = _isolation_forest(X, contamination)
        channel_scores.append((if_scores, 0.25))
        algos_used.append("Isolation Forest")

        for i in np.where(if_preds == -1)[0]:
            row_reasons[i].append("multivariate_outlier(IsolationForest)")

    # ── 4. Numeric — multivariate (LOF) ──────────────────────────────
    if numeric_cols and n_rows >= _MIN_SAMPLES_ML:
        # Re-use X from above
        lof_preds, lof_scores = _local_outlier_factor(X, contamination)
        channel_scores.append((lof_scores, 0.20))
        algos_used.append("Local Outlier Factor (LOF)")

        for i in np.where(lof_preds == -1)[0]:
            row_reasons[i].append("density_outlier(LOF)")

    # ── 4b. Numeric — multivariate (DBSCAN Clustering) ───────────────
    if numeric_cols and n_rows >= _MIN_SAMPLES_ML:
        db_preds, db_scores = _dbscan_clustering(X)
        channel_scores.append((db_scores, 0.15))
        algos_used.append("DBSCAN (Density-Based Clustering)")

        for i in np.where(db_preds == -1)[0]:
            row_reasons[i].append("density_cluster_noise(DBSCAN)")

    # ── 4c. Time-Series Trend Forecaster (Prophet / Trend) ───────────
    if dt_cols and numeric_cols and n_rows >= 10:
        ts_scores, ts_reasons, ts_algos = _prophet_or_trend_forecaster(df, dt_cols, numeric_cols)
        if ts_algos:
            channel_scores.append((ts_scores, 0.15))
            algos_used += ts_algos
            for i, r in enumerate(ts_reasons):
                if r:
                    row_reasons[i].append(r)

    # ── 5. Categorical rarity ────────────────────────────────────────
    if cat_cols:
        cat_max = np.zeros(n_rows)
        for col in cat_cols:
            cs = _categorical_rarity_scores(df[col])
            cat_max = np.maximum(cat_max, cs)
            for i in np.where(cs > 0.10)[0]:
                row_col_scores[i][col] = float(cs[i])
                row_reasons[i].append(f"rare_category({col}='{df[col].iloc[i]}')")
        channel_scores.append((cat_max, 0.08))
        algos_used.append("Categorical Rarity")

    # ── 6. Datetime range checks ─────────────────────────────────────
    if dt_cols:
        dt_max = np.zeros(n_rows)
        for col in dt_cols:
            ds = _datetime_range_scores(df[col])
            dt_max = np.maximum(dt_max, ds)
            for i in np.where(ds > 0.10)[0]:
                row_col_scores[i][col] = float(ds[i])
                label = "future_date" if ds[i] >= 0.65 else "historical_date"
                row_reasons[i].append(f"{label}({col})")
        channel_scores.append((dt_max, 0.07))
        algos_used.append("Datetime Range Check")

    # ── 7. Missing-value patterns ────────────────────────────────────
    miss_scores = _missing_pattern_scores(df)
    channel_scores.append((miss_scores, 0.10))
    algos_used.append("Missing Pattern Analysis")
    for i in np.where(miss_scores > 0.30)[0]:
        n_miss = int(df.iloc[i].isna().sum())
        row_reasons[i].append(f"excessive_missing({n_miss}/{n_cols} cols)")

    # ── 8. Duplicate rows ────────────────────────────────────────────
    dup_flags = _duplicate_flags(df)
    dup_count = int(dup_flags.sum())
    channel_scores.append((dup_flags * 0.65, 0.10))
    algos_used.append("Duplicate Detection")
    for i in np.where(dup_flags > 0)[0]:
        row_reasons[i].append("duplicate_row")

    # ── 9. Type-violation detection ──────────────────────────────────
    type_viol_score = np.zeros(n_rows)
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            is_viol = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            for i in np.where(is_viol)[0]:
                row_reasons[i].append(f"type_violation({col}='{df[col].iloc[i]}')")
                type_viol_score[i] = max(type_viol_score[i], 0.70)
    if type_viol_score.any():
        channel_scores.append((type_viol_score, 0.10))
        algos_used.append("Type Violation Detection")

    # ── 10. Stage 3: Weighted Majority Vote Ensemble ─────────────────
    total_weight = sum(w for _, w in channel_scores)
    if total_weight > 0:
        weighted_avg = sum(s * w for s, w in channel_scores) / total_weight
        channel_max = np.maximum.reduce([s for s, _ in channel_scores])
        final = np.maximum(weighted_avg, channel_max * 0.85)
    else:
        final = np.zeros(n_rows)
    final = np.clip(final, 0.0, 1.0)
    algos_used.append("Stage 3: Weighted Majority Vote Ensemble")

    # ── 11. Build per-row results ────────────────────────────────────
    row_anomalies: list[RowAnomaly] = []
    for i in range(n_rows):
        sev = _severity(float(final[i]))
        row_anomalies.append(
            RowAnomaly(
                row_index=i,
                anomaly_score=float(final[i]),
                severity=sev,
                is_anomaly=sev != Severity.NORMAL,
                reasons=row_reasons[i],
                column_scores=row_col_scores[i],
            )
        )

    # ── 12. Aggregate stats ──────────────────────────────────────────
    anom = [r for r in row_anomalies if r.is_anomaly]
    type_counts: Counter[str] = Counter()
    for ra in row_anomalies:
        for reason in ra.reasons:
            type_counts[reason.split("(")[0].strip()] += 1

    elapsed = time.perf_counter() - t0

    return DetectionSummary(
        source_id=source_id,
        total_records=n_rows,
        total_columns=n_cols,
        column_profiles=col_profiles,
        normal_count=n_rows - len(anom),
        anomaly_count=len(anom),
        high_count=sum(1 for r in anom if r.severity == Severity.HIGH),
        moderate_count=sum(1 for r in anom if r.severity == Severity.MODERATE),
        low_count=sum(1 for r in anom if r.severity == Severity.LOW),
        duplicate_count=dup_count,
        row_anomalies=row_anomalies,
        anomaly_type_counts=dict(type_counts.most_common()),
        execution_time_seconds=elapsed,
        throughput_rows_per_sec=n_rows / (elapsed or 1e-6),
        algorithms_used=list(dict.fromkeys(algos_used)),
    )


# ── Auto-Cleaning ────────────────────────────────────────────────────────


def clean_dataset(
    csv_path: str | Path,
    source_id: str | None = None,
    *,
    remove_duplicates: bool = True,
    clip_outliers: bool = True,
    impute_missing: bool = True,
    fix_type_violations: bool = True,
) -> CleaningSummary:
    """Auto-clean any CSV while preserving column names, order, and types.

    Actions (all optional via flags):
        • Numeric outliers  → clip to [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
        • Missing numerics  → median imputation
        • Missing categoricals → mode imputation
        • Duplicate rows    → keep first, remove rest
        • Type violations   → coerce to inferred type (invalid → NaN → impute)

    Parameters
    ----------
    csv_path
        Path to the input CSV.
    source_id
        Identifier (defaults to filename stem).
    remove_duplicates, clip_outliers, impute_missing, fix_type_violations
        Toggle individual cleaning steps.

    Returns
    -------
    CleaningSummary
        Includes the cleaned DataFrame and per-column action counts.
    """
    p = Path(csv_path)
    source_id = source_id or p.stem

    df = _read_csv_safe(p)
    original_rows = len(df)
    original_columns = list(df.columns)
    original_dtypes = df.dtypes.copy()

    col_types: dict[str, ColumnType] = {
        col: _infer_column_type(df[col], col) for col in df.columns
    }

    nulls_imputed: dict[str, int] = {}
    outliers_clipped: dict[str, int] = {}
    type_violations_fixed: dict[str, int] = {}

    cleaned = df.copy()

    # 1. Duplicates
    dups_removed = 0
    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
        dups_removed = before - len(cleaned)

    # 2. Per-column cleaning
    for col in cleaned.columns:
        ct = col_types[col]

        if ct == ColumnType.NUMERIC:
            # Coerce non-numeric strings → NaN
            if fix_type_violations and not pd.api.types.is_numeric_dtype(cleaned[col]):
                before_nn = int(cleaned[col].notna().sum())
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                after_nn = int(cleaned[col].notna().sum())
                viols = before_nn - after_nn
                if viols > 0:
                    type_violations_fixed[col] = viols

            # Median imputation
            if impute_missing:
                n_miss = int(cleaned[col].isna().sum())
                if n_miss > 0:
                    med = cleaned[col].median()
                    if pd.notna(med):
                        cleaned[col] = cleaned[col].fillna(med)
                        nulls_imputed[col] = n_miss

            # Clip to IQR fences
            if clip_outliers:
                valid = cleaned[col].dropna()
                if len(valid) >= 4:
                    q1 = float(valid.quantile(0.25))
                    q3 = float(valid.quantile(0.75))
                    iqr_val = q3 - q1
                    if iqr_val > 0:
                        lo = q1 - _IQR_MODERATE * iqr_val
                        hi = q3 + _IQR_MODERATE * iqr_val
                        mask = (cleaned[col] < lo) | (cleaned[col] > hi)
                        n_clip = int(mask.sum())
                        if n_clip > 0:
                            cleaned[col] = cleaned[col].clip(lower=lo, upper=hi)
                            outliers_clipped[col] = n_clip

            # Restore integer dtype when possible
            orig_dt = str(original_dtypes.get(col, ""))
            if "int" in orig_dt.lower():
                try:
                    cleaned[col] = cleaned[col].round().astype(original_dtypes[col])
                except (ValueError, TypeError):
                    pass

        elif ct == ColumnType.CATEGORICAL:
            if impute_missing:
                n_miss = int(cleaned[col].isna().sum())
                if n_miss > 0:
                    mode = cleaned[col].mode()
                    if len(mode) > 0:
                        cleaned[col] = cleaned[col].fillna(mode.iloc[0])
                        nulls_imputed[col] = n_miss

        elif ct == ColumnType.DATETIME:
            if fix_type_violations and not pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                try:
                    before_nn = int(cleaned[col].notna().sum())
                    parsed = pd.to_datetime(cleaned[col], errors="coerce")
                    after_nn = int(parsed.notna().sum())
                    viols = before_nn - after_nn
                    if viols > 0:
                        type_violations_fixed[col] = viols
                    cleaned[col] = parsed
                except Exception:
                    pass

        elif ct == ColumnType.BOOLEAN:
            if impute_missing:
                n_miss = int(cleaned[col].isna().sum())
                if n_miss > 0:
                    mode = cleaned[col].mode()
                    if len(mode) > 0:
                        cleaned[col] = cleaned[col].fillna(mode.iloc[0])
                        nulls_imputed[col] = n_miss

    # Guarantee column order
    cleaned = cleaned[original_columns]

    return CleaningSummary(
        source_id=source_id,
        original_rows=original_rows,
        cleaned_rows=len(cleaned),
        duplicates_removed=dups_removed,
        nulls_imputed=nulls_imputed,
        outliers_clipped=outliers_clipped,
        type_violations_fixed=type_violations_fixed,
        columns_preserved=original_columns,
        cleaned_df=cleaned,
    )
