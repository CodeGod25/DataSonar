"""DataSonar Anomaly Detector — Terminal CLI.

Usage:
    python -m app.main demo              Full pipeline demo with synthetic data
    python -m app.main train             Train on synthetic profiles
    python -m app.main train --csv FILE  Train from a CSV file
    python -m app.main detect --csv FILE Detect anomalies in any CSV file
    python -m app.main clean  --csv FILE Auto-clean a CSV and export result
"""

from __future__ import annotations

import argparse
import logging
import io
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

from app.core.config import get_settings
from app.processor import EventProcessor
from app.services.data_generator import generate_training_dataframe
from app.services.local_storage import LocalModelStorage
from app.services.model_manager import ModelManager
from app.services.outlier_detection import QualityOutlierDetector
from app.services.time_series import TimeSeriesAnomalyDetector

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console(force_terminal=True)


# ── Helpers ──────────────────────────────────────────────────────────


def _setup_logging(level: str = "WARNING") -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _build_components() -> tuple[
    ModelManager, EventProcessor, QualityOutlierDetector, TimeSeriesAnomalyDetector
]:
    """Wire up all components — no external services needed."""
    storage = LocalModelStorage()
    quality_detector = QualityOutlierDetector()
    ts_detector = TimeSeriesAnomalyDetector()
    manager = ModelManager(storage, quality_detector, ts_detector)
    processor = EventProcessor(ts_detector, quality_detector, manager)
    return manager, processor, quality_detector, ts_detector


def _print_banner() -> None:
    banner = Text()
    banner.append("DataSonar ", style="bold cyan")
    banner.append("Anomaly Detector", style="bold white")
    banner.append("  v1.0", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
    console.print()


def _print_baseline_table(stats: dict, source_id: str) -> None:
    """Print baseline statistics in a compact table."""
    table = Table(
        title=f"Baseline - {source_id}",
        box=box.ROUNDED,
        title_style="bold green",
        border_style="green",
        show_lines=False,
    )
    table.add_column("Metric", style="cyan", min_width=18)
    table.add_column("Value", style="white", justify="right", min_width=12)

    for key in [
        "quality_mean",
        "quality_std",
        "quality_median",
        "quality_p5",
        "quality_p95",
        "volume_mean",
        "volume_std",
        "data_points",
        "n_estimators",
        "n_features",
    ]:
        val = stats.get(key)
        if val is not None:
            if isinstance(val, float):
                table.add_row(key, f"{val:.4f}")
            else:
                table.add_row(key, str(val))

    console.print(table)
    console.print()


def _print_detection_results(results: list, source_id: str) -> None:
    """Print anomaly detection results as a rich table."""
    from app.processor import DetectionResult

    anomalies = [r for r in results if r.is_anomaly]
    normals = len(results) - len(anomalies)
    high = sum(1 for r in anomalies if r.severity == "HIGH")
    medium = sum(1 for r in anomalies if r.severity == "MEDIUM")

    # Summary line
    summary = Text()
    summary.append(f"  {normals} normal", style="green")
    summary.append(" | ", style="dim")
    summary.append(f"{len(anomalies)} anomalies", style="red bold" if anomalies else "green")
    if high:
        summary.append(f" ({high} high", style="red")
        if medium:
            summary.append(f", {medium} medium", style="yellow")
        summary.append(")", style="red" if not medium else "yellow")
    elif medium:
        summary.append(f" ({medium} medium)", style="yellow")
    console.print(summary)
    console.print()

    if not anomalies:
        console.print("  [green][OK][/green] No anomalies detected!")
        console.print()
        return

    # Anomaly detail table
    table = Table(
        title=f"Anomalies Detected - {source_id}",
        box=box.ROUNDED,
        title_style="bold red",
        border_style="red",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Timestamp", style="cyan", min_width=20)
    table.add_column("Quality", justify="right", min_width=8)
    table.add_column("Volume", justify="right", min_width=10)
    table.add_column("Types", style="magenta", min_width=16)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Severity", justify="center", min_width=8)

    for r in anomalies[:50]:  # Cap display at 50 anomalies
        sev_style = "red bold" if r.severity == "HIGH" else "yellow"
        qual_style = "red" if r.quality_score < 0.7 else "white"
        table.add_row(
            str(r.event_index),
            r.timestamp.strftime("%Y-%m-%d %H:%M"),
            f"[{qual_style}]{r.quality_score:.3f}[/{qual_style}]",
            f"{r.record_count:,}",
            ", ".join(r.anomaly_types),
            f"{r.anomaly_score:.2f}",
            f"[{sev_style}]{r.severity}[/{sev_style}]",
        )

    if len(anomalies) > 50:
        table.add_row("...", f"and {len(anomalies) - 50} more", "", "", "", "", "")

    console.print(table)
    console.print()


# ── Commands ─────────────────────────────────────────────────────────


def cmd_demo(args: argparse.Namespace) -> None:
    """Full pipeline demo: generate → train → detect."""
    _print_banner()
    manager, processor, _, _ = _build_components()
    settings = get_settings()
    profiles = args.profiles.split(",") if args.profiles else settings.PRETRAIN_PROFILES

    t0 = time.perf_counter()

    # Step 1: Generate & Train
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training models...", total=len(profiles))

        for profile in profiles:
            source_id = f"pretrained-{profile}"
            df = generate_training_dataframe(profile=profile, n_points=args.n_points)
            stats = manager.train_from_dataframe(source_id=source_id, df=df)
            progress.update(task, advance=1, description=f"[cyan]Trained: {profile}")

    console.print()
    console.print(f"  [green][OK][/green] Trained {len(profiles)} models in {time.perf_counter() - t0:.2f}s")
    console.print()

    # Show baselines
    for profile in profiles:
        source_id = f"pretrained-{profile}"
        baseline = manager.ensure_source_baseline(source_id)
        # Get full stats from the storage
        full_baseline = manager._storage.load_model(source_id, "history.joblib")
        if full_baseline and "baseline_stats" in full_baseline:
            _print_baseline_table(full_baseline["baseline_stats"], source_id)

    # Step 2: Detect on test data
    console.rule("[bold cyan]Anomaly Detection", style="cyan")
    console.print()

    t1 = time.perf_counter()

    for profile in profiles:
        source_id = f"pretrained-{profile}"
        test_df = generate_training_dataframe(
            profile=profile, n_points=args.test_points, random_state=99
        )
        results = processor.process_dataframe(source_id, test_df)

        console.print(f"  [bold]{source_id}[/bold] - {len(results)} events processed")
        _print_detection_results(results, source_id)

    elapsed = time.perf_counter() - t0
    console.print(f"  [dim]Total time: {elapsed:.2f}s[/dim]")
    console.print()


def cmd_train(args: argparse.Namespace) -> None:
    """Train models from CSV or synthetic data."""
    _print_banner()
    manager, _, _, _ = _build_components()

    t0 = time.perf_counter()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            console.print(f"[red]Error:[/red] File not found: {csv_path}")
            sys.exit(1)

        source_id = args.source_id or csv_path.stem
        console.print(f"  Training from CSV: [cyan]{csv_path}[/cyan]")

        try:
            stats = manager.train_from_csv(source_id=source_id, csv_path=str(csv_path))
        except ValueError as e:
            console.print(f"  [red]Validation error:[/red] {e}")
            sys.exit(1)

        elapsed = time.perf_counter() - t0
        console.print(f"  [green][OK][/green] Model trained in {elapsed:.2f}s")
        console.print()
        _print_baseline_table(stats, source_id)
    else:
        settings = get_settings()
        profiles = args.profiles.split(",") if args.profiles else settings.PRETRAIN_PROFILES

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training...", total=len(profiles))
            for profile in profiles:
                source_id = f"pretrained-{profile}"
                df = generate_training_dataframe(profile=profile, n_points=args.n_points)
                stats = manager.train_from_dataframe(source_id=source_id, df=df)
                progress.update(task, advance=1, description=f"[cyan]Trained: {profile}")

        elapsed = time.perf_counter() - t0
        console.print()
        console.print(f"  [green][OK][/green] {len(profiles)} models trained in {elapsed:.2f}s")
        console.print()

        for profile in profiles:
            source_id = f"pretrained-{profile}"
            full = manager._storage.load_model(source_id, "history.joblib")
            if full and "baseline_stats" in full:
                _print_baseline_table(full["baseline_stats"], source_id)


def _print_tabular_results(summary: Any) -> None:
    """Print tabular anomaly detection results as a rich report."""
    # Summary line
    summary_text = Text()
    summary_text.append(f"  {summary.normal_count} normal", style="green")
    summary_text.append(" | ", style="dim")
    summary_text.append(f"{summary.anomaly_count} anomalies", style="red bold" if summary.anomaly_count else "green")
    if summary.high_count:
        summary_text.append(f" ({summary.high_count} high", style="red bold")
        if summary.medium_count:
            summary_text.append(f", {summary.medium_count} medium", style="yellow")
        summary_text.append(")", style="red bold" if not summary.medium_count else "yellow")
    elif summary.medium_count:
        summary_text.append(f" ({summary.medium_count} medium)", style="yellow")

    summary_text.append(" | ", style="dim")
    summary_text.append(f"Quality: {summary.mean_quality_score:.3f}/1.000", style="cyan bold")
    summary_text.append(" | ", style="dim")
    summary_text.append(f"Speed: {summary.execution_time_seconds*1000:.1f}ms ({summary.throughput_rows_per_sec:,.0f} rows/s)", style="magenta")
    console.print(summary_text)
    console.print()

    # Anomaly breakdown by category
    if summary.anomaly_type_counts:
        cat_table = Table(
            title=f"Anomaly Breakdown — {summary.source_id}",
            box=box.ROUNDED,
            title_style="bold yellow",
            border_style="yellow",
            show_lines=False,
        )
        cat_table.add_column("Anomaly Category", style="cyan", min_width=28)
        cat_table.add_column("Count", justify="right", style="white", min_width=8)
        cat_table.add_column("% of Records", justify="right", style="magenta", min_width=14)

        for cat, cnt in summary.anomaly_type_counts.items():
            pct = (cnt / summary.total_records) * 100
            cat_table.add_row(cat, str(cnt), f"{pct:.1f}%")

        console.print(cat_table)
        console.print()

    # Detail table
    anomalies = [r for r in summary.results if r.is_anomaly]
    if not anomalies:
        console.print("  [green][OK][/green] No anomalies detected!")
        console.print()
        return

    table = Table(
        title=f"Anomalies Detected — {summary.source_id}",
        box=box.ROUNDED,
        title_style="bold red",
        border_style="red",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Record ID", style="cyan", min_width=12)
    table.add_column("Student / Name", style="white", min_width=18)
    table.add_column("Quality", justify="right", min_width=8)
    table.add_column("Anomaly Types", style="magenta", min_width=32)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Severity", justify="center", min_width=8)

    for r in anomalies[:50]:  # Cap display at 50
        sev_style = "red bold" if r.severity == "HIGH" else "yellow"
        qual_style = "red" if r.quality_score < 0.5 else ("yellow" if r.quality_score < 0.75 else "green")
        issues_summary = ", ".join(r.anomaly_types[:3])
        if len(r.anomaly_types) > 3:
            issues_summary += f" (+{len(r.anomaly_types)-3} more)"

        table.add_row(
            str(r.row_index),
            r.record_id,
            r.display_name,
            f"[{qual_style}]{r.quality_score:.2f}[/{qual_style}]",
            issues_summary,
            f"{r.anomaly_score:.2f}",
            f"[{sev_style}]{r.severity}[/{sev_style}]",
        )

    if len(anomalies) > 50:
        table.add_row("...", f"and {len(anomalies) - 50} more records", "", "", "", "", "")

    console.print(table)
    console.print()


def _print_universal_results(summary: Any) -> None:
    """Rich display for universal anomaly-detection results."""
    from app.services.universal_detector import DetectionSummary, Severity

    # ── Dataset Overview ─────────────────────────────────────────────
    ov = Table(
        title="Dataset Overview",
        box=box.ROUNDED,
        title_style="bold cyan",
        border_style="cyan",
        show_lines=False,
    )
    ov.add_column("Property", style="cyan", min_width=24)
    ov.add_column("Value", style="white", justify="right", min_width=14)
    ov.add_row("Total Rows", f"{summary.total_records:,}")
    ov.add_row("Total Columns", str(summary.total_columns))
    n_num = sum(1 for p in summary.column_profiles if p.inferred_type.value == "numeric")
    n_cat = sum(1 for p in summary.column_profiles if p.inferred_type.value == "categorical")
    n_dt = sum(1 for p in summary.column_profiles if p.inferred_type.value == "datetime")
    n_other = summary.total_columns - n_num - n_cat - n_dt
    ov.add_row("Numeric Columns", str(n_num))
    ov.add_row("Categorical Columns", str(n_cat))
    ov.add_row("Datetime Columns", str(n_dt))
    if n_other:
        ov.add_row("Other Columns (ID/Text/Bool)", str(n_other))
    ov.add_row("Duplicates Found", str(summary.duplicate_count))
    console.print(ov)
    console.print()

    # ── Algorithms Used ──────────────────────────────────────────────
    algo_text = Text("  Algorithms: ", style="dim")
    algo_text.append(", ".join(summary.algorithms_used), style="cyan")
    console.print(algo_text)
    console.print()

    # ── Summary Line ─────────────────────────────────────────────────
    sl = Text()
    sl.append(f"  {summary.normal_count} normal", style="green")
    sl.append(" | ", style="dim")
    sl.append(
        f"{summary.anomaly_count} anomalies",
        style="red bold" if summary.anomaly_count else "green",
    )
    if summary.high_count:
        sl.append(f" ({summary.high_count} HIGH", style="red bold")
        if summary.moderate_count:
            sl.append(f", {summary.moderate_count} MODERATE", style="yellow")
        if summary.low_count:
            sl.append(f", {summary.low_count} LOW", style="blue")
        sl.append(")", style="dim")
    elif summary.moderate_count:
        sl.append(f" ({summary.moderate_count} MODERATE", style="yellow")
        if summary.low_count:
            sl.append(f", {summary.low_count} LOW", style="blue")
        sl.append(")", style="dim")
    elif summary.low_count:
        sl.append(f" ({summary.low_count} LOW)", style="blue")
    sl.append(" | ", style="dim")
    sl.append(
        f"Speed: {summary.execution_time_seconds * 1000:.1f}ms "
        f"({summary.throughput_rows_per_sec:,.0f} rows/s)",
        style="magenta",
    )
    console.print(sl)
    console.print()

    # ── Column Profiles ──────────────────────────────────────────────
    col_table = Table(
        title=f"Column Profiles — {summary.source_id}",
        box=box.ROUNDED,
        title_style="bold green",
        border_style="green",
        show_lines=False,
    )
    col_table.add_column("Column", style="cyan", min_width=20)
    col_table.add_column("Type", style="magenta", min_width=12)
    col_table.add_column("Nulls", justify="right", min_width=8)
    col_table.add_column("Null %", justify="right", min_width=8)
    col_table.add_column("Unique", justify="right", min_width=8)
    col_table.add_column("Stats", style="white", min_width=28)

    for p in summary.column_profiles:
        null_style = "red" if p.null_rate > 0.20 else ("yellow" if p.null_rate > 0.05 else "green")
        stats = ""
        if p.inferred_type.value == "numeric" and p.mean is not None:
            std_val = p.std if p.std is not None else 0.0
            med_str = f"  med={p.median:.2f}" if p.median is not None else ""
            stats = f"μ={p.mean:.2f}  σ={std_val:.2f}{med_str}"
        elif p.inferred_type.value == "categorical" and p.mode_value is not None:
            stats = f"mode='{p.mode_value}'  cats={p.unique_count}"
        col_table.add_row(
            p.name,
            p.inferred_type.value,
            f"[{null_style}]{p.null_count}[/{null_style}]",
            f"[{null_style}]{p.null_rate:.1%}[/{null_style}]",
            str(p.unique_count),
            stats,
        )
    console.print(col_table)
    console.print()

    # ── Anomaly Breakdown ────────────────────────────────────────────
    if summary.anomaly_type_counts:
        br = Table(
            title=f"Anomaly Breakdown — {summary.source_id}",
            box=box.ROUNDED,
            title_style="bold yellow",
            border_style="yellow",
            show_lines=False,
        )
        br.add_column("Anomaly Type", style="cyan", min_width=32)
        br.add_column("Count", justify="right", style="white", min_width=8)
        br.add_column("% of Rows", justify="right", style="magenta", min_width=10)
        for atype, cnt in summary.anomaly_type_counts.items():
            pct = cnt / summary.total_records * 100
            br.add_row(atype, str(cnt), f"{pct:.1f}%")
        console.print(br)
        console.print()

    # ── Anomaly Detail Table ─────────────────────────────────────────
    anom_rows = [r for r in summary.row_anomalies if r.is_anomaly]
    if not anom_rows:
        console.print("  [green][OK][/green] No anomalies detected!")
        console.print()
        return

    # Sort by score descending so worst anomalies are first
    anom_rows = sorted(anom_rows, key=lambda r: r.anomaly_score, reverse=True)

    detail = Table(
        title=f"Top Anomalous Rows — {summary.source_id}",
        box=box.ROUNDED,
        title_style="bold red",
        border_style="red",
        show_lines=False,
    )
    detail.add_column("Row", style="dim", width=6, justify="right")
    detail.add_column("Score", justify="right", min_width=7)
    detail.add_column("Severity", justify="center", min_width=10)
    detail.add_column("Reasons", style="white", min_width=50)

    for r in anom_rows[:60]:  # Cap display
        sev_style = {
            "HIGH": "red bold",
            "MODERATE": "yellow",
            "LOW": "blue",
        }.get(r.severity.value, "dim")
        score_style = "red" if r.anomaly_score >= 0.7 else ("yellow" if r.anomaly_score >= 0.4 else "blue")
        reasons = ", ".join(r.reasons[:4])
        if len(r.reasons) > 4:
            reasons += f" (+{len(r.reasons) - 4} more)"
        detail.add_row(
            str(r.row_index),
            f"[{score_style}]{r.anomaly_score:.2f}[/{score_style}]",
            f"[{sev_style}]{r.severity.value}[/{sev_style}]",
            reasons,
        )
    if len(anom_rows) > 60:
        detail.add_row("...", "", "", f"and {len(anom_rows) - 60} more rows")

    console.print(detail)
    console.print()


def _print_cleaning_results(summary: Any) -> None:
    """Rich display for auto-cleaning results."""
    cl = Table(
        title=f"Cleaning Report — {summary.source_id}",
        box=box.ROUNDED,
        title_style="bold green",
        border_style="green",
        show_lines=False,
    )
    cl.add_column("Action", style="cyan", min_width=28)
    cl.add_column("Details", style="white", min_width=40)

    cl.add_row("Original Rows", f"{summary.original_rows:,}")
    cl.add_row("Cleaned Rows", f"{summary.cleaned_rows:,}")
    cl.add_row("Duplicates Removed", f"{summary.duplicates_removed:,}")
    cl.add_row("Columns Preserved", f"{len(summary.columns_preserved)} columns")

    if summary.nulls_imputed:
        items = ", ".join(f"{c}: {n}" for c, n in summary.nulls_imputed.items())
        cl.add_row("Nulls Imputed", items)
    else:
        cl.add_row("Nulls Imputed", "None")

    if summary.outliers_clipped:
        items = ", ".join(f"{c}: {n}" for c, n in summary.outliers_clipped.items())
        cl.add_row("Outliers Clipped", items)
    else:
        cl.add_row("Outliers Clipped", "None")

    if summary.type_violations_fixed:
        items = ", ".join(f"{c}: {n}" for c, n in summary.type_violations_fixed.items())
        cl.add_row("Type Violations Fixed", items)
    else:
        cl.add_row("Type Violations Fixed", "None")

    console.print(cl)
    console.print()


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect anomalies in a CSV file (auto-detects pipeline vs tabular dataset)."""
    _print_banner()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        console.print(f"[red]Error:[/red] File not found: {csv_path}")
        sys.exit(1)

    source_id = args.source_id or csv_path.stem

    # Check if the file is a pipeline metrics CSV or a general tabular dataset
    is_pipeline_csv = False
    try:
        with open(csv_path, "rb") as f:
            first_line = f.readline().decode("utf-8", errors="ignore").lower()
            if all(k in first_line for k in ["timestamp", "record_count", "overall_score"]):
                is_pipeline_csv = True
    except Exception:
        pass

    if not is_pipeline_csv:
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        use_stream = getattr(args, "stream", False) or (file_size_mb > 50.0 and not getattr(args, "in_memory", False))

        if use_stream:
            from app.services.industrial_stream import IndustrialStreamDetector

            console.print(f"  [bold yellow]Industrial Streaming Mode[/bold yellow] ({file_size_mb:.1f} MB: [cyan]{csv_path.name}[/cyan])")
            console.print("  Processing with Out-Of-Core Reservoir ML + Online Welford Statistics...")
            console.print()

            streamer = IndustrialStreamDetector(chunk_size=getattr(args, "chunk_size", 50_000))
            summary = streamer.detect_stream(csv_path, source_id=source_id)
            _print_universal_results(summary)
            return

        # Route to universal anomaly detector
        from app.services.universal_detector import detect_anomalies

        console.print(f"  Detected tabular dataset: [cyan]{csv_path.name}[/cyan]")
        console.print(f"  Running multi-algorithm ensemble anomaly detection...")
        console.print()

        summary = detect_anomalies(csv_path, source_id)
        _print_universal_results(summary)
        return

    # Standard pipeline detection
    manager, processor, _, _ = _build_components()

    # Try to load existing model, otherwise train from the CSV itself
    manager.initialize()
    if not manager.is_loaded(source_id):
        console.print(f"  No existing model for [cyan]{source_id}[/cyan], training from CSV...")
        try:
            manager.train_from_csv(source_id=source_id, csv_path=str(csv_path))
        except ValueError as e:
            console.print(f"  [red]Validation error:[/red] {e}")
            sys.exit(1)

    t0 = time.perf_counter()

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    results = processor.process_dataframe(source_id, df)

    elapsed = time.perf_counter() - t0

    console.print(f"  [bold]{source_id}[/bold] - {len(results)} events in {elapsed:.2f}s")
    _print_detection_results(results, source_id)


# ── Clean Command ────────────────────────────────────────────────────


def cmd_clean(args: argparse.Namespace) -> None:
    """Auto-clean a CSV dataset and export the result."""
    _print_banner()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        console.print(f"[red]Error:[/red] File not found: {csv_path}")
        sys.exit(1)

    source_id = args.source_id or csv_path.stem
    output_path = Path(args.output) if args.output else csv_path.with_name(f"{csv_path.stem}_cleaned.csv")

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    use_stream = getattr(args, "stream", False) or (file_size_mb > 50.0 and not getattr(args, "in_memory", False))

    if use_stream:
        from app.services.industrial_stream import IndustrialStreamDetector
        console.print(f"  [bold yellow]Industrial Streaming Auto-Clean[/bold yellow] ({file_size_mb:.1f} MB: [cyan]{csv_path.name}[/cyan])")
        console.print()
        streamer = IndustrialStreamDetector(chunk_size=getattr(args, "chunk_size", 50_000))
        t0 = time.perf_counter()
        res = streamer.stream_clean_to_file(csv_path, output_path)
        elapsed = time.perf_counter() - t0
        console.print(f"  [green][OK][/green] Industrial dataset cleaned ({res['rows_cleaned']:,} rows) -> [cyan]{output_path}[/cyan]")
        console.print(f"  [dim]Out-of-core stream cleaning completed in {elapsed:.2f}s with constant memory[/dim]")
        console.print()
        return

    from app.services.universal_detector import clean_dataset, detect_anomalies

    # Step 1: Detection
    console.print(f"  Analysing: [cyan]{csv_path.name}[/cyan]")
    console.print()
    summary = detect_anomalies(csv_path, source_id)
    _print_universal_results(summary)

    # Step 2: Cleaning
    console.rule("[bold green]Auto-Cleaning", style="green")
    console.print()
    t0 = time.perf_counter()

    clean_summary = clean_dataset(
        csv_path,
        source_id,
        remove_duplicates=not args.keep_duplicates,
        clip_outliers=not args.no_clip,
        impute_missing=not args.no_impute,
        fix_type_violations=True,
    )

    elapsed = time.perf_counter() - t0
    _print_cleaning_results(clean_summary)

    # Step 3: Export
    clean_summary.cleaned_df.to_csv(output_path, index=False)
    console.print(f"  [green][OK][/green] Cleaned dataset saved to: [cyan]{output_path}[/cyan]")
    console.print(f"  [dim]Cleaning completed in {elapsed:.2f}s[/dim]")
    console.print()


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-detector",
        description="DataSonar Anomaly Detector — Terminal CLI",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # demo
    demo = sub.add_parser("demo", help="Full pipeline demo with synthetic data")
    demo.add_argument("--profiles", type=str, default=None, help="Comma-separated profiles (default: stable,volatile,degrading)")
    demo.add_argument("--n-points", type=int, default=1000, help="Training data points per profile (default: 1000)")
    demo.add_argument("--test-points", type=int, default=200, help="Test data points per profile (default: 200)")

    # train
    train = sub.add_parser("train", help="Train models from CSV or synthetic data")
    train.add_argument("--csv", type=str, default=None, help="Path to CSV file for training")
    train.add_argument("--source-id", type=str, default=None, help="Source ID (default: CSV filename stem)")
    train.add_argument("--profiles", type=str, default=None, help="Comma-separated profiles for synthetic training")
    train.add_argument("--n-points", type=int, default=1000, help="Data points per synthetic profile")

    # detect
    detect = sub.add_parser("detect", help="Detect anomalies in any CSV file")
    detect.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    detect.add_argument("--source-id", type=str, default=None, help="Source ID (default: CSV filename stem)")
    detect.add_argument("--stream", action="store_true", help="Force out-of-core streaming mode for large/industrial GB datasets")
    detect.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming processing (default: 50,000)")
    detect.add_argument("--in-memory", action="store_true", help="Force in-memory mode even for large files")

    # clean
    clean = sub.add_parser("clean", help="Auto-clean a CSV file and export result")
    clean.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    clean.add_argument("--output", "-o", type=str, default=None, help="Output path (default: <name>_cleaned.csv)")
    clean.add_argument("--source-id", type=str, default=None, help="Source ID (default: CSV filename stem)")
    clean.add_argument("--keep-duplicates", action="store_true", help="Do not remove duplicate rows")
    clean.add_argument("--no-clip", action="store_true", help="Do not clip numeric outliers")
    clean.add_argument("--no-impute", action="store_true", help="Do not impute missing values")
    clean.add_argument("--stream", action="store_true", help="Force out-of-core streaming mode for large/industrial GB datasets")
    clean.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming processing (default: 50,000)")
    clean.add_argument("--in-memory", action="store_true", help="Force in-memory mode even for large files")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging("DEBUG" if args.verbose else "WARNING")

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
