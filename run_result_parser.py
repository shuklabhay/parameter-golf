from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "final_sliding_window_exact",
        re.compile(
            r"final_sliding_window(?:_eval)?_exact[^\n]*val_loss:(?P<val_loss>[-+0-9.eE]+)\s+val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
        ),
    ),
    (
        "final_int8_zlib_roundtrip_exact",
        re.compile(
            r"final_int8_zlib_roundtrip_exact[^\n]*val_loss:(?P<val_loss>[-+0-9.eE]+)\s+val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
        ),
    ),
    (
        "final_int6_roundtrip_exact",
        re.compile(
            r"final_int6_roundtrip_exact[^\n]*val_loss:(?P<val_loss>[-+0-9.eE]+)\s+val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
        ),
    ),
)
TOTAL_SIZE_PATTERN: re.Pattern[str] = re.compile(
    r"Total submission size(?: int8\+zlib)?:\s+(?P<bytes_total>\d+)\s+bytes"
)
TRAIN_TIME_PATTERN: re.Pattern[str] = re.compile(r"train_time:(?P<train_time_ms>\d+)ms")


@dataclass(frozen=True)
class ParsedRun:
    """Stores parsed final metrics and run-level checks from a training log."""

    final_metric_name: str | None
    val_loss: float | None
    val_bpb: float | None
    bytes_total: int | None
    max_train_time_ms: int | None
    saw_stop_early: bool


def parse_training_log(log_text: str) -> ParsedRun:
    """Parses a training log string and returns final metrics and status checks."""

    final_metric_name: str | None = None
    val_loss: float | None = None
    val_bpb: float | None = None
    for metric_name, pattern in FINAL_PATTERNS:
        matches: list[re.Match[str]] = list(pattern.finditer(log_text))
        if not matches:
            continue
        match: re.Match[str] = matches[-1]
        final_metric_name = metric_name
        val_loss = float(match.group("val_loss"))
        val_bpb = float(match.group("val_bpb"))
        break

    size_matches: list[re.Match[str]] = list(TOTAL_SIZE_PATTERN.finditer(log_text))
    bytes_total: int | None = (
        int(size_matches[-1].group("bytes_total")) if size_matches else None
    )

    time_matches: list[re.Match[str]] = list(TRAIN_TIME_PATTERN.finditer(log_text))
    max_train_time_ms: int | None = (
        max(int(match.group("train_time_ms")) for match in time_matches)
        if time_matches
        else None
    )

    return ParsedRun(
        final_metric_name=final_metric_name,
        val_loss=val_loss,
        val_bpb=val_bpb,
        bytes_total=bytes_total,
        max_train_time_ms=max_train_time_ms,
        saw_stop_early="stopping_early: wallclock_cap" in log_text,
    )


def build_status(parsed: ParsedRun) -> dict[str, bool | None]:
    """Builds boolean status checks for eval completion, time cap, and size cap."""

    return {
        "has_final_eval": parsed.final_metric_name is not None,
        "under_16mb": (
            None if parsed.bytes_total is None else parsed.bytes_total <= 16_000_000
        ),
        "within_10min_cap": (
            None
            if parsed.max_train_time_ms is None
            else parsed.max_train_time_ms <= 610_000
        ),
        "saw_wallclock_stop": parsed.saw_stop_early,
    }


def build_results_text(
    run_id: str, parsed: ParsedRun, status: dict[str, bool | None]
) -> str:
    """Formats a concise plain-text summary for LLM and Flywheel ingestion."""

    lines: list[str] = [f"run_id: {run_id}"]
    lines.append(f"final_metric: {parsed.final_metric_name}")
    lines.append(f"val_loss: {parsed.val_loss}")
    lines.append(f"val_bpb: {parsed.val_bpb}")
    lines.append(f"bytes_total: {parsed.bytes_total}")
    lines.append(f"max_train_time_ms: {parsed.max_train_time_ms}")
    lines.append(f"has_final_eval: {status['has_final_eval']}")
    lines.append(f"under_16mb: {status['under_16mb']}")
    lines.append(f"within_10min_cap: {status['within_10min_cap']}")
    lines.append(f"saw_wallclock_stop: {status['saw_wallclock_stop']}")
    return "\n".join(lines) + "\n"


def write_result_files(run_dir: Path, run_id: str, parsed: ParsedRun) -> dict[str, Any]:
    """Writes metrics, status, and summary text artifacts into the provided run directory."""

    status: dict[str, bool | None] = build_status(parsed)
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "final_metric_name": parsed.final_metric_name,
        "val_loss": parsed.val_loss,
        "val_bpb": parsed.val_bpb,
        "bytes_total": parsed.bytes_total,
        "max_train_time_ms": parsed.max_train_time_ms,
    }

    metrics_path: Path = run_dir / "metrics.json"
    status_path: Path = run_dir / "status.json"
    results_path: Path = run_dir / "results.txt"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results_path.write_text(
        build_results_text(run_id, parsed, status), encoding="utf-8"
    )

    return {
        "metrics_path": str(metrics_path),
        "status_path": str(status_path),
        "results_path": str(results_path),
        "metrics": metrics,
        "status": status,
    }
