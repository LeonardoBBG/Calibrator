import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import make_source_slug


PER_CASE_JSON_DIRS = (
    "calibration_raw",
    "calibration_validated",
    "calibration_repaired",
    "compression",
    "outcome_optimized",
    "human_review_queue",
)

INDEX_PATH_COLUMNS = (
    "judgment_pdf_path",
    "et_path",
    "pdf_path",
    "path",
)

INDEX_STATUS_COLUMNS = (
    "calibrator_case_slug",
    "calibrator_status",
    "processing_state",
    "calibrator_runnable",
    "calibrator_reason",
    "calibrator_latest_artifact",
    "calibrator_artifact_counts",
    "calibrator_output_root",
    "calibrator_status_updated_at",
)


def in_progress_sentinel_path(pdf_path: Path, output_root: Path) -> Path:
    """Return the per-case in-progress sentinel path."""
    return output_root / "in_progress" / f"{make_source_slug(pdf_path)}.inprogress"


def claim_judgment_run(pdf_path: Path, output_root: Path, run_id: str) -> Path | None:
    """Atomically claim a judgment for processing, or return None if already claimed."""
    sentinel_path = in_progress_sentinel_path(pdf_path, output_root)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "judgment_path": str(pdf_path),
        "case_slug": make_source_slug(pdf_path),
        "claimed_at": datetime.now().isoformat(timespec="seconds"),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(sentinel_path, flags, 0o644)
    except FileExistsError:
        return None

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception:
        sentinel_path.unlink(missing_ok=True)
        raise
    return sentinel_path


def release_judgment_run_claim(sentinel_path: Path | None) -> None:
    """Release a previously acquired in-progress sentinel."""
    if sentinel_path is not None:
        sentinel_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class JudgmentRunStatus:
    pdf_path: Path
    case_slug: str
    status: str
    runnable: bool
    reason: str
    artifact_counts: Dict[str, int]
    artifacts: List[Path]

    @property
    def latest_artifact(self) -> Path | None:
        if not self.artifacts:
            return None
        return max(self.artifacts, key=lambda path: path.stat().st_mtime)


def _matches_case_slug(path: Path, case_slug: str) -> bool:
    name = path.name
    return f"_{case_slug}_" in name or name.startswith(f"{case_slug}_")


def find_case_json_artifacts(output_root: Path, case_slug: str) -> Dict[str, List[Path]]:
    """Find downstream per-case JSON artifacts for a judgment filename slug."""
    artifacts: Dict[str, List[Path]] = {}
    for folder_name in PER_CASE_JSON_DIRS:
        folder = output_root / folder_name
        if not folder.exists():
            artifacts[folder_name] = []
            continue
        artifacts[folder_name] = sorted(
            path
            for path in folder.glob("*.json")
            if _matches_case_slug(path, case_slug)
        )
    return artifacts


def get_judgment_run_status(pdf_path: Path, output_root: Path) -> JudgmentRunStatus:
    case_slug = make_source_slug(pdf_path)
    artifacts_by_dir = find_case_json_artifacts(output_root, case_slug)
    artifact_counts = {
        folder_name: len(paths)
        for folder_name, paths in artifacts_by_dir.items()
        if paths
    }
    artifacts = [
        path
        for paths in artifacts_by_dir.values()
        for path in paths
    ]

    if artifacts_by_dir.get("outcome_optimized"):
        return JudgmentRunStatus(
            pdf_path=pdf_path,
            case_slug=case_slug,
            status="complete",
            runnable=False,
            reason="Final outcome JSON already exists.",
            artifact_counts=artifact_counts,
            artifacts=artifacts,
        )

    sentinel_path = in_progress_sentinel_path(pdf_path, output_root)
    if sentinel_path.exists():
        return JudgmentRunStatus(
            pdf_path=pdf_path,
            case_slug=case_slug,
            status="in_progress",
            runnable=False,
            reason="Case is currently being processed by another worker.",
            artifact_counts=artifact_counts,
            artifacts=artifacts,
        )

    if artifacts:
        return JudgmentRunStatus(
            pdf_path=pdf_path,
            case_slug=case_slug,
            status="blocked_partial",
            runnable=False,
            reason="Downstream JSON exists but final outcome JSON is missing.",
            artifact_counts=artifact_counts,
            artifacts=artifacts,
        )

    if not pdf_path.exists():
        return JudgmentRunStatus(
            pdf_path=pdf_path,
            case_slug=case_slug,
            status="missing_pdf",
            runnable=False,
            reason="No downstream per-case JSON found, and source PDF path is not available.",
            artifact_counts={},
            artifacts=[],
        )

    return JudgmentRunStatus(
        pdf_path=pdf_path,
        case_slug=case_slug,
        status="not_run",
        runnable=True,
        reason="No downstream per-case JSON found.",
        artifact_counts={},
        artifacts=[],
    )


def _row_path_value(row: Dict[str, str]) -> str:
    for column in INDEX_PATH_COLUMNS:
        value = str(row.get(column) or "").strip()
        if value:
            return value
    return ""


def load_judgment_index_rows(index_path: Path) -> List[Dict[str, str]]:
    """Load Moltie judgment index rows from CSV."""
    if not index_path.exists():
        return []
    with open(index_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_judgment_index_paths(index_path: Path) -> List[Path]:
    """Return judgment PDF paths from a Moltie-generated index, preserving rank order."""
    paths: List[Path] = []
    seen = set()
    for row in load_judgment_index_rows(index_path):
        raw = _row_path_value(row)
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def load_judgment_index_paths_with_band_caps(
    index_path: Path, band_caps: Dict[str, int]
) -> List[Path]:
    """Return not-yet-run paths applying per-band caps.
    The cap counts only runnable rows (calibrator_runnable=true), so already-complete
    cases never consume quota. Rows are visited in CSV rank order (best first)."""
    rows = load_judgment_index_rows(index_path)
    if not rows:
        return []
    band_counts: Dict[str, int] = {}
    paths: List[Path] = []
    seen: set = set()
    for row in rows:
        is_runnable = str(row.get("calibrator_runnable") or "").strip().lower() == "true"
        if not is_runnable:
            continue
        band = str(row.get("composite_band") or "").strip() or "UNKNOWN"
        cap = int(band_caps.get(band, 0))
        if cap == 0 or band_counts.get(band, 0) >= cap:
            continue
        raw = _row_path_value(row)
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        band_counts[band] = band_counts.get(band, 0) + 1
        paths.append(path)
    return paths


def load_judgment_index_paths_capped_by_band(index_path: Path, max_per_band: int) -> List[Path]:
    """Return paths from the index capped to max_per_band per composite_band, preserving rank order."""
    rows = load_judgment_index_rows(index_path)
    if not rows:
        return []
    band_counts: Dict[str, int] = {}
    paths: List[Path] = []
    seen: set = set()
    for row in rows:
        band = str(row.get("composite_band") or "").strip() or "UNKNOWN"
        if max_per_band > 0 and band_counts.get(band, 0) >= max_per_band:
            continue
        raw = _row_path_value(row)
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        band_counts[band] = band_counts.get(band, 0) + 1
        paths.append(path)
    return paths


def get_index_band_summary(index_path: Path, max_per_band: int = 0) -> List[Dict[str, Any]]:
    """Return per-band counts from the index CSV using the cached calibrator_runnable column."""
    rows = load_judgment_index_rows(index_path)
    if not rows:
        return []
    band_data: Dict[str, Dict[str, int]] = {}
    band_order: List[str] = []
    band_sel_counts: Dict[str, int] = {}
    for row in rows:
        band = str(row.get("composite_band") or "").strip() or "UNKNOWN"
        raw = _row_path_value(row)
        if not raw:
            continue
        if band not in band_data:
            band_data[band] = {"total": 0, "selected": 0, "runnable_selected": 0}
            band_order.append(band)
        band_data[band]["total"] += 1
        if max_per_band > 0 and band_sel_counts.get(band, 0) >= max_per_band:
            continue
        band_sel_counts[band] = band_sel_counts.get(band, 0) + 1
        band_data[band]["selected"] += 1
        if str(row.get("calibrator_runnable") or "").strip().lower() == "true":
            band_data[band]["runnable_selected"] += 1
    return [{"band": band, **band_data[band]} for band in band_order]


def scan_judgment_run_statuses(
    judgments_dir: Path,
    output_root: Path,
    judgment_index_path: Path | None = None,
) -> List[JudgmentRunStatus]:
    """Scan indexed or local input judgment PDFs and classify whether each can be run."""
    if judgment_index_path and judgment_index_path.exists():
        indexed_paths = load_judgment_index_paths(judgment_index_path)
        if indexed_paths:
            return [
                get_judgment_run_status(path, output_root)
                for path in indexed_paths
            ]

    return [
        get_judgment_run_status(path, output_root)
        for path in sorted(judgments_dir.glob("*.pdf"))
    ]


def plan_judgment_run(
    judgment_paths: Iterable[Path],
    output_root: Path,
) -> tuple[List[Path], List[JudgmentRunStatus]]:
    """Return runnable PDFs and skipped statuses, enforcing the no-rerun guard."""
    runnable_paths: List[Path] = []
    skipped_statuses: List[JudgmentRunStatus] = []
    for path in judgment_paths:
        status = get_judgment_run_status(path, output_root)
        if status.runnable:
            runnable_paths.append(path)
        else:
            skipped_statuses.append(status)
    return runnable_paths, skipped_statuses


def _processing_state(status: JudgmentRunStatus) -> str:
    if status.status == "complete":
        return "processed"
    if status.status == "not_run":
        return "pending"
    if status.status == "blocked_partial":
        return "partial"
    if status.status == "missing_pdf":
        return "missing_pdf"
    if status.status == "in_progress":
        return "in_progress"
    return status.status


def _write_index_rows(index_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, index_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def refresh_judgment_index_statuses(index_path: Path, output_root: Path) -> Dict[str, int | str]:
    """Update a Moltie judgment index with current Calibrator run status columns."""
    rows = load_judgment_index_rows(index_path)
    if not rows:
        return {
            "index_path": str(index_path),
            "total": 0,
            "processed": 0,
            "pending": 0,
            "partial": 0,
            "missing_pdf": 0,
            "in_progress": 0,
        }

    fieldnames = list(rows[0].keys())
    for column in INDEX_STATUS_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    status_counts = {
        "processed": 0,
        "pending": 0,
        "partial": 0,
        "missing_pdf": 0,
        "in_progress": 0,
    }
    updated_at = datetime.now().isoformat(timespec="seconds")

    for row in rows:
        raw = _row_path_value(row)
        if raw:
            status = get_judgment_run_status(Path(raw).expanduser(), output_root)
        else:
            status = JudgmentRunStatus(
                pdf_path=Path(""),
                case_slug="",
                status="missing_pdf",
                runnable=False,
                reason="Index row has no judgment PDF path column.",
                artifact_counts={},
                artifacts=[],
            )

        state = _processing_state(status)
        if state in status_counts:
            status_counts[state] += 1

        row["calibrator_case_slug"] = status.case_slug
        row["calibrator_status"] = status.status
        row["processing_state"] = state
        row["calibrator_runnable"] = "true" if status.runnable else "false"
        row["calibrator_reason"] = status.reason
        row["calibrator_latest_artifact"] = str(status.latest_artifact) if status.latest_artifact else ""
        row["calibrator_artifact_counts"] = json.dumps(status.artifact_counts, sort_keys=True)
        row["calibrator_output_root"] = str(output_root)
        row["calibrator_status_updated_at"] = updated_at

    _write_index_rows(index_path, rows, fieldnames)

    return {
        "index_path": str(index_path),
        "total": len(rows),
        **status_counts,
    }
