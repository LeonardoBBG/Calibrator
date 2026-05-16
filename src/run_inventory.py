from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .io_utils import make_source_slug


PER_CASE_JSON_DIRS = (
    "calibration_raw",
    "calibration_validated",
    "calibration_repaired",
    "compression",
    "outcome_optimized",
    "human_review_queue",
)


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

    return JudgmentRunStatus(
        pdf_path=pdf_path,
        case_slug=case_slug,
        status="not_run",
        runnable=True,
        reason="No downstream per-case JSON found.",
        artifact_counts={},
        artifacts=[],
    )


def scan_judgment_run_statuses(judgments_dir: Path, output_root: Path) -> List[JudgmentRunStatus]:
    """Scan all input judgment PDFs and classify whether each can be run."""
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
