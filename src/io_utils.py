import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from config import Config

def ensure_dirs(config: "Config") -> None:
    """Ensure all output directories exist."""
    dirs = [
        config.output_root / "extracted_text",
        config.output_root / "calibration_raw",
        config.output_root / "calibration_validated",
        config.output_root / "calibration_repaired",
        config.output_root / "ws_tagging",
        config.output_root / "compression",
        config.output_root / "outcome_optimized",
        config.output_root / "outcome_aggregation",
        config.output_root / "theme_store",
        config.output_root / "human_review_queue",
        config.output_root / "in_progress",
        config.cache_root / "text",
        config.cache_root / "llm",
        config.output_root / "logs"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def read_text_file(path: Path) -> str:
    """Read text file as UTF-8."""
    return path.read_text(encoding='utf-8')

def read_json(path: Path) -> Any:
    """Read JSON file as UTF-8."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _temporary_sibling_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")

def write_json(path: Path, data: Any, validate_reload: bool = False) -> None:
    """Write dict to JSON file atomically, optionally validate reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling_path(path)
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    if validate_reload:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)

def write_text(path: Path, text: str) -> None:
    """Write text to file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling_path(path)
    try:
        temp_path.write_text(text, encoding='utf-8')
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def make_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_source_slug(path: Path) -> str:
    """Create a filesystem-safe slug while preserving the source filename stem."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return slug or "source"

def make_run_scope_slug(run_id: str, case_slugs: Iterable[str]) -> str:
    """Create a run-level artifact slug that preserves single-case source identity."""
    slugs = [slug for slug in case_slugs if slug]
    if len(slugs) == 1:
        return f"{run_id}_{slugs[0]}"
    if len(slugs) > 1:
        return f"{run_id}_batch_{len(slugs)}_cases"
    return f"{run_id}_no_successful_cases"
