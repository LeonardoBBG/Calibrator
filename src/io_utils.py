import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, TYPE_CHECKING

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
        config.output_root / "human_review_queue",
        config.cache_root / "text",
        config.cache_root / "llm",
        config.output_root / "logs"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def read_text_file(path: Path) -> str:
    """Read text file as UTF-8."""
    return path.read_text(encoding='utf-8')

def write_json(path: Path, data: Any, validate_reload: bool = False) -> None:
    """Write dict to JSON file, optionally validate reload."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if validate_reload:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)

def write_text(path: Path, text: str) -> None:
    """Write text to file."""
    path.write_text(text, encoding='utf-8')

def make_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_source_slug(path: Path) -> str:
    """Create a filesystem-safe slug while preserving the source filename stem."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return slug or "source"
