"""Small deterministic I/O helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


def _prepare_private_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass


def _finish_private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    _prepare_private_file(path)
    path.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _finish_private_file(path)


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    _prepare_private_file(path)
    rows = [json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True) for value in values]
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    _finish_private_file(path)


def _safe_csv_cell(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _prepare_private_file(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [{key: _safe_csv_cell(value) for key, value in row.items()} for row in rows]
        )
    _finish_private_file(path)


def write_text(path: Path, text: str) -> None:
    _prepare_private_file(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    _finish_private_file(path)
