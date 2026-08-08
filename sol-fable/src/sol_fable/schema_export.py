"""Export reviewable JSON Schema snapshots from the authoritative Pydantic models."""

from __future__ import annotations

from .config import project_root
from .io_utils import write_json
from .models import (
    Argument,
    AtomicProposition,
    CaseLawSearchPackage,
    DebateSummary,
    DebateTurn,
    Document,
    Issue,
)


def main() -> None:
    destination = project_root() / "schemas"
    for model in (
        Document,
        AtomicProposition,
        Issue,
        DebateTurn,
        DebateSummary,
        Argument,
        CaseLawSearchPackage,
    ):
        write_json(destination / f"{model.__name__}.schema.json", model.model_json_schema())


if __name__ == "__main__":
    main()
