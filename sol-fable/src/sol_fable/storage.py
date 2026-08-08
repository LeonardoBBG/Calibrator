"""SQLite persistence. SQLite is the authoritative record for every run."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from .models import (
    Argument,
    AtomicProposition,
    CaseLawSearchPackage,
    Document,
    HumanReviewItem,
    Issue,
    Paragraph,
    ReferencePlaceholder,
    RunRecord,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    backend TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    stage INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(document_type, sha256, parser_version)
);
CREATE TABLE IF NOT EXISTS run_documents (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    document_type TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    PRIMARY KEY (run_id, document_type)
);
CREATE TABLE IF NOT EXISTS run_document_metadata (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    document_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, document_type)
);
CREATE TABLE IF NOT EXISTS paragraphs (
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    paragraph_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (document_id, paragraph_id)
);
CREATE TABLE IF NOT EXISTS references_found (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    reference_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, reference_id)
);
CREATE TABLE IF NOT EXISTS issues (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    issue_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, issue_id)
);
CREATE TABLE IF NOT EXISTS propositions (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    proposition_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, proposition_id)
);
CREATE TABLE IF NOT EXISTS arguments (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    argument_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, argument_id)
);
CREATE TABLE IF NOT EXISTS search_packages (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    argument_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, argument_id)
);
CREATE TABLE IF NOT EXISTS human_reviews (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    review_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, review_id)
);
CREATE INDEX IF NOT EXISTS idx_paragraph_document ON paragraphs(document_id, sequence);
CREATE INDEX IF NOT EXISTS idx_review_run ON human_reviews(run_id);
"""


class CaseStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self.migrate()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            row = connection.execute("SELECT MAX(version) AS version FROM schema_info").fetchone()
            current = row["version"] if row else None
            if current is None:
                connection.execute(
                    "INSERT INTO schema_info(version, applied_at) VALUES(?, ?)",
                    (3, datetime.now(timezone.utc).isoformat()),
                )
            elif int(current) < 3:
                connection.commit()
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.executescript(
                    """
                    BEGIN;
                    DROP TABLE IF EXISTS documents_v3;
                    DROP TABLE IF EXISTS paragraphs_v3;
                    DROP TABLE IF EXISTS run_documents_v3;
                    CREATE TABLE documents_v3 (
                        document_id TEXT PRIMARY KEY,
                        document_type TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        ingested_at TEXT NOT NULL,
                        parser_version TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(document_type, sha256, parser_version)
                    );
                    CREATE TABLE paragraphs_v3 (
                        document_id TEXT NOT NULL REFERENCES documents_v3(document_id),
                        paragraph_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (document_id, paragraph_id)
                    );
                    CREATE TABLE run_documents_v3 (
                        run_id TEXT NOT NULL REFERENCES runs(run_id),
                        document_type TEXT NOT NULL,
                        document_id TEXT NOT NULL REFERENCES documents_v3(document_id),
                        PRIMARY KEY (run_id, document_type)
                    );
                    INSERT INTO documents_v3 SELECT * FROM documents;
                    INSERT INTO paragraphs_v3 SELECT * FROM paragraphs;
                    INSERT INTO run_documents_v3 SELECT * FROM run_documents;
                    DROP TABLE run_documents;
                    DROP TABLE paragraphs;
                    DROP TABLE documents;
                    ALTER TABLE documents_v3 RENAME TO documents;
                    ALTER TABLE paragraphs_v3 RENAME TO paragraphs;
                    ALTER TABLE run_documents_v3 RENAME TO run_documents;
                    DROP INDEX IF EXISTS idx_paragraph_document;
                    CREATE INDEX idx_paragraph_document ON paragraphs(document_id, sequence);
                    COMMIT;
                    """
                )
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute(
                    "INSERT INTO schema_info(version, applied_at) VALUES(?, ?)",
                    (3, datetime.now(timezone.utc).isoformat()),
                )

    @staticmethod
    def _payload(model: BaseModel) -> str:
        validated = type(model).model_validate(model.model_dump(mode="python"))
        return validated.model_dump_json()

    def create_run(self, run: RunRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.started_at,
                    run.completed_at,
                    run.status,
                    run.backend,
                    run.prompt_version,
                    json.dumps(run.config_snapshot, sort_keys=True),
                ),
            )

    def update_run(self, run_id: str, status: str, completed_at: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, completed_at, run_id),
            )

    def update_run_config(self, run_id: str, config_snapshot: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET config_json = ? WHERE run_id = ?",
                (json.dumps(config_snapshot, sort_keys=True), run_id),
            )

    def update_run_backend(self, run_id: str, backend: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET backend = ? WHERE run_id = ?",
                (backend, run_id),
            )

    def update_run_provenance(
        self,
        run_id: str,
        *,
        backend: str,
        prompt_version: str,
        config_snapshot: dict[str, Any],
    ) -> None:
        """Update assessment provenance in one transaction."""

        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET backend = ?, prompt_version = ?, config_json = ? WHERE run_id = ?",
                (
                    backend,
                    prompt_version,
                    json.dumps(config_snapshot, sort_keys=True),
                    run_id,
                ),
            )

    def latest_run_id(self) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return str(row["run_id"]) if row else None

    def load_run(self, run_id: str) -> RunRecord:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return RunRecord(
            run_id=row["run_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=row["status"],
            backend=row["backend"],
            prompt_version=row["prompt_version"],
            config_snapshot=json.loads(row["config_json"]),
        )

    def record_stage(
        self, run_id: str, stage: int, name: str, status: str, details: dict[str, Any] | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO stage_events(run_id, stage, name, status, recorded_at, details_json) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    stage,
                    name,
                    status,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(details or {}, sort_keys=True),
                ),
            )

    def save_document(self, run_id: str, document: Document, paragraphs: list[Paragraph]) -> Document:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM documents WHERE document_type = ? AND sha256 = ? "
                "AND parser_version = ?",
                (document.document_type, document.sha256, document.parser_version),
            ).fetchone()
            if existing:
                stored = Document.model_validate_json(existing["payload_json"])
            else:
                stored = document
                connection.execute(
                    "INSERT INTO documents VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored.document_id,
                        stored.document_type,
                        stored.filename,
                        stored.sha256,
                        stored.ingested_at,
                        stored.parser_version,
                        self._payload(stored),
                    ),
                )
                connection.executemany(
                    "INSERT INTO paragraphs(document_id, paragraph_id, sequence, payload_json) "
                    "VALUES(?, ?, ?, ?)",
                    [
                        (stored.document_id, item.paragraph_id, item.sequence, self._payload(item))
                        for item in paragraphs
                    ],
                )
            run_document = document.model_copy(update={"document_id": stored.document_id})
            connection.execute(
                "INSERT OR REPLACE INTO run_documents(run_id, document_type, document_id) VALUES(?, ?, ?)",
                (run_id, document.document_type, stored.document_id),
            )
            connection.execute(
                "INSERT OR REPLACE INTO run_document_metadata(run_id, document_type, payload_json) "
                "VALUES(?, ?, ?)",
                (run_id, document.document_type, self._payload(run_document)),
            )
        return run_document

    def load_documents(self, run_id: str) -> list[Document]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT COALESCE(rdm.payload_json, d.payload_json) AS payload_json "
                "FROM documents d JOIN run_documents rd ON rd.document_id = d.document_id "
                "LEFT JOIN run_document_metadata rdm ON rdm.run_id = rd.run_id "
                "AND rdm.document_type = rd.document_type "
                "WHERE rd.run_id = ? ORDER BY d.document_type",
                (run_id,),
            ).fetchall()
        return [Document.model_validate_json(row["payload_json"]) for row in rows]

    def load_paragraphs(self, run_id: str, document_type: str | None = None) -> list[Paragraph]:
        query = (
            "SELECT p.payload_json FROM paragraphs p JOIN run_documents rd "
            "ON rd.document_id = p.document_id WHERE rd.run_id = ?"
        )
        params: list[Any] = [run_id]
        if document_type:
            query += " AND rd.document_type = ?"
            params.append(document_type)
        query += " ORDER BY rd.document_type, p.sequence"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [Paragraph.model_validate_json(row["payload_json"]) for row in rows]

    def _replace_models(self, table: str, run_id: str, key_name: str, models: list[BaseModel]) -> None:
        with self.connect() as connection:
            connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            connection.executemany(
                f"INSERT INTO {table}(run_id, {key_name}, payload_json) VALUES(?, ?, ?)",
                [
                    (run_id, str(getattr(model, key_name)), self._payload(model))
                    for model in models
                ],
            )

    def _load_models(self, table: str, run_id: str, model: type[ModelT]) -> list[ModelT]:
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM {table} WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [model.model_validate_json(row["payload_json"]) for row in rows]

    def save_references(self, run_id: str, values: list[ReferencePlaceholder]) -> None:
        self._replace_models("references_found", run_id, "reference_id", values)

    def load_references(self, run_id: str) -> list[ReferencePlaceholder]:
        return self._load_models("references_found", run_id, ReferencePlaceholder)

    def save_issues(self, run_id: str, values: list[Issue]) -> None:
        self._replace_models("issues", run_id, "issue_id", values)

    def load_issues(self, run_id: str) -> list[Issue]:
        return self._load_models("issues", run_id, Issue)

    def save_propositions(self, run_id: str, values: list[AtomicProposition]) -> None:
        self._replace_models("propositions", run_id, "proposition_id", values)

    def load_propositions(self, run_id: str) -> list[AtomicProposition]:
        return self._load_models("propositions", run_id, AtomicProposition)

    def save_arguments(self, run_id: str, values: list[Argument]) -> None:
        self._replace_models("arguments", run_id, "argument_id", values)

    def upsert_argument(self, run_id: str, value: Argument) -> None:
        """Checkpoint one argument without rewriting every argument in the run."""

        with self.connect() as connection:
            connection.execute(
                "INSERT INTO arguments(run_id, argument_id, payload_json) VALUES(?, ?, ?) "
                "ON CONFLICT(run_id, argument_id) DO UPDATE SET payload_json = excluded.payload_json",
                (run_id, value.argument_id, self._payload(value)),
            )

    def load_arguments(self, run_id: str) -> list[Argument]:
        return self._load_models("arguments", run_id, Argument)

    def save_search_packages(self, run_id: str, values: list[CaseLawSearchPackage]) -> None:
        self._replace_models("search_packages", run_id, "argument_id", values)

    def load_search_packages(self, run_id: str) -> list[CaseLawSearchPackage]:
        return self._load_models("search_packages", run_id, CaseLawSearchPackage)

    def save_reviews(self, run_id: str, values: list[HumanReviewItem], append: bool = False) -> None:
        with self.connect() as connection:
            if not append:
                connection.execute("DELETE FROM human_reviews WHERE run_id = ?", (run_id,))
            connection.executemany(
                "INSERT OR REPLACE INTO human_reviews(run_id, review_id, payload_json) VALUES(?, ?, ?)",
                [(run_id, value.review_id, self._payload(value)) for value in values],
            )

    def delete_reviews_for_entities(
        self, run_id: str, gate: str, entity_ids: list[str]
    ) -> None:
        if not entity_ids:
            return
        placeholders = ",".join("?" for _ in entity_ids)
        with self.connect() as connection:
            connection.execute(
                f"DELETE FROM human_reviews WHERE run_id = ? AND json_extract(payload_json, '$.gate') = ? "
                f"AND json_extract(payload_json, '$.entity_id') IN ({placeholders})",
                [run_id, gate, *entity_ids],
            )

    def delete_reviews_for_gate(self, run_id: str, gate: str) -> None:
        """Remove obsolete review output before deterministically rebuilding a gate."""

        with self.connect() as connection:
            connection.execute(
                "DELETE FROM human_reviews WHERE run_id = ? "
                "AND json_extract(payload_json, '$.gate') = ?",
                (run_id, gate),
            )

    def load_reviews(self, run_id: str) -> list[HumanReviewItem]:
        return self._load_models("human_reviews", run_id, HumanReviewItem)

    def stage_events(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT stage, name, status, recorded_at, details_json FROM stage_events "
                "WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "stage": row["stage"],
                "name": row["name"],
                "status": row["status"],
                "recorded_at": row["recorded_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]
