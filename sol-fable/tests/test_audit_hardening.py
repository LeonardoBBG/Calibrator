"""Regression tests for provenance, state invalidation and local-data safety."""

from __future__ import annotations

import csv
import sqlite3
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_fable.analysis_backend import DeterministicAssessmentBackend
from sol_fable.errors import ConfigurationError, PipelineStateError
from sol_fable.ingest import ingest_document
from sol_fable.io_utils import write_csv
from sol_fable.llm_backend import prompt_bundle_sha256
from sol_fable.models import (
    Argument,
    ArgumentCategory,
    DocumentType,
    Importance,
    PleadingStatus,
    RunRecord,
    SourceOrigin,
)
from sol_fable.orchestrator import PipelineOrchestrator
from sol_fable.storage import CaseStore


FIXTURES = Path(__file__).parent / "fixtures"


def _build_to_arguments(orchestrator: PipelineOrchestrator) -> str:
    run_id = orchestrator.start_run()
    orchestrator.ingest(
        FIXTURES / "ET1.txt",
        FIXTURES / "ET3.txt",
        FIXTURES / "WS.txt",
        run_id,
    )
    orchestrator.parse_references(run_id)
    orchestrator.build_issues(run_id)
    orchestrator.extract_propositions(run_id)
    orchestrator.match_pleadings(run_id)
    orchestrator.build_arguments(run_id)
    return run_id


class _AuditedLiveStub(DeterministicAssessmentBackend):
    is_live = True
    supports_argument_cap = True

    def __init__(self, generation_revision: int):
        self.name = "audited-live-stub"
        self.generation_revision = generation_revision

    def metadata(self):
        return {
            "type": "test-live",
            "name": self.name,
            "live": True,
            "model": "test-model",
            "generation_revision": self.generation_revision,
        }


def test_live_provenance_cap_accounting_fingerprint_and_true_noop(tmp_path: Path) -> None:
    backend = _AuditedLiveStub(generation_revision=1)
    orchestrator = PipelineOrchestrator(tmp_path / "project", backend=backend)
    run_id = _build_to_arguments(orchestrator)

    orchestrator.assess(run_id, n_rounds=1, live_argument_cap=1)
    run = orchestrator.store.load_run(run_id)
    arguments = orchestrator.store.load_arguments(run_id)
    assert run.prompt_version == "0.3.0"
    assert run.config_snapshot["prompt_version"] == "0.3.0"
    assert len(run.config_snapshot["prompt_bundle_sha256"]) == 64
    assert run.config_snapshot["planned_live_arguments"] == 1
    assert run.config_snapshot["planned_provider_calls"] == 5
    assert run.config_snapshot["assessment_backend_counts"][
        "SOL:audited-live-stub"
    ] == 1
    assert run.config_snapshot["assessment_backend_counts"][
        "FABLE:audited-live-stub"
    ] == 1
    assert sum(item.assessment_mode == "LIVE" for item in arguments) == 1
    assert sum(item.assessment_mode == "DETERMINISTIC_FALLBACK" for item in arguments) == (
        len(arguments) - 1
    )

    event_count = len(orchestrator.store.stage_events(run_id))
    orchestrator.assess(run_id, n_rounds=1, live_argument_cap=1)
    assert len(orchestrator.store.stage_events(run_id)) == event_count

    # The visible name is unchanged, but content-affecting metadata differs and
    # must invalidate the live-owned argument's fingerprint.
    orchestrator.backend = _AuditedLiveStub(generation_revision=2)
    orchestrator.assess(run_id, n_rounds=1, live_argument_cap=1)
    assert orchestrator.store.stage_events(run_id)[-1]["details"][
        "newly_processed_arguments"
    ] == 1


def test_prompt_version_must_resolve_exact_bundle() -> None:
    assert len(prompt_bundle_sha256("0.3.0")) == 64
    with pytest.raises(ConfigurationError, match="not bundled"):
        prompt_bundle_sha256("9.9.9")


def test_completed_run_inputs_are_immutable_and_upstream_rerun_invalidates(tmp_path: Path) -> None:
    orchestrator = PipelineOrchestrator(
        tmp_path / "project", backend=DeterministicAssessmentBackend()
    )
    run_id = orchestrator.run_all(
        FIXTURES / "ET1.txt",
        FIXTURES / "ET3.txt",
        FIXTURES / "WS.txt",
    )
    original_arguments = orchestrator.store.load_arguments(run_id)

    with pytest.raises(PipelineStateError, match="immutable"):
        orchestrator.ingest(
            FIXTURES / "ET1.txt",
            FIXTURES / "ET3.txt",
            FIXTURES / "WS.txt",
            run_id,
        )
    assert orchestrator.store.load_run(run_id).status == "COMPLETED"
    assert orchestrator.store.load_arguments(run_id) == original_arguments

    orchestrator.extract_propositions(run_id)
    assert orchestrator.store.load_run(run_id).status == "RUNNING"
    assert orchestrator.store.load_arguments(run_id) == []
    assert orchestrator.store.load_search_packages(run_id) == []
    with pytest.raises(PipelineStateError, match="built arguments"):
        orchestrator.report(run_id)


def test_parser_version_is_part_of_cached_parse_identity(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "case.sqlite")
    for suffix in ("ONE", "TWO"):
        store.create_run(
            RunRecord(
                run_id=f"RUN-{suffix}",
                started_at="2026-01-01T00:00:00+00:00",
                status="RUNNING",
                backend="deterministic-v1",
                prompt_version="0.3.0",
                config_snapshot={},
            )
        )
    first, first_paragraphs = ingest_document(
        FIXTURES / "ET1.txt", DocumentType.ET1, "parser-v1"
    )
    second, second_paragraphs = ingest_document(
        FIXTURES / "ET1.txt", DocumentType.ET1, "parser-v2"
    )
    assert first.sha256 == second.sha256
    assert first.document_id != second.document_id

    stored_first = store.save_document("RUN-ONE", first, first_paragraphs)
    stored_second = store.save_document("RUN-TWO", second, second_paragraphs)
    assert stored_first.document_id != stored_second.document_id
    assert {item.document_id for item in store.load_paragraphs("RUN-ONE")} == {
        stored_first.document_id
    }
    assert {item.document_id for item in store.load_paragraphs("RUN-TWO")} == {
        stored_second.document_id
    }


def test_v2_database_migrates_without_losing_cached_paragraphs(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite"
    document, paragraphs = ingest_document(
        FIXTURES / "ET1.txt", DocumentType.ET1, "parser-v1"
    )
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_info VALUES(2, '2026-01-01T00:00:00+00:00');
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
                status TEXT NOT NULL, backend TEXT NOT NULL, prompt_version TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY, document_type TEXT NOT NULL,
                filename TEXT NOT NULL, sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL,
                parser_version TEXT NOT NULL, payload_json TEXT NOT NULL,
                UNIQUE(document_type, sha256)
            );
            CREATE TABLE run_documents (
                run_id TEXT NOT NULL REFERENCES runs(run_id), document_type TEXT NOT NULL,
                document_id TEXT NOT NULL REFERENCES documents(document_id),
                PRIMARY KEY (run_id, document_type)
            );
            CREATE TABLE run_document_metadata (
                run_id TEXT NOT NULL REFERENCES runs(run_id), document_type TEXT NOT NULL,
                payload_json TEXT NOT NULL, PRIMARY KEY (run_id, document_type)
            );
            CREATE TABLE paragraphs (
                document_id TEXT NOT NULL REFERENCES documents(document_id),
                paragraph_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL, PRIMARY KEY (document_id, paragraph_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                "RUN-LEGACY",
                "2026-01-01T00:00:00+00:00",
                None,
                "RUNNING",
                "deterministic-v1",
                "0.3.0",
                "{}",
            ),
        )
        connection.execute(
            "INSERT INTO documents VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.document_type,
                document.filename,
                document.sha256,
                document.ingested_at,
                document.parser_version,
                document.model_dump_json(),
            ),
        )
        connection.executemany(
            "INSERT INTO paragraphs VALUES(?, ?, ?, ?)",
            [
                (
                    document.document_id,
                    paragraph.paragraph_id,
                    paragraph.sequence,
                    paragraph.model_dump_json(),
                )
                for paragraph in paragraphs
            ],
        )
        connection.execute(
            "INSERT INTO run_documents VALUES(?, ?, ?)",
            ("RUN-LEGACY", "ET1", document.document_id),
        )

    migrated = CaseStore(database)

    assert migrated.load_documents("RUN-LEGACY")[0].parser_version == "parser-v1"
    assert len(migrated.load_paragraphs("RUN-LEGACY", "ET1")) == len(paragraphs)
    with migrated.connect() as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
    assert version == 3


def test_storage_revalidates_model_copy_updates(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "case.sqlite")
    store.create_run(
        RunRecord(
            run_id="RUN-VALIDATE",
            started_at="2026-01-01T00:00:00+00:00",
            status="RUNNING",
            backend="deterministic-v1",
            prompt_version="0.3.0",
            config_snapshot={},
        )
    )
    argument = Argument(
        argument_id="ARG-SUB-001",
        title="A source-bound point",
        proposition="The witness records the point.",
        category=ArgumentCategory.SUBSTANTIVE,
        importance=Importance.HIGH,
        issue_ids=["ISSUE-001"],
        ws_paragraphs=["WS-P001"],
        et1_paragraphs=["ET1-P001"],
        et3_paragraphs=[],
        source_types=[SourceOrigin.SELF_ACCOUNT, SourceOrigin.ET1],
        document_placeholders=[],
        pleading_statuses=[PleadingStatus.PLEADED_ET1_EXPLICIT],
        proposition_ids=["WS-P001-C01"],
    )
    invalid = argument.model_copy(update={"ranking_score": 999.0})
    with pytest.raises(ValidationError):
        store.save_arguments("RUN-VALIDATE", [invalid])


def test_csv_formula_cells_are_escaped_and_case_store_is_private(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    write_csv(path, [{"reason": "=HYPERLINK(\"bad\")"}], ["reason"])
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["reason"].startswith("'=")

    store = CaseStore(tmp_path / "private" / "case.sqlite")
    if stat.S_IMODE(store.path.stat().st_mode):
        assert stat.S_IMODE(store.path.stat().st_mode) & 0o077 == 0
        assert stat.S_IMODE(store.path.parent.stat().st_mode) & 0o077 == 0
