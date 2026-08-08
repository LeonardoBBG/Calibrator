from __future__ import annotations

import json
from pathlib import Path

from sol_fable.agents import assess_fable, assess_sol
from sol_fable.config import Thresholds
from sol_fable.ingest import ingest_document
from sol_fable.issue_builder import build_issues, respondent_stance
from sol_fable.models import (
    Argument,
    ArgumentCategory,
    ArgumentTreatment,
    AtomicProposition,
    DocumentType,
    EvidentialCharacter,
    Importance,
    Paragraph,
    PleadingStatus,
    RespondentPosition,
    SourceOrigin,
)
from sol_fable.orchestrator import PipelineOrchestrator
from sol_fable.pleading_matcher import match_propositions
from sol_fable.proposition_extractor import extract_propositions
from sol_fable.ranker import rank_arguments
from sol_fable.reference_parser import parse_references
from sol_fable.search_package_builder import build_search_packages


def paragraph(document_type: DocumentType, sequence: int, text: str, number: str | None = None) -> Paragraph:
    return Paragraph(
        paragraph_id=f"{document_type.value}-P{sequence:03d}",
        document_id=f"DOC-{document_type.value}-TEST",
        document_type=document_type,
        original_number=number or str(sequence),
        text=text,
        raw_text=f"{number or sequence}. {text}",
        section_heading=None,
        sequence=sequence,
    )


def test_at_001_stable_paragraph_ids(tmp_path: Path) -> None:
    source = tmp_path / "ET1.txt"
    source.write_text("1. First pleaded point.\n2. Second pleaded point.\n", encoding="utf-8")
    first_document, first_paragraphs = ingest_document(source, DocumentType.ET1, "0.1.0")
    second_document, second_paragraphs = ingest_document(source, DocumentType.ET1, "0.1.0")
    assert first_document.sha256 == second_document.sha256
    assert first_document.document_id == second_document.document_id
    assert [item.paragraph_id for item in first_paragraphs] == [item.paragraph_id for item in second_paragraphs]


def test_at_002_multiple_atomic_propositions() -> None:
    ws = [
        paragraph(
            DocumentType.WS,
            1,
            "I sent the email. I spoke to my manager. I continued the project work.",
        )
    ]
    propositions = extract_propositions(ws)
    assert [item.proposition_id for item in propositions] == ["WS-P001-C01", "WS-P001-C02", "WS-P001-C03"]
    assert {item.ws_paragraph_id for item in propositions} == {"WS-P001"}


def test_at_003_bracket_is_unresolved_document_placeholder() -> None:
    references = parse_references(
        [paragraph(DocumentType.WS, 1, "I received positive feedback [Performance review].")]
    )
    assert len(references) == 1
    assert references[0].raw_text == "[Performance review]"
    assert references[0].reference_type == "EVIDENCE_REFERENCE"
    assert references[0].resolved is False
    propositions = extract_propositions(
        [paragraph(DocumentType.WS, 1, "I received positive feedback [Performance review].")]
    )
    assert EvidentialCharacter.EVIDENCE_NOT_YET_INGESTED in propositions[0].evidential_character


def test_at_004_et1_ws_refinement_is_reviewed_not_declared_contradiction() -> None:
    et1 = [paragraph(DocumentType.ET1, 1, "The Claimant says no warning was given.")]
    et3 = [paragraph(DocumentType.ET3, 1, "The Respondent says a clear warning was given.")]
    ws = [
        paragraph(
            DocumentType.WS,
            1,
            "I had a discussion, but I did not understand it as a warning.",
        )
    ]
    propositions = extract_propositions(ws)
    matched, reviews = match_propositions("RUN-TEST", propositions, et1, et3, Thresholds())
    assert len(matched) == 1
    assert matched[0].pleading_status == PleadingStatus.ET1_DETAIL_OR_ELABORATION
    assert matched[0].requires_human_review
    assert any("formulation tension" in item.reason for item in reviews)


def test_at_005_et3_allegation_remains_a_party_position() -> None:
    et1 = [paragraph(DocumentType.ET1, 1, "The Claimant denies receiving an instruction to stop.")]
    et3 = [paragraph(DocumentType.ET3, 1, "The Respondent denies that account and alleges an instruction was given.")]
    issues = build_issues(et1, et3)
    assert issues[0].status == "DISPUTED"
    assert "Respondent" in issues[0].respondent_position
    assert respondent_stance(et3[0].text) == RespondentPosition.DENIED


def test_at_006_advantage_with_concession_risk_can_be_both() -> None:
    argument = Argument(
        argument_id="ARG-MIX-001",
        title="Discussion not understood as warning",
        proposition="A discussion occurred, but it was not understood as a warning.",
        category=ArgumentCategory.MIXED,
        importance=Importance.CRITICAL,
        issue_ids=["ISSUE-001"],
        ws_paragraphs=["WS-P001"],
        et1_paragraphs=["ET1-P001"],
        et3_paragraphs=["ET3-P001"],
        source_types=[SourceOrigin.ET1, SourceOrigin.ET3, SourceOrigin.SELF_ACCOUNT],
        document_placeholders=["[July communications]"],
        pleading_statuses=[PleadingStatus.PLEADED_ET1_EXPLICIT],
        proposition_ids=["WS-P001-C01"],
    )
    argument = argument.model_copy(
        update={
            "sol_assessment": assess_sol(argument, "deterministic-v1", "0.1.0"),
            "fable_assessment": assess_fable(argument, "deterministic-v1", "0.1.0"),
        }
    )
    weights = {
        "outcome_materiality": 0.30,
        "pleading_alignment": 0.15,
        "ws_support": 0.10,
        "respondent_attack_strength": 0.15,
        "cross_examination_vulnerability": 0.15,
        "case_law_calibration_value": 0.15,
    }
    ranked = rank_arguments([argument], weights)
    assert ranked[0].treatment == [ArgumentTreatment.BOTH]


def test_at_007_focused_search_package_has_required_dimensions() -> None:
    argument = Argument(
        argument_id="ARG-MIX-001",
        title="Continued conduct after alleged warning",
        proposition="The conduct continued for eight months after the alleged warning without formal action.",
        category=ArgumentCategory.MIXED,
        importance=Importance.CRITICAL,
        issue_ids=["ISSUE-001"],
        ws_paragraphs=["WS-P001"],
        et1_paragraphs=["ET1-P001"],
        et3_paragraphs=["ET3-P001"],
        source_types=[SourceOrigin.SELF_ACCOUNT],
        document_placeholders=[],
        pleading_statuses=[PleadingStatus.ET1_DETAIL_OR_ELABORATION],
        proposition_ids=["WS-P001-C01"],
    )
    package = build_search_packages([argument])[0]
    assert package.legal_dimensions
    assert package.factual_analogues
    assert package.broad_queries
    assert package.targeted_queries
    assert package.questions_for_cases


def test_at_008_completed_run_has_full_audit_trace(tmp_path: Path) -> None:
    et1 = tmp_path / "ET1.txt"
    et3 = tmp_path / "ET3.txt"
    ws = tmp_path / "WS.txt"
    et1.write_text(
        "1. The Claimant says no warning was given about ticket routing.\n"
        "2. The dismissal was unfair because the investigation ignored the Claimant's explanation.\n",
        encoding="utf-8",
    )
    et3.write_text(
        "1. The Respondent denies that no warning was given and says the Claimant was instructed to stop.\n"
        "2. The Respondent denies an unfair process and says a reasonable investigation occurred.\n",
        encoding="utf-8",
    )
    ws.write_text(
        "1. In July I had a discussion, but I did not understand it as a warning [July 2023 communications].\n"
        "2. I continued the ticket work for eight months. My manager knew it continued.\n"
        "3. At the disciplinary meeting I explained that project work had priority.\n",
        encoding="utf-8",
    )
    orchestrator = PipelineOrchestrator(project_dir=tmp_path / "standalone")
    run_id = orchestrator.run_all(et1, et3, ws)
    report_path = orchestrator.reports_for(run_id) / "ws_calibration_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["run"]["run_id"] == run_id
    assert payload["run"]["status"] == "COMPLETED"
    assert len(payload["audit_trail"]) == 11
    assert payload["arguments"]
    first = payload["arguments"][0]
    assert first["ws_paragraphs"]
    assert first["sol_assessment"]["backend"] == "deterministic-v1"
    assert first["fable_assessment"]["paragraph_citations"]
    assert payload["run"]["config_snapshot"]["n_rounds"] == 2
    assert payload["run"]["config_snapshot"]["claimant_barrister"] == "SOL"
    assert payload["run"]["config_snapshot"]["respondent_barrister"] == "FABLE"
    assert first["debate_summary"]["n_rounds"] == 2
    assert len(first["debate_summary"]["turns"]) == 4
    assert len(payload["documents"]) == 3
    assert all(len(item["sha256"]) == 64 for item in payload["documents"])
