from __future__ import annotations

from sol_fable.argument_builder import build_arguments
from sol_fable.models import (
    Argument,
    ArgumentCategory,
    ArgumentTreatment,
    AtomicProposition,
    BarristerPersona,
    DebateSummary,
    DebateTurn,
    EvidentialCharacter,
    Importance,
    Issue,
    MatchType,
    ParagraphMatch,
    PartySide,
    PleadingStatus,
    ReferencePlaceholder,
    ReferenceType,
    RespondentPosition,
    SourceOrigin,
)
from sol_fable.ranker import rank_arguments


WEIGHTS = {
    "outcome_materiality": 0.30,
    "pleading_alignment": 0.15,
    "ws_support": 0.10,
    "respondent_attack_strength": 0.15,
    "cross_examination_vulnerability": 0.15,
    "case_law_calibration_value": 0.15,
}


def _argument(
    argument_id: str,
    *,
    statuses: list[PleadingStatus],
    source_types: list[SourceOrigin],
    et1: list[str],
    et3: list[str],
    placeholders: list[str],
    review_status: str,
) -> Argument:
    return Argument(
        argument_id=argument_id,
        title=argument_id,
        proposition="The manager knew that the work continued after the alleged warning.",
        category=ArgumentCategory.SUBSTANTIVE,
        importance=Importance.HIGH,
        issue_ids=["ISSUE-001"],
        ws_paragraphs=["WS-P001"],
        et1_paragraphs=et1,
        et3_paragraphs=et3,
        source_types=source_types,
        document_placeholders=placeholders,
        pleading_statuses=statuses,
        proposition_ids=[f"WS-P001-{argument_id}"],
        human_review_status=review_status,
    )


def test_source_led_treatment_can_protect_defend_and_do_both() -> None:
    protect = _argument(
        "ARG-SUB-001",
        statuses=[PleadingStatus.PLEADED_ET1_EXPLICIT],
        source_types=[SourceOrigin.SELF_ACCOUNT, SourceOrigin.ET1],
        et1=["ET1-P001"],
        et3=[],
        placeholders=[],
        review_status="NOT_REQUIRED",
    )
    defend = _argument(
        "ARG-SUB-002",
        statuses=[PleadingStatus.POTENTIALLY_NEW_CASE],
        source_types=[SourceOrigin.UNKNOWN, SourceOrigin.DOCUMENT_PLACEHOLDER, SourceOrigin.ET3],
        et1=[],
        et3=["ET3-P001"],
        placeholders=["[unverified email]"],
        review_status="PENDING",
    )
    both = _argument(
        "ARG-SUB-003",
        statuses=[PleadingStatus.PLEADED_ET1_EXPLICIT],
        source_types=[
            SourceOrigin.SELF_ACCOUNT,
            SourceOrigin.DOCUMENT_PLACEHOLDER,
            SourceOrigin.ET1,
            SourceOrigin.ET3,
        ],
        et1=["ET1-P001"],
        et3=["ET3-P001"],
        placeholders=["[unverified email]"],
        review_status="PENDING",
    )

    ranked = {item.argument_id: item for item in rank_arguments([protect, defend, both], WEIGHTS)}

    assert ranked[protect.argument_id].treatment == [ArgumentTreatment.PROTECT]
    assert ranked[defend.argument_id].treatment == [ArgumentTreatment.DEFEND]
    assert ranked[both.argument_id].treatment == [ArgumentTreatment.BOTH]


def _debate_summary(argument_id: str, bullet_count: int) -> DebateSummary:
    turn = DebateTurn(
        turn_id=f"{argument_id}-R01-RESPONDENT",
        argument_id=argument_id,
        round_number=1,
        barrister=BarristerPersona.FABLE,
        side=PartySide.RESPONDENT,
        position="The proposition should be tested against its sources.",
        supporting_points=[f"support {index}" for index in range(bullet_count)],
        responses_to_opponent=[f"response {index}" for index in range(bullet_count)],
        challenges_for_opponent=[f"challenge {index}" for index in range(bullet_count)],
        concessions_or_risks=[f"risk {index}" for index in range(bullet_count)],
        hidden_assumptions=[],
        evidence_gaps=[],
        protect_or_defend_points=[],
        paragraph_citations=["WS-P001"],
        confidence=0.7,
        backend="test",
        prompt_version="test",
    )
    return DebateSummary(
        argument_id=argument_id,
        n_rounds=1,
        claimant_barrister=BarristerPersona.SOL,
        respondent_barrister=BarristerPersona.FABLE,
        turns=[turn],
        methodological_agreements=[],
        remaining_disputes=[],
        hidden_assumptions=[],
        evidence_gaps=[],
        human_review_reasons=[],
        stop_reason="round limit",
    )


def test_generated_bullet_volume_does_not_change_ranking() -> None:
    base = _argument(
        "ARG-SUB-001",
        statuses=[PleadingStatus.PLEADED_ET1_EXPLICIT],
        source_types=[SourceOrigin.SELF_ACCOUNT, SourceOrigin.ET1, SourceOrigin.ET3],
        et1=["ET1-P001"],
        et3=["ET3-P001"],
        placeholders=[],
        review_status="NOT_REQUIRED",
    )
    terse = base.model_copy(update={"debate_summary": _debate_summary(base.argument_id, 1)})
    verbose = base.model_copy(update={"debate_summary": _debate_summary(base.argument_id, 20)})

    terse_ranked = rank_arguments([terse], WEIGHTS)[0]
    verbose_ranked = rank_arguments([verbose], WEIGHTS)[0]

    assert terse_ranked.ranking_score == verbose_ranked.ranking_score
    assert terse_ranked.ranking_components == verbose_ranked.ranking_components
    assert terse_ranked.treatment == verbose_ranked.treatment

    reranked = rank_arguments([terse_ranked], WEIGHTS)[0]
    assert reranked.ranking_score == terse_ranked.ranking_score
    assert reranked.ranking_components == terse_ranked.ranking_components
    assert reranked.treatment == terse_ranked.treatment


def test_bounded_mediated_verdict_can_resolve_a_close_treatment() -> None:
    base = _argument(
        "ARG-SUB-010",
        statuses=[PleadingStatus.RESPONDS_TO_ET3],
        source_types=[SourceOrigin.SELF_ACCOUNT],
        et1=[],
        et3=[],
        placeholders=[],
        review_status="NOT_REQUIRED",
    )
    summary = _debate_summary(base.argument_id, 1)
    helpful = base.model_copy(
        update={
            "debate_summary": summary.model_copy(
                update={
                    "claimant_strength": "VERY_HIGH",
                    "respondent_strength": "LOW",
                    "recommended_treatment": "PROTECT",
                }
            )
        }
    )
    damaging = base.model_copy(
        update={
            "debate_summary": summary.model_copy(
                update={
                    "claimant_strength": "LOW",
                    "respondent_strength": "VERY_HIGH",
                    "recommended_treatment": "DEFEND",
                }
            )
        }
    )

    helpful_ranked = rank_arguments([helpful], WEIGHTS)[0]
    damaging_ranked = rank_arguments([damaging], WEIGHTS)[0]

    assert helpful_ranked.treatment == [ArgumentTreatment.PROTECT]
    assert damaging_ranked.treatment == [ArgumentTreatment.DEFEND]
    assert helpful_ranked.ranking_components != damaging_ranked.ranking_components


def _proposition(
    proposition_id: str,
    ws_id: str,
    text: str,
    et1_id: str,
) -> AtomicProposition:
    return AtomicProposition(
        proposition_id=proposition_id,
        ws_paragraph_id=ws_id,
        text=text,
        source_origins=[SourceOrigin.SELF_ACCOUNT],
        pleading_status=PleadingStatus.PLEADED_ET1_EXPLICIT,
        et1_matches=[
            ParagraphMatch(
                paragraph_id=et1_id,
                match_type=MatchType.DIRECT_SEMANTIC,
                confidence=0.9,
                respondent_position=RespondentPosition.DENIED,
            )
        ],
        evidential_character=[EvidentialCharacter.PERSONAL_RECOLLECTION],
        materiality=Importance.HIGH,
    )


def test_consolidation_preserves_traceability_and_opposite_meaning() -> None:
    issue = Issue(
        issue_id="ISSUE-001",
        title="Whether a written warning was received",
        claimant_position="The Claimant did not receive a written warning.",
        respondent_position="The Respondent says a written warning was received.",
        et1_paragraphs=["ET1-P001"],
        et3_paragraphs=["ET3-P001"],
        category=ArgumentCategory.SUBSTANTIVE,
        status="DISPUTED",
    )
    first = _proposition(
        "WS-P001-C01",
        "WS-P001",
        "I did not receive any written warning about ticket routing.",
        "ET1-P001",
    )
    repeated = _proposition(
        "WS-P002-C01",
        "WS-P002",
        "I did not receive a written warning about ticket routing.",
        "ET1-P001",
    )
    opposite = _proposition(
        "WS-P003-C01",
        "WS-P003",
        "I did receive a written warning about ticket routing.",
        "ET1-P001",
    )
    reference = ReferencePlaceholder(
        reference_id="REF-001",
        ws_paragraph_id="WS-P002",
        raw_text="[warning email]",
        reference_type=ReferenceType.EVIDENCE_REFERENCE,
    )

    arguments = build_arguments([first, repeated, opposite], [issue], [reference])

    assert len(arguments) == 2
    merged = next(item for item in arguments if first.proposition_id in item.proposition_ids)
    unmerged = next(item for item in arguments if opposite.proposition_id in item.proposition_ids)
    assert merged.proposition_ids == [first.proposition_id, repeated.proposition_id]
    assert merged.ws_paragraphs == ["WS-P001", "WS-P002"]
    assert merged.et1_paragraphs == ["ET1-P001"]
    assert merged.document_placeholders == ["[warning email]"]
    assert unmerged.proposition_ids == [opposite.proposition_id]


def test_consolidation_merges_same_issue_and_source_map_across_paragraphs() -> None:
    issue = Issue(
        issue_id="ISSUE-002",
        title="Explanation at the disciplinary meeting",
        claimant_position="The Claimant explained that project work had priority.",
        respondent_position="The Respondent disputes that explanation.",
        et1_paragraphs=["ET1-P002"],
        et3_paragraphs=["ET3-P002"],
        category=ArgumentCategory.PROCEDURAL,
        status="DISPUTED",
    )
    first = _proposition(
        "WS-P010-C01",
        "WS-P010",
        "At the disciplinary meeting I explained that project work had priority.",
        "ET1-P002",
    )
    reformulation = _proposition(
        "WS-P011-C01",
        "WS-P011",
        "At the disciplinary meeting I said the project work was my priority.",
        "ET1-P002",
    )
    unrelated = _proposition(
        "WS-P012-C01",
        "WS-P012",
        "I sent the payroll spreadsheet on Friday.",
        "ET1-P099",
    )

    arguments = build_arguments([first, reformulation, unrelated], [issue], [])

    assert len(arguments) == 2
    merged = next(item for item in arguments if first.proposition_id in item.proposition_ids)
    assert merged.proposition_ids == [first.proposition_id, reformulation.proposition_id]
    assert unrelated.proposition_id in {
        proposition_id
        for argument in arguments
        for proposition_id in argument.proposition_ids
    }


def test_paragraph_units_are_bounded_complete_and_stable() -> None:
    issue = Issue(
        issue_id="ISSUE-020",
        title="Conduct of the meeting",
        claimant_position="The meeting did not fairly examine the explanation.",
        respondent_position="The meeting fairly examined the explanation.",
        et1_paragraphs=["ET1-P020"],
        et3_paragraphs=["ET3-P020"],
        category=ArgumentCategory.PROCEDURAL,
        status="DISPUTED",
    )
    texts = [
        "The meeting began with an explanation about the work allocation.",
        "The chair asked who had authorised the task.",
        "The manager described the expected approval route.",
        "A colleague confirmed the team allocation.",
        "The minutes recorded the central explanation.",
        "The outcome letter omitted the colleague's confirmation.",
        "The appeal later considered the missing confirmation.",
    ]
    propositions = [
        _proposition(
            f"WS-P020-C{index:02d}",
            "WS-P020",
            text,
            f"ET1-P{20 + index:03d}",
        )
        for index, text in enumerate(texts, start=1)
    ]
    references = [
        ReferencePlaceholder(
            reference_id="REF-020-B",
            ws_paragraph_id="WS-P020",
            raw_text="[outcome letter]",
            reference_type=ReferenceType.EVIDENCE_REFERENCE,
        ),
        ReferencePlaceholder(
            reference_id="REF-020-A",
            ws_paragraph_id="WS-P020",
            raw_text="[meeting minutes]",
            reference_type=ReferenceType.EVIDENCE_REFERENCE,
        ),
    ]

    forward = build_arguments(propositions, [issue], references)
    reversed_input = build_arguments(
        list(reversed(propositions)), [issue], list(reversed(references))
    )

    assert [item.model_dump() for item in forward] == [
        item.model_dump() for item in reversed_input
    ]
    assert len(forward) == 2
    assert max(len(item.proposition_ids) for item in forward) == 5
    assert [
        proposition_id
        for argument in forward
        for proposition_id in argument.proposition_ids
    ] == [item.proposition_id for item in propositions]
    assert {
        paragraph_id
        for argument in forward
        for paragraph_id in argument.et1_paragraphs
    }.issuperset({f"ET1-P{20 + index:03d}" for index in range(1, 8)})
    assert all(argument.ws_paragraphs == ["WS-P020"] for argument in forward)
    assert all(
        argument.document_placeholders == ["[meeting minutes]", "[outcome letter]"]
        for argument in forward
    )


def test_paragraph_unit_keeps_risk_and_legal_profiles_separate() -> None:
    issue = Issue(
        issue_id="ISSUE-030",
        title="Authority for the work",
        claimant_position="The work was authorised.",
        respondent_position="The work was not authorised.",
        et1_paragraphs=["ET1-P030"],
        et3_paragraphs=["ET3-P030"],
        category=ArgumentCategory.SUBSTANTIVE,
        status="DISPUTED",
    )
    pleaded = _proposition(
        "WS-P030-C01",
        "WS-P030",
        "The manager allocated the work to me.",
        "ET1-P030",
    )
    response = _proposition(
        "WS-P030-C02",
        "WS-P030",
        "I explained the allocation in response to the allegation.",
        "ET1-P030",
    ).model_copy(update={"pleading_status": PleadingStatus.RESPONDS_TO_ET3})
    new_case = _proposition(
        "WS-P030-C03",
        "WS-P030",
        "A different manager later approved another task.",
        "ET1-P030",
    ).model_copy(update={"pleading_status": PleadingStatus.POTENTIALLY_NEW_CASE})
    legal_submission = _proposition(
        "WS-P030-C04",
        "WS-P030",
        "The allocation amounted to contractual authority.",
        "ET1-P030",
    ).model_copy(
        update={
            "source_origins": [SourceOrigin.LEGAL_SUBMISSION],
            "evidential_character": [EvidentialCharacter.LEGAL_CHARACTERISATION],
        }
    )

    arguments = build_arguments(
        [pleaded, response, new_case, legal_submission], [issue], []
    )

    assert len(arguments) == 3
    assert any(
        argument.proposition_ids == [pleaded.proposition_id, response.proposition_id]
        for argument in arguments
    )
    assert any(
        argument.proposition_ids == [new_case.proposition_id] for argument in arguments
    )
    assert any(
        argument.proposition_ids == [legal_submission.proposition_id]
        for argument in arguments
    )


def test_same_et3_admission_and_denial_are_not_hidden_in_one_unit() -> None:
    issue = Issue(
        issue_id="ISSUE-040",
        title="Receipt of the report",
        claimant_position="The report was received.",
        respondent_position="Receipt is disputed.",
        et1_paragraphs=["ET1-P040"],
        et3_paragraphs=["ET3-P040"],
        category=ArgumentCategory.SUBSTANTIVE,
        status="DISPUTED",
    )
    admitted = _proposition(
        "WS-P040-C01",
        "WS-P040",
        "The Respondent received the report from the Claimant.",
        "ET1-P040",
    ).model_copy(
        update={
            "et3_matches": [
                ParagraphMatch(
                    paragraph_id="ET3-P040",
                    match_type=MatchType.DIRECT_SEMANTIC,
                    confidence=0.9,
                    respondent_position=RespondentPosition.ADMITTED,
                )
            ]
        }
    )
    denied = _proposition(
        "WS-P040-C02",
        "WS-P040",
        "The Respondent received the same report that afternoon.",
        "ET1-P040",
    ).model_copy(
        update={
            "et3_matches": [
                ParagraphMatch(
                    paragraph_id="ET3-P040",
                    match_type=MatchType.DIRECT_SEMANTIC,
                    confidence=0.9,
                    respondent_position=RespondentPosition.DENIED,
                )
            ]
        }
    )

    arguments = build_arguments([admitted, denied], [issue], [])

    assert len(arguments) == 2
    assert [argument.proposition_ids for argument in arguments] == [
        [admitted.proposition_id],
        [denied.proposition_id],
    ]
