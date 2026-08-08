from __future__ import annotations

import pytest

from sol_fable.analysis_backend import DeterministicAssessmentBackend
from sol_fable.config import load_settings
from sol_fable.debate import conduct_debate
from sol_fable.models import (
    Argument,
    ArgumentCategory,
    BarristerPersona,
    EvidenceLink,
    Importance,
    PartySide,
    PleadingStatus,
    SourceOrigin,
)


def argument() -> Argument:
    return Argument(
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
        pleading_statuses=[PleadingStatus.ET1_DETAIL_OR_ELABORATION],
        proposition_ids=["WS-P001-C01"],
        human_review_status="PENDING",
    )


def test_default_debate_configuration_is_two_rounds() -> None:
    settings = load_settings()

    assert settings.n_rounds == 2
    assert settings.claimant_barrister == BarristerPersona.SOL


def test_debate_is_mediated_in_claimant_then_respondent_order() -> None:
    summary = conduct_debate(
        argument(),
        DeterministicAssessmentBackend(),
        "0.2.0",
        n_rounds=2,
        claimant_barrister=BarristerPersona.SOL,
    )

    assert summary.n_rounds == 2
    assert len(summary.turns) == 4
    assert [turn.side for turn in summary.turns] == [
        PartySide.CLAIMANT,
        PartySide.RESPONDENT,
        PartySide.CLAIMANT,
        PartySide.RESPONDENT,
    ]
    assert summary.turns[1].opponent_turn_id == summary.turns[0].turn_id
    assert summary.turns[2].opponent_turn_id == summary.turns[1].turn_id
    assert summary.turns[3].opponent_turn_id == summary.turns[2].turn_id
    assert summary.turns[2].responses_to_opponent
    assert summary.turns[3].responses_to_opponent
    assert all(
        set(turn.paragraph_citations) <= {"ET1-P001", "ET3-P001", "WS-P001"}
        for turn in summary.turns
    )
    assert summary.recommended_treatment in {"PROTECT", "DEFEND", "BOTH"}
    assert summary.stop_reason == "CONFIGURED_ROUND_LIMIT_REACHED:2"
    assert {
        paragraph_id
        for link in summary.evidence_links
        for paragraph_id in link.paragraph_ids
    } == set(summary.paragraph_citations)


def test_claimant_barrister_switch_assigns_other_persona_to_respondent() -> None:
    summary = conduct_debate(
        argument(),
        DeterministicAssessmentBackend(),
        "0.2.0",
        n_rounds=3,
        claimant_barrister=BarristerPersona.FABLE,
    )

    assert summary.claimant_barrister == BarristerPersona.FABLE
    assert summary.respondent_barrister == BarristerPersona.SOL
    assert len(summary.turns) == 6
    assert all(
        turn.barrister == BarristerPersona.FABLE
        for turn in summary.turns
        if turn.side == PartySide.CLAIMANT
    )
    assert all(
        turn.barrister == BarristerPersona.SOL
        for turn in summary.turns
        if turn.side == PartySide.RESPONDENT
    )


class _IgnoringBackend(DeterministicAssessmentBackend):
    def debate_turn(self, *args, **kwargs):
        turn = super().debate_turn(*args, **kwargs)
        if turn.opponent_turn_id:
            return turn.model_copy(update={"responses_to_opponent": []})
        return turn


def test_turn_with_opponent_must_provide_a_response() -> None:
    with pytest.raises(ValueError, match="did not answer"):
        conduct_debate(
            argument(),
            _IgnoringBackend(),
            "0.3.0",
            n_rounds=1,
            claimant_barrister=BarristerPersona.SOL,
        )


class _PleadingAsProofBackend(DeterministicAssessmentBackend):
    def debate_turn(self, *args, **kwargs):
        turn = super().debate_turn(*args, **kwargs)
        if turn.side == PartySide.CLAIMANT:
            return turn.model_copy(
                update={
                    "paragraph_citations": ["ET3-P001"],
                    "evidence_links": [
                        EvidenceLink(
                            claim="The pleaded allegation proves the event.",
                            paragraph_ids=["ET3-P001"],
                            relationship="SUPPORTS",
                        )
                    ],
                }
            )
        return turn


def test_pleading_paragraphs_cannot_be_labelled_as_factual_support() -> None:
    with pytest.raises(ValueError, match="CONTEXT only"):
        conduct_debate(
            argument(),
            _PleadingAsProofBackend(),
            "0.3.0",
            n_rounds=1,
            claimant_barrister=BarristerPersona.SOL,
        )
