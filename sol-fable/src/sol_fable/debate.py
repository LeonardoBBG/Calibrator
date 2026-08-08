"""Orchestrator-mediated, citation-bound barrister debate workflow."""

from __future__ import annotations

from typing import Protocol

from .agents._shared import confidence
from .models import (
    Argument,
    BarristerPersona,
    DebateSummary,
    DebateTurn,
    EvidenceLink,
    PartySide,
    PleadingStatus,
)
from .text_analysis import unique


class DebateBackend(Protocol):
    name: str

    def debate_turn(
        self,
        argument: Argument,
        round_number: int,
        barrister: BarristerPersona,
        side: PartySide,
        opponent_turn: DebateTurn | None,
        own_previous_turn: DebateTurn | None,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateTurn: ...

    def summarize_debate(
        self,
        argument: Argument,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        respondent_barrister: BarristerPersona,
        turns: list[DebateTurn],
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateSummary: ...


def other_barrister(barrister: BarristerPersona | str) -> BarristerPersona:
    selected = BarristerPersona(barrister)
    return BarristerPersona.FABLE if selected == BarristerPersona.SOL else BarristerPersona.SOL


def citations(argument: Argument) -> list[str]:
    return unique(argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs)


_citations = citations


def validate_pleading_evidence_links(
    evidence_links: list[EvidenceLink],
    pleading_paragraph_ids: set[str] | list[str],
    owner: str,
) -> None:
    """Keep ET1/ET3 citations as party-position context, never factual proof."""

    pleading_ids = set(pleading_paragraph_ids)
    mischaracterised = sorted(
        {
            paragraph_id
            for link in evidence_links
            if link.relationship != "CONTEXT"
            for paragraph_id in link.paragraph_ids
            if paragraph_id in pleading_ids
        }
    )
    if mischaracterised:
        raise ValueError(
            f"{owner} must label ET1/ET3 pleading paragraphs as CONTEXT only: "
            f"{mischaracterised}"
        )


def _claimant_points(argument: Argument) -> list[str]:
    values = [
        f"The formulation is anchored to the WS at {', '.join(argument.ws_paragraphs)}.",
    ]
    if argument.et1_paragraphs:
        values.append(f"The identified pleading basis is {', '.join(argument.et1_paragraphs)}.")
    if argument.et3_paragraphs:
        values.append(
            f"The contrary ET3 position at {', '.join(argument.et3_paragraphs)} remains a disputed party position."
        )
    return values


def _respondent_points(argument: Argument) -> list[str]:
    values = []
    if argument.et3_paragraphs:
        values.append(
            f"The Respondent can put the contrary pleaded position identified at {', '.join(argument.et3_paragraphs)}."
        )
    values.append("The proposition must be tested for pleading consistency, personal knowledge and evidential support.")
    if argument.et1_paragraphs:
        values.append(f"Any elaboration must remain faithful to {', '.join(argument.et1_paragraphs)}.")
    return values


def _evidence_gaps(argument: Argument) -> list[str]:
    values = [
        f"Unresolved documentary placeholder: {item}" for item in argument.document_placeholders
    ]
    if not argument.et1_paragraphs:
        values.append("No sufficiently strong ET1 paragraph link was identified automatically.")
    if not argument.et3_paragraphs:
        values.append("No sufficiently strong ET3 paragraph link was identified automatically.")
    return values


def _source_links(argument: Argument, position: str, side: PartySide) -> list[EvidenceLink]:
    """Describe source character without treating either pleading as evidence."""

    links: list[EvidenceLink] = []
    if argument.ws_paragraphs:
        links.append(
            EvidenceLink(
                claim=(
                    position
                    if side == PartySide.CLAIMANT
                    else "The WS records the claimant account being challenged."
                ),
                paragraph_ids=argument.ws_paragraphs,
                relationship="SUPPORTS" if side == PartySide.CLAIMANT else "CONTEXT",
            )
        )
    if argument.et1_paragraphs:
        links.append(
            EvidenceLink(
                claim="The ET1 supplies claimant pleading context, not proof.",
                paragraph_ids=argument.et1_paragraphs,
                relationship="CONTEXT",
            )
        )
    if argument.et3_paragraphs:
        links.append(
            EvidenceLink(
                claim="The ET3 supplies respondent pleading context, not proof.",
                paragraph_ids=argument.et3_paragraphs,
                relationship="CONTEXT",
            )
        )
    return links


def deterministic_debate_turn(
    argument: Argument,
    round_number: int,
    barrister: BarristerPersona,
    side: PartySide,
    opponent_turn: DebateTurn | None,
    own_previous_turn: DebateTurn | None,
    backend: str,
    prompt_version: str,
    paragraph_texts: dict[str, str] | None = None,
) -> DebateTurn:
    persona = BarristerPersona(barrister)
    party = PartySide(side)
    is_claimant = party == PartySide.CLAIMANT
    responses: list[str] = []
    if opponent_turn:
        opponent_focus = (
            opponent_turn.challenges_for_opponent[0]
            if opponent_turn.challenges_for_opponent
            else opponent_turn.position
        )
        responses.append(
            f"In response to {opponent_turn.turn_id}, address this without expanding beyond the cited record: {opponent_focus}"
        )
    if own_previous_turn:
        responses.append(
            f"Preserve the source-bound core of {own_previous_turn.turn_id} while dealing with any new challenge."
        )

    if is_claimant:
        position = argument.proposition
        supporting = _claimant_points(argument)
        challenges = [
            "Require the Respondent to identify evidence supporting any contrary ET3 allegation.",
            "Require any alternative inference to be tested against the same paragraph-level source map.",
        ]
        risks = []
        if PleadingStatus.POTENTIALLY_NEW_CASE in argument.pleading_statuses:
            risks.append("The formulation may be challenged as a materially new case rather than legitimate detail.")
        if argument.document_placeholders:
            risks.append("The Claimant cannot treat unresolved brackets as verified documentary support.")
        protect = [
            "Keep the proposition no broader than the cited WS wording.",
            "Preserve the distinction between personal recollection, inference and documentary proof.",
        ]
    else:
        position = (
            f"The Respondent disputes the Claimant proposition: {argument.proposition}"
            if argument.et3_paragraphs
            else "No aligned ET3 proposition was identified; the challenge must remain limited to testing the Claimant's proof."
        )
        supporting = _respondent_points(argument)
        challenges = [
            "Require a precise explanation of consistency between the WS formulation and the ET1.",
            "Test timing, personal knowledge, hindsight and the factual basis for every inference.",
        ]
        risks = ["An ET3 allegation is a party position and cannot be presented as proof."]
        if not argument.et3_paragraphs:
            risks.append("Inventing a positive Respondent account would exceed the available source material.")
        protect = [
            "Defend the strongest source-supported alternative without overstating the ET3.",
            "Keep cross-examination themes tied to exact paragraph citations.",
        ]

    if persona == BarristerPersona.SOL:
        supporting.append("SOL focuses on a coherent formulation and the factual findings it would require.")
    else:
        challenges.append("FABLE expressly tests overstatement, retrospective reconstruction and damaging concessions.")

    assumptions = [
        "The cited paragraphs concern the same material factual issue.",
        "The proposition's legal significance remains subject to later authority research and factual findings.",
    ]
    return DebateTurn(
        turn_id=f"{argument.argument_id}-R{round_number:02d}-{party.value}",
        argument_id=argument.argument_id,
        round_number=round_number,
        barrister=persona,
        side=party,
        opponent_turn_id=opponent_turn.turn_id if opponent_turn else None,
        position=position,
        supporting_points=unique(supporting),
        responses_to_opponent=unique(responses),
        challenges_for_opponent=unique(challenges),
        concessions_or_risks=unique(risks),
        hidden_assumptions=assumptions,
        evidence_gaps=_evidence_gaps(argument),
        protect_or_defend_points=protect,
        paragraph_citations=_citations(argument),
        evidence_links=_source_links(argument, position, party),
        confidence=confidence(argument),
        backend=backend,
        prompt_version=prompt_version,
    )


def deterministic_debate_summary(
    argument: Argument,
    n_rounds: int,
    claimant_barrister: BarristerPersona,
    respondent_barrister: BarristerPersona,
    turns: list[DebateTurn],
    paragraph_texts: dict[str, str] | None = None,
) -> DebateSummary:
    assumptions = unique(
        [item for turn in turns for item in turn.hidden_assumptions]
    )
    evidence_gaps = unique([item for turn in turns for item in turn.evidence_gaps])
    review_reasons: list[str] = []
    if argument.human_review_status == "PENDING":
        review_reasons.append("The underlying proposition already requires human review.")
    if argument.document_placeholders:
        review_reasons.append("Documentary placeholders must be resolved before they can support either side.")
    if PleadingStatus.POTENTIALLY_NEW_CASE in argument.pleading_statuses:
        review_reasons.append("Review legitimate factual elaboration versus a materially new case.")
    claimant_strength = (
        "HIGH"
        if argument.ws_paragraphs and argument.et1_paragraphs
        else "MEDIUM"
        if argument.ws_paragraphs
        else "LOW"
    )
    respondent_strength = (
        "HIGH"
        if PleadingStatus.POTENTIALLY_NEW_CASE in argument.pleading_statuses
        or not argument.et1_paragraphs
        else "MEDIUM"
        if argument.et3_paragraphs or argument.document_placeholders
        else "LOW"
    )
    if claimant_strength == "HIGH" and respondent_strength == "LOW":
        treatment = "PROTECT"
    elif respondent_strength == "HIGH" and claimant_strength != "HIGH":
        treatment = "DEFEND"
    else:
        treatment = "BOTH"
    source_links = _source_links(
        argument,
        "The witness statement records the claimant proposition under debate.",
        PartySide.CLAIMANT,
    )
    return DebateSummary(
        argument_id=argument.argument_id,
        n_rounds=n_rounds,
        claimant_barrister=claimant_barrister,
        respondent_barrister=respondent_barrister,
        turns=turns,
        methodological_agreements=[
            "Both barristers are confined to the same stable paragraph source map.",
            "Neither ET1 nor ET3 allegations are treated as proof merely because they are pleaded.",
            "Unresolved documentary placeholders do not increase evidential confidence.",
        ],
        remaining_disputes=unique(
            [
                f"Whether the fact-finder should accept the Claimant proposition: {argument.proposition}",
                "Whether the proposition is sufficiently supported and remains within the pleaded case.",
                *[
                    challenge
                    for turn in turns[-2:]
                    for challenge in turn.challenges_for_opponent
                ],
            ]
        )[:8],
        hidden_assumptions=assumptions,
        evidence_gaps=evidence_gaps,
        human_review_reasons=review_reasons,
        stop_reason=f"CONFIGURED_ROUND_LIMIT_REACHED:{n_rounds}",
        claimant_strength=claimant_strength,
        respondent_strength=respondent_strength,
        recommended_treatment=treatment,
        treatment_rationale=(
            "A bounded source-led synthesis of pleading alignment, the WS source map, "
            "the respondent pleading context, and unresolved evidence gaps."
        ),
        paragraph_citations=_citations(argument),
        evidence_links=source_links,
        backend="deterministic-v1",
        prompt_version=turns[0].prompt_version if turns else "unknown",
    )


def _validate_turn(
    argument: Argument,
    turn: DebateTurn,
    *,
    round_number: int,
    barrister: BarristerPersona,
    side: PartySide,
    opponent_turn: DebateTurn | None,
) -> None:
    if turn.argument_id != argument.argument_id:
        raise ValueError(f"Debate turn {turn.turn_id} refers to the wrong argument")
    expected_id = f"{argument.argument_id}-R{round_number:02d}-{side.value}"
    if turn.turn_id != expected_id:
        raise ValueError(f"Debate turn ID {turn.turn_id} should be {expected_id}")
    if turn.round_number != round_number or turn.side != side or turn.barrister != barrister:
        raise ValueError(f"Debate turn {turn.turn_id} changed orchestrator-owned role metadata")
    expected_opponent = opponent_turn.turn_id if opponent_turn else None
    if turn.opponent_turn_id != expected_opponent:
        raise ValueError(f"Debate turn {turn.turn_id} has the wrong opponent handoff")
    if opponent_turn is not None and not turn.responses_to_opponent:
        raise ValueError(f"Debate turn {turn.turn_id} did not answer the opponent turn")
    allowed = set(_citations(argument))
    unknown = set(turn.paragraph_citations) - allowed
    if unknown:
        raise ValueError(
            f"Debate turn {turn.turn_id} invented or imported unknown citations: {sorted(unknown)}"
        )
    evidence_ids = {
        paragraph_id
        for link in turn.evidence_links
        for paragraph_id in link.paragraph_ids
    }
    unknown_evidence = evidence_ids - allowed
    if unknown_evidence:
        raise ValueError(
            f"Debate turn {turn.turn_id} has unknown evidence links: {sorted(unknown_evidence)}"
        )
    if allowed and not turn.evidence_links:
        raise ValueError(f"Debate turn {turn.turn_id} has no claim-level evidence links")
    if evidence_ids != set(turn.paragraph_citations):
        raise ValueError(
            f"Debate turn {turn.turn_id} citations do not match its claim-level evidence links"
        )
    validate_pleading_evidence_links(
        turn.evidence_links,
        set(argument.et1_paragraphs + argument.et3_paragraphs),
        f"Debate turn {turn.turn_id}",
    )


def conduct_debate(
    argument: Argument,
    backend: DebateBackend,
    prompt_version: str,
    n_rounds: int,
    claimant_barrister: BarristerPersona,
    paragraph_texts: dict[str, str] | None = None,
) -> DebateSummary:
    if not 1 <= n_rounds <= 10:
        raise ValueError("n_rounds must be between 1 and 10")
    claimant = BarristerPersona(claimant_barrister)
    respondent = other_barrister(claimant)
    turns: list[DebateTurn] = []
    claimant_previous: DebateTurn | None = None
    respondent_previous: DebateTurn | None = None
    for round_number in range(1, n_rounds + 1):
        claimant_turn = backend.debate_turn(
            argument,
            round_number,
            claimant,
            PartySide.CLAIMANT,
            respondent_previous,
            claimant_previous,
            prompt_version,
            paragraph_texts,
        )
        _validate_turn(
            argument,
            claimant_turn,
            round_number=round_number,
            barrister=claimant,
            side=PartySide.CLAIMANT,
            opponent_turn=respondent_previous,
        )
        turns.append(claimant_turn)
        respondent_turn = backend.debate_turn(
            argument,
            round_number,
            respondent,
            PartySide.RESPONDENT,
            claimant_turn,
            respondent_previous,
            prompt_version,
            paragraph_texts,
        )
        _validate_turn(
            argument,
            respondent_turn,
            round_number=round_number,
            barrister=respondent,
            side=PartySide.RESPONDENT,
            opponent_turn=claimant_turn,
        )
        turns.append(respondent_turn)
        claimant_previous = claimant_turn
        respondent_previous = respondent_turn
    summary = backend.summarize_debate(
        argument,
        n_rounds,
        claimant,
        respondent,
        turns,
        paragraph_texts,
    )
    if [item.turn_id for item in summary.turns] != [item.turn_id for item in turns]:
        raise ValueError("Debate summary did not preserve the orchestrator's validated turn sequence")
    allowed = set(_citations(argument))
    summary_evidence_ids = {
        paragraph_id
        for link in summary.evidence_links
        for paragraph_id in link.paragraph_ids
    }
    unknown_summary = set(summary.paragraph_citations).union(summary_evidence_ids) - allowed
    if unknown_summary:
        raise ValueError(f"Debate summary contains unknown citations: {sorted(unknown_summary)}")
    if allowed and not summary.evidence_links:
        raise ValueError("Debate summary has no claim-level evidence links")
    if set(summary.paragraph_citations) != summary_evidence_ids:
        raise ValueError("Debate summary citations do not match its claim-level evidence links")
    validate_pleading_evidence_links(
        summary.evidence_links,
        set(argument.et1_paragraphs + argument.et3_paragraphs),
        "Debate summary",
    )
    if summary.recommended_treatment not in {"PROTECT", "DEFEND", "BOTH"}:
        raise ValueError("Debate summary did not provide a bounded treatment recommendation")
    required_review_reasons: list[str] = []
    if argument.human_review_status == "PENDING":
        required_review_reasons.append("The underlying proposition already requires human review.")
    if argument.document_placeholders:
        required_review_reasons.append(
            "Documentary placeholders must be resolved before they can support either side."
        )
    if PleadingStatus.POTENTIALLY_NEW_CASE in argument.pleading_statuses:
        required_review_reasons.append(
            "Review legitimate factual elaboration versus a materially new case."
        )
    return summary.model_copy(
        update={
            "argument_id": argument.argument_id,
            "n_rounds": n_rounds,
            "claimant_barrister": claimant,
            "respondent_barrister": respondent,
            "turns": turns,
            "human_review_reasons": unique(
                required_review_reasons + summary.human_review_reasons
            ),
            "stop_reason": f"CONFIGURED_ROUND_LIMIT_REACHED:{n_rounds}",
            "prompt_version": prompt_version,
        }
    )
