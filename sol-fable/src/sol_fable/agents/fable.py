"""Deterministic respondent-side stress test, constrained to linked sources."""

from __future__ import annotations

from . import _shared  # type: ignore[attr-defined]
from ..models import Argument, EvidenceLink, FableAssessment, PleadingStatus


def assess_fable(
    argument: Argument,
    backend: str,
    prompt_version: str,
    paragraph_texts: dict[str, str] | None = None,
) -> FableAssessment:
    citations = _shared.citations(argument)
    statuses = set(argument.pleading_statuses)
    new_case = PleadingStatus.POTENTIALLY_NEW_CASE in statuses
    unsupported = not argument.et1_paragraphs
    placeholder_only_risk = bool(argument.document_placeholders)
    risk_level = "HIGH" if new_case or unsupported else "MEDIUM" if placeholder_only_risk else "LOW"
    pleading_attack = (
        "Challenge whether this is legitimate factual detail or a materially new case not identifiable in the ET1."
        if new_case or unsupported
        else "Test whether the WS formulation remains within, and is consistent with, the linked ET1 position."
    )
    evidence_attack = (
        "The cited brackets are unresolved placeholders and do not verify their suggested documentary content."
        if placeholder_only_risk
        else "Test the proposition against contemporaneous evidence; the present source map does not independently prove the account."
    )
    respondent_case = (
        f"The linked ET3 position at {', '.join(argument.et3_paragraphs)} should be put as a disputed party account, not assumed true."
        if argument.et3_paragraphs
        else "No aligned ET3 paragraph was identified; avoid inventing a respondent position."
    )
    return FableAssessment(
        argument_id=argument.argument_id,
        respondent_case=respondent_case,
        pleading_attack=pleading_attack,
        evidence_attack=evidence_attack,
        alternative_explanation=(
            "A decision-maker could draw a different inference from the same events; that inference must be tested against the cited sources."
        ),
        damaging_concessions=[
            f"An unqualified acceptance or retreat from the wording at {', '.join(argument.ws_paragraphs)} could weaken this point.",
            "Treating an unresolved document reference as verified could undermine credibility.",
        ],
        cross_examination_risk=(
            "Test precise wording, timing, personal knowledge, consistency with the ET1, and the basis for any inference."
        ),
        defend_points=[
            "Be able to identify the exact source for each factual component.",
            "Do not overstate what the ET1, ET3 or unresolved brackets establish.",
        ],
        risk_level=risk_level,
        hidden_assumptions=[
            "The linked ET3 wording is sufficiently connected to the same factual issue.",
            "A plausible alternative inference is not itself evidence that the proposition is false.",
        ],
        paragraph_citations=citations,
        evidence_links=[
            *(
                [
                    EvidenceLink(
                        claim="The ET3 supplies the respondent's pleaded position, not proof.",
                        paragraph_ids=argument.et3_paragraphs,
                        relationship="CONTEXT",
                    )
                ]
                if argument.et3_paragraphs
                else []
            ),
            *(
                [
                    EvidenceLink(
                        claim="The ET1 defines the claimant pleading against which overreach is tested.",
                        paragraph_ids=argument.et1_paragraphs,
                        relationship="CONTEXT",
                    )
                ]
                if argument.et1_paragraphs
                else []
            ),
            *(
                [
                    EvidenceLink(
                        claim="The WS records the account being stress-tested.",
                        paragraph_ids=argument.ws_paragraphs,
                        relationship="CONTEXT",
                    )
                ]
                if argument.ws_paragraphs
                else []
            ),
        ],
        confidence=_shared.confidence(argument),
        backend=backend,
        prompt_version=prompt_version,
    )
