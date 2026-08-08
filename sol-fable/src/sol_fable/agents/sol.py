"""Deterministic claimant-side assessment, constrained to linked source IDs."""

from __future__ import annotations

from . import _shared  # type: ignore[attr-defined]
from ..models import Argument, EvidenceLink, PleadingStatus, SolAssessment


def assess_sol(
    argument: Argument,
    backend: str,
    prompt_version: str,
    paragraph_texts: dict[str, str] | None = None,
) -> SolAssessment:
    citations = _shared.citations(argument)
    statuses = set(argument.pleading_statuses)
    aligned = bool(statuses.intersection({
        PleadingStatus.PLEADED_ET1_EXPLICIT,
        PleadingStatus.PLEADED_ET1_IMPLICIT,
        PleadingStatus.ET1_DETAIL_OR_ELABORATION,
    }))
    new_case = PleadingStatus.POTENTIALLY_NEW_CASE in statuses
    strength = "HIGH" if aligned and argument.et1_paragraphs else "MEDIUM" if citations else "LOW"
    basis = (
        f"The formulation is linked to {', '.join(argument.et1_paragraphs)}."
        if argument.et1_paragraphs
        else "No sufficiently strong ET1 paragraph link was identified automatically."
    )
    assumptions = [
        "The cited WS wording is accurate and remains consistent under challenge.",
        "Any disputed party position requires a factual finding and is not treated as proof.",
    ]
    if argument.document_placeholders:
        assumptions.append("The unresolved documentary placeholders may or may not support the proposition once checked.")
    risks = []
    if new_case:
        risks.append("The proposition may be characterised as a materially new case rather than factual elaboration.")
    if not argument.et1_paragraphs:
        risks.append("The current automated source map does not establish an ET1 basis.")
    return SolAssessment(
        argument_id=argument.argument_id,
        best_formulation=argument.proposition,
        pleading_basis=basis,
        claimant_relevance=(
            "Preserving this proposition may support the Claimant's account on a material disputed issue, subject to the Tribunal's findings."
        ),
        required_findings=[
            f"The fact-finder accepts the account recorded at {', '.join(argument.ws_paragraphs)}.",
            "The proposition is relevant to at least one pleaded issue and is not a materially different case.",
        ],
        hidden_assumptions=assumptions,
        strength=strength,
        new_case_risk="HIGH" if new_case else "LOW" if aligned else "MEDIUM",
        protect_points=[
            "Keep the wording no broader than the cited WS paragraph.",
            "Maintain the distinction between recollection, inference and unresolved documents.",
        ],
        risks=risks,
        paragraph_citations=citations,
        evidence_links=[
            *(
                [
                    EvidenceLink(
                        claim="The witness statement records this proposition.",
                        paragraph_ids=argument.ws_paragraphs,
                        relationship="SUPPORTS",
                    )
                ]
                if argument.ws_paragraphs
                else []
            ),
            *(
                [
                    EvidenceLink(
                        claim="The ET1 is the pleading context for the proposition, not proof of it.",
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
                        claim="The ET3 supplies the disputed respondent pleading context.",
                        paragraph_ids=argument.et3_paragraphs,
                        relationship="CONTEXT",
                    )
                ]
                if argument.et3_paragraphs
                else []
            ),
        ],
        confidence=_shared.confidence(argument),
        backend=backend,
        prompt_version=prompt_version,
    )
