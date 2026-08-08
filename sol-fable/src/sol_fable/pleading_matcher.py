"""Traceable proposition-to-pleading matching with explicit review gates."""

from __future__ import annotations

import re

from .config import Thresholds
from .issue_builder import respondent_stance
from .models import (
    AtomicProposition,
    HumanReviewItem,
    Importance,
    MatchType,
    Paragraph,
    ParagraphMatch,
    PleadingStatus,
)
from .text_analysis import lexical_similarity


def _explicit_ids(text: str, document_type: str, paragraphs: list[Paragraph]) -> list[str]:
    numbers = re.findall(
        rf"\b{document_type}\s*(?:para(?:graph)?\s*)?(\d+(?:\.\d+)*)\b",
        text,
        re.IGNORECASE,
    )
    by_original = {paragraph.original_number: paragraph.paragraph_id for paragraph in paragraphs}
    return [by_original[number] for number in numbers if number in by_original]


def _matches(
    proposition: AtomicProposition,
    paragraphs: list[Paragraph],
    document_type: str,
    thresholds: Thresholds,
) -> list[ParagraphMatch]:
    explicit = _explicit_ids(proposition.text, document_type, paragraphs)
    if explicit:
        return [
            ParagraphMatch(paragraph_id=item, match_type=MatchType.EXACT_REFERENCE, confidence=1.0)
            for item in explicit
        ]
    candidates = sorted(
        ((lexical_similarity(proposition.text, item.text), item) for item in paragraphs),
        key=lambda pair: (-pair[0], pair[1].sequence),
    )
    output: list[ParagraphMatch] = []
    for similarity, paragraph in candidates[:3]:
        if similarity < thresholds.lexical_candidate_floor:
            continue
        confidence = round(min(0.99, 0.35 + (similarity * 0.65)), 3)
        match_type = (
            MatchType.DIRECT_SEMANTIC if confidence >= thresholds.high_confidence
            else MatchType.CONTEXTUAL if confidence >= thresholds.medium_confidence
            else MatchType.ISSUE_LEVEL
        )
        output.append(
            ParagraphMatch(
                paragraph_id=paragraph.paragraph_id,
                match_type=match_type,
                confidence=confidence,
                respondent_position=respondent_stance(paragraph.text)
                if document_type == "ET3"
                else None,
            )
        )
    return output


def _warning_refinement_tension(text: str, et1_matches: list[ParagraphMatch], et1: list[Paragraph]) -> bool:
    if not re.search(r"\b(discussion|conversation|meeting)\b", text, re.IGNORECASE):
        return False
    if not re.search(r"\b(not|never|did not|wasn't|was not)\b.*\b(warning|instruction)\b", text, re.IGNORECASE):
        return False
    by_id = {paragraph.paragraph_id: paragraph for paragraph in et1}
    return any(
        re.search(r"\b(no|not|never)\b.*\b(warning|instruction)\b", by_id[item.paragraph_id].text, re.IGNORECASE)
        for item in et1_matches
        if item.paragraph_id in by_id
    )


def match_propositions(
    run_id: str,
    propositions: list[AtomicProposition],
    et1: list[Paragraph],
    et3: list[Paragraph],
    thresholds: Thresholds,
) -> tuple[list[AtomicProposition], list[HumanReviewItem]]:
    matched: list[AtomicProposition] = []
    reviews: list[HumanReviewItem] = []
    for proposition in propositions:
        et1_matches = _matches(proposition, et1, "ET1", thresholds)
        et3_matches = _matches(proposition, et3, "ET3", thresholds)
        reasons: list[str] = []
        explicit = any(item.match_type == MatchType.EXACT_REFERENCE for item in et1_matches)
        best_et1 = et1_matches[0].confidence if et1_matches else 0.0
        best_et3 = et3_matches[0].confidence if et3_matches else 0.0
        tension = _warning_refinement_tension(proposition.text, et1_matches, et1)
        if explicit:
            status = PleadingStatus.PLEADED_ET1_EXPLICIT
        elif tension:
            status = PleadingStatus.ET1_DETAIL_OR_ELABORATION
            reasons.append(
                "Possible ET1-to-WS formulation tension: a discussion is acknowledged but its status as a warning or instruction is disputed."
            )
        elif best_et1 >= thresholds.high_confidence:
            status = PleadingStatus.PLEADED_ET1_IMPLICIT
        elif best_et1 >= thresholds.possible_new_case_below:
            status = PleadingStatus.ET1_DETAIL_OR_ELABORATION
        elif best_et3 >= thresholds.possible_new_case_below:
            status = PleadingStatus.RESPONDS_TO_ET3
        elif proposition.materiality in (Importance.CRITICAL, Importance.HIGH):
            status = PleadingStatus.POTENTIALLY_NEW_CASE
            reasons.append("Material WS proposition has no sufficiently similar ET1 position; review for legitimate elaboration versus a new case.")
        else:
            status = PleadingStatus.NOT_MATERIAL_TO_PLEADINGS
        best_available = max(best_et1, best_et3)
        if (et1_matches or et3_matches) and best_available < thresholds.human_review_below:
            reasons.append("The best automated pleading match is below the configured human-review threshold.")
        if status == PleadingStatus.POTENTIALLY_NEW_CASE and not reasons:
            reasons.append("Every potentially new case classification requires human review.")
        updated = proposition.model_copy(
            update={
                "pleading_status": status,
                "et1_matches": et1_matches,
                "et3_matches": et3_matches,
                "requires_human_review": bool(reasons),
                "review_reasons": reasons,
            }
        )
        matched.append(updated)
        for reason in reasons:
            reviews.append(
                HumanReviewItem(
                    review_id=f"REV-B-{len(reviews) + 1:04d}",
                    run_id=run_id,
                    gate="B",
                    entity_type="PROPOSITION",
                    entity_id=proposition.proposition_id,
                    reason=reason,
                    confidence=best_available or None,
                )
            )
    return matched, reviews

