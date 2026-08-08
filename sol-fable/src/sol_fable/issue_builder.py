"""Build a neutral issue registry from the ET1 and ET3 party positions."""

from __future__ import annotations

import re

from .models import Issue, Paragraph, RespondentPosition
from .text_analysis import classify_category, concise_title, lexical_similarity


def respondent_stance(text: str) -> RespondentPosition:
    lowered = text.lower()
    if "partly admit" in lowered or "admitted in part" in lowered:
        return RespondentPosition.PARTLY_ADMITTED
    if re.search(r"\bnot admitted\b|\brequire(?:s)? proof\b", lowered):
        return RespondentPosition.NOT_ADMITTED
    if re.search(r"\bden(?:y|ies|ied)\b", lowered):
        return RespondentPosition.DENIED
    if re.search(r"\badmit(?:s|ted)?\b|\bagreed\b|\bcommon ground\b", lowered):
        return RespondentPosition.ADMITTED
    if re.search(r"\binstead\b|\bthe respondent'?s case\b|\bin fact\b|\balternatively\b", lowered):
        return RespondentPosition.ALTERNATIVE_ACCOUNT
    return RespondentPosition.RESPONDENT_ALLEGATION


def build_issues(et1: list[Paragraph], et3: list[Paragraph]) -> list[Issue]:
    issues: list[Issue] = []
    used_et3: set[str] = set()
    for claimant_paragraph in et1:
        candidates = sorted(
            ((lexical_similarity(claimant_paragraph.text, item.text), item) for item in et3),
            key=lambda pair: (-pair[0], pair[1].sequence),
        )
        score, respondent_paragraph = candidates[0] if candidates else (0.0, None)
        linked_et3: list[str] = []
        response = "Not addressed in the ET3."
        status = "UNANSWERED"
        if respondent_paragraph is not None and score >= 0.16:
            linked_et3 = [respondent_paragraph.paragraph_id]
            used_et3.add(respondent_paragraph.paragraph_id)
            stance = respondent_stance(respondent_paragraph.text)
            response = respondent_paragraph.text
            status = "COMMON_GROUND" if stance == RespondentPosition.ADMITTED else "DISPUTED"
        issues.append(
            Issue(
                issue_id=f"ISSUE-{len(issues) + 1:03d}",
                title=concise_title(claimant_paragraph.text),
                claimant_position=claimant_paragraph.text,
                respondent_position=response,
                et1_paragraphs=[claimant_paragraph.paragraph_id],
                et3_paragraphs=linked_et3,
                category=classify_category(claimant_paragraph.text, response),
                status=status,
            )
        )
    for respondent_paragraph in et3:
        if respondent_paragraph.paragraph_id in used_et3:
            continue
        issues.append(
            Issue(
                issue_id=f"ISSUE-{len(issues) + 1:03d}",
                title=concise_title(respondent_paragraph.text),
                claimant_position="No aligned ET1 paragraph was identified automatically.",
                respondent_position=respondent_paragraph.text,
                et1_paragraphs=[],
                et3_paragraphs=[respondent_paragraph.paragraph_id],
                category=classify_category(respondent_paragraph.text),
                status="ET3_ONLY_POSITION",
            )
        )
    return issues


def pleadings_map_markdown(issues: list[Issue]) -> str:
    lines = ["# Pleadings issue map", "", "Party positions are allegations or admissions, not proof.", ""]
    for issue in issues:
        lines.extend(
            [
                f"## {issue.issue_id}: {issue.title}",
                "",
                f"- Category: {issue.category}",
                f"- Status: {issue.status}",
                f"- ET1: {', '.join(issue.et1_paragraphs) or 'none'} — {issue.claimant_position}",
                f"- ET3: {', '.join(issue.et3_paragraphs) or 'none'} — {issue.respondent_position}",
                "",
            ]
        )
    return "\n".join(lines)

