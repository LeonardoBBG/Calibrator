"""Transparent, source-led ranking and PROTECT/DEFEND/BOTH treatment."""

from __future__ import annotations

import re

from .models import (
    Argument,
    ArgumentTreatment,
    Importance,
    PleadingStatus,
    RankingComponents,
    SourceOrigin,
)

IMPORTANCE_SCORE = {
    Importance.CRITICAL: 1.0,
    Importance.HIGH: 0.82,
    Importance.MEDIUM: 0.58,
    Importance.LOW: 0.32,
}
ALIGNMENT_SCORE = {
    PleadingStatus.PLEADED_ET1_EXPLICIT: 1.0,
    PleadingStatus.PLEADED_ET1_IMPLICIT: 0.88,
    PleadingStatus.ET1_DETAIL_OR_ELABORATION: 0.68,
    PleadingStatus.COMMON_GROUND: 0.82,
    PleadingStatus.RESPONDS_TO_ET3: 0.55,
    PleadingStatus.ET3_ONLY_ALLEGATION: 0.30,
    PleadingStatus.POTENTIALLY_NEW_CASE: 0.12,
    PleadingStatus.NOT_MATERIAL_TO_PLEADINGS: 0.20,
    PleadingStatus.UNRESOLVED: 0.25,
}

_NORMALISED_LABELS = {
    "VERY HIGH": 0.96,
    "CRITICAL": 0.96,
    "HIGH": 0.86,
    "SEVERE": 0.86,
    "STRONG": 0.86,
    "MEDIUM": 0.60,
    "MODERATE": 0.60,
    "MIXED": 0.60,
    "LOW": 0.30,
    "WEAK": 0.30,
    "MINIMAL": 0.18,
    "NONE": 0.12,
    "NEGLIGIBLE": 0.12,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalised_label(value: str | None) -> float | None:
    """Translate explicit assessment labels without interpreting free-form prose."""

    if not value:
        return None
    cleaned = re.sub(r"[^A-Z]+", " ", value.upper()).strip()
    if cleaned in _NORMALISED_LABELS:
        return _NORMALISED_LABELS[cleaned]
    # Structured-output providers occasionally add a noun (for example
    # ``HIGH RISK``). Accept a leading canonical label, but do not infer a risk
    # score from arbitrary generated sentences.
    for label in ("VERY HIGH", "CRITICAL", "HIGH", "SEVERE", "STRONG", "MEDIUM", "MODERATE", "LOW", "WEAK", "MINIMAL", "NONE", "NEGLIGIBLE"):
        if cleaned.startswith(f"{label} "):
            return _NORMALISED_LABELS[label]
    return None


def _blend_assessment(
    source_score: float,
    label: str | None,
    confidence: float | None,
    *,
    maximum_influence: float,
) -> float:
    """Use model labels as a bounded calibration, never as the source score."""

    assessment_score = _normalised_label(label)
    if assessment_score is None:
        return source_score
    reliable = _clamp(confidence if confidence is not None else 0.5, 0.25, 0.95)
    influence = maximum_influence * reliable
    return _clamp((source_score * (1.0 - influence)) + (assessment_score * influence))


def _components(argument: Argument) -> RankingComponents:
    statuses = {PleadingStatus(value) for value in argument.pleading_statuses}
    source_types = {SourceOrigin(value) for value in argument.source_types}

    alignment_values = [ALIGNMENT_SCORE[status] for status in statuses]
    alignment = sum(alignment_values) / len(alignment_values) if alignment_values else 0.20
    if PleadingStatus.POTENTIALLY_NEW_CASE in statuses:
        # A safe status on one merged proposition must not conceal a new-case
        # concern on another proposition in the same argument.
        alignment = min(alignment, 0.52)
    if not argument.et1_paragraphs and PleadingStatus.COMMON_GROUND not in statuses:
        alignment = min(alignment, 0.42)

    # A WS paragraph is evidence of what the witness says, not independent
    # corroboration. Source character therefore matters more than paragraph count.
    if SourceOrigin.AGREED_FACT in source_types:
        ws_support = 0.82
    elif SourceOrigin.SELF_ACCOUNT in source_types:
        ws_support = 0.68
    elif SourceOrigin.OTHER_WITNESS in source_types:
        ws_support = 0.58
    elif SourceOrigin.UNKNOWN in source_types:
        ws_support = 0.40
    else:
        ws_support = 0.52
    if SourceOrigin.INFERENCE in source_types:
        ws_support -= 0.08
    if SourceOrigin.LEGAL_SUBMISSION in source_types:
        ws_support -= 0.10
    if argument.document_placeholders or SourceOrigin.DOCUMENT_PLACEHOLDER in source_types:
        ws_support -= 0.14
    if not argument.et1_paragraphs and PleadingStatus.COMMON_GROUND not in statuses:
        ws_support -= 0.08
    if argument.human_review_status == "PENDING":
        ws_support -= 0.06
    ws_support = _clamp(ws_support, 0.15, 0.90)
    if argument.sol_assessment:
        ws_support = _blend_assessment(
            ws_support,
            argument.sol_assessment.strength,
            argument.sol_assessment.confidence,
            maximum_influence=0.12,
        )

    common_ground_only = statuses == {PleadingStatus.COMMON_GROUND}
    respondent_attack = 0.18 if common_ground_only else 0.28
    if argument.et3_paragraphs and not common_ground_only:
        respondent_attack += 0.14
    if not argument.et1_paragraphs and not common_ground_only:
        respondent_attack += 0.18
    if PleadingStatus.POTENTIALLY_NEW_CASE in statuses:
        respondent_attack += 0.32
    elif PleadingStatus.ET3_ONLY_ALLEGATION in statuses:
        respondent_attack += 0.24
    elif PleadingStatus.UNRESOLVED in statuses:
        respondent_attack += 0.16
    elif PleadingStatus.RESPONDS_TO_ET3 in statuses:
        respondent_attack += 0.10
    if argument.document_placeholders:
        respondent_attack += 0.10
    if argument.human_review_status == "PENDING":
        respondent_attack += 0.07
    respondent_attack = _clamp(respondent_attack, 0.12, 0.95)

    vulnerability = 0.25 if common_ground_only else 0.30
    if SourceOrigin.SELF_ACCOUNT in source_types:
        vulnerability += 0.06
    if SourceOrigin.UNKNOWN in source_types:
        vulnerability += 0.18
    if SourceOrigin.INFERENCE in source_types:
        vulnerability += 0.12
    if SourceOrigin.LEGAL_SUBMISSION in source_types:
        vulnerability += 0.10
    if argument.et3_paragraphs and not common_ground_only:
        vulnerability += 0.07
    if not argument.et1_paragraphs and not common_ground_only:
        vulnerability += 0.12
    if argument.document_placeholders:
        vulnerability += 0.14
    if PleadingStatus.POTENTIALLY_NEW_CASE in statuses:
        vulnerability += 0.25
    elif PleadingStatus.ET3_ONLY_ALLEGATION in statuses:
        vulnerability += 0.18
    elif PleadingStatus.UNRESOLVED in statuses:
        vulnerability += 0.14
    if argument.human_review_status == "PENDING":
        vulnerability += 0.08
    vulnerability = _clamp(vulnerability, 0.12, 0.97)

    # These explicit, confidence-bearing labels are deliberately bounded. The
    # number or length of generated bullets never enters the score, so local and
    # hosted backends remain comparable.
    if argument.fable_assessment:
        respondent_attack = _blend_assessment(
            respondent_attack,
            argument.fable_assessment.risk_level,
            argument.fable_assessment.confidence,
            maximum_influence=0.18,
        )
        vulnerability = _blend_assessment(
            vulnerability,
            argument.fable_assessment.risk_level,
            argument.fable_assessment.confidence,
            maximum_influence=0.14,
        )
    if argument.sol_assessment:
        respondent_attack = _blend_assessment(
            respondent_attack,
            argument.sol_assessment.new_case_risk,
            argument.sol_assessment.confidence,
            maximum_influence=0.08,
        )
    if argument.debate_summary:
        turn_confidences = [turn.confidence for turn in argument.debate_summary.turns]
        debate_confidence = (
            sum(turn_confidences) / len(turn_confidences) if turn_confidences else 0.5
        )
        ws_support = _blend_assessment(
            ws_support,
            argument.debate_summary.claimant_strength,
            debate_confidence,
            maximum_influence=0.12,
        )
        respondent_attack = _blend_assessment(
            respondent_attack,
            argument.debate_summary.respondent_strength,
            debate_confidence,
            maximum_influence=0.12,
        )
        vulnerability = _blend_assessment(
            vulnerability,
            argument.debate_summary.respondent_strength,
            debate_confidence,
            maximum_influence=0.08,
        )

    combined = argument.proposition.lower()
    case_value = 0.82 if any(
        term in combined
        for term in ("warning", "instruction", "dishon", "investigat", "appeal", "dismiss", "misconduct")
    ) else 0.55
    return RankingComponents(
        outcome_materiality=IMPORTANCE_SCORE[Importance(argument.importance)],
        pleading_alignment=round(alignment, 4),
        ws_support=round(ws_support, 4),
        respondent_attack_strength=round(respondent_attack, 4),
        cross_examination_vulnerability=round(vulnerability, 4),
        case_law_calibration_value=case_value,
    )


def rank_arguments(arguments: list[Argument], weights: dict[str, float]) -> list[Argument]:
    ranked: list[Argument] = []
    for argument in arguments:
        components = _components(argument)
        component_data = components.model_dump()
        score = round(100 * sum(component_data[name] * weight for name, weight in weights.items()), 1)

        # Treatment describes direction, while ranking_score and importance describe
        # priority. Keeping those concepts separate prevents a low-scoring helpful
        # point from being relabelled merely because it is less material.
        claimant_value = (
            (components.pleading_alignment * 0.56)
            + (components.ws_support * 0.44)
        )
        exposure = (
            (components.respondent_attack_strength * 0.52)
            + (components.cross_examination_vulnerability * 0.48)
        )
        debate_recommendation = (
            argument.debate_summary.recommended_treatment
            if argument.debate_summary
            else None
        )
        # The mediated verdict is a bounded calibration, not an override of the
        # paragraph-led score. It can resolve a close case, but cannot turn a
        # clearly unsupported proposition into a protected point by itself.
        if debate_recommendation == ArgumentTreatment.PROTECT:
            claimant_value = _clamp(claimant_value + 0.035)
        elif debate_recommendation == ArgumentTreatment.DEFEND:
            exposure = _clamp(exposure + 0.035)
        elif debate_recommendation == ArgumentTreatment.BOTH:
            claimant_value = _clamp(claimant_value + 0.02)
            exposure = _clamp(exposure + 0.02)
        if claimant_value >= 0.58 and exposure >= 0.58:
            treatment = [ArgumentTreatment.BOTH]
        elif claimant_value >= 0.50 and claimant_value >= exposure:
            treatment = [ArgumentTreatment.PROTECT]
        else:
            treatment = [ArgumentTreatment.DEFEND]

        ranked.append(
            argument.model_copy(
                update={
                    "treatment": treatment,
                    "ranking_score": score,
                    "ranking_components": components,
                }
            )
        )
    return sorted(ranked, key=lambda item: (-(item.ranking_score or 0), item.argument_id))


def top_arguments_markdown(arguments: list[Argument]) -> str:
    lines = ["# Ranked arguments", "", "Scores are transparent priorities, not legal conclusions.", ""]
    for argument in arguments:
        lines.extend(
            [
                f"## {argument.argument_id}: {argument.title}",
                "",
                f"- Score: {argument.ranking_score}/100",
                f"- Category: {argument.category}",
                f"- Treatment: {', '.join(argument.treatment)}",
                f"- Importance: {argument.importance}",
                f"- Sources: {', '.join(argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs)}",
                "",
            ]
        )
    return "\n".join(lines)
