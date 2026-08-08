"""Human-readable and machine-readable audited report generation."""

from __future__ import annotations

from typing import Any

from .models import (
    Argument,
    ArgumentCategory,
    ArgumentTreatment,
    CaseLawSearchPackage,
    Document,
    HumanReviewItem,
    Issue,
    PleadingStatus,
    ReferencePlaceholder,
    RunRecord,
)


_TREATMENT_EFFECT = {
    ArgumentTreatment.PROTECT: "HELPS THE WS — preserve this point",
    ArgumentTreatment.DEFEND: "DAMAGES / EXPOSES THE WS — prepare a response",
    ArgumentTreatment.BOTH: "MIXED — helpful but materially vulnerable",
    ArgumentTreatment.SECONDARY: "SECONDARY — lower-priority point",
    ArgumentTreatment.DISTRACTING: "DISTRACTING — may dilute the WS",
    ArgumentTreatment.UNSUPPORTED: "DAMAGES THE WS — presently unsupported",
}


def _argument_effect(argument: Argument) -> str:
    if not argument.treatment:
        return "NOT YET RANKED"
    return "; ".join(
        _TREATMENT_EFFECT.get(ArgumentTreatment(value), str(value))
        for value in argument.treatment
    )


def _evidence_link_lines(prefix: str, assessment: Any) -> list[str]:
    links = getattr(assessment, "evidence_links", [])
    if not links:
        return [f"- {prefix} claim-to-source links: none recorded"]
    return [
        f"- {prefix} source link ({link.relationship}): {link.claim} "
        f"[{', '.join(link.paragraph_ids)}]"
        for link in links
    ]


def _argument_summary(argument: Argument) -> list[str]:
    citations = argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs
    return [
        f"### {argument.argument_id}: {argument.title}",
        "",
        argument.proposition,
        "",
        f"- Category: {argument.category}",
        f"- Effect on WS: {_argument_effect(argument)}",
        f"- Treatment: {', '.join(argument.treatment)}",
        f"- Importance / score: {argument.importance} / {argument.ranking_score}",
        f"- Analysis mode: {argument.assessment_mode or 'NOT ASSESSED'}",
        f"- Source paragraphs: {', '.join(citations) or 'none'}",
        f"- Source classification: {', '.join(argument.source_types)}",
        f"- Documentary placeholders: {', '.join(argument.document_placeholders) or 'none'}",
        f"- Pleading status: {', '.join(argument.pleading_statuses)}",
        f"- Human review: {argument.human_review_status}",
        "",
    ]


def _debate_lines(argument: Argument) -> list[str]:
    summary = argument.debate_summary
    if not summary:
        return ["- No mediated barrister debate is stored for this argument.", ""]
    lines = [
        f"- Claimant barrister: {summary.claimant_barrister}",
        f"- Respondent barrister: {summary.respondent_barrister}",
        f"- Configured rounds: {summary.n_rounds}",
        f"- Summary method: {summary.backend} (prompt {summary.prompt_version})",
        f"- Mediated strengths: Claimant {summary.claimant_strength}; Respondent {summary.respondent_strength}",
        f"- Mediated recommendation: {summary.recommended_treatment or 'NOT RECORDED'}",
        f"- Recommendation rationale: {summary.treatment_rationale or 'Not recorded.'}",
        *_evidence_link_lines("Summary", summary),
        "",
    ]
    for turn in summary.turns:
        lines.extend(
            [
                f"#### Round {turn.round_number} — {turn.side}: {turn.barrister}",
                "",
                turn.position,
                "",
                *(["Responses to opponent:", *[f"- {item}" for item in turn.responses_to_opponent], ""] if turn.responses_to_opponent else []),
                "Supporting points:",
                *[f"- {item}" for item in turn.supporting_points],
                "",
                "Challenges for opponent:",
                *[f"- {item}" for item in turn.challenges_for_opponent],
                "",
                f"Citations: {', '.join(turn.paragraph_citations)}",
                f"Generation: {turn.backend} (prompt {turn.prompt_version})",
                *_evidence_link_lines("Turn", turn),
                "",
            ]
        )
    lines.extend(
        [
            "Remaining disputes:",
            *[f"- {item}" for item in summary.remaining_disputes],
            "",
            "Evidence gaps:",
            *([f"- {item}" for item in summary.evidence_gaps] or ["- None automatically identified."]),
            "",
            f"Stop reason: {summary.stop_reason}",
            "",
        ]
    )
    return lines


def barrister_debate_markdown(run: RunRecord, arguments: list[Argument]) -> str:
    lines = [
        "# Mediated barrister debate",
        "",
        f"Run ID: `{run.run_id}`  ",
        f"Configured rounds: `{run.config_snapshot.get('n_rounds', 2)}`  ",
        f"Claimant barrister: `{run.config_snapshot.get('claimant_barrister', 'SOL')}`  ",
        f"Respondent barrister: `{run.config_snapshot.get('respondent_barrister', 'FABLE')}`",
        f"Assessment modes: `{run.config_snapshot.get('assessment_mode_counts', {})}`  ",
        f"Assessment providers: `{run.config_snapshot.get('assessment_backend_counts', {})}`  ",
        f"Recorded token usage: `{run.config_snapshot.get('token_usage', {})}`",
        "",
        "> The orchestrator mediates every turn. Barristers do not communicate directly, and every cited paragraph must exist in the argument source map.",
        "",
    ]
    for argument in arguments:
        lines.extend(
            [
                f"## {argument.argument_id}: {argument.title}",
                "",
                argument.proposition,
                "",
                *_debate_lines(argument),
            ]
        )
    return "\n".join(lines)


def build_markdown_report(
    run: RunRecord,
    documents: list[Document],
    issues: list[Issue],
    arguments: list[Argument],
    references: list[ReferencePlaceholder],
    packages: list[CaseLawSearchPackage],
    reviews: list[HumanReviewItem],
    stage_events: list[dict[str, Any]],
) -> str:
    lines = [
        "# Sol-Fable WS calibration report",
        "",
        f"Run ID: `{run.run_id}`  ",
        f"Backend: `{run.backend}`  ",
        f"Prompt version: `{run.prompt_version}`",
        f"Debate rounds: `{run.config_snapshot.get('n_rounds', 2)}`  ",
        f"Claimant barrister: `{run.config_snapshot.get('claimant_barrister', 'SOL')}`  ",
        f"Respondent barrister: `{run.config_snapshot.get('respondent_barrister', 'FABLE')}`",
        "",
        "> This report assists legal preparation. Pleadings are party positions, unresolved brackets are not verified evidence, and no final legal conclusion is made.",
        "",
        "## Executive argument map",
        "",
    ]
    for category in ArgumentCategory:
        selected = [item for item in arguments if item.category == category][:8]
        lines.extend([f"### Top {category.lower()} arguments", ""])
        lines.extend(
            [
                f"- {item.argument_id} ({item.importance}, {item.ranking_score}): {item.title}"
                for item in selected
            ]
            or ["- None identified."]
        )
        lines.append("")
    lines.extend(["## Treatment priorities", ""])
    for treatment in (ArgumentTreatment.PROTECT, ArgumentTreatment.DEFEND, ArgumentTreatment.BOTH):
        selected = [item for item in arguments if treatment in item.treatment][:8]
        lines.extend(
            [f"### {treatment}", ""],
        )
        lines.extend([f"- {item.argument_id}: {item.title}" for item in selected] or ["- None identified."])
        lines.append("")

    potential_new = [
        item for item in arguments if PleadingStatus.POTENTIALLY_NEW_CASE in item.pleading_statuses
    ]
    tensions = [item for item in reviews if "tension" in item.reason.lower()]
    et3_only = [item for item in issues if item.status == "ET3_ONLY_POSITION"]
    lines.extend(
        [
            "## Pleading and evidence review",
            "",
            "### Potentially unpleaded propositions",
            "",
            *([f"- {item.argument_id}: {item.proposition}" for item in potential_new] or ["- None automatically flagged."]),
            "",
            "### ET1-to-WS tensions",
            "",
            *([f"- {item.entity_id}: {item.reason}" for item in tensions] or ["- None automatically flagged."]),
            "",
            "### ET3 positions not aligned to an ET1 issue",
            "",
            *([f"- {item.issue_id}: {item.respondent_position}" for item in et3_only] or ["- None automatically flagged."]),
            "",
            "### Evidence placeholders awaiting bundle ingestion",
            "",
            *([f"- {item.reference_id} at {item.ws_paragraph_id}: {item.raw_text} ({item.reference_type})" for item in references] or ["- None found."]),
            "",
            "### Most valuable future case-law research concepts",
            "",
            *([f"- {item.argument_id}: {', '.join(item.legal_dimensions)}" for item in packages[:8]] or ["- None generated from the current ranking."]),
            "",
            "## Detailed arguments",
            "",
        ]
    )
    for argument in arguments:
        lines.extend(_argument_summary(argument))
        if argument.debate_summary:
            lines.extend(["#### Mediated barrister exchange", "", *_debate_lines(argument)])
        elif argument.sol_assessment:
            lines.extend(
                [
                    f"- Sol: {argument.sol_assessment.claimant_relevance}",
                    f"- Sol hidden assumptions: {'; '.join(argument.sol_assessment.hidden_assumptions)}",
                    f"- Sol generation: {argument.sol_assessment.backend}",
                    *_evidence_link_lines("Sol", argument.sol_assessment),
                ]
            )
        if not argument.debate_summary and argument.fable_assessment:
            lines.extend(
                [
                    f"- Fable: {argument.fable_assessment.pleading_attack}",
                    f"- Cross-examination vulnerability: {argument.fable_assessment.cross_examination_risk}",
                    f"- Likely damaging concessions: {'; '.join(argument.fable_assessment.damaging_concessions)}",
                    f"- Fable generation: {argument.fable_assessment.backend}",
                    *_evidence_link_lines("Fable", argument.fable_assessment),
                ]
            )
        if argument.search_package:
            lines.append(
                f"- Search concepts: {', '.join(argument.search_package.get('legal_dimensions', []))}"
            )
        lines.append("")
    lines.extend(
        [
            "## Human-review queue",
            "",
            *([f"- Gate {item.gate} — {item.entity_id}: {item.reason}" for item in reviews] or ["- No review items."]),
            "",
            "## Audit trail",
            "",
            *[f"- Stage {item['stage']} {item['name']}: {item['status']} at {item['recorded_at']}" for item in stage_events],
            "",
            "### Input integrity",
            "",
            *[f"- {item.document_type}: `{item.filename}` — SHA-256 `{item.sha256}`" for item in documents],
        ]
    )
    return "\n".join(lines)


def build_json_report(
    run: RunRecord,
    documents: list[Document],
    issues: list[Issue],
    arguments: list[Argument],
    references: list[ReferencePlaceholder],
    packages: list[CaseLawSearchPackage],
    reviews: list[HumanReviewItem],
    stage_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "report_version": "0.3.0",
        "legal_status": "ASSISTIVE_NOT_A_FINAL_LEGAL_DETERMINATION",
        "run": run.model_dump(mode="json"),
        "documents": [item.model_dump(mode="json") for item in documents],
        "issues": [item.model_dump(mode="json") for item in issues],
        "arguments": [item.model_dump(mode="json") for item in arguments],
        "reference_placeholders": [item.model_dump(mode="json") for item in references],
        "case_law_search_packages": [item.model_dump(mode="json") for item in packages],
        "human_review_queue": [item.model_dump(mode="json") for item in reviews],
        "audit_trail": stage_events,
    }


def review_csv_rows(reviews: list[HumanReviewItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in reviews]
