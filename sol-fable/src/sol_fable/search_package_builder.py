"""Generate focused case-law research concepts without retrieving authorities."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Argument, CaseLawSearchPackage, Importance
from .text_analysis import tokens, unique


@dataclass(frozen=True)
class ConceptTemplate:
    name: str
    triggers: tuple[str, ...]
    dimensions: tuple[str, ...]
    queries: tuple[str, ...]


TEMPLATES = (
    ConceptTemplate(
        "continued conduct after alleged warning",
        ("continued", "after warning", "months", "no action", "management knew"),
        ("clarity of warning", "management knowledge", "tolerance or acquiescence", "reasonable belief", "proportionality"),
        (
            "UK unfair dismissal continued conduct after warning employer took no action",
            "EAT management knew conduct continued but delayed disciplinary action",
        ),
    ),
    ConceptTemplate(
        "unclear or informal instruction",
        ("instruction", "discussion", "informal", "not understood", "warning"),
        ("clarity of instruction", "deliberate disobedience", "reasonable understanding", "gross misconduct"),
        ("UK unfair dismissal ambiguous instruction gross misconduct", "EAT dismissal for disobeying unclear instruction"),
    ),
    ConceptTemplate(
        "knowledge compared with dishonest intention",
        ("knew", "intend", "dishonest", "motive", "deliberate"),
        ("knowledge", "purpose", "intention", "dishonesty", "reasonable grounds for belief"),
        ("UK unfair dismissal deliberate act disputed dishonest intention", "EAT knowledge of consequence dishonest motive"),
    ),
    ConceptTemplate(
        "competing management priorities",
        ("project", "priority", "conflicting", "duties", "manager"),
        ("authorised work", "workload prioritisation", "reasonable understanding of role", "culpability"),
        ("UK unfair dismissal competing management instructions workload priorities", "EAT authorised working practice misconduct"),
    ),
    ConceptTemplate(
        "failure to investigate employee explanation",
        ("investigation", "ignored", "not interviewed", "not obtained", "explanation"),
        ("reasonable investigation", "exculpatory evidence", "Burchell test", "procedural fairness"),
        ("EAT failure to investigate exculpatory evidence unfair dismissal", "Burchell alternative explanation not investigated"),
    ),
    ConceptTemplate(
        "positive appraisal during alleged misconduct",
        ("appraisal", "review", "praised", "performance", "positive"),
        ("credibility of alleged warning", "management knowledge", "seriousness", "contemporaneous evidence"),
        ("UK unfair dismissal positive appraisal after alleged misconduct", "EAT performance review gross misconduct allegation"),
    ),
    ConceptTemplate(
        "clean record and proportionality",
        ("no prior", "long service", "clean record", "first offence", "summary dismissal"),
        ("range of reasonable responses", "mitigation", "lesser sanction", "gross misconduct"),
        ("UK unfair dismissal clean disciplinary record gross misconduct", "EAT summary dismissal long service no prior warnings"),
    ),
    ConceptTemplate(
        "appeal failed to cure defects",
        ("appeal", "fresh investigation", "independent", "procedural defect"),
        ("appeal curing defects", "rehearing versus review", "overall fairness", "independence"),
        ("EAT disciplinary appeal failed to cure investigation defects", "UK unfair dismissal whether appeal cures procedural unfairness"),
    ),
)


def _select_template(argument: Argument) -> ConceptTemplate:
    text = argument.proposition.lower()
    scored = [
        (sum(trigger in text for trigger in template.triggers), index, template)
        for index, template in enumerate(TEMPLATES)
    ]
    score, _, template = max(scored, key=lambda item: (item[0], -item[1]))
    if score:
        return template
    return ConceptTemplate(
        "fact-specific unfair-dismissal issue",
        tuple(),
        ("reasonable investigation", "reasonable belief", "range of reasonable responses"),
        ("UK Employment Appeal Tribunal unfair dismissal analogous facts",),
    )


def build_search_packages(arguments: list[Argument]) -> list[CaseLawSearchPackage]:
    packages: list[CaseLawSearchPackage] = []
    for argument in arguments:
        if argument.importance not in (Importance.CRITICAL, Importance.HIGH):
            continue
        template = _select_template(argument)
        search_terms = unique(list(template.dimensions) + tokens(argument.title)[:5])
        broad = list(template.queries[:1])
        targeted = list(template.queries[1:]) or [
            f"EAT {template.name} unfair dismissal"
        ]
        packages.append(
            CaseLawSearchPackage(
                argument_id=argument.argument_id,
                core_proposition=argument.proposition,
                legal_dimensions=list(template.dimensions),
                factual_analogues=[template.name, argument.proposition],
                search_terms=search_terms,
                broad_queries=broad,
                targeted_queries=targeted,
                authority_queries=[
                    f"site:gov.uk EAT {template.name}",
                    f"site:bailii.org employment {template.name}",
                ],
                questions_for_cases=[
                    "Which facts were established by evidence rather than merely pleaded?",
                    "How closely do the timing, knowledge and decision-making context compare?",
                    "Was the point relevant to belief, investigation, procedure, sanction or more than one dimension?",
                    "What court or tribunal decided the case, and what precedential weight does it carry?",
                ],
                do_not_assume=[
                    "Factual similarity determines the legal outcome.",
                    "A first-instance Employment Tribunal judgment is binding authority.",
                    "The search wording establishes any disputed fact in this case.",
                ],
            )
        )
    return packages


def research_plan_markdown(packages: list[CaseLawSearchPackage]) -> str:
    lines = [
        "# Case-law research plan",
        "",
        "These are search concepts only. No authority has been retrieved or verified.",
        "",
    ]
    for package in packages:
        lines.extend(
            [
                f"## {package.argument_id}",
                "",
                f"Core proposition: {package.core_proposition}",
                "",
                "Targeted queries:",
                *[f"- {query}" for query in package.targeted_queries + package.authority_queries],
                "",
                "Comparability questions:",
                *[f"- {question}" for question in package.questions_for_cases],
                "",
            ]
        )
    return "\n".join(lines)

