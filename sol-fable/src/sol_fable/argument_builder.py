"""Create precise, traceable arguments from matched atomic propositions."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from .models import (
    Argument,
    ArgumentCategory,
    AtomicProposition,
    Importance,
    Issue,
    PleadingStatus,
    ReferencePlaceholder,
    SourceOrigin,
)
from .text_analysis import classify_category, concise_title, lexical_similarity, unique


@dataclass
class _DraftGroup:
    """A bounded argument unit whose original propositions remain auditable."""

    propositions: list[AtomicProposition] = field(default_factory=list)
    proposition_issues: list[Issue | None] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def add(self, proposition: AtomicProposition, issue: Issue | None) -> None:
        self.propositions.append(proposition)
        self.proposition_issues.append(issue)
        if issue and all(existing.issue_id != issue.issue_id for existing in self.issues):
            self.issues.append(issue)

    def absorb(self, other: _DraftGroup) -> None:
        for proposition, issue in zip(
            other.propositions, other.proposition_issues, strict=True
        ):
            self.add(proposition, issue)


_MATERIALITY_ORDER = {
    Importance.LOW: 0,
    Importance.MEDIUM: 1,
    Importance.HIGH: 2,
    Importance.CRITICAL: 3,
}
_NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|without|deny|denied|denies|didn't|wasn't|weren't)\b", re.I)
_SPECIFIC_RE = re.compile(
    r"\b(?:\d+(?:[.:/-]\d+)*|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.I,
)

# Five clauses are enough to turn atomised sentence output into a useful unit while
# keeping the result short enough for assessment, debate and human review.
MAX_PROPOSITIONS_PER_ARGUMENT = 5

_PLEADED_OR_RESPONSE = {
    PleadingStatus.PLEADED_ET1_EXPLICIT,
    PleadingStatus.PLEADED_ET1_IMPLICIT,
    PleadingStatus.ET1_DETAIL_OR_ELABORATION,
    PleadingStatus.RESPONDS_TO_ET3,
}
_ADVERSE_RESPONDENT_POSITIONS = {
    "DENIED",
    "NOT_ADMITTED",
    "ALTERNATIVE_ACCOUNT",
    "RESPONDENT_ALLEGATION",
}


def _best_issue(proposition: AtomicProposition, issues: list[Issue]) -> Issue | None:
    linked_ids = {
        match.paragraph_id for match in proposition.et1_matches + proposition.et3_matches
    }
    linked = [
        issue
        for issue in issues
        if linked_ids.intersection(issue.et1_paragraphs + issue.et3_paragraphs)
    ]
    if linked:
        return linked[0]
    candidates = sorted(
        (
            (
                lexical_similarity(
                    proposition.text,
                    f"{issue.claimant_position} {issue.respondent_position}",
                ),
                issue,
            )
            for issue in issues
        ),
        key=lambda pair: (-pair[0], pair[1].issue_id),
    )
    return candidates[0][1] if candidates and candidates[0][0] >= 0.18 else None


def _linked_paragraphs(proposition: AtomicProposition) -> set[str]:
    return {
        match.paragraph_id
        for match in proposition.et1_matches + proposition.et3_matches
    }


def _compatible_meaning(left: str, right: str) -> bool:
    """Reject direct conflicts without splitting an ordinary narrative sequence.

    A paragraph can properly contain different dates, figures, or a mixture of
    positive and negative statements.  Those surface differences are treated as a
    conflict only where the propositions otherwise describe substantially the same
    thing.  This still separates pairs such as "I received the warning" and "I did
    not receive the warning".
    """

    similarity = lexical_similarity(left, right)
    if (
        bool(_NEGATION_RE.search(left)) != bool(_NEGATION_RE.search(right))
        and similarity >= 0.45
    ):
        return False
    left_specifics = {item.lower() for item in _SPECIFIC_RE.findall(left)}
    right_specifics = {item.lower() for item in _SPECIFIC_RE.findall(right)}
    return not (
        left_specifics
        and right_specifics
        and left_specifics != right_specifics
        and similarity >= 0.55
    )


def _respondent_positions(proposition: AtomicProposition) -> dict[str, set[str]]:
    positions: dict[str, set[str]] = defaultdict(set)
    for match in proposition.et3_matches:
        if match.respondent_position:
            positions[match.paragraph_id].add(
                getattr(match.respondent_position, "value", str(match.respondent_position))
            )
    return positions


def _compatible_match_positions(
    left: AtomicProposition, right: AtomicProposition
) -> bool:
    """Do not hide an admission/denial conflict for the same ET3 paragraph."""

    left_positions = _respondent_positions(left)
    right_positions = _respondent_positions(right)
    for paragraph_id in left_positions.keys() & right_positions.keys():
        left_values = left_positions[paragraph_id]
        right_values = right_positions[paragraph_id]
        if (
            "ADMITTED" in left_values
            and right_values.intersection(_ADVERSE_RESPONDENT_POSITIONS)
        ) or (
            "ADMITTED" in right_values
            and left_values.intersection(_ADVERSE_RESPONDENT_POSITIONS)
        ):
            return False
    return True


def _compatible_propositions(
    left: AtomicProposition, right: AtomicProposition
) -> bool:
    return _compatible_meaning(left.text, right.text) and _compatible_match_positions(
        left, right
    )


def _pleading_family(proposition: AtomicProposition) -> str:
    status = PleadingStatus(proposition.pleading_status)
    if status in _PLEADED_OR_RESPONSE:
        return "PLEADED_OR_RESPONSE"
    return status.value


def _evidence_mode(proposition: AtomicProposition) -> str:
    """Keep legal characterisation separate from propositions of primary fact."""

    origins = {SourceOrigin(value) for value in proposition.source_origins}
    characters = {
        getattr(value, "value", str(value)) for value in proposition.evidential_character
    }
    if SourceOrigin.LEGAL_SUBMISSION in origins or "LEGAL_CHARACTERISATION" in characters:
        return "LEGAL"
    return "FACTUAL"


def _group_profile(group: _DraftGroup) -> tuple[str, str]:
    anchor = group.propositions[0]
    return _pleading_family(anchor), _evidence_mode(anchor)


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    """Make output independent of database or caller insertion order."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _merge_score(
    anchor: AtomicProposition,
    anchor_issue: Issue | None,
    candidate: AtomicProposition,
    candidate_issue: Issue | None,
) -> float | None:
    """Return a merge score only for near duplicates or a shared source issue."""

    if not _compatible_propositions(anchor, candidate):
        return None
    similarity = lexical_similarity(anchor.text, candidate.text)

    # Near-identical wording is safe to collapse even if the issue linker chose a
    # different issue for each copy.
    if similarity >= 0.90:
        return similarity

    same_issue = bool(
        anchor_issue
        and candidate_issue
        and anchor_issue.issue_id == candidate_issue.issue_id
    )
    if not same_issue:
        return None

    anchor_links = _linked_paragraphs(anchor)
    candidate_links = _linked_paragraphs(candidate)
    shared_link = bool(anchor_links.intersection(candidate_links))
    same_nonempty_map = bool(anchor_links) and anchor_links == candidate_links
    same_ws_paragraph = anchor.ws_paragraph_id == candidate.ws_paragraph_id

    # These lower thresholds are available only where the issue and underlying
    # paragraph map agree. They merge repeated formulations, not merely every
    # sentence that happens to have the same broad issue label.
    if similarity >= 0.80 and (shared_link or same_ws_paragraph):
        return similarity
    if similarity >= 0.72 and same_nonempty_map:
        return similarity
    return None


def _paragraph_group_score(
    group: _DraftGroup, proposition: AtomicProposition
) -> float | None:
    if len(group.propositions) >= MAX_PROPOSITIONS_PER_ARGUMENT:
        return None
    if _group_profile(group) != (
        _pleading_family(proposition),
        _evidence_mode(proposition),
    ):
        return None
    if not all(
        _compatible_propositions(existing, proposition)
        for existing in group.propositions
    ):
        return None
    return fmean(
        lexical_similarity(existing.text, proposition.text)
        for existing in group.propositions
    )


def _cross_paragraph_group_score(
    left: _DraftGroup, right: _DraftGroup
) -> float | None:
    """Merge only groups that are near-duplicates in their entirety."""

    if len(left.propositions) + len(right.propositions) > MAX_PROPOSITIONS_PER_ARGUMENT:
        return None
    if _group_profile(left) != _group_profile(right):
        return None
    scores: list[float] = []
    for left_proposition, left_issue in zip(
        left.propositions, left.proposition_issues, strict=True
    ):
        for right_proposition, right_issue in zip(
            right.propositions, right.proposition_issues, strict=True
        ):
            score = _merge_score(
                left_proposition,
                left_issue,
                right_proposition,
                right_issue,
            )
            if score is None:
                return None
            scores.append(score)
    return fmean(scores) if scores else None


def _draft_groups(
    propositions: list[AtomicProposition], issues: list[Issue]
) -> list[_DraftGroup]:
    """Build local, bounded units, then collapse only genuine repetitions."""

    ordered_propositions = sorted(
        propositions,
        key=lambda item: (
            _natural_key(item.ws_paragraph_id),
            _natural_key(item.proposition_id),
        ),
    )
    paragraph_groups: list[_DraftGroup] = []
    groups_by_paragraph: dict[str, list[int]] = defaultdict(list)

    for proposition in ordered_propositions:
        issue = _best_issue(proposition, issues)
        candidates: list[tuple[float, int]] = []
        for index in groups_by_paragraph[proposition.ws_paragraph_id]:
            score = _paragraph_group_score(paragraph_groups[index], proposition)
            if score is not None:
                candidates.append((score, index))

        if candidates:
            # Prefer the most coherent compatible unit; earlier creation is the
            # deterministic tie-breaker.
            _, selected_index = max(candidates, key=lambda item: (item[0], -item[1]))
            paragraph_groups[selected_index].add(proposition, issue)
            continue

        group = _DraftGroup()
        group.add(proposition, issue)
        groups_by_paragraph[proposition.ws_paragraph_id].append(len(paragraph_groups))
        paragraph_groups.append(group)

    consolidated: list[_DraftGroup] = []
    for group in paragraph_groups:
        candidates = [
            (score, index)
            for index, existing in enumerate(consolidated)
            if (score := _cross_paragraph_group_score(existing, group)) is not None
        ]
        if not candidates:
            consolidated.append(group)
            continue
        _, selected_index = max(candidates, key=lambda item: (item[0], -item[1]))
        consolidated[selected_index].absorb(group)
    return consolidated


def _consolidated_text(propositions: list[AtomicProposition]) -> str:
    texts = list(dict.fromkeys(item.text.strip() for item in propositions if item.text.strip()))
    if len(texts) == 1:
        return texts[0]

    # For true near duplicates, keep the most informative wording. For related
    # formulations, retain each distinct formulation in the consolidated argument.
    if all(lexical_similarity(texts[0], text) >= 0.90 for text in texts[1:]):
        return max(texts, key=lambda text: (len(text.split()), len(text)))
    return "; ".join(text.rstrip(" .;:") for text in texts) + "."


def _group_category(group: _DraftGroup, proposition_text: str) -> ArgumentCategory:
    categories = {ArgumentCategory(issue.category) for issue in group.issues}
    if len(categories) == 1:
        return next(iter(categories))
    if len(categories) > 1:
        return ArgumentCategory.MIXED
    return classify_category(proposition_text)


def _group_importance(group: _DraftGroup) -> Importance:
    return max(
        (Importance(item.materiality) for item in group.propositions),
        key=lambda value: _MATERIALITY_ORDER[value],
    )


def build_arguments(
    propositions: list[AtomicProposition],
    issues: list[Issue],
    references: list[ReferencePlaceholder],
) -> list[Argument]:
    references_by_paragraph: dict[str, list[str]] = defaultdict(list)
    for reference in sorted(
        references,
        key=lambda item: (
            _natural_key(item.ws_paragraph_id),
            _natural_key(item.reference_id),
        ),
    ):
        references_by_paragraph[reference.ws_paragraph_id].append(reference.raw_text)

    ordered_issues = sorted(issues, key=lambda item: _natural_key(item.issue_id))
    groups = _draft_groups(propositions, ordered_issues)

    category_counts: dict[str, int] = defaultdict(int)
    arguments: list[Argument] = []
    for group in groups:
        proposition_text = _consolidated_text(group.propositions)
        category = _group_category(group, proposition_text)
        category_value = category.value if hasattr(category, "value") else str(category)
        prefix = {"SUBSTANTIVE": "SUB", "PROCEDURAL": "PRO", "MIXED": "MIX"}[category_value]
        category_counts[prefix] += 1

        et1_ids = unique(
            [
                match.paragraph_id
                for proposition in group.propositions
                for match in proposition.et1_matches
            ]
            + [
                paragraph_id
                for issue in group.issues
                for paragraph_id in issue.et1_paragraphs
            ]
        )
        et3_ids = unique(
            [
                match.paragraph_id
                for proposition in group.propositions
                for match in proposition.et3_matches
            ]
            + [
                paragraph_id
                for issue in group.issues
                for paragraph_id in issue.et3_paragraphs
            ]
        )
        ws_ids = unique([item.ws_paragraph_id for item in group.propositions])
        source_types = unique(
            [
                SourceOrigin(origin)
                for proposition in group.propositions
                for origin in proposition.source_origins
            ]
        )
        if et1_ids:
            source_types.append(SourceOrigin.ET1)
        if et3_ids:
            source_types.append(SourceOrigin.ET3)

        issue_ids = unique([issue.issue_id for issue in group.issues])
        title = (
            group.issues[0].title
            if len(issue_ids) == 1
            else concise_title(proposition_text)
        )
        arguments.append(
            Argument(
                argument_id=f"ARG-{prefix}-{category_counts[prefix]:03d}",
                title=title,
                proposition=proposition_text,
                category=category,
                importance=_group_importance(group),
                issue_ids=issue_ids,
                ws_paragraphs=ws_ids,
                et1_paragraphs=et1_ids,
                et3_paragraphs=et3_ids,
                source_types=unique(source_types),
                document_placeholders=unique(
                    [
                        raw_text
                        for ws_id in ws_ids
                        for raw_text in references_by_paragraph[ws_id]
                    ]
                ),
                pleading_statuses=unique(
                    [item.pleading_status for item in group.propositions]
                ),
                proposition_ids=[item.proposition_id for item in group.propositions],
                human_review_status=(
                    "PENDING"
                    if any(item.requires_human_review for item in group.propositions)
                    else "NOT_REQUIRED"
                ),
            )
        )
    return arguments
