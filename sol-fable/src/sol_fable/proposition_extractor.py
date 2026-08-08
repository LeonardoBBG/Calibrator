"""Conservative, deterministic extraction of atomic WS propositions."""

from __future__ import annotations

import re

from .models import (
    AtomicProposition,
    EvidentialCharacter,
    Importance,
    Paragraph,
    SourceOrigin,
)

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
CLAUSE_VERBS = re.compile(
    r"\b(am|are|asked|believe|continued|did|discussed|gave|had|has|heard|instructed|"
    r"is|knew|know|made|met|received|said|saw|sent|spoke|told|understood|was|were|worked|wrote)\b",
    re.IGNORECASE,
)


def _split_conjunction(clause: str) -> list[str]:
    parts = re.split(r"\s*;\s*", clause)
    output: list[str] = []
    for part in parts:
        # The acknowledgement of a discussion and the qualification that it was not
        # understood as a warning form one proposition; splitting would lose meaning.
        if re.search(r"\b(discussion|conversation|meeting)\b", part, re.IGNORECASE) and re.search(
            r"\bbut\b.*\b(not|never)\b.*\b(warning|instruction)\b", part, re.IGNORECASE
        ):
            output.append(part)
            continue
        comma_parts = re.split(r",\s+(?:and|but)\s+", part, maxsplit=1, flags=re.IGNORECASE)
        if len(comma_parts) == 2 and all(CLAUSE_VERBS.search(item) for item in comma_parts):
            output.extend(comma_parts)
        else:
            and_parts = re.split(r"\s+(?:and|but)\s+", part, maxsplit=1, flags=re.IGNORECASE)
            if len(and_parts) == 2 and all(CLAUSE_VERBS.search(item) for item in and_parts):
                output.extend(and_parts)
            else:
                output.append(part)
    return [re.sub(r"\s+", " ", item).strip(" ,;") for item in output if item.strip(" ,;")]


def split_atomic(text: str) -> list[str]:
    sentences = SENTENCE_BOUNDARY_RE.split(re.sub(r"\s+", " ", text).strip())
    propositions: list[str] = []
    for sentence in sentences:
        propositions.extend(_split_conjunction(sentence))
    return propositions or [text.strip()]


def _source_origins(text: str) -> list[SourceOrigin]:
    lowered = text.lower()
    origins: list[SourceOrigin] = []
    if re.search(r"\[[^]]+\]", text):
        origins.append(SourceOrigin.DOCUMENT_PLACEHOLDER)
    if re.search(r"\b(i|me|my|mine)\b", lowered):
        origins.append(SourceOrigin.SELF_ACCOUNT)
    if re.search(r"\b(told me|said to me|according to|informed me)\b", lowered):
        origins.append(SourceOrigin.OTHER_WITNESS)
    if re.search(r"\b(i infer|i concluded|must have|therefore|this shows)\b", lowered):
        origins.append(SourceOrigin.INFERENCE)
    if re.search(r"\b(unfair|unlawful|breach|legally|reasonable responses|burchell)\b", lowered):
        origins.append(SourceOrigin.LEGAL_SUBMISSION)
    return origins or [SourceOrigin.UNKNOWN]


def _evidential_character(text: str, origins: list[SourceOrigin]) -> list[EvidentialCharacter]:
    lowered = text.lower()
    values: list[EvidentialCharacter] = []
    if SourceOrigin.DOCUMENT_PLACEHOLDER in origins:
        values.extend(
            [
                EvidentialCharacter.CONTEMPORANEOUS_DOCUMENT_PLACEHOLDER,
                EvidentialCharacter.EVIDENCE_NOT_YET_INGESTED,
            ]
        )
    if SourceOrigin.SELF_ACCOUNT in origins:
        values.append(EvidentialCharacter.PERSONAL_RECOLLECTION)
    if SourceOrigin.OTHER_WITNESS in origins:
        values.append(EvidentialCharacter.HEARSAY_ACCOUNT)
    if SourceOrigin.INFERENCE in origins:
        values.append(EvidentialCharacter.INFERENCE_FROM_FACTS)
    if SourceOrigin.LEGAL_SUBMISSION in origins:
        values.append(EvidentialCharacter.LEGAL_CHARACTERISATION)
    if re.search(r"\b(i accept|i admit|I acknowledge)\b", text, re.IGNORECASE):
        values.append(EvidentialCharacter.ADMISSION)
    if not values or origins == [SourceOrigin.UNKNOWN]:
        values.append(EvidentialCharacter.UNSUPPORTED_ASSERTION)
    return list(dict.fromkeys(values))


def _materiality(text: str) -> Importance:
    lowered = text.lower()
    critical = ("dismiss" , "dishonest", "gross misconduct", "warning", "instruction")
    high = ("investigat", "appeal", "disciplinary", "manager", "sanction", "admit")
    if any(term in lowered for term in critical):
        return Importance.CRITICAL
    if any(term in lowered for term in high):
        return Importance.HIGH
    if len(text.split()) >= 8:
        return Importance.MEDIUM
    return Importance.LOW


def extract_propositions(ws_paragraphs: list[Paragraph]) -> list[AtomicProposition]:
    propositions: list[AtomicProposition] = []
    for paragraph in ws_paragraphs:
        for sequence, text in enumerate(split_atomic(paragraph.text), start=1):
            origins = _source_origins(text)
            propositions.append(
                AtomicProposition(
                    proposition_id=f"{paragraph.paragraph_id}-C{sequence:02d}",
                    ws_paragraph_id=paragraph.paragraph_id,
                    text=text,
                    source_origins=origins,
                    evidential_character=_evidential_character(text, origins),
                    materiality=_materiality(text),
                )
            )
    return propositions
