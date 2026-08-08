"""Bracketed WS reference detection and conservative classification."""

from __future__ import annotations

import re

from .models import Paragraph, ReferencePlaceholder, ReferenceType

BRACKET_RE = re.compile(r"\[[^\[\]\n]+\]")


def classify_reference(raw_text: str) -> ReferenceType:
    lowered = raw_text.lower()
    if re.search(r"\b(et1|et3)\b|\bplead(?:ing|ed)?\b", lowered):
        return ReferenceType.PLEADING_REFERENCE
    if re.search(r"\b(ws|witness statement|para(?:graph)?\s*\d+)\b", lowered):
        return ReferenceType.INTERNAL_WS_REFERENCE
    if re.search(r"\b(todo|insert|draft|check|editor(?:ial)?|note to)\b", lowered):
        return ReferenceType.EDITORIAL_NOTE
    if re.search(
        r"\b(email|letter|message|communications?|records?|review|appraisal|report|bundle|"
        r"exhibit|document|minutes|meeting note|policy|contract|log|data|telemetry)\b",
        lowered,
    ):
        return ReferenceType.EVIDENCE_REFERENCE
    if re.search(r"\b(to be provided|tbc|placeholder|date|page|pp?\.)\b", lowered):
        return ReferenceType.PLACEHOLDER
    return ReferenceType.UNKNOWN


def parse_references(ws_paragraphs: list[Paragraph]) -> list[ReferencePlaceholder]:
    references: list[ReferencePlaceholder] = []
    for paragraph in ws_paragraphs:
        for raw in BRACKET_RE.findall(paragraph.raw_text):
            references.append(
                ReferencePlaceholder(
                    reference_id=f"REF-{len(references) + 1:04d}",
                    ws_paragraph_id=paragraph.paragraph_id,
                    raw_text=raw,
                    reference_type=classify_reference(raw),
                    resolved=False,
                    resolved_document_id=None,
                )
            )
    return references
