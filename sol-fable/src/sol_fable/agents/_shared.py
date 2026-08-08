"""Shared assessment helpers."""

from __future__ import annotations

from ..models import Argument
from ..text_analysis import unique


def citations(argument: Argument) -> list[str]:
    return unique(argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs)


def confidence(argument: Argument) -> float:
    base = 0.45
    if argument.et1_paragraphs:
        base += 0.18
    if argument.et3_paragraphs:
        base += 0.08
    if argument.ws_paragraphs:
        base += 0.12
    if argument.document_placeholders:
        base -= 0.08
    if argument.human_review_status == "PENDING":
        base -= 0.08
    return round(max(0.2, min(0.9, base)), 2)

