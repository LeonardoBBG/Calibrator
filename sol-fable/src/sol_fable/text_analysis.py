"""Deterministic text features shared by analysis stages."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import ArgumentCategory

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "me", "my", "not", "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "with", "would", "you",
}

PROCEDURAL_TERMS = {
    "appeal", "disciplinary", "hearing", "investigation", "investigate", "interview",
    "procedure", "procedural", "notice", "disclosure", "decision-maker", "adjournment",
    "representative", "accompanied", "minutes", "outcome letter",
}
SUBSTANTIVE_TERMS = {
    "dishonest", "dishonesty", "misconduct", "conduct", "intention", "intent", "motive",
    "warning", "instruction", "performance", "knowledge", "dismissal", "sanction", "work",
    "role", "duty", "duties", "reason", "belief",
}


def tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def lexical_similarity(left: str, right: str) -> float:
    left_counts = Counter(tokens(left))
    right_counts = Counter(tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[word] * right_counts[word] for word in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return round(dot / (left_norm * right_norm), 4) if left_norm and right_norm else 0.0


def classify_category(*texts: str) -> ArgumentCategory:
    combined = " ".join(texts).lower()
    procedural = sum(term in combined for term in PROCEDURAL_TERMS)
    substantive = sum(term in combined for term in SUBSTANTIVE_TERMS)
    if procedural and substantive:
        return ArgumentCategory.MIXED
    if procedural:
        return ArgumentCategory.PROCEDURAL
    return ArgumentCategory.SUBSTANTIVE


def concise_title(text: str, limit: int = 10) -> str:
    clean = re.sub(r"\[[^]]+\]", "", re.sub(r"\s+", " ", text)).strip(" .:;-")
    words = clean.split()
    title = " ".join(words[:limit])
    if len(words) > limit:
        title += "…"
    return title[:1].upper() + title[1:] if title else "Unlabelled issue"


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))

