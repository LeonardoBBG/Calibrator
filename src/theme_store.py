"""Deterministic theme-store exports for outcome-optimized case matches."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

CASE_EFFECT_SCORE = {
    "WIN_DRIVER": 1.00,
    "STRONG_SUPPORT": 0.85,
    "MODERATE_SUPPORT": 0.65,
    "WEAK_SUPPORT": 0.40,
    "NEUTRAL": 0.20,
    "NEUTRAL_CONTEXT": 0.20,
    "ADVERSE": -1.00,
}

CONFIDENCE_SCORE = {
    "HIGH": 1.00,
    "MEDIUM": 0.70,
    "LOW": 0.40,
}

EFFECT_SCORE = {
    "REINFORCE": 1.00,
    "ADD FACT": 1.00,
    "ADD EVIDENCE ANCHOR": 1.00,
    "DISTINGUISH": 0.50,
    "REVIEW_MANUALLY": 0.20,
    "NEUTRAL": 0.20,
    "UNDERMINE": -1.00,
}

DEFAULT_SUBTHEME = "unassigned"
DEFAULT_REVIEW_STATUS = "UNREVIEWED"


def build_theme_store(
    aggregation: Dict,
    outcome_cases: List[Dict],
    source_filenames: Optional[Dict[int, str]] = None,
    top_n_per_theme: int = 10,
) -> Dict:
    """Build grouped review exports from aggregation keys and outcome case items."""
    source_filenames = source_filenames or {}
    theme_rows = aggregation.get("theme_strength_matrix", [])
    theme_store = {
        row.get("theme_id"): {
            "theme_label": row.get("theme_name", row.get("theme_id")),
            "theme_rank_data": _theme_rank_data(row),
            "n_matches": 0,
            "n_high_confidence": 0,
            "n_win_drivers": 0,
            "action_lanes": {},
        }
        for row in theme_rows
        if row.get("theme_id")
    }

    flat_matches = []
    duplicates = []
    dedup_seen = {}

    for case_index, case in enumerate(outcome_cases):
        filename = source_filenames.get(case_index)
        for match in _matches_from_outcome_case(case, filename):
            flat_matches.append(match)
            theme_id = match["theme"]
            if theme_id not in theme_store:
                theme_store[theme_id] = {
                    "theme_label": match.get("theme_label") or theme_id,
                    "theme_rank_data": {},
                    "n_matches": 0,
                    "n_high_confidence": 0,
                    "n_win_drivers": 0,
                    "action_lanes": {},
                }

            key = _dedup_key(match)
            previous = dedup_seen.get(key)
            if previous is None:
                dedup_seen[key] = match
                _add_match(theme_store[theme_id], match)
                continue

            kept, duplicate = _higher_ranked(previous, match)
            duplicates.append(duplicate)
            duplicate["review_status"] = "DUPLICATE"
            if kept is match:
                _remove_match(theme_store[theme_id], previous)
                _add_match(theme_store[theme_id], match)
                dedup_seen[key] = match

    _sort_store_matches(theme_store)
    review_queue = _build_review_queue(theme_store)
    top_matches = _build_top_matches(theme_store, top_n_per_theme)
    summary = _build_theme_summary(theme_store)

    return {
        "theme_store": theme_store,
        "theme_summary": summary,
        "review_queue": review_queue,
        "top_matches_per_theme": top_matches,
        "duplicates": duplicates,
        "flat_matches": flat_matches,
    }


def write_theme_store_outputs(theme_store_bundle: Dict, output_dir: Path) -> Dict[str, Path]:
    """Write theme_store.json, theme_summary.csv, review_queue.csv, and top matches CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "theme_store": output_dir / "theme_store.json",
        "theme_summary": output_dir / "theme_summary.csv",
        "review_queue": output_dir / "review_queue.csv",
        "top_matches_per_theme": output_dir / "top_matches_per_theme.csv",
    }

    paths["theme_store"].write_text(
        json.dumps(theme_store_bundle["theme_store"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(paths["theme_summary"], theme_store_bundle["theme_summary"], _summary_fields())
    _write_csv(paths["review_queue"], theme_store_bundle["review_queue"], _match_fields())
    _write_csv(paths["top_matches_per_theme"], theme_store_bundle["top_matches_per_theme"], _match_fields())
    return paths


def load_outcome_cases(paths: Iterable[Path]) -> tuple[List[Dict], Dict[int, str]]:
    """Load outcome-optimized case JSON files and retain source filenames."""
    cases = []
    source_filenames = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            cases.append(json.load(f))
        source_filenames[len(cases) - 1] = path.name
    return cases, source_filenames


def load_flat_match_records(path: Path) -> List[Dict]:
    """Load flat classified match records from CSV, JSON, or JSONL."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("matches", [])
    raise ValueError(f"Unsupported match-record input type: {path}")


def _matches_from_outcome_case(case: Dict, filename: Optional[str]) -> List[Dict]:
    metadata = case.get("case_metadata", {})
    outcome = case.get("outcome_optimization", {})
    causal_by_signal = {
        item.get("signal_id"): item
        for item in outcome.get("signal_causal_weights", [])
        if isinstance(item, dict)
    }
    matches = []
    for signal in case.get("judgment_signals", []):
        signal_id = signal.get("signal_id")
        causal = causal_by_signal.get(signal_id, {})
        match = {
            "case_id": metadata.get("case_number"),
            "case_name": metadata.get("case_name", "Unknown"),
            "filename": filename,
            "source_pointer": _format_source_pointer(signal.get("judgment_references", [])),
            "paragraph_reference": "; ".join(str(ref) for ref in signal.get("judgment_references", [])),
            "source_text": signal.get("source_text"),
            "ws_atom_id": signal.get("ws_atom_id"),
            "ws_reference": signal.get("mapped_theme_id"),
            "ws_text": signal.get("ws_text"),
            "theme": signal.get("mapped_theme_id", "UNKNOWN_THEME"),
            "theme_label": signal.get("theme_name"),
            "subtheme": signal.get("subtheme") or DEFAULT_SUBTHEME,
            "effect": _normalize_enum(signal.get("effect") or signal.get("recommended_action"), EFFECT_SCORE, "NEUTRAL"),
            "case_effect": _normalize_enum(signal.get("case_effect"), CASE_EFFECT_SCORE, "NEUTRAL"),
            "confidence": _normalize_enum(signal.get("dictionary_match_confidence"), CONFIDENCE_SCORE, "MEDIUM"),
            "summary": signal.get("signal_summary"),
            "relevance_to_ws": signal.get("relevance_to_ws"),
            "causal_weight": causal.get("causal_weight", "UNKNOWN"),
            "causal_weight_reason": causal.get("causal_weight_reason"),
            "factual_hooks": _as_list(signal.get("factual_hooks", [])),
            "legal_functions": _as_list(signal.get("legal_functions", [])),
            "review_status": signal.get("review_status") or DEFAULT_REVIEW_STATUS,
        }
        match["action_lane"] = match["effect"]
        match["rank_score"] = _rank_score(match)
        match["review_priority_score"] = _review_priority_score(match)
        match["normalization_flags"] = _normalization_flags(signal, match)
        matches.append(match)
    return matches


def _theme_rank_data(row: Dict) -> Dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"theme_id", "theme_name"}
    }


def _rank_score(match: Dict) -> float:
    return round(
        0.50 * CASE_EFFECT_SCORE.get(match["case_effect"], CASE_EFFECT_SCORE["NEUTRAL"])
        + 0.30 * CONFIDENCE_SCORE.get(match["confidence"], CONFIDENCE_SCORE["MEDIUM"])
        + 0.20 * EFFECT_SCORE.get(match["effect"], EFFECT_SCORE["NEUTRAL"]),
        4,
    )


def _review_priority_score(match: Dict) -> float:
    confidence = CONFIDENCE_SCORE.get(match["confidence"], CONFIDENCE_SCORE["MEDIUM"])
    return round(abs(match["rank_score"]) + (0.15 if match["case_effect"] == "ADVERSE" else 0) + (0.10 * confidence), 4)


def _normalize_enum(value, lookup: Dict[str, float], default: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in lookup:
        return normalized
    return default


def _normalization_flags(signal: Dict, match: Dict) -> List[str]:
    flags = []
    if signal.get("case_effect") != match["case_effect"]:
        flags.append("case_effect_normalized")
    if signal.get("dictionary_match_confidence") != match["confidence"]:
        flags.append("confidence_normalized")
    if (signal.get("effect") or signal.get("recommended_action")) != match["effect"]:
        flags.append("effect_normalized")
    if match["subtheme"] == DEFAULT_SUBTHEME:
        flags.append("subtheme_missing")
    if not match.get("factual_hooks"):
        flags.append("factual_hooks_missing")
    if not match.get("legal_functions"):
        flags.append("legal_functions_missing")
    if not match.get("source_pointer"):
        flags.append("source_pointer_missing")
    return flags


def _format_source_pointer(references: List[str]) -> Optional[str]:
    if not references:
        return None
    return "judgment paragraphs: " + ", ".join(str(ref) for ref in references)


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dedup_key(match: Dict) -> str:
    return "|".join([
        str(match.get("filename") or ""),
        str(match.get("theme") or ""),
        str(match.get("action_lane") or ""),
        str(match.get("subtheme") or ""),
        _normalize_summary(match.get("summary") or ""),
    ])


def _normalize_summary(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    return re.sub(r"\s+", " ", value)


def _higher_ranked(left: Dict, right: Dict) -> tuple[Dict, Dict]:
    if right["rank_score"] > left["rank_score"]:
        return right, left
    if right["rank_score"] == left["rank_score"] and right["review_priority_score"] > left["review_priority_score"]:
        return right, left
    return left, right


def _add_match(theme: Dict, match: Dict) -> None:
    action_lane = match["action_lane"]
    subtheme = match["subtheme"]
    theme["action_lanes"].setdefault(action_lane, {"n_matches": 0, "subthemes": {}})
    lane = theme["action_lanes"][action_lane]
    lane["subthemes"].setdefault(subtheme, {"n_matches": 0, "matches": []})
    lane["subthemes"][subtheme]["matches"].append(match)
    lane["subthemes"][subtheme]["n_matches"] += 1
    lane["n_matches"] += 1
    theme["n_matches"] += 1
    if match["confidence"] == "HIGH":
        theme["n_high_confidence"] += 1
    if match["case_effect"] == "WIN_DRIVER":
        theme["n_win_drivers"] += 1


def _remove_match(theme: Dict, match: Dict) -> None:
    lane = theme["action_lanes"].get(match["action_lane"])
    if not lane:
        return
    subtheme = lane["subthemes"].get(match["subtheme"])
    if not subtheme:
        return
    subtheme["matches"] = [item for item in subtheme["matches"] if item is not match]
    subtheme["n_matches"] = len(subtheme["matches"])
    lane["n_matches"] -= 1
    theme["n_matches"] -= 1
    if match["confidence"] == "HIGH":
        theme["n_high_confidence"] -= 1
    if match["case_effect"] == "WIN_DRIVER":
        theme["n_win_drivers"] -= 1


def _sort_store_matches(theme_store: Dict) -> None:
    for theme in theme_store.values():
        for lane in theme["action_lanes"].values():
            for subtheme in lane["subthemes"].values():
                subtheme["matches"].sort(
                    key=lambda item: (item["rank_score"], item["review_priority_score"], item["case_name"]),
                    reverse=True,
                )


def _build_review_queue(theme_store: Dict) -> List[Dict]:
    rows = []
    for theme_id, theme in theme_store.items():
        for action_lane, lane in theme["action_lanes"].items():
            for subtheme_id, subtheme in lane["subthemes"].items():
                for match in subtheme["matches"]:
                    if match.get("review_status") == "UNREVIEWED":
                        rows.append(_flatten_match(theme_id, theme["theme_label"], action_lane, subtheme_id, match))
    return sorted(rows, key=lambda row: (row["theme"], row["action_lane"], row["subtheme"], -float(row["rank_score"])))


def _build_top_matches(theme_store: Dict, top_n_per_theme: int) -> List[Dict]:
    rows = []
    for theme_id, theme in theme_store.items():
        matches = []
        for action_lane, lane in theme["action_lanes"].items():
            for subtheme_id, subtheme in lane["subthemes"].items():
                matches.extend(_flatten_match(theme_id, theme["theme_label"], action_lane, subtheme_id, match) for match in subtheme["matches"])
        rows.extend(sorted(matches, key=lambda row: float(row["rank_score"]), reverse=True)[:top_n_per_theme])
    return rows


def _build_theme_summary(theme_store: Dict) -> List[Dict]:
    rows = []
    for theme_id, theme in theme_store.items():
        matches = []
        subtheme_ids = set()
        for lane in theme["action_lanes"].values():
            for subtheme_id, subtheme in lane["subthemes"].items():
                subtheme_ids.add(subtheme_id)
                matches.extend(subtheme["matches"])
        top_matches = sorted(matches, key=lambda item: item["rank_score"], reverse=True)
        rows.append({
            "theme": theme_id,
            "theme_label": theme["theme_label"],
            "number_of_matches": theme["n_matches"],
            "number_of_high_confidence_matches": theme["n_high_confidence"],
            "number_of_win_drivers": theme["n_win_drivers"],
            "number_of_action_lanes": len(theme["action_lanes"]),
            "action_lanes": "; ".join(sorted(theme["action_lanes"])),
            "number_of_subthemes": len(subtheme_ids),
            "top_case_names": "; ".join(dict.fromkeys(item["case_name"] for item in top_matches[:5])),
            "top_rank_score": top_matches[0]["rank_score"] if top_matches else None,
        })
    return sorted(rows, key=lambda row: (row["top_rank_score"] is not None, row["top_rank_score"] or -999), reverse=True)


def _flatten_match(theme_id: str, theme_label: str, action_lane: str, subtheme_id: str, match: Dict) -> Dict:
    row = dict(match)
    row["theme"] = theme_id
    row["theme_label"] = theme_label
    row["action_lane"] = action_lane
    row["subtheme"] = subtheme_id
    row["factual_hooks"] = "; ".join(match.get("factual_hooks", []))
    row["legal_functions"] = "; ".join(match.get("legal_functions", []))
    row["normalization_flags"] = "; ".join(match.get("normalization_flags", []))
    return row


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_fields() -> List[str]:
    return [
        "theme",
        "theme_label",
        "number_of_matches",
        "number_of_high_confidence_matches",
        "number_of_win_drivers",
        "number_of_action_lanes",
        "action_lanes",
        "number_of_subthemes",
        "top_case_names",
        "top_rank_score",
    ]


def _match_fields() -> List[str]:
    return [
        "theme",
        "theme_label",
        "action_lane",
        "subtheme",
        "case_id",
        "case_name",
        "filename",
        "source_pointer",
        "paragraph_reference",
        "effect",
        "case_effect",
        "confidence",
        "summary",
        "relevance_to_ws",
        "causal_weight",
        "causal_weight_reason",
        "factual_hooks",
        "legal_functions",
        "review_status",
        "rank_score",
        "review_priority_score",
        "normalization_flags",
    ]
