"""Corpus-level aggregation for outcome-optimized calibration files."""

from collections import Counter, defaultdict
from typing import Dict, List

RISK_CONTROL_THEME_ID = "T20_RISK_CONTROL"
MIN_PRIMARY_CASES = 2

try:
    from dictionary_loader import get_priority_by_theme
    from outcome_lookup_tables import (
        CONFIDENCE_MULTIPLIER_LOOKUP,
        FACTUAL_PROXIMITY_LOOKUP,
        LIABILITY_OUTCOME_STRENGTH_LOOKUP,
        SCORING_PROFILE_VERSION,
        TRANSFERABILITY_LOOKUP,
    )
    from outcome_scoring import (
        apply_ranking_thresholds,
        compute_claimant_usefulness_score,
        compute_negative_penalty,
        compute_pt_score,
        compute_remedy_outcome_strength,
        compute_signal_optimization_score,
        get_threshold_profile,
    )
except ImportError:
    from src.dictionary_loader import get_priority_by_theme
    from src.outcome_lookup_tables import (
        CONFIDENCE_MULTIPLIER_LOOKUP,
        FACTUAL_PROXIMITY_LOOKUP,
        LIABILITY_OUTCOME_STRENGTH_LOOKUP,
        SCORING_PROFILE_VERSION,
        TRANSFERABILITY_LOOKUP,
    )
    from src.outcome_scoring import (
        apply_ranking_thresholds,
        compute_claimant_usefulness_score,
        compute_negative_penalty,
        compute_pt_score,
        compute_remedy_outcome_strength,
        compute_signal_optimization_score,
        get_threshold_profile,
    )


def aggregate_outcome_optimized_cases(
    cases: List[Dict],
    dictionary: Dict,
    threshold_profile: str = "pilot_20_to_30",
) -> Dict:
    """Aggregate validated outcome-optimized cases into pilot reports."""
    theme_name_by_id = {
        theme["theme_id"]: theme.get("theme_name", theme["theme_id"])
        for theme in dictionary.get("ws_theme_dictionary", [])
    }
    theme_priority_by_id = get_priority_by_theme(dictionary)
    active_profile = get_threshold_profile(threshold_profile)

    theme_rows = _initial_theme_rows(theme_name_by_id)
    case_shortlist = []
    negative_summary = defaultdict(lambda: {"count": 0, "severity_counts": Counter(), "theme_counts": Counter()})
    other_descriptions = Counter()
    remedy_null_cases = []
    risk_control_summary = _initial_risk_control_summary()
    factual_distribution = Counter()
    transferability_distribution = Counter()
    causal_weight_distribution = Counter()

    for case in cases:
        case_metadata = case.get("case_metadata", {})
        case_name = case_metadata.get("case_name", "Unknown")
        case_number = case_metadata.get("case_number")
        outcome = case.get("outcome_optimization", {})
        signal_weights = {
            item.get("signal_id"): item.get("causal_weight")
            for item in outcome.get("signal_causal_weights", [])
            if isinstance(item, dict)
        }

        factual_proximity = outcome.get("factual_proximity", "UNKNOWN")
        transferability = outcome.get("transferability_rating", "UNKNOWN")
        liability_band = outcome.get("liability_outcome_strength_band", "UNKNOWN")
        reduction_status = outcome.get("reduction_findings_status", "UNKNOWN")
        polkey_pct = outcome.get("polkey_reduction_pct")
        contribution_pct = outcome.get("contributory_fault_pct")
        negative_flags = outcome.get("negative_theme_flags", [])

        factual_distribution[factual_proximity] += 1
        transferability_distribution[transferability] += 1
        for causal_weight in signal_weights.values():
            causal_weight_distribution[causal_weight] += 1

        remedy_strength = compute_remedy_outcome_strength(
            polkey_pct,
            contribution_pct,
            reduction_status,
        )
        if remedy_strength is None:
            remedy_null_cases.append({
                "case_name": case_name,
                "case_number": case_number,
                "reduction_findings_status": reduction_status,
                "award_reduction_notes": outcome.get("award_reduction_notes"),
            })

        negative_penalty = compute_negative_penalty(negative_flags)
        pt_score = compute_pt_score(factual_proximity, transferability)
        liability_score = LIABILITY_OUTCOME_STRENGTH_LOOKUP.get(liability_band, 0.50)
        liability_signal_positive_score = 0.0

        for flag in negative_flags:
            if not isinstance(flag, dict):
                continue
            pattern = flag.get("negative_pattern")
            severity = flag.get("severity")
            theme_id = flag.get("theme_id")
            negative_summary[pattern]["count"] += 1
            negative_summary[pattern]["severity_counts"][severity] += 1
            negative_summary[pattern]["theme_counts"][theme_id] += 1
            _update_risk_control_summary(risk_control_summary, case_name, pattern, theme_id)
            if pattern == "OTHER":
                other_descriptions[flag.get("reason", "").strip()] += 1
            if theme_id in theme_rows:
                penalty = compute_negative_penalty([flag])
                theme_rows[theme_id]["total_negative_penalty"] += penalty
                theme_rows[theme_id]["net_theme_score"] += penalty

        seen_themes_in_case = set()
        high_confidence_themes_in_case = set()
        for signal in case.get("judgment_signals", []):
            theme_id = signal.get("mapped_theme_id")
            if theme_id not in theme_rows:
                continue
            causal_weight = signal_weights.get(signal.get("signal_id"), "UNKNOWN")
            signal_score = compute_signal_optimization_score(
                causal_weight,
                liability_band,
                factual_proximity,
                transferability,
                signal.get("dictionary_match_confidence", "MEDIUM"),
            )
            row = theme_rows[theme_id]
            row["signal_scores"].append(signal_score)
            row["liability_scores"].append(liability_score)
            row["pt_scores"].append(pt_score)
            row["confidence_scores"].append(
                CONFIDENCE_MULTIPLIER_LOOKUP.get(signal.get("dictionary_match_confidence"), 0.75)
            )
            if theme_id == RISK_CONTROL_THEME_ID:
                risk_control_summary["risk_control_signal_count"] += 1
            else:
                positive_signal_score = max(0.0, signal_score)
                row["total_positive_score"] += positive_signal_score
                row["net_theme_score"] += signal_score
                liability_signal_positive_score += positive_signal_score
            if causal_weight == "DECISIVE":
                row["decisive_signal_count"] += 1
            if causal_weight == "CONTRIBUTING":
                row["contributing_signal_count"] += 1
            seen_themes_in_case.add(theme_id)
            if signal.get("dictionary_match_confidence") == "HIGH" and causal_weight in {"DECISIVE", "CONTRIBUTING"}:
                high_confidence_themes_in_case.add(theme_id)

        for theme_id in seen_themes_in_case:
            theme_rows[theme_id]["supporting_cases"].add(case_name)
        for theme_id in high_confidence_themes_in_case:
            theme_rows[theme_id]["high_confidence_cases"].add(case_name)

        claimant_usefulness_score = compute_claimant_usefulness_score(
            liability_band,
            polkey_pct,
            contribution_pct,
            reduction_status,
            factual_proximity,
            transferability,
            negative_flags,
        )
        remedy_usefulness_score = remedy_strength if remedy_strength is not None else 0.50
        liability_usefulness_score = round(liability_signal_positive_score, 4)

        case_shortlist.append({
            "case_name": case_name,
            "case_number": case_number,
            "claimant_usefulness_score": claimant_usefulness_score,
            "liability_usefulness_score": liability_usefulness_score,
            "liability_usefulness_band": _liability_usefulness_band(liability_usefulness_score),
            "remedy_usefulness_score": remedy_usefulness_score,
            "remedy_usefulness_band": _remedy_usefulness_band(remedy_strength, polkey_pct, contribution_pct, reduction_status),
            "overall_use_mode": _overall_use_mode(liability_usefulness_score, remedy_strength, negative_penalty),
            "liability_outcome_strength_band": liability_band,
            "factual_proximity": factual_proximity,
            "transferability_rating": transferability,
            "negative_penalty": negative_penalty,
        })

    theme_strength_matrix = _finalize_theme_rows(
        theme_rows,
        theme_priority_by_id,
        active_profile,
        len(cases),
    )
    ranked_theme_optimization_table = _ranked_table(theme_strength_matrix)
    finalized_risk_control_summary = _finalize_risk_control_summary(risk_control_summary)
    return {
        "aggregation_metadata": {
            "scoring_profile_version": SCORING_PROFILE_VERSION,
            "threshold_profile": threshold_profile,
            "case_count": len(cases),
            "min_primary_cases": MIN_PRIMARY_CASES,
        },
        "theme_strength_matrix": theme_strength_matrix,
        "ranked_theme_optimization_table": ranked_theme_optimization_table,
        "risk_control_summary": finalized_risk_control_summary,
        "negative_theme_summary": _format_negative_summary(negative_summary),
        "other_negative_pattern_review_queue": _format_other_queue(other_descriptions, len(cases)),
        "remedy_null_cases_report": remedy_null_cases,
        "factual_proximity_distribution": dict(factual_distribution),
        "transferability_distribution": dict(transferability_distribution),
        "signal_causal_weight_distribution": dict(causal_weight_distribution),
        "case_shortlist": sorted(
            case_shortlist,
            key=lambda row: (row["claimant_usefulness_score"], row["liability_usefulness_score"]),
            reverse=True
        ),
        "ws_optimization_mapping": {
            "status": "blocked_until_ws_theme_anchor_map",
            "note": "Post-pilot ws_theme_anchor_map is required before automated WS paragraph mapping.",
        },
        "pilot_review_report": {
            "threshold_profile_used": threshold_profile,
            "review_required": True,
            "review_points": [
                "Review blended pt_score 0.40/0.60 weights.",
                "Review pilot_20_to_30 thresholds before scaling.",
                "Review DISMISSAL_WITHIN_REASONABLE_RESPONSES penalty.",
                "Review OTHER negative pattern queue for enum promotion.",
            ],
        },
    }


def _initial_theme_rows(theme_name_by_id):
    rows = {}
    for theme_id, theme_name in theme_name_by_id.items():
        rows[theme_id] = {
            "theme_id": theme_id,
            "theme_name": theme_name,
            "supporting_cases": set(),
            "decisive_signal_count": 0,
            "contributing_signal_count": 0,
            "liability_scores": [],
            "pt_scores": [],
            "confidence_scores": [],
            "signal_scores": [],
            "total_positive_score": 0.0,
            "total_negative_penalty": 0.0,
            "net_theme_score": 0.0,
            "high_confidence_cases": set(),
        }
    return rows


def _finalize_theme_rows(theme_rows, theme_priority_by_id, active_profile, case_count):
    finalized = []
    for theme_id, row in theme_rows.items():
        high_confidence_case_count = len(row["high_confidence_cases"])
        net_theme_score = round(row["net_theme_score"], 4)
        supporting_case_count = len(row["supporting_cases"])
        recommendation = _theme_recommendation(
            theme_id,
            row,
            net_theme_score,
            high_confidence_case_count,
            active_profile,
            case_count,
        )
        finalized.append({
            "theme_id": theme_id,
            "theme_name": row["theme_name"],
            "supporting_case_count": supporting_case_count,
            "decisive_signal_count": row["decisive_signal_count"],
            "contributing_signal_count": row["contributing_signal_count"],
            "avg_liability_strength": _avg(row["liability_scores"]),
            "avg_pt_score": _avg(row["pt_scores"]),
            "avg_dictionary_match_confidence": _avg(row["confidence_scores"]),
            "total_positive_score": round(row["total_positive_score"], 4),
            "total_negative_penalty": round(row["total_negative_penalty"], 4),
            "net_theme_score": net_theme_score,
            "high_confidence_case_count": high_confidence_case_count,
            "recommendation": recommendation,
        })
    return sorted(
        finalized,
        key=lambda item: (-item["net_theme_score"], theme_priority_by_id.get(item["theme_id"], 999)),
    )


def _ranked_table(theme_strength_matrix):
    table = {
        "REINFORCE_PRIMARY": [],
        "REINFORCE_SUPPORTING": [],
        "REFRAME": [],
        "MONITOR": [],
        "AVOID": [],
        "RISK_CONTROL": [],
        "NO_SIGNAL": [],
    }
    for row in theme_strength_matrix:
        table[row["recommendation"]].append(row)
    return table


def _theme_recommendation(
    theme_id,
    row,
    net_theme_score,
    high_confidence_case_count,
    active_profile,
    case_count,
):
    if (
        len(row["supporting_cases"]) == 0
        and not row["signal_scores"]
        and round(row["total_negative_penalty"], 4) == 0
    ):
        return "NO_SIGNAL"
    if theme_id == RISK_CONTROL_THEME_ID:
        return "RISK_CONTROL"

    recommendation = apply_ranking_thresholds(
        net_theme_score,
        high_confidence_case_count,
        active_profile,
    )
    if recommendation == "REINFORCE_PRIMARY" and case_count < MIN_PRIMARY_CASES:
        return "REINFORCE_SUPPORTING"
    return recommendation


def _liability_usefulness_band(score):
    if score >= 1.5:
        return "HIGH"
    if score >= 0.75:
        return "MEDIUM_HIGH"
    if score >= 0.25:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "NONE"


def _remedy_usefulness_band(remedy_strength, polkey_pct, contribution_pct, reduction_status):
    if reduction_status != "DETERMINED" or remedy_strength is None:
        return "UNKNOWN_OR_NOT_DETERMINED"
    if (polkey_pct or 0) >= 50 or (contribution_pct or 0) >= 50 or remedy_strength < 0.30:
        return "ADVERSE"
    if remedy_strength >= 0.85:
        return "STRONG"
    if remedy_strength >= 0.60:
        return "MODERATE"
    return "LIMITED"


def _overall_use_mode(liability_usefulness_score, remedy_strength, negative_penalty):
    if liability_usefulness_score > 0 and remedy_strength is not None and remedy_strength < 0.30:
        return "LIABILITY_ONLY_PROCEDURAL_ANALOGY"
    if liability_usefulness_score > 0 and negative_penalty < 0:
        return "CAUTIOUS_LIABILITY_ANALOGY"
    if liability_usefulness_score > 0:
        return "GENERAL_SUPPORT"
    if negative_penalty < 0:
        return "RISK_ONLY_QUARANTINE"
    return "NO_USE_IDENTIFIED"


def _initial_risk_control_summary():
    return {
        "risk_control_cases": set(),
        "polkey_risk_cases": set(),
        "contribution_risk_cases": set(),
        "low_remedy_win_cases": set(),
        "negative_pattern_counts": Counter(),
        "risk_control_signal_count": 0,
    }


def _update_risk_control_summary(summary, case_name, pattern, theme_id):
    if theme_id != RISK_CONTROL_THEME_ID and pattern not in {
        "HIGH_POLKEY_REDUCTION",
        "HIGH_CONTRIBUTORY_FAULT",
        "PROCEDURAL_ONLY_WIN_LOW_REMEDY",
    }:
        return

    summary["risk_control_cases"].add(case_name)
    summary["negative_pattern_counts"][pattern] += 1
    if pattern == "HIGH_POLKEY_REDUCTION":
        summary["polkey_risk_cases"].add(case_name)
    if pattern == "HIGH_CONTRIBUTORY_FAULT":
        summary["contribution_risk_cases"].add(case_name)
    if pattern == "PROCEDURAL_ONLY_WIN_LOW_REMEDY":
        summary["low_remedy_win_cases"].add(case_name)


def _finalize_risk_control_summary(summary):
    return {
        "risk_control_case_count": len(summary["risk_control_cases"]),
        "polkey_risk_cases": len(summary["polkey_risk_cases"]),
        "contribution_risk_cases": len(summary["contribution_risk_cases"]),
        "low_remedy_win_cases": len(summary["low_remedy_win_cases"]),
        "risk_control_signal_count": summary["risk_control_signal_count"],
        "negative_pattern_counts": dict(summary["negative_pattern_counts"]),
        "recommended_use": "QUARANTINE" if summary["risk_control_cases"] else "NO_RISK_CONTROL_SIGNAL",
    }


def _format_negative_summary(negative_summary):
    formatted = []
    for pattern, data in negative_summary.items():
        formatted.append({
            "negative_pattern": pattern,
            "count": data["count"],
            "severity_counts": dict(data["severity_counts"]),
            "theme_counts": dict(data["theme_counts"]),
        })
    return sorted(formatted, key=lambda item: (-item["count"], item["negative_pattern"] or ""))


def _format_other_queue(other_descriptions, case_count):
    threshold = 3 if case_count < 100 else 5
    return [
        {
            "description": description,
            "count": count,
            "promotion_threshold": threshold,
            "promote_for_enum_review": count >= threshold,
        }
        for description, count in other_descriptions.most_common()
    ]


def _avg(values):
    if not values:
        return None
    return round(sum(values) / len(values), 4)
