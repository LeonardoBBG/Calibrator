"""Deterministic scoring functions for the outcome optimization layer."""

try:
    from outcome_lookup_tables import (
        CAUSAL_WEIGHT_LOOKUP,
        CONFIDENCE_MULTIPLIER_LOOKUP,
        FACTUAL_PROXIMITY_LOOKUP,
        LIABILITY_OUTCOME_STRENGTH_LOOKUP,
        NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP,
        PT_SCORE_BLEND_WEIGHTS,
        RANKING_THRESHOLD_PROFILES,
        REMEDY_NULL_AGGREGATION_DEFAULT,
        SEVERITY_MULTIPLIER,
        TRANSFERABILITY_LOOKUP,
    )
except ImportError:
    from src.outcome_lookup_tables import (
        CAUSAL_WEIGHT_LOOKUP,
        CONFIDENCE_MULTIPLIER_LOOKUP,
        FACTUAL_PROXIMITY_LOOKUP,
        LIABILITY_OUTCOME_STRENGTH_LOOKUP,
        NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP,
        PT_SCORE_BLEND_WEIGHTS,
        RANKING_THRESHOLD_PROFILES,
        REMEDY_NULL_AGGREGATION_DEFAULT,
        SEVERITY_MULTIPLIER,
        TRANSFERABILITY_LOOKUP,
    )


def compute_remedy_outcome_strength(
    polkey_pct,
    contribution_pct,
    reduction_findings_status,
):
    """Compute remedy strength from express Polkey/contribution findings."""
    not_determined = {"NOT_DETERMINED", "NOT_ADDRESSED", "UNKNOWN"}
    if reduction_findings_status in not_determined:
        return None
    polkey = (polkey_pct or 0) / 100
    contribution = (contribution_pct or 0) / 100
    return round((1 - polkey) * (1 - contribution), 2)


def remedy_strength_for_aggregation(
    polkey_pct,
    contribution_pct,
    reduction_findings_status,
):
    """Resolve undetermined remedy findings to the neutral aggregation default."""
    computed = compute_remedy_outcome_strength(
        polkey_pct,
        contribution_pct,
        reduction_findings_status,
    )
    return REMEDY_NULL_AGGREGATION_DEFAULT if computed is None else computed


def compute_pt_score(
    factual_proximity,
    transferability_rating,
    factual_proximity_lookup=None,
    transferability_lookup=None,
    pt_score_blend_weights=None,
):
    """Compute blended factual-proximity/transferability score."""
    factual_proximity_lookup = factual_proximity_lookup or FACTUAL_PROXIMITY_LOOKUP
    transferability_lookup = transferability_lookup or TRANSFERABILITY_LOOKUP
    pt_score_blend_weights = pt_score_blend_weights or PT_SCORE_BLEND_WEIGHTS

    fp = factual_proximity_lookup.get(factual_proximity, 0.30)
    tr = transferability_lookup.get(transferability_rating, 0.30)
    fp_w = pt_score_blend_weights["factual_proximity_weight"]
    tr_w = pt_score_blend_weights["transferability_weight"]
    return round((fp_w * fp) + (tr_w * tr), 4)


def compute_signal_optimization_score(
    causal_weight,
    liability_outcome_strength_band,
    factual_proximity,
    transferability_rating,
    dictionary_match_confidence,
    causal_weight_lookup=None,
    liability_lookup=None,
    factual_proximity_lookup=None,
    transferability_lookup=None,
    pt_score_blend_weights=None,
    confidence_multiplier_lookup=None,
):
    """Compute per-signal optimization score from categorical inputs."""
    causal_weight_lookup = causal_weight_lookup or CAUSAL_WEIGHT_LOOKUP
    liability_lookup = liability_lookup or LIABILITY_OUTCOME_STRENGTH_LOOKUP
    confidence_multiplier_lookup = confidence_multiplier_lookup or CONFIDENCE_MULTIPLIER_LOOKUP

    cw = causal_weight_lookup.get(causal_weight, 0.0)
    ls = liability_lookup.get(liability_outcome_strength_band, 0.50)
    pt = compute_pt_score(
        factual_proximity,
        transferability_rating,
        factual_proximity_lookup or FACTUAL_PROXIMITY_LOOKUP,
        transferability_lookup or TRANSFERABILITY_LOOKUP,
        pt_score_blend_weights or PT_SCORE_BLEND_WEIGHTS,
    )
    cm = confidence_multiplier_lookup.get(dictionary_match_confidence, 0.75)
    return round(cw * ls * pt * cm, 4)


def compute_negative_penalty(
    negative_theme_flags,
    base_penalty_lookup=None,
    severity_multiplier_lookup=None,
):
    """Compute total case penalty from closed-enum negative flags."""
    base_penalty_lookup = base_penalty_lookup or NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP
    severity_multiplier_lookup = severity_multiplier_lookup or SEVERITY_MULTIPLIER

    total = 0.0
    for flag in negative_theme_flags:
        base = base_penalty_lookup.get(flag["negative_pattern"], -0.30)
        multiplier = severity_multiplier_lookup.get(flag["severity"], 0.75)
        total += base * multiplier
    return round(total, 4)


def compute_claimant_usefulness_score(
    liability_outcome_strength_band,
    polkey_pct,
    contribution_pct,
    reduction_findings_status,
    factual_proximity,
    transferability_rating,
    negative_theme_flags,
    liability_lookup=None,
    factual_proximity_lookup=None,
    transferability_lookup=None,
    pt_score_blend_weights=None,
    base_penalty_lookup=None,
    severity_multiplier_lookup=None,
):
    """Compute overall case usefulness for aggregation only."""
    liability_lookup = liability_lookup or LIABILITY_OUTCOME_STRENGTH_LOOKUP
    liability_score = liability_lookup.get(liability_outcome_strength_band, 0.50)
    remedy_score = remedy_strength_for_aggregation(
        polkey_pct,
        contribution_pct,
        reduction_findings_status,
    )
    pt = compute_pt_score(
        factual_proximity,
        transferability_rating,
        factual_proximity_lookup or FACTUAL_PROXIMITY_LOOKUP,
        transferability_lookup or TRANSFERABILITY_LOOKUP,
        pt_score_blend_weights or PT_SCORE_BLEND_WEIGHTS,
    )
    base = liability_score * 0.50 + remedy_score * 0.30 + pt * 0.20
    penalty = compute_negative_penalty(
        negative_theme_flags,
        base_penalty_lookup or NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP,
        severity_multiplier_lookup or SEVERITY_MULTIPLIER,
    )
    return round(max(0.0, base + penalty), 4)


def apply_ranking_thresholds(
    net_theme_score,
    high_confidence_case_count,
    active_profile,
):
    """Apply a ranking threshold profile to a net theme score."""
    rp = active_profile["REINFORCE_PRIMARY"]
    rs = active_profile["REINFORCE_SUPPORTING"]
    rf = active_profile["REFRAME"]
    mo = active_profile["MONITOR"]

    if (
        net_theme_score >= rp["net_theme_score_gte"]
        and high_confidence_case_count >= rp.get("high_confidence_case_count_gte", 0)
    ):
        return "REINFORCE_PRIMARY"
    if net_theme_score >= rs["net_theme_score_gte"]:
        return "REINFORCE_SUPPORTING"
    if net_theme_score >= rf["net_theme_score_gte"]:
        return "REFRAME"
    if net_theme_score >= mo["net_theme_score_gte"]:
        return "MONITOR"
    return "AVOID"


def get_threshold_profile(profile_name=None):
    """Return a configured ranking threshold profile."""
    profile_name = profile_name or RANKING_THRESHOLD_PROFILES["active_profile"]
    return RANKING_THRESHOLD_PROFILES[profile_name]
