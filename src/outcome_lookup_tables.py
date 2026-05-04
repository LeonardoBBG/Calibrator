"""Versioned lookup tables for outcome optimization scoring."""

SCORING_PROFILE_VERSION = "0.2_pilot"

LIABILITY_OUTCOME_STRENGTH_LOOKUP = {
    "STRONG_WIN": 1.00,
    "MODERATE_WIN": 0.80,
    "NARROW_WIN": 0.65,
    "PARTIAL": 0.50,
    "LOSS": 0.00,
    "UNKNOWN": 0.50,
}

CAUSAL_WEIGHT_LOOKUP = {
    "DECISIVE": 1.00,
    "CONTRIBUTING": 0.65,
    "PERIPHERAL": 0.30,
    "NEUTRAL": 0.00,
    "NEGATIVE": -0.70,
    "UNKNOWN": 0.00,
}

FACTUAL_PROXIMITY_LOOKUP = {
    "HIGH": 1.00,
    "MEDIUM": 0.70,
    "LOW": 0.40,
    "ANALOGY_ONLY": 0.25,
    "UNKNOWN": 0.30,
}

TRANSFERABILITY_LOOKUP = {
    "HIGH": 1.00,
    "MEDIUM": 0.70,
    "LOW": 0.40,
    "ANALOGY_ONLY": 0.25,
    "NON_TRANSFERABLE": 0.00,
    "UNKNOWN": 0.30,
}

PT_SCORE_BLEND_WEIGHTS = {
    "factual_proximity_weight": 0.40,
    "transferability_weight": 0.60,
    "review_required_after": "smoke_test_20_to_30_cases",
}

CONFIDENCE_MULTIPLIER_LOOKUP = {
    "HIGH": 1.00,
    "MEDIUM": 0.75,
    "LOW": 0.50,
}

REMEDY_NULL_AGGREGATION_DEFAULT = 0.50

NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP = {
    "EXPLANATION_RAISED_LATE": -0.40,
    "EXPLANATION_INCONSISTENT": -0.50,
    "CLAIMANT_ADMITTED_CORE_MISCONDUCT": -0.65,
    "CLAIMANT_FAILED_TO_ENGAGE": -0.40,
    "CLAIMANT_CREDIBILITY_REJECTED": -0.65,
    "EMPLOYER_HAD_STRONG_CONTEMPORANEOUS_EVIDENCE": -0.60,
    "EMPLOYER_INVESTIGATION_FOUND_REASONABLE": -0.60,
    "APPEAL_CURED_DEFECT": -0.55,
    "PROCEDURAL_ONLY_WIN_LOW_REMEDY": -0.50,
    "HIGH_POLKEY_REDUCTION": -0.55,
    "HIGH_CONTRIBUTORY_FAULT": -0.55,
    "DISMISSAL_WITHIN_REASONABLE_RESPONSES": -0.70,
    "NO_MATERIAL_PROCEDURAL_PREJUDICE": -0.45,
    "CLEAN_RECORD_GIVEN_LOW_WEIGHT": -0.30,
    "COMPARATOR_ARGUMENT_REJECTED": -0.35,
    "TECHNICAL_OR_SYSTEM_EXPLANATION_REJECTED": -0.60,
    "ALTERNATIVE_SANCTION_NOT_REQUIRED": -0.35,
    "OTHER": -0.30,
}

SEVERITY_MULTIPLIER = {
    "LOW": 0.50,
    "MEDIUM": 0.75,
    "HIGH": 1.00,
}

RANKING_THRESHOLD_PROFILES = {
    "active_profile": "pilot_20_to_30",
    "pilot_20_to_30": {
        "version": "0.2_pilot",
        "REINFORCE_PRIMARY": {
            "net_theme_score_gte": 5,
            "high_confidence_case_count_gte": 2,
        },
        "REINFORCE_SUPPORTING": {
            "net_theme_score_gte": 2,
            "net_theme_score_lt": 5,
        },
        "REFRAME": {
            "net_theme_score_gte": 0,
            "net_theme_score_lt": 2,
        },
        "MONITOR": {
            "net_theme_score_gte": -2,
            "net_theme_score_lt": 0,
        },
        "AVOID": {
            "net_theme_score_lt": -2,
        },
    },
    "scale_500": {
        "version": "to_be_set_post_pilot",
        "REINFORCE_PRIMARY": {
            "net_theme_score_gte": 50,
            "high_confidence_case_count_gte": 10,
        },
        "REINFORCE_SUPPORTING": {
            "net_theme_score_gte": 20,
            "net_theme_score_lt": 50,
        },
        "REFRAME": {
            "net_theme_score_gte": 0,
            "net_theme_score_lt": 20,
        },
        "MONITOR": {
            "net_theme_score_gte": -20,
            "net_theme_score_lt": 0,
        },
        "AVOID": {
            "net_theme_score_lt": -20,
        },
    },
}

FACTUAL_PROXIMITY_ENUM = set(FACTUAL_PROXIMITY_LOOKUP)
TRANSFERABILITY_ENUM = set(TRANSFERABILITY_LOOKUP)
CLAIMANT_LIABILITY_OUTCOME_ENUM = {"WIN", "LOSS", "PARTIAL", "MIXED", "UNKNOWN"}
LIABILITY_OUTCOME_STRENGTH_ENUM = set(LIABILITY_OUTCOME_STRENGTH_LOOKUP)
REMEDY_STATUS_ENUM = {"COMPLETED", "RESERVED", "OUTSTANDING", "NOT_ADDRESSED", "UNKNOWN"}
REDUCTION_FINDINGS_STATUS_ENUM = {"DETERMINED", "NOT_DETERMINED", "NOT_ADDRESSED", "UNKNOWN"}
CAUSAL_WEIGHT_ENUM = set(CAUSAL_WEIGHT_LOOKUP)
NEGATIVE_PATTERN_ENUM = set(NEGATIVE_PATTERN_BASE_PENALTY_LOOKUP)
SEVERITY_ENUM = set(SEVERITY_MULTIPLIER)

REQUIRED_OUTCOME_QC_TRUE_FLAGS = {
    "outcome_optimization_added",
    "no_authority_weight_generated",
    "no_claimant_usefulness_score_generated",
    "no_causal_weight_score_generated",
    "no_remedy_outcome_strength_generated",
    "liability_strength_band_used",
    "factual_proximity_assigned",
    "transferability_rating_assigned",
    "remedy_status_assigned",
    "reduction_findings_status_assigned",
    "remedy_null_rule_followed",
    "negative_pattern_closed_enum_used",
    "signal_causal_weights_match_signal_ids",
    "negative_theme_flags_checked",
    "no_existing_schema_fields_modified",
}

FORBIDDEN_OUTCOME_FIELDS = {
    "authority_level",
    "authority_weight",
    "claimant_usefulness_score",
    "causal_weight_score",
    "remedy_outcome_strength",
}
