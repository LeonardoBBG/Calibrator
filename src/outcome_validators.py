"""Validation for outcome-optimized calibration JSON files."""

try:
    from dictionary_loader import get_theme_ids
    from outcome_lookup_tables import (
        CAUSAL_WEIGHT_ENUM,
        CLAIMANT_LIABILITY_OUTCOME_ENUM,
        FACTUAL_PROXIMITY_ENUM,
        FORBIDDEN_OUTCOME_FIELDS,
        LIABILITY_OUTCOME_STRENGTH_ENUM,
        NEGATIVE_PATTERN_ENUM,
        REDUCTION_FINDINGS_STATUS_ENUM,
        REMEDY_STATUS_ENUM,
        REQUIRED_OUTCOME_QC_TRUE_FLAGS,
        SEVERITY_ENUM,
        TRANSFERABILITY_ENUM,
    )
    from validators import CalibrationValidationContext, validate_calibration_output
except ImportError:
    from src.dictionary_loader import get_theme_ids
    from src.outcome_lookup_tables import (
        CAUSAL_WEIGHT_ENUM,
        CLAIMANT_LIABILITY_OUTCOME_ENUM,
        FACTUAL_PROXIMITY_ENUM,
        FORBIDDEN_OUTCOME_FIELDS,
        LIABILITY_OUTCOME_STRENGTH_ENUM,
        NEGATIVE_PATTERN_ENUM,
        REDUCTION_FINDINGS_STATUS_ENUM,
        REMEDY_STATUS_ENUM,
        REQUIRED_OUTCOME_QC_TRUE_FLAGS,
        SEVERITY_ENUM,
        TRANSFERABILITY_ENUM,
    )
    from src.validators import CalibrationValidationContext, validate_calibration_output


def validate_outcome_optimized_calibration(
    calibration,
    dictionary=None,
    context=None,
):
    """Validate existing calibration fields plus the outcome optimization block."""
    errors = []
    if context is None and dictionary is not None:
        context = CalibrationValidationContext.from_dictionary(dictionary)
    if context is not None:
        errors.extend(validate_calibration_output(calibration, context=context))

    outcome = calibration.get("outcome_optimization")
    if not isinstance(outcome, dict):
        errors.append({
            "path": "outcome_optimization",
            "error": "Missing required outcome_optimization block",
            "value": outcome,
        })
        return errors

    theme_ids = context.theme_ids if context is not None else get_theme_ids(dictionary or {})
    _validate_required_outcome_fields(errors, outcome)
    _validate_enum(errors, outcome, "factual_proximity", FACTUAL_PROXIMITY_ENUM)
    _validate_enum(errors, outcome, "transferability_rating", TRANSFERABILITY_ENUM)
    _validate_enum(errors, outcome, "claimant_liability_outcome", CLAIMANT_LIABILITY_OUTCOME_ENUM)
    _validate_enum(errors, outcome, "liability_outcome_strength_band", LIABILITY_OUTCOME_STRENGTH_ENUM)
    _validate_enum(errors, outcome, "remedy_status", REMEDY_STATUS_ENUM)
    _validate_enum(errors, outcome, "reduction_findings_status", REDUCTION_FINDINGS_STATUS_ENUM)
    _validate_pct(errors, outcome, "polkey_reduction_pct")
    _validate_pct(errors, outcome, "contributory_fault_pct")
    _validate_min_text(errors, outcome, "award_reduction_notes", 20)
    _validate_min_text(errors, outcome, "optimization_notes", 20)
    _validate_forbidden_fields(errors, outcome)
    _validate_liability_consistency(errors, outcome)
    _validate_remedy_consistency(errors, outcome)
    _validate_signal_causal_weights(errors, calibration, outcome)
    _validate_negative_theme_flags(errors, outcome, theme_ids)
    _validate_quality_control(errors, calibration, outcome)
    return errors


def _validate_required_outcome_fields(errors, outcome):
    required_fields = [
        "factual_proximity",
        "transferability_rating",
        "claimant_liability_outcome",
        "liability_outcome_strength_band",
        "remedy_status",
        "reduction_findings_status",
        "polkey_reduction_pct",
        "contributory_fault_pct",
        "award_reduction_notes",
        "signal_causal_weights",
        "negative_theme_flags",
        "optimization_notes",
    ]
    for field in required_fields:
        if field not in outcome:
            errors.append({
                "path": f"outcome_optimization.{field}",
                "error": "Missing required field",
                "value": None,
            })


def _validate_enum(errors, outcome, field, allowed):
    value = outcome.get(field)
    if value is not None and value not in allowed:
        errors.append({
            "path": f"outcome_optimization.{field}",
            "error": "Invalid enum value",
            "value": value,
        })


def _validate_pct(errors, outcome, field):
    value = outcome.get(field)
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 <= value <= 100):
        errors.append({
            "path": f"outcome_optimization.{field}",
            "error": "Must be a number in [0,100] or null",
            "value": value,
        })


def _validate_min_text(errors, outcome, field, min_length):
    value = outcome.get(field)
    if not isinstance(value, str) or len(value.strip()) < min_length:
        errors.append({
            "path": f"outcome_optimization.{field}",
            "error": f"Must be text with length >= {min_length}",
            "value": value,
        })


def _validate_forbidden_fields(errors, outcome):
    for field in FORBIDDEN_OUTCOME_FIELDS:
        if field in outcome:
            errors.append({
                "path": f"outcome_optimization.{field}",
                "error": "Forbidden field in LLM outcome output",
                "value": outcome.get(field),
            })


def _validate_liability_consistency(errors, outcome):
    liability = outcome.get("claimant_liability_outcome")
    band = outcome.get("liability_outcome_strength_band")
    allowed_by_liability = {
        "WIN": {"STRONG_WIN", "MODERATE_WIN", "NARROW_WIN"},
        "LOSS": {"LOSS"},
        "PARTIAL": {"PARTIAL"},
        "MIXED": {"PARTIAL"},
        "UNKNOWN": {"UNKNOWN"},
    }
    allowed = allowed_by_liability.get(liability)
    if allowed is not None and band not in allowed:
        errors.append({
            "path": "outcome_optimization.liability_outcome_strength_band",
            "error": f"liability_outcome_strength_band inconsistent with {liability}",
            "value": band,
        })


def _validate_remedy_consistency(errors, outcome):
    reduction_status = outcome.get("reduction_findings_status")
    polkey = outcome.get("polkey_reduction_pct")
    contribution = outcome.get("contributory_fault_pct")
    notes = outcome.get("award_reduction_notes")
    notes_lower = notes.lower() if isinstance(notes, str) else ""

    if reduction_status != "DETERMINED":
        if polkey is not None:
            errors.append({
                "path": "outcome_optimization.polkey_reduction_pct",
                "error": "Must be null unless reduction_findings_status is DETERMINED",
                "value": polkey,
            })
        if contribution is not None:
            errors.append({
                "path": "outcome_optimization.contributory_fault_pct",
                "error": "Must be null unless reduction_findings_status is DETERMINED",
                "value": contribution,
            })

    if (polkey is None or contribution is None) and not _notes_explain_null(notes_lower):
        errors.append({
            "path": "outcome_optimization.award_reduction_notes",
            "error": "Must explain why null reduction percentages are null",
            "value": notes,
        })

    if reduction_status == "DETERMINED" and polkey == 0 and contribution == 0:
        if not _notes_confirm_express_zero(notes_lower):
            errors.append({
                "path": "outcome_optimization.award_reduction_notes",
                "error": "Must confirm zero reductions are express tribunal findings",
                "value": notes,
            })


def _notes_explain_null(notes_lower):
    if not notes_lower:
        return False
    return any(
        marker in notes_lower
        for marker in (
            "null",
            "not determined",
            "not addressed",
            "unknown",
            "unclear",
            "no percentage",
            "not yet",
        )
    )


def _notes_confirm_express_zero(notes_lower):
    if not notes_lower or "express" not in notes_lower:
        return False
    return any(marker in notes_lower for marker in ("zero", "0", "no polkey", "no contribut"))


def _validate_signal_causal_weights(errors, calibration, outcome):
    signals = calibration.get("judgment_signals", [])
    expected_ids = [signal.get("signal_id") for signal in signals if signal.get("signal_id")]
    expected_set = set(expected_ids)
    weights = outcome.get("signal_causal_weights")
    if not isinstance(weights, list):
        errors.append({
            "path": "outcome_optimization.signal_causal_weights",
            "error": "Must be a list",
            "value": type(weights),
        })
        return

    seen = {}
    for index, item in enumerate(weights):
        path = f"outcome_optimization.signal_causal_weights[{index}]"
        if not isinstance(item, dict):
            errors.append({"path": path, "error": "Must be an object", "value": item})
            continue
        signal_id = item.get("signal_id")
        if signal_id not in expected_set:
            errors.append({
                "path": f"{path}.signal_id",
                "error": "Unknown signal_id",
                "value": signal_id,
            })
        seen[signal_id] = seen.get(signal_id, 0) + 1
        causal_weight = item.get("causal_weight")
        if causal_weight not in CAUSAL_WEIGHT_ENUM:
            errors.append({
                "path": f"{path}.causal_weight",
                "error": "Invalid causal_weight",
                "value": causal_weight,
            })
        reason = item.get("causal_weight_reason")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append({
                "path": f"{path}.causal_weight_reason",
                "error": "Must be text with length >= 20",
                "value": reason,
            })

    missing = sorted(expected_set - set(seen))
    extras = sorted(set(seen) - expected_set)
    duplicates = sorted(signal_id for signal_id, count in seen.items() if count > 1)
    if missing:
        errors.append({
            "path": "outcome_optimization.signal_causal_weights",
            "error": "Missing signal_id entries",
            "value": missing,
        })
    if extras:
        errors.append({
            "path": "outcome_optimization.signal_causal_weights",
            "error": "Extra signal_id entries",
            "value": extras,
        })
    if duplicates:
        errors.append({
            "path": "outcome_optimization.signal_causal_weights",
            "error": "Duplicate signal_id entries",
            "value": duplicates,
        })


def _validate_negative_theme_flags(errors, outcome, theme_ids):
    flags = outcome.get("negative_theme_flags")
    if not isinstance(flags, list):
        errors.append({
            "path": "outcome_optimization.negative_theme_flags",
            "error": "Must be a list",
            "value": type(flags),
        })
        return
    for index, flag in enumerate(flags):
        path = f"outcome_optimization.negative_theme_flags[{index}]"
        if not isinstance(flag, dict):
            errors.append({"path": path, "error": "Must be an object", "value": flag})
            continue
        theme_id = flag.get("theme_id")
        if theme_ids and theme_id not in theme_ids:
            errors.append({
                "path": f"{path}.theme_id",
                "error": "Unknown theme_id",
                "value": theme_id,
            })
        if flag.get("negative_pattern") not in NEGATIVE_PATTERN_ENUM:
            errors.append({
                "path": f"{path}.negative_pattern",
                "error": "Invalid negative_pattern",
                "value": flag.get("negative_pattern"),
            })
        if flag.get("severity") not in SEVERITY_ENUM:
            errors.append({
                "path": f"{path}.severity",
                "error": "Invalid severity",
                "value": flag.get("severity"),
            })
        reason = flag.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append({
                "path": f"{path}.reason",
                "error": "Must be text with length >= 20",
                "value": reason,
            })


def _validate_quality_control(errors, calibration, outcome):
    qc = calibration.get("quality_control", {})
    if not isinstance(qc, dict):
        errors.append({"path": "quality_control", "error": "Must be an object", "value": qc})
        return

    for flag in sorted(REQUIRED_OUTCOME_QC_TRUE_FLAGS):
        if qc.get(flag) is not True:
            errors.append({
                "path": f"quality_control.{flag}",
                "error": "Required outcome QC flag must be true",
                "value": qc.get(flag),
            })

    has_other = any(
        flag.get("negative_pattern") == "OTHER"
        for flag in outcome.get("negative_theme_flags", [])
        if isinstance(flag, dict)
    )
    if has_other and qc.get("other_negative_pattern_human_review_required") is not True:
        errors.append({
            "path": "quality_control.other_negative_pattern_human_review_required",
            "error": "Must be true when any negative_pattern is OTHER",
            "value": qc.get("other_negative_pattern_human_review_required"),
        })
