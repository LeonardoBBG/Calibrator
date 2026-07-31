from dataclasses import dataclass
from typing import Dict, List, Set

try:
    from dictionary_loader import get_theme_ids, get_priority_by_theme, get_allowed_actions_by_theme, get_theme_by_id
except ImportError:
    from src.dictionary_loader import get_theme_ids, get_priority_by_theme, get_allowed_actions_by_theme, get_theme_by_id

VALID_CASE_EFFECTS = {
    "WIN_DRIVER",
    "LOSS_DRIVER",
    "POLKEY_RISK",
    "CONTRIBUTION_RISK",
    "NEUTRAL_CONTEXT",
    "NON_TRANSFERABLE"
}

VALID_WS_PRESENCE = {
    "PRESENT",
    "PARTIAL",
    "ABSENT",
    "UNCLEAR"
}

VALID_CONFIDENCE = {
    "HIGH",
    "MEDIUM",
    "LOW"
}

REINFORCEMENT_ACTIONS = {
    "REINFORCE",
    "ADD FACT",
    "ADD EVIDENCE ANCHOR"
}

VALID_SCOPE_STATUS = {
    "in_scope",
    "out_of_scope_claim_type",
}

VALID_SENSITIVITY_FLAGS = {
    "discrimination_implication",
}


@dataclass(frozen=True)
class CalibrationValidationContext:
    theme_ids: Set[str]
    priority_map: Dict[str, int]
    actions_map: Dict[str, Set[str]]
    theme_map: Dict[str, Dict]
    aliases: Dict[str, List[str]]
    action_glossary: Set[str]
    mapping_schema_required: bool
    pre_check_theme_ids: Set[str]
    t20_reason_codes: Set[str]

    @classmethod
    def from_dictionary(cls, dictionary: Dict) -> "CalibrationValidationContext":
        theme_map = get_theme_by_id(dictionary)
        t20 = theme_map.get("T20_RISK_CONTROL", {})
        return cls(
            theme_ids=get_theme_ids(dictionary),
            priority_map=get_priority_by_theme(dictionary),
            actions_map=get_allowed_actions_by_theme(dictionary),
            theme_map=theme_map,
            aliases=dictionary.get("aliases", {}).get("definitions", {}),
            action_glossary=set(dictionary.get("action_glossary", {}).get("actions", {}).keys()),
            mapping_schema_required=bool(dictionary.get("mapping_output_schema")),
            pre_check_theme_ids={
                theme_id
                for theme_id, theme in theme_map.items()
                if theme.get("pre_check_required")
            },
            t20_reason_codes=set(t20.get("t20_reason_codes", {}).keys())
        )


def validate_calibration_output(
    calibration: Dict,
    dictionary: Dict = None,
    context: CalibrationValidationContext = None,
    ws_tagging_summary: Dict = None
) -> List[Dict]:
    """Validate the calibration output against rules."""
    if context is None:
        if dictionary is None:
            raise ValueError("dictionary or context is required")
        context = CalibrationValidationContext.from_dictionary(dictionary)

    errors = []

    # Required top-level keys
    required_keys = ["case_metadata", "case_relevance_to_ws", "judgment_signals", "quality_control"]
    for key in required_keys:
        if key not in calibration:
            errors.append({"path": key, "error": "Missing required key", "value": None})

    if "judgment_signals" not in calibration:
        return errors  # Can't validate further

    signals = calibration["judgment_signals"]
    if not isinstance(signals, list):
        errors.append({"path": "judgment_signals", "error": "Must be a list", "value": type(signals)})
        return errors

    for i, signal in enumerate(signals):
        path_prefix = f"judgment_signals[{i}]"

        required_signal_keys = ["signal_id", "mapped_theme_id", "theme_priority", "recommended_action", "case_effect"]
        if context.mapping_schema_required:
            required_signal_keys.extend([
                "primary_theme_id",
                "action",
                "signal_verbatim",
                "signal_source_reference",
                "scope_status",
                "no_external_judgment_facts_imported",
                "rationale",
            ])
        for key in required_signal_keys:
            if key not in signal:
                errors.append({"path": f"{path_prefix}.{key}", "error": "Missing required key", "value": None})

        mapped_theme_id = signal.get("mapped_theme_id") or signal.get("primary_theme_id")
        primary_theme_id = signal.get("primary_theme_id")
        if mapped_theme_id and primary_theme_id and mapped_theme_id != primary_theme_id:
            errors.append({
                "path": f"{path_prefix}.primary_theme_id",
                "error": "primary_theme_id must match mapped_theme_id",
                "value": primary_theme_id
            })
        if mapped_theme_id and mapped_theme_id not in context.theme_ids:
            errors.append({"path": f"{path_prefix}.mapped_theme_id", "error": "Unknown mapped_theme_id", "value": mapped_theme_id})

        theme_priority = signal.get("theme_priority")
        if mapped_theme_id and theme_priority is not None:
            expected_priority = context.priority_map.get(mapped_theme_id)
            if theme_priority != expected_priority:
                errors.append({"path": f"{path_prefix}.theme_priority", "error": "Incorrect theme_priority", "value": theme_priority})

        recommended_action = signal.get("recommended_action") or signal.get("action")
        action = signal.get("action")
        if recommended_action and action and recommended_action != action:
            errors.append({
                "path": f"{path_prefix}.action",
                "error": "action must match recommended_action",
                "value": action
            })
        if mapped_theme_id and recommended_action:
            allowed_actions = context.actions_map.get(mapped_theme_id, set())
            if recommended_action not in allowed_actions:
                errors.append({"path": f"{path_prefix}.recommended_action", "error": "Action not permitted for theme", "value": recommended_action})
            if context.action_glossary and recommended_action not in context.action_glossary:
                errors.append({"path": f"{path_prefix}.action", "error": "Action missing from action_glossary", "value": recommended_action})

        case_effect = signal.get("case_effect")
        if case_effect and case_effect not in VALID_CASE_EFFECTS:
            errors.append({"path": f"{path_prefix}.case_effect", "error": "Invalid case_effect", "value": case_effect})

        ws_presence = signal.get("ws_presence")
        if ws_presence and ws_presence not in VALID_WS_PRESENCE:
            errors.append({"path": f"{path_prefix}.ws_presence", "error": "Invalid ws_presence", "value": ws_presence})

        if mapped_theme_id and ws_tagging_summary:
            _validate_ws_baseline_coupling(
                errors,
                path_prefix,
                mapped_theme_id,
                recommended_action,
                ws_presence,
                ws_tagging_summary
            )

        confidence = signal.get("dictionary_match_confidence")
        if confidence and confidence not in VALID_CONFIDENCE:
            errors.append({"path": f"{path_prefix}.dictionary_match_confidence", "error": "Invalid confidence", "value": confidence})

        cross_refs = signal.get("cross_reference_theme_ids", [])
        if not isinstance(cross_refs, list):
            errors.append({"path": f"{path_prefix}.cross_reference_theme_ids", "error": "Must be a list", "value": type(cross_refs)})
        else:
            for ref in cross_refs:
                if ref not in context.theme_ids:
                    errors.append({"path": f"{path_prefix}.cross_reference_theme_ids", "error": "Unknown cross_reference_theme_id", "value": ref})
            if mapped_theme_id and mapped_theme_id in cross_refs:
                errors.append({"path": f"{path_prefix}.cross_reference_theme_ids", "error": "Cannot include mapped_theme_id", "value": mapped_theme_id})

        if context.mapping_schema_required:
            _validate_mapping_schema_fields(
                errors,
                path_prefix,
                signal,
                context,
                mapped_theme_id,
                recommended_action,
            )

    qc = calibration.get("quality_control", {})
    if qc.get("new_themes_created") is not False:
        errors.append({"path": "quality_control.new_themes_created", "error": "Must be false", "value": qc.get("new_themes_created")})
    if qc.get("ws_rewrite_performed") is not False:
        errors.append({"path": "quality_control.ws_rewrite_performed", "error": "Must be false", "value": qc.get("ws_rewrite_performed")})
    if qc.get("new_allegations_created") is not False:
        errors.append({"path": "quality_control.new_allegations_created", "error": "Must be false", "value": qc.get("new_allegations_created")})

    return errors


def _validate_mapping_schema_fields(
    errors: List[Dict],
    path_prefix: str,
    signal: Dict,
    context: CalibrationValidationContext,
    mapped_theme_id: str,
    recommended_action: str,
) -> None:
    scope_status = signal.get("scope_status")
    if scope_status and scope_status not in VALID_SCOPE_STATUS:
        errors.append({"path": f"{path_prefix}.scope_status", "error": "Invalid scope_status", "value": scope_status})

    if mapped_theme_id and mapped_theme_id != "T20_RISK_CONTROL" and scope_status and scope_status != "in_scope":
        errors.append({
            "path": f"{path_prefix}.scope_status",
            "error": "Non-T20 mappings must use scope_status='in_scope'",
            "value": scope_status
        })

    if scope_status == "out_of_scope_claim_type":
        if mapped_theme_id != "T20_RISK_CONTROL":
            errors.append({
                "path": f"{path_prefix}.mapped_theme_id",
                "error": "out_of_scope_claim_type must map to T20_RISK_CONTROL",
                "value": mapped_theme_id
            })
        if signal.get("t20_reason_code") != "T20_OUT_OF_SCOPE_CLAIM_TYPE":
            errors.append({
                "path": f"{path_prefix}.t20_reason_code",
                "error": "out_of_scope_claim_type requires T20_OUT_OF_SCOPE_CLAIM_TYPE",
                "value": signal.get("t20_reason_code")
            })

    if mapped_theme_id == "T20_RISK_CONTROL":
        t20_reason_code = signal.get("t20_reason_code")
        if not t20_reason_code:
            errors.append({"path": f"{path_prefix}.t20_reason_code", "error": "Missing required key", "value": None})
        elif t20_reason_code not in context.t20_reason_codes:
            errors.append({"path": f"{path_prefix}.t20_reason_code", "error": "Invalid t20_reason_code", "value": t20_reason_code})

    if signal.get("no_external_judgment_facts_imported") is not True:
        errors.append({
            "path": f"{path_prefix}.no_external_judgment_facts_imported",
            "error": "Must be true",
            "value": signal.get("no_external_judgment_facts_imported")
        })

    if recommended_action != "DO_NOT_USE":
        ws_anchor = signal.get("ws_anchor")
        if not isinstance(ws_anchor, str) or not ws_anchor.strip():
            errors.append({"path": f"{path_prefix}.ws_anchor", "error": "ws_anchor is required unless action is DO_NOT_USE", "value": ws_anchor})

    for key in ("signal_verbatim", "signal_source_reference", "rationale"):
        value = signal.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append({"path": f"{path_prefix}.{key}", "error": "Must be non-empty text", "value": value})

    if mapped_theme_id in context.pre_check_theme_ids:
        _validate_pre_check_log(errors, path_prefix, signal.get("pre_check_log"), context, mapped_theme_id)

    sensitivity_flags = signal.get("sensitivity_flags", [])
    if sensitivity_flags is None:
        return
    if not isinstance(sensitivity_flags, list):
        errors.append({"path": f"{path_prefix}.sensitivity_flags", "error": "Must be a list", "value": type(sensitivity_flags)})
        return
    for flag in sensitivity_flags:
        if flag not in VALID_SENSITIVITY_FLAGS:
            errors.append({"path": f"{path_prefix}.sensitivity_flags", "error": "Invalid sensitivity flag", "value": flag})


def _validate_pre_check_log(
    errors: List[Dict],
    path_prefix: str,
    pre_check_log,
    context: CalibrationValidationContext,
    mapped_theme_id: str,
) -> None:
    if not isinstance(pre_check_log, list) or not pre_check_log:
        errors.append({"path": f"{path_prefix}.pre_check_log", "error": "pre_check_log is required for this theme", "value": pre_check_log})
        return

    resolved_ids = set()
    for index, item in enumerate(pre_check_log):
        item_path = f"{path_prefix}.pre_check_log[{index}]"
        if not isinstance(item, dict):
            errors.append({"path": item_path, "error": "Must be an object", "value": item})
            continue
        theme_id = item.get("theme_id")
        alias = item.get("theme_id_alias")
        if theme_id:
            if theme_id not in context.theme_ids:
                errors.append({"path": f"{item_path}.theme_id", "error": "Unknown pre-check theme_id", "value": theme_id})
            else:
                resolved_ids.add(theme_id)
        elif alias:
            alias_ids = context.aliases.get(alias)
            if not alias_ids:
                errors.append({"path": f"{item_path}.theme_id_alias", "error": "Unknown pre-check alias", "value": alias})
            else:
                resolved_ids.update(alias_ids)
        else:
            errors.append({"path": item_path, "error": "Missing theme_id or theme_id_alias", "value": item})
        reason = item.get("reason") or item.get("exclusion_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append({"path": f"{item_path}.reason", "error": "Pre-check requires an exclusion reason", "value": reason})

    if mapped_theme_id == "T20_RISK_CONTROL":
        required_ids = set(context.aliases.get("ALL_SUBSTANTIVE_THEMES_T01_TO_T19", []))
        if not required_ids:
            errors.append({
                "path": f"{path_prefix}.pre_check_log",
                "error": "Dictionary missing ALL_SUBSTANTIVE_THEMES_T01_TO_T19 alias",
                "value": None
            })
        elif not required_ids.issubset(resolved_ids):
            missing = sorted(required_ids - resolved_ids)
            errors.append({"path": f"{path_prefix}.pre_check_log", "error": "T20 pre_check_log must cover all substantive themes", "value": missing})


def _validate_ws_baseline_coupling(
    errors: List[Dict],
    path_prefix: str,
    mapped_theme_id: str,
    recommended_action: str,
    ws_presence: str,
    ws_tagging_summary: Dict
) -> None:
    presence_by_id = ws_tagging_summary.get("theme_presence_by_id", {})
    baseline_presence = presence_by_id.get(mapped_theme_id)
    if not baseline_presence:
        return

    if baseline_presence == "ABSENT":
        if ws_presence == "PRESENT":
            errors.append({
                "path": f"{path_prefix}.ws_presence",
                "error": "WS tagging summary marks mapped theme ABSENT; ws_presence cannot be PRESENT",
                "value": ws_presence
            })
        if recommended_action in REINFORCEMENT_ACTIONS:
            errors.append({
                "path": f"{path_prefix}.recommended_action",
                "error": "WS tagging summary marks mapped theme ABSENT; reinforcement actions are not permitted",
                "value": recommended_action
            })

    if baseline_presence == "LATENT" and recommended_action in REINFORCEMENT_ACTIONS:
        errors.append({
            "path": f"{path_prefix}.recommended_action",
            "error": "WS tagging summary marks mapped theme LATENT; use REVIEW_MANUALLY unless manually approved outside the automated pipeline",
            "value": recommended_action
        })

    if baseline_presence == "RISK_ONLY":
        if ws_presence == "PRESENT":
            errors.append({
                "path": f"{path_prefix}.ws_presence",
                "error": "WS tagging summary marks mapped theme RISK_ONLY; ws_presence cannot be PRESENT",
                "value": ws_presence
            })
        if recommended_action in REINFORCEMENT_ACTIONS:
            errors.append({
                "path": f"{path_prefix}.recommended_action",
                "error": "WS tagging summary marks mapped theme RISK_ONLY; reinforcement actions are not permitted",
                "value": recommended_action
            })
