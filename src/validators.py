from dataclasses import dataclass
from typing import Dict, List, Set

try:
    from dictionary_loader import get_theme_ids, get_priority_by_theme, get_allowed_actions_by_theme
except ImportError:
    from src.dictionary_loader import get_theme_ids, get_priority_by_theme, get_allowed_actions_by_theme

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


@dataclass(frozen=True)
class CalibrationValidationContext:
    theme_ids: Set[str]
    priority_map: Dict[str, int]
    actions_map: Dict[str, Set[str]]

    @classmethod
    def from_dictionary(cls, dictionary: Dict) -> "CalibrationValidationContext":
        return cls(
            theme_ids=get_theme_ids(dictionary),
            priority_map=get_priority_by_theme(dictionary),
            actions_map=get_allowed_actions_by_theme(dictionary)
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
        for key in required_signal_keys:
            if key not in signal:
                errors.append({"path": f"{path_prefix}.{key}", "error": "Missing required key", "value": None})

        mapped_theme_id = signal.get("mapped_theme_id")
        if mapped_theme_id and mapped_theme_id not in context.theme_ids:
            errors.append({"path": f"{path_prefix}.mapped_theme_id", "error": "Unknown mapped_theme_id", "value": mapped_theme_id})

        theme_priority = signal.get("theme_priority")
        if mapped_theme_id and theme_priority is not None:
            expected_priority = context.priority_map.get(mapped_theme_id)
            if theme_priority != expected_priority:
                errors.append({"path": f"{path_prefix}.theme_priority", "error": "Incorrect theme_priority", "value": theme_priority})

        recommended_action = signal.get("recommended_action")
        if mapped_theme_id and recommended_action:
            allowed_actions = context.actions_map.get(mapped_theme_id, set())
            if recommended_action not in allowed_actions:
                errors.append({"path": f"{path_prefix}.recommended_action", "error": "Action not permitted for theme", "value": recommended_action})

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

    qc = calibration.get("quality_control", {})
    if qc.get("new_themes_created") is not False:
        errors.append({"path": "quality_control.new_themes_created", "error": "Must be false", "value": qc.get("new_themes_created")})
    if qc.get("ws_rewrite_performed") is not False:
        errors.append({"path": "quality_control.ws_rewrite_performed", "error": "Must be false", "value": qc.get("ws_rewrite_performed")})
    if qc.get("new_allegations_created") is not False:
        errors.append({"path": "quality_control.new_allegations_created", "error": "Must be false", "value": qc.get("new_allegations_created")})

    return errors


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
