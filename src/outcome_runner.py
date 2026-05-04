"""LLM runner for adding the outcome optimization block."""

from copy import deepcopy
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient


def run_outcome_optimization(
    validated_calibration: Dict,
    outcome_prompt: str,
    llm_client: "LLMClient",
) -> Dict:
    """Generate and merge outcome_optimization without altering existing fields."""
    payload = {
        "VALIDATED_CALIBRATION_JSON": validated_calibration,
    }
    response = llm_client.complete_json(outcome_prompt, payload)
    return merge_outcome_optimization(validated_calibration, response)


def repair_outcome_optimization(
    invalid_outcome_optimized: Dict,
    validation_errors,
    outcome_repair_prompt: str,
    llm_client: "LLMClient",
) -> Dict:
    """Repair a failed outcome optimization using validator feedback."""
    payload = {
        "INVALID_OUTCOME_OPTIMIZED_JSON": invalid_outcome_optimized,
        "VALIDATION_ERRORS": validation_errors,
    }
    response = llm_client.complete_json(outcome_repair_prompt, payload)
    return merge_outcome_optimization(invalid_outcome_optimized, response)


def merge_outcome_optimization(validated_calibration: Dict, outcome_response: Dict) -> Dict:
    """Merge outcome response into a calibration copy."""
    outcome_block = outcome_response.get("outcome_optimization")
    if outcome_block is None:
        outcome_block = outcome_response

    qc_updates = outcome_response.get("quality_control", {})
    optimized = deepcopy(validated_calibration)
    optimized["outcome_optimization"] = outcome_block
    existing_qc = optimized.setdefault("quality_control", {})
    existing_qc.update(qc_updates)

    has_other = any(
        flag.get("negative_pattern") == "OTHER"
        for flag in outcome_block.get("negative_theme_flags", [])
        if isinstance(flag, dict)
    )
    existing_qc.setdefault("other_negative_pattern_human_review_required", bool(has_other))
    return optimized
