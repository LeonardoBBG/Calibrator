from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def repair_calibration_output(
    invalid_calibration: Dict,
    validation_errors: List[Dict],
    compact_dict: Dict,
    ws_tagging_summary: Dict,
    repair_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Repair the invalid calibration output using LLM."""
    payload = {
        "WS_THEME_DICTIONARY_JSON": compact_dict,
        "WS_TAGGING_SUMMARY_JSON": ws_tagging_summary,
        "INVALID_CALIBRATION_JSON": invalid_calibration,
        "VALIDATION_ERRORS": validation_errors,
    }
    response = llm_client.complete_json(repair_prompt, payload)
    return response
