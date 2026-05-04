from typing import Dict, List, TYPE_CHECKING

try:
    from dictionary_loader import compact_dictionary_for_llm
except ImportError:
    from src.dictionary_loader import compact_dictionary_for_llm

if TYPE_CHECKING:
    from llm_client import LLMClient

def repair_calibration_output(
    invalid_calibration: Dict,
    validation_errors: List[Dict],
    dictionary: Dict,
    ws_tagging_summary: Dict,
    repair_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Repair the invalid calibration output using LLM."""
    payload = {
        "INVALID_CALIBRATION_JSON": invalid_calibration,
        "VALIDATION_ERRORS": validation_errors,
        "WS_THEME_DICTIONARY_JSON": compact_dictionary_for_llm(dictionary),
        "WS_TAGGING_SUMMARY_JSON": ws_tagging_summary
    }
    response = llm_client.complete_json(repair_prompt, payload)
    return response
