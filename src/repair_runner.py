from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def repair_calibration_output(
    invalid_calibration: Dict,
    validation_errors: List[Dict],
    dictionary: Dict,
    repair_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Repair the invalid calibration output using LLM."""
    payload = {
        "INVALID_CALIBRATION_JSON": invalid_calibration,
        "VALIDATION_ERRORS": validation_errors,
        "WS_THEME_DICTIONARY_JSON": dictionary
    }
    response = llm_client.complete_json(repair_prompt, payload)
    return response