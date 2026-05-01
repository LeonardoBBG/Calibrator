from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def run_compression(
    validated_calibration: Dict,
    dictionary: Dict,
    compression_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Run the compression step to create non-duplicative reinforcement plan."""
    payload = {
        "VALIDATED_CALIBRATION_JSON": validated_calibration,
        "WS_THEME_DICTIONARY_JSON": dictionary
    }
    response = llm_client.complete_json(compression_prompt, payload)
    return response