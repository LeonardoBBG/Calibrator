from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def run_calibration(
    ws_text: str,
    judgment_text: str,
    dictionary: Dict,
    calibration_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Run the calibration step using LLM."""
    payload = {
        "WS_TEXT": ws_text,
        "PDF_TEXT": judgment_text,
        "WS_THEME_DICTIONARY_JSON": dictionary
    }
    # The prompt is the system prompt, payload as user message
    response = llm_client.complete_json(calibration_prompt, payload)
    return response