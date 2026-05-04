from typing import Dict, TYPE_CHECKING

try:
    from dictionary_loader import compact_dictionary_for_llm
except ImportError:
    from src.dictionary_loader import compact_dictionary_for_llm

if TYPE_CHECKING:
    from llm_client import LLMClient

def run_calibration(
    ws_text: str,
    judgment_text: str,
    dictionary: Dict,
    ws_tagging_summary: Dict,
    calibration_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Run the calibration step using LLM."""
    payload = {
        "WS_TEXT": ws_text,
        "PDF_TEXT": judgment_text,
        "WS_THEME_DICTIONARY_JSON": compact_dictionary_for_llm(dictionary),
        "WS_TAGGING_SUMMARY_JSON": ws_tagging_summary
    }
    # The prompt is the system prompt, payload as user message
    response = llm_client.complete_json(calibration_prompt, payload)
    return response
