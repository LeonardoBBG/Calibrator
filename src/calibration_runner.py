from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def run_calibration(
    ws_text: str,
    judgment_text: str,
    compact_dict: Dict,
    ws_tagging_summary: Dict,
    calibration_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Run the calibration step using LLM."""
    payload = {
        "WS_THEME_DICTIONARY_JSON": compact_dict,
        "WS_TAGGING_SUMMARY_JSON": ws_tagging_summary,
        "WS_TEXT": ws_text,
        "PDF_TEXT": judgment_text,
    }
    response = llm_client.complete_json(calibration_prompt, payload)
    return response
