from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

def run_ws_tagging(
    ws_text: str,
    dictionary: Dict,
    tagging_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Tag/analyze the WS using the controlled theme dictionary."""
    payload = {
        "WS_TEXT": ws_text,
        "WS_THEME_DICTIONARY_JSON": dictionary
    }
    response = llm_client.complete_json(tagging_prompt, payload)
    return response
