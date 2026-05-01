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

def build_ws_tagging_summary(ws_tagging: Dict) -> Dict:
    """Build compact WS baseline for repeated judgment calibration calls."""
    summary = {
        "theme_presence_by_id": {},
        "recommended_action_by_id": {},
        "theme_priority_by_id": {},
        "cross_reference_theme_ids_by_id": {},
        "risk_or_rationale_by_id": {}
    }

    for mapping in ws_tagging.get("theme_mappings", []):
        theme_id = mapping.get("mapped_theme_id")
        if not theme_id:
            continue

        summary["theme_presence_by_id"][theme_id] = mapping.get("theme_presence")
        summary["recommended_action_by_id"][theme_id] = mapping.get("recommended_action")
        summary["theme_priority_by_id"][theme_id] = mapping.get("theme_priority")
        summary["cross_reference_theme_ids_by_id"][theme_id] = mapping.get("cross_reference_theme_ids", [])

        notes = []
        duplication_risk = mapping.get("duplication_risk")
        mapping_rationale = mapping.get("mapping_rationale")
        if duplication_risk:
            notes.append(f"duplication_risk: {duplication_risk}")
        if mapping_rationale:
            notes.append(f"mapping_rationale: {mapping_rationale}")
        if notes:
            summary["risk_or_rationale_by_id"][theme_id] = " ".join(notes)

    return summary
