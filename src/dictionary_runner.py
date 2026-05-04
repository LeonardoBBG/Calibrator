from typing import Dict, TYPE_CHECKING

try:
    from dictionary_loader import compact_dictionary_for_llm
except ImportError:
    from src.dictionary_loader import compact_dictionary_for_llm

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
        "WS_THEME_DICTIONARY_JSON": compact_dictionary_for_llm(dictionary)
    }
    response = llm_client.complete_json(tagging_prompt, payload)
    return response

def build_ws_tagging_summary(ws_tagging: Dict) -> Dict:
    """Build compact WS baseline for repeated judgment calibration calls."""
    reinforcement_actions = {"REINFORCE", "ADD FACT", "ADD EVIDENCE ANCHOR"}
    rationale_presence = {"LATENT", "ABSENT", "RISK_ONLY"}
    summary = {
        "theme_presence_by_id": {},
        "recommended_action_by_id": {},
        "cross_reference_theme_ids_by_id": {},
        "risk_or_rationale_by_id": {}
    }

    for mapping in ws_tagging.get("theme_mappings", []):
        theme_id = mapping.get("mapped_theme_id")
        if not theme_id:
            continue

        theme_presence = mapping.get("theme_presence")
        recommended_action = mapping.get("recommended_action")

        notes = []
        if theme_presence in rationale_presence and recommended_action in reinforcement_actions:
            notes.append(
                f"summary_action_sanitized: {recommended_action} is not valid for {theme_presence}; "
                "using REVIEW_MANUALLY"
            )
            recommended_action = "REVIEW_MANUALLY"

        summary["theme_presence_by_id"][theme_id] = theme_presence
        summary["recommended_action_by_id"][theme_id] = recommended_action
        summary["cross_reference_theme_ids_by_id"][theme_id] = mapping.get("cross_reference_theme_ids", [])

        duplication_risk = mapping.get("duplication_risk")
        mapping_rationale = mapping.get("mapping_rationale")
        if theme_presence in rationale_presence:
            if duplication_risk:
                notes.append(f"duplication_risk: {duplication_risk}")
            if mapping_rationale:
                notes.append(f"mapping_rationale: {mapping_rationale}")
        if notes and theme_presence in rationale_presence:
            summary["risk_or_rationale_by_id"][theme_id] = " ".join(notes)

    return summary
