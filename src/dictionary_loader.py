import json
from pathlib import Path
from typing import Dict, Set

ALLOWED_ACTIONS = {
    "ADD FACT",
    "ADD EVIDENCE ANCHOR",
    "REINFORCE",
    "REFRAME",
    "SOFTEN",
    "REMOVE",
    "DO_NOT_USE",
    "REVIEW_MANUALLY"
}

COMPACT_LLM_THEME_FIELDS = (
    "theme_id",
    "theme_priority",
    "theme_name",
    "definition",
    "include_when",
    "exclude_when",
    "duplication_guardrail",
    "permitted_actions",
    "preferred_ws_destination",
)

def load_dictionary(dictionary_path: Path) -> Dict:
    """Load the WS theme dictionary from JSON."""
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compact_dictionary_for_llm(dictionary: Dict) -> Dict:
    """Return the dictionary fields needed by repeated LLM mapping calls."""
    compact_themes = []
    for theme in dictionary.get('ws_theme_dictionary', []):
        compact_themes.append({
            field: theme[field]
            for field in COMPACT_LLM_THEME_FIELDS
            if field in theme
        })
    return {
        "global_mapping_rules": dictionary.get("global_mapping_rules", {}),
        "ws_theme_dictionary": compact_themes,
    }

def get_theme_ids(dictionary: Dict) -> Set[str]:
    """Get set of all theme_ids."""
    return {theme['theme_id'] for theme in dictionary.get('ws_theme_dictionary', [])}

def get_priority_by_theme(dictionary: Dict) -> Dict[str, int]:
    """Get dict of theme_id to theme_priority."""
    return {theme['theme_id']: theme['theme_priority'] for theme in dictionary.get('ws_theme_dictionary', [])}

def get_allowed_actions_by_theme(dictionary: Dict) -> Dict[str, Set[str]]:
    """Get dict of theme_id to set of permitted_actions."""
    return {theme['theme_id']: set(theme.get('permitted_actions', [])) for theme in dictionary.get('ws_theme_dictionary', [])}

def validate_dictionary(dictionary: Dict) -> None:
    """Validate the dictionary structure and content."""
    themes = dictionary.get('ws_theme_dictionary', [])
    if len(themes) != 20:
        raise ValueError(f"Dictionary must have exactly 20 themes, found {len(themes)}")

    theme_ids = []
    priorities = []
    for theme in themes:
        theme_id = theme.get('theme_id')
        if not theme_id:
            raise ValueError("Theme missing theme_id")
        theme_ids.append(theme_id)

        priority = theme.get('theme_priority')
        if not isinstance(priority, int) or not (1 <= priority <= 20):
            raise ValueError(f"Invalid theme_priority for {theme_id}: {priority}")
        priorities.append(priority)

        actions = theme.get('permitted_actions', [])
        if not actions:
            raise ValueError(f"No permitted_actions for {theme_id}")
        for action in actions:
            if action not in ALLOWED_ACTIONS:
                raise ValueError(f"Invalid action '{action}' for {theme_id}")

    if len(set(theme_ids)) != 20:
        raise ValueError("theme_ids are not unique")

    if len(set(priorities)) != 20:
        raise ValueError("theme_priorities are not unique")

    if 'global_mapping_rules' not in dictionary:
        raise ValueError("Missing global_mapping_rules")

    if 'quality_control' not in dictionary:
        raise ValueError("Missing quality_control")
