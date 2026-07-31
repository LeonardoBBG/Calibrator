import hashlib
import json
from copy import deepcopy
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
    "theme_status",
    "definition",
    "include_when",
    "exclude_when",
    "common_subthemes",
    "duplication_guardrail",
    "pre_check_required",
    "permitted_actions",
    "preferred_ws_destination",
    "evidence_anchor_types",
    "forbidden_uses",
    "t20_reason_codes",
)

COMPACT_LLM_TOP_LEVEL_FIELDS = (
    "dictionary_metadata",
    "action_glossary",
    "aliases",
    "mapping_output_schema",
    "global_mapping_rules",
    "quality_control",
)

def load_dictionary(dictionary_path: Path) -> Dict:
    """Load the WS theme dictionary from JSON."""
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compact_dictionary_for_llm(dictionary: Dict) -> Dict:
    """Return the dictionary contract needed by repeated LLM mapping calls."""
    compact_themes = []
    for theme in dictionary.get('ws_theme_dictionary', []):
        compact_themes.append({
            field: theme[field]
            for field in COMPACT_LLM_THEME_FIELDS
            if field in theme
        })
    compact = {
        field: deepcopy(dictionary[field])
        for field in COMPACT_LLM_TOP_LEVEL_FIELDS
        if field in dictionary
    }
    compact["ws_theme_dictionary"] = compact_themes
    compact.setdefault("dictionary_metadata", {})
    compact["dictionary_metadata"]["payload_sha256"] = dictionary_payload_sha256(compact)
    return compact


def dictionary_payload_sha256(dictionary: Dict) -> str:
    """Stable hash for dictionary payload compatibility checks."""
    payload = json.dumps(dictionary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def get_theme_ids(dictionary: Dict) -> Set[str]:
    """Get set of all theme_ids."""
    return {theme['theme_id'] for theme in dictionary.get('ws_theme_dictionary', [])}

def get_priority_by_theme(dictionary: Dict) -> Dict[str, int]:
    """Get dict of theme_id to theme_priority."""
    return {theme['theme_id']: theme['theme_priority'] for theme in dictionary.get('ws_theme_dictionary', [])}

def get_allowed_actions_by_theme(dictionary: Dict) -> Dict[str, Set[str]]:
    """Get dict of theme_id to set of permitted_actions."""
    return {theme['theme_id']: set(theme.get('permitted_actions', [])) for theme in dictionary.get('ws_theme_dictionary', [])}


def get_theme_by_id(dictionary: Dict) -> Dict[str, Dict]:
    """Get dict of theme_id to theme object."""
    return {theme['theme_id']: theme for theme in dictionary.get('ws_theme_dictionary', [])}

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

    action_glossary = dictionary.get("action_glossary", {}).get("actions", {})
    if action_glossary:
        for action in {action for theme in themes for action in theme.get("permitted_actions", [])}:
            if action not in action_glossary:
                raise ValueError(f"Action '{action}' missing from action_glossary")

    aliases = dictionary.get("aliases", {}).get("definitions", {})
    theme_id_set = set(theme_ids)
    if "mapping_output_schema" in dictionary:
        required_alias = "ALL_SUBSTANTIVE_THEMES_T01_TO_T19"
        substantive_theme_ids = {
            theme_id
            for theme_id in theme_id_set
            if theme_id != "T20_RISK_CONTROL"
        }
        alias_values = set(aliases.get(required_alias, []))
        if not alias_values:
            raise ValueError(f"Missing required alias '{required_alias}'")
        if alias_values != substantive_theme_ids:
            raise ValueError(
                f"Alias '{required_alias}' must resolve exactly to all substantive T01-T19 theme_ids"
            )
    for alias, values in aliases.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Alias '{alias}' must resolve to a non-empty list")
        unknown = [value for value in values if value not in theme_id_set]
        if unknown:
            raise ValueError(f"Alias '{alias}' contains unknown theme_ids: {unknown}")

    for theme in themes:
        for item in theme.get("pre_check_required", []):
            if not isinstance(item, dict):
                raise ValueError(f"pre_check_required for {theme.get('theme_id')} must contain objects")
            ref = item.get("theme_id")
            alias = item.get("theme_id_alias")
            if ref and ref not in theme_id_set:
                raise ValueError(f"pre_check_required references unknown theme_id '{ref}'")
            if alias and alias not in aliases:
                raise ValueError(f"pre_check_required references unknown alias '{alias}'")

    t20 = next((theme for theme in themes if theme.get("theme_id") == "T20_RISK_CONTROL"), None)
    if t20 and "mapping_output_schema" in dictionary and not t20.get("t20_reason_codes"):
        raise ValueError("T20_RISK_CONTROL must define t20_reason_codes")
