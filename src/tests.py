import json
from pathlib import Path

from .dictionary_loader import load_dictionary, validate_dictionary, get_theme_ids, get_priority_by_theme, get_allowed_actions_by_theme
from .validators import validate_calibration_output

def test_dictionary():
    """Test dictionary loading and validation."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    validate_dictionary(dictionary)
    themes = dictionary['ws_theme_dictionary']
    assert len(themes) == 20
    theme_ids = get_theme_ids(dictionary)
    assert len(theme_ids) == 20
    priorities = list(get_priority_by_theme(dictionary).values())
    assert sorted(priorities) == list(range(1, 21))
    print("Dictionary test passed")

def test_fake_calibration():
    """Test validation with a fake correct calibration."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    fake_calibration = {
        "case_metadata": {"case_name": "Test Case"},
        "case_relevance_to_ws": {"overall_similarity_score_0_to_10": 5},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 3,
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "ws_presence": "PRESENT",
                "dictionary_match_confidence": "HIGH",
                "cross_reference_theme_ids": []
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    errors = validate_calibration_output(fake_calibration, dictionary)
    assert len(errors) == 0, f"Errors: {errors}"
    print("Fake calibration test passed")

def test_invalid_theme_id():
    """Test validation fails with unknown theme_id."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    invalid_calibration = {
        "case_metadata": {},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T99_FAKE",
                "theme_priority": 3,
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER"
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    errors = validate_calibration_output(invalid_calibration, dictionary)
    assert len(errors) > 0
    assert any("Unknown mapped_theme_id" in e['error'] for e in errors)
    print("Invalid theme_id test passed")

def test_invalid_priority():
    """Test validation fails with incorrect priority."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    invalid_calibration = {
        "case_metadata": {},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 2,  # Should be 3
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER"
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    errors = validate_calibration_output(invalid_calibration, dictionary)
    assert len(errors) > 0
    assert any("Incorrect theme_priority" in e['error'] for e in errors)
    print("Invalid priority test passed")

def test_invalid_action():
    """Test validation fails with action not permitted."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    invalid_calibration = {
        "case_metadata": {},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 3,
                "recommended_action": "ADD FACT",  # Not permitted for T01
                "case_effect": "WIN_DRIVER"
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    errors = validate_calibration_output(invalid_calibration, dictionary)
    assert len(errors) > 0
    assert any("Action not permitted" in e['error'] for e in errors)
    print("Invalid action test passed")

def test_invalid_cross_ref():
    """Test validation fails with unknown cross_ref."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    invalid_calibration = {
        "case_metadata": {},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T01_CONTRACT_TERMS",
                "theme_priority": 1,
                "recommended_action": "ADD FACT",
                "case_effect": "WIN_DRIVER",
                "cross_reference_theme_ids": ["T99_FAKE"]
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    errors = validate_calibration_output(invalid_calibration, dictionary)
    assert len(errors) > 0
    assert any("Unknown cross_reference_theme_id" in e['error'] for e in errors)
    print("Invalid cross_ref test passed")

def test_json_reload():
    """Test that written JSON can be reloaded."""
    import tempfile
    from .io_utils import write_json
    data = {"test": "value", "number": 42}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    write_json(temp_path, data)
    with open(temp_path, 'r') as f:
        reloaded = json.load(f)
    assert reloaded == data
    temp_path.unlink()
    print("JSON reload test passed")

def test_llm_cache():
    """Test that LLM responses are cached by prompt and payload."""
    import tempfile
    from .llm_client import LLMClient

    class FakeLLMClient(LLMClient):
        def __init__(self, cache_dir):
            super().__init__(
                provider="local",
                model="fake",
                temperature=0.0,
                max_tokens=100,
                require_temperature_support=True,
                cache_dir=cache_dir,
                cache_enabled=True
            )
            self.calls = 0

        def _local_complete(self, system_prompt, user_payload):
            self.calls += 1
            return {"calls": self.calls, "payload": user_payload}

    with tempfile.TemporaryDirectory() as temp_dir:
        client = FakeLLMClient(Path(temp_dir))
        first = client.complete_json("prompt", {"value": 1})
        second = client.complete_json("prompt", {"value": 1})

    assert first == second
    assert client.calls == 1
    print("LLM cache test passed")

def test_judgment_path_selection():
    """Test debug and batch judgment path selection."""
    import tempfile
    from .config import Config
    from .io_utils import make_run_id

    config = Config.default(make_run_id())
    config.run_mode = "debug"
    assert config.selected_judgment_paths() == [config.judgment_path]

    with tempfile.TemporaryDirectory() as temp_dir:
        batch_dir = Path(temp_dir)
        second = batch_dir / "b_second.pdf"
        first = batch_dir / "a_first.pdf"
        ignored = batch_dir / "notes.txt"
        second.write_text("", encoding="utf-8")
        first.write_text("", encoding="utf-8")
        ignored.write_text("", encoding="utf-8")

        config.run_mode = "batch"
        config.judgments_dir = batch_dir
        assert config.selected_judgment_paths() == [first, second]

    print("Judgment path selection test passed")

def test_default_require_temperature_support():
    """Test model-aware temperature policy defaults."""
    from .config import default_require_temperature_support

    assert default_require_temperature_support("gpt-4.1-mini") is True
    assert default_require_temperature_support("gpt-5.5") is False
    assert default_require_temperature_support("  GPT-5.4  ") is False
    print("Temperature support default test passed")

def test_ws_tagging_summary():
    """Test compact WS tagging summary generation."""
    from .dictionary_runner import build_ws_tagging_summary

    ws_tagging = {
        "theme_mappings": [
            {
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 3,
                "theme_presence": "PRESENT",
                "recommended_action": "REINFORCE",
                "cross_reference_theme_ids": ["T02_MANAGEMENT_DIRECTION"],
                "duplication_risk": "Keep distinct from management direction.",
                "mapping_rationale": "WS expressly describes role evolution."
            }
        ]
    }

    summary = build_ws_tagging_summary(ws_tagging)
    assert summary["theme_presence_by_id"]["T01_ROLE_EVOLUTION"] == "PRESENT"
    assert summary["recommended_action_by_id"]["T01_ROLE_EVOLUTION"] == "REINFORCE"
    assert "theme_priority_by_id" not in summary
    assert summary["cross_reference_theme_ids_by_id"]["T01_ROLE_EVOLUTION"] == ["T02_MANAGEMENT_DIRECTION"]
    assert "duplication_risk" in summary["risk_or_rationale_by_id"]["T01_ROLE_EVOLUTION"]
    print("WS tagging summary test passed")


def test_ws_tagging_summary_sanitizes_reinforcement_actions():
    """Test ABSENT and RISK_ONLY baselines cannot carry reinforcement actions downstream."""
    from .dictionary_runner import build_ws_tagging_summary

    ws_tagging = {
        "theme_mappings": [
            {
                "mapped_theme_id": "T20_RISK_CONTROL",
                "theme_presence": "RISK_ONLY",
                "recommended_action": "REINFORCE",
                "mapping_rationale": "Quarantine-only issue."
            },
            {
                "mapped_theme_id": "T09_MISSING_LOGS",
                "theme_presence": "ABSENT",
                "recommended_action": "ADD EVIDENCE ANCHOR",
                "mapping_rationale": "Not currently in the WS."
            }
        ]
    }

    summary = build_ws_tagging_summary(ws_tagging)
    assert summary["recommended_action_by_id"]["T20_RISK_CONTROL"] == "REVIEW_MANUALLY"
    assert summary["recommended_action_by_id"]["T09_MISSING_LOGS"] == "REVIEW_MANUALLY"
    assert "summary_action_sanitized" in summary["risk_or_rationale_by_id"]["T20_RISK_CONTROL"]
    print("WS tagging summary sanitization test passed")


def test_ws_baseline_validation_rules():
    """Test calibration validation enforces WS tagging baseline coupling."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    calibration = {
        "case_metadata": {"case_name": "Test Case"},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "SIG_001",
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 3,
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "ws_presence": "PRESENT",
                "dictionary_match_confidence": "HIGH",
                "cross_reference_theme_ids": []
            }
        ],
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False
        }
    }
    ws_tagging_summary = {
        "theme_presence_by_id": {"T01_ROLE_EVOLUTION": "ABSENT"},
        "recommended_action_by_id": {"T01_ROLE_EVOLUTION": "REVIEW_MANUALLY"}
    }

    errors = validate_calibration_output(
        calibration,
        dictionary,
        ws_tagging_summary=ws_tagging_summary
    )
    assert any("ws_presence cannot be PRESENT" in e["error"] for e in errors)
    assert any("reinforcement actions are not permitted" in e["error"] for e in errors)
    print("WS baseline validation test passed")


if __name__ == "__main__":
    test_dictionary()
    test_fake_calibration()
    test_invalid_theme_id()
    test_invalid_priority()
    test_invalid_action()
    test_invalid_cross_ref()
    test_json_reload()
    test_llm_cache()
    test_judgment_path_selection()
    test_default_require_temperature_support()
    test_ws_tagging_summary()
    test_ws_tagging_summary_sanitizes_reinforcement_actions()
    test_ws_baseline_validation_rules()
    print("All tests passed!")
