import json
import os
from pathlib import Path

from .dictionary_loader import (
    compact_dictionary_for_llm,
    load_dictionary,
    validate_dictionary,
    get_theme_ids,
    get_priority_by_theme,
    get_allowed_actions_by_theme,
)
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

def test_compact_dictionary_for_llm():
    """Test repeated LLM calls receive only required dictionary fields."""
    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    compact = compact_dictionary_for_llm(dictionary)

    assert "global_mapping_rules" in compact
    assert "ws_theme_dictionary" in compact
    assert len(compact["ws_theme_dictionary"]) == 20
    first_theme = compact["ws_theme_dictionary"][0]
    assert "theme_id" in first_theme
    assert "theme_priority" in first_theme
    assert "include_when" in first_theme
    assert "exclude_when" in first_theme
    assert "duplication_guardrail" in first_theme
    assert "permitted_actions" in first_theme
    assert "common_subthemes" not in first_theme
    assert "example_mapping_language" not in first_theme
    assert len(json.dumps(compact, separators=(',', ':'))) < len(json.dumps(dictionary, separators=(',', ':')))
    print("Compact dictionary test passed")

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
    from .io_utils import read_json, write_json
    data = {"test": "value", "number": 42}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    write_json(temp_path, data)
    reloaded = read_json(temp_path)
    assert reloaded == data
    temp_path.unlink()
    print("JSON reload test passed")

def test_run_scope_slug_preserves_single_case_name():
    """Test run-level artifact naming preserves single-case source identity."""
    from .io_utils import make_run_scope_slug

    assert make_run_scope_slug("RUN", ["case_a"]) == "RUN_case_a"
    assert make_run_scope_slug("RUN", ["case_a", "case_b"]) == "RUN_batch_2_cases"
    assert make_run_scope_slug("RUN", []) == "RUN_no_successful_cases"
    print("Run scope slug test passed")

def test_prepare_ws_tagging_loads_existing_summary():
    """Test run_ws=False loads a prior summary without creating WS output files."""
    import tempfile
    from types import SimpleNamespace
    from .io_utils import write_json
    from .main import prepare_ws_tagging

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        summary_path = temp_root / "existing_summary.json"
        expected = {
            "theme_presence_by_id": {"T01_ROLE_EVOLUTION": "PRESENT"},
            "recommended_action_by_id": {"T01_ROLE_EVOLUTION": "REINFORCE"},
        }
        write_json(summary_path, expected)
        config = SimpleNamespace(
            run_ws=False,
            ws_tagging_summary_path=summary_path,
            output_root=temp_root / "output",
            validate_json_writes=True,
        )

        actual = prepare_ws_tagging(config, "TEST_RUN", "ws text", {}, "prompt", None)

        assert actual == expected
        assert not (config.output_root / "ws_tagging").exists()
    print("WS tagging summary load test passed")

def test_prepare_ws_tagging_reuses_existing_summary_before_llm():
    """Test run_ws=True still reuses a local summary before making a WS LLM call."""
    import tempfile
    from types import SimpleNamespace
    from .io_utils import write_json
    from .main import prepare_ws_tagging

    class ExplodingLLMClient:
        def complete_json(self, system_prompt, user_payload):
            raise AssertionError("WS tagging should have reused the existing summary")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        summary_path = temp_root / "existing_summary.json"
        expected = {
            "theme_presence_by_id": {"T01_ROLE_EVOLUTION": "PRESENT"},
            "recommended_action_by_id": {"T01_ROLE_EVOLUTION": "REINFORCE"},
        }
        write_json(summary_path, expected)
        config = SimpleNamespace(
            run_ws=True,
            reuse_existing_ws_tagging=True,
            ws_tagging_summary_path=summary_path,
            output_root=temp_root / "output",
            validate_json_writes=True,
        )

        actual = prepare_ws_tagging(config, "TEST_RUN", "ws text", {}, "prompt", ExplodingLLMClient())

        assert actual == expected
        assert not (config.output_root / "ws_tagging").exists()
    print("WS tagging reuse-before-LLM test passed")

def test_prepare_ws_tagging_runs_and_writes_summary():
    """Test run_ws=True writes exactly the full WS tagging and derived summary."""
    import tempfile
    from types import SimpleNamespace
    from .main import prepare_ws_tagging

    class FakeLLMClient:
        def complete_json(self, system_prompt, user_payload):
            return {
                "theme_mappings": [
                    {
                        "mapped_theme_id": "T01_ROLE_EVOLUTION",
                        "theme_presence": "PRESENT",
                        "recommended_action": "REINFORCE",
                        "cross_reference_theme_ids": [],
                        "mapping_rationale": "Baseline present",
                    }
                ]
            }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        output_root = temp_root / "output"
        (output_root / "ws_tagging").mkdir(parents=True)
        config = SimpleNamespace(
            run_ws=True,
            ws_tagging_summary_path=temp_root / "unused.json",
            output_root=output_root,
            validate_json_writes=True,
        )

        summary = prepare_ws_tagging(config, "TEST_RUN", "ws text", {}, "prompt", FakeLLMClient())
        output_files = sorted(path.name for path in (output_root / "ws_tagging").iterdir())

        assert summary["theme_presence_by_id"] == {"T01_ROLE_EVOLUTION": "PRESENT"}
        assert output_files == [
            "TEST_RUN_ws_tagging.json",
            "TEST_RUN_ws_tagging_summary.json",
        ]
    print("WS tagging run/write test passed")

def test_record_case_failure_writes_review_artifact():
    """Test failed cases are captured as human-review artifacts."""
    import tempfile
    from types import SimpleNamespace
    from .main import record_case_failure

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        config = SimpleNamespace(
            output_root=temp_root / "output",
            validate_json_writes=True,
        )
        (config.output_root / "human_review_queue").mkdir(parents=True)
        judgment_path = temp_root / "failed_case.pdf"
        try:
            raise RuntimeError("simulated failure")
        except RuntimeError as exc:
            failure = record_case_failure(
                config,
                "TEST_RUN",
                judgment_path,
                "failed_case",
                "calibration",
                exc,
            )

        failure_path = config.output_root / "human_review_queue" / "TEST_RUN_failed_case_case_failure.json"
        assert failure_path.exists()
        assert failure["judgment"] == "failed_case.pdf"
        assert failure["failed_stage"] == "calibration"
        assert failure["error_type"] == "RuntimeError"
        assert "simulated failure" in failure["error"]
    print("Case failure artifact test passed")

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

def test_llm_timeout_configuration_does_not_affect_cache_key():
    """Test request timeout is configurable but does not invalidate the response cache."""
    import tempfile
    from .llm_client import LLMClient

    with tempfile.TemporaryDirectory() as temp_dir:
        base_kwargs = {
            "provider": "local",
            "model": "fake",
            "temperature": 0.0,
            "max_tokens": 100,
            "require_temperature_support": True,
            "cache_dir": Path(temp_dir),
            "cache_enabled": True,
        }
        fast = LLMClient(**base_kwargs, request_timeout_seconds=300)
        slow = LLMClient(**base_kwargs, request_timeout_seconds=600)

        assert fast.request_timeout_seconds == 300
        assert slow.request_timeout_seconds == 600
        assert fast._cache_path("prompt", {"value": 1}) == slow._cache_path("prompt", {"value": 1})
    print("LLM timeout configuration test passed")

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
        config.judgment_index_path = batch_dir / "missing_index.csv"
        assert config.selected_judgment_paths() == [first, second]

        index = batch_dir / "moltie_judgment_index.csv"
        index.write_text(
            "rank,judgment_pdf_path,composite_pct,percentile_rank\n"
            f"1,{second},98.2,99.9\n"
            f"2,{first},74.1,88.5\n",
            encoding="utf-8",
        )
        config.judgment_index_path = index
        assert config.selected_judgment_paths() == [second, first]

    print("Judgment path selection test passed")

def test_run_inventory_blocks_existing_downstream_json():
    """Test per-case downstream JSON blocks reruns for matching PDFs."""
    import tempfile
    from .io_utils import write_json
    from .run_inventory import claim_judgment_run, get_judgment_run_status, release_judgment_run_claim

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        pdf_path = temp_root / "input" / "judgments" / "case_one.pdf"
        output_root = temp_root / "output"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        raw_dir = output_root / "calibration_raw"
        raw_dir.mkdir(parents=True)
        write_json(raw_dir / "RUN_case_one_calibration_raw.json", {"ok": True})

        partial_status = get_judgment_run_status(pdf_path, output_root)
        assert partial_status.status == "blocked_partial"
        assert partial_status.runnable is False

        outcome_dir = output_root / "outcome_optimized"
        outcome_dir.mkdir(parents=True)
        write_json(outcome_dir / "RUN_case_one_outcome_optimized.json", {"ok": True})

        complete_status = get_judgment_run_status(pdf_path, output_root)
        assert complete_status.status == "complete"
        assert complete_status.runnable is False

        other_pdf = temp_root / "input" / "judgments" / "case_two.pdf"
        other_pdf.write_bytes(b"%PDF-1.4")
        claim_path = claim_judgment_run(other_pdf, output_root, "RUN_A")
        assert claim_path is not None
        in_progress_status = get_judgment_run_status(other_pdf, output_root)
        assert in_progress_status.status == "in_progress"
        assert in_progress_status.runnable is False
        assert claim_judgment_run(other_pdf, output_root, "RUN_B") is None
        release_judgment_run_claim(claim_path)
        ready_status = get_judgment_run_status(other_pdf, output_root)
        assert ready_status.status == "not_run"
        assert ready_status.runnable is True

    print("Run inventory downstream JSON guard test passed")

def test_run_calibrator_skips_existing_artifacts_before_shared_input_load():
    """Test the runner exits before loading shared inputs when selected PDFs are already run."""
    import tempfile
    from .config import Config
    from .io_utils import write_json
    from .main import run_calibrator

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        pdf_path = temp_root / "input" / "judgments" / "case_one.pdf"
        output_root = temp_root / "output"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        raw_dir = output_root / "calibration_raw"
        raw_dir.mkdir(parents=True)
        write_json(raw_dir / "RUN_case_one_calibration_raw.json", {"ok": True})

        config = Config.default("TEST_RUN")
        config.project_root = temp_root
        config.ws_path = temp_root / "missing_ws.pdf"
        config.judgment_path = pdf_path
        config.judgments_dir = pdf_path.parent
        config.run_mode = "debug"
        config.output_root = output_root
        config.cache_root = output_root / "cache"

        result = run_calibrator(config)

        assert result["processed_case_count"] == 0
        assert result["skipped_case_count"] == 1
        assert result["skipped_cases"][0]["status"] == "blocked_partial"

    print("Run calibrator skip-before-load test passed")

def test_default_require_temperature_support():
    """Test model-aware temperature policy defaults."""
    from .config import default_require_temperature_support

    assert default_require_temperature_support("gpt-4.1-mini") is True
    assert default_require_temperature_support("gpt-5.5") is False
    assert default_require_temperature_support("  GPT-5.4  ") is False
    print("Temperature support default test passed")

def test_openai_request_omits_temperature_when_not_required():
    """Test provider-default-temperature models do not send temperature."""
    from .llm_client import LLMClient

    class CapturingLLMClient(LLMClient):
        def __init__(self):
            super().__init__(
                provider="openai",
                model="gpt-5.5",
                temperature=0.0,
                max_tokens=100,
                require_temperature_support=False,
            )
            self.captured_body = None

        def _post_openai_chat_completion(self, request_body, api_key):
            self.captured_body = request_body
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "{\"ok\": true}"}
                    }
                ]
            }

    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        client = CapturingLLMClient()
        assert client._openai_complete("prompt", {"value": 1}) == {"ok": True}
    finally:
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key

    assert "temperature" not in client.captured_body
    assert client.captured_body["messages"][1]["content"] == "{\"value\":1}"
    print("OpenAI default-temperature request test passed")

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
    assert "T01_ROLE_EVOLUTION" not in summary["risk_or_rationale_by_id"]
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
            },
            {
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_presence": "LATENT",
                "recommended_action": "ADD FACT",
                "mapping_rationale": "Only weakly organised in the WS."
            }
        ]
    }

    summary = build_ws_tagging_summary(ws_tagging)
    assert summary["recommended_action_by_id"]["T20_RISK_CONTROL"] == "REVIEW_MANUALLY"
    assert summary["recommended_action_by_id"]["T09_MISSING_LOGS"] == "REVIEW_MANUALLY"
    assert summary["recommended_action_by_id"]["T01_ROLE_EVOLUTION"] == "REVIEW_MANUALLY"
    assert "summary_action_sanitized" in summary["risk_or_rationale_by_id"]["T20_RISK_CONTROL"]
    print("WS tagging summary sanitization test passed")


def test_reinforcement_cluster_count():
    """Test compression cluster count uses current bucket names."""
    from .compression_runner import count_reinforcement_clusters

    reinforcement_plan = {
        "reinforcement_clusters": [{}, {}],
        "manual_review_clusters": [{}],
        "risk_control_clusters": [{}],
    }
    legacy_plan = {"compressed_reinforcement_plan": [{}, {}]}

    assert count_reinforcement_clusters(reinforcement_plan) == 4
    assert count_reinforcement_clusters(legacy_plan) == 2
    print("Reinforcement cluster count test passed")


def _valid_outcome_calibration():
    from .outcome_lookup_tables import REQUIRED_OUTCOME_QC_TRUE_FLAGS

    calibration = {
        "case_metadata": {"case_name": "Test Case", "case_number": "123"},
        "case_relevance_to_ws": {},
        "judgment_signals": [
            {
                "signal_id": "JS01",
                "mapped_theme_id": "T01_ROLE_EVOLUTION",
                "theme_priority": 3,
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "ws_presence": "PRESENT",
                "dictionary_match_confidence": "HIGH",
                "signal_summary": "Tribunal finding supports the claimant.",
                "judgment_references": ["1"],
                "relevance_to_ws": "This maps to the role evolution theme.",
                "cross_reference_theme_ids": []
            },
            {
                "signal_id": "JS02",
                "mapped_theme_id": "T02_MANAGEMENT_DIRECTION",
                "theme_priority": 4,
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "ws_presence": "PRESENT",
                "dictionary_match_confidence": "HIGH",
                "signal_summary": "Tribunal finding supports management direction.",
                "judgment_references": ["2"],
                "relevance_to_ws": "This maps to the management direction theme.",
                "cross_reference_theme_ids": []
            }
        ],
        "outcome_optimization": {
            "factual_proximity": "ANALOGY_ONLY",
            "transferability_rating": "HIGH",
            "claimant_liability_outcome": "WIN",
            "liability_outcome_strength_band": "STRONG_WIN",
            "remedy_status": "OUTSTANDING",
            "reduction_findings_status": "NOT_DETERMINED",
            "polkey_reduction_pct": None,
            "contributory_fault_pct": None,
            "award_reduction_notes": "Both percentages are null because reductions are not determined.",
            "signal_causal_weights": [
                {
                    "signal_id": "JS01",
                    "causal_weight": "DECISIVE",
                    "causal_weight_reason": "The Tribunal treated this issue as central to the outcome."
                },
                {
                    "signal_id": "JS02",
                    "causal_weight": "CONTRIBUTING",
                    "causal_weight_reason": "The Tribunal treated this issue as a material supporting point."
                }
            ],
            "negative_theme_flags": [],
            "optimization_notes": "Factually distant but structurally useful for claimant-side WS review."
        },
        "quality_control": {
            "ws_rewrite_performed": False,
            "new_allegations_created": False,
            "new_themes_created": False,
            "other_negative_pattern_human_review_required": False
        }
    }
    for flag in REQUIRED_OUTCOME_QC_TRUE_FLAGS:
        calibration["quality_control"][flag] = True
    return calibration


def test_outcome_scoring_functions():
    """Test deterministic outcome scoring formulas."""
    from .outcome_scoring import (
        apply_ranking_thresholds,
        compute_pt_score,
        compute_remedy_outcome_strength,
        compute_signal_optimization_score,
        get_threshold_profile,
        remedy_strength_for_aggregation,
    )

    assert compute_pt_score("ANALOGY_ONLY", "HIGH") == 0.7
    assert compute_signal_optimization_score(
        "DECISIVE",
        "STRONG_WIN",
        "ANALOGY_ONLY",
        "HIGH",
        "HIGH",
    ) == 0.7
    assert compute_remedy_outcome_strength(25, 20, "DETERMINED") == 0.6
    assert compute_remedy_outcome_strength(None, None, "NOT_DETERMINED") is None
    assert remedy_strength_for_aggregation(None, None, "NOT_DETERMINED") == 0.5
    assert apply_ranking_thresholds(5, 2, get_threshold_profile("pilot_20_to_30")) == "REINFORCE_PRIMARY"
    assert apply_ranking_thresholds(7, 1, get_threshold_profile("pilot_20_to_30")) == "REINFORCE_SUPPORTING"
    print("Outcome scoring functions test passed")


def test_outcome_validation_rules():
    """Test valid outcome layer and key schema constraints."""
    from copy import deepcopy
    from .outcome_validators import validate_outcome_optimized_calibration

    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    calibration = _valid_outcome_calibration()

    assert validate_outcome_optimized_calibration(calibration, dictionary) == []

    mixed_invalid = deepcopy(calibration)
    mixed_invalid["outcome_optimization"]["claimant_liability_outcome"] = "MIXED"
    mixed_invalid["outcome_optimization"]["liability_outcome_strength_band"] = "STRONG_WIN"
    mixed_errors = validate_outcome_optimized_calibration(mixed_invalid, dictionary)
    assert any("inconsistent with MIXED" in error["error"] for error in mixed_errors)

    missing_weight = deepcopy(calibration)
    missing_weight["outcome_optimization"]["signal_causal_weights"] = missing_weight["outcome_optimization"]["signal_causal_weights"][:1]
    weight_errors = validate_outcome_optimized_calibration(missing_weight, dictionary)
    assert any("Missing signal_id entries" in error["error"] for error in weight_errors)

    bad_zero_notes = deepcopy(calibration)
    bad_zero_notes["outcome_optimization"]["reduction_findings_status"] = "DETERMINED"
    bad_zero_notes["outcome_optimization"]["polkey_reduction_pct"] = 0
    bad_zero_notes["outcome_optimization"]["contributory_fault_pct"] = 0
    bad_zero_notes["outcome_optimization"]["award_reduction_notes"] = "Both reductions are zero."
    zero_errors = validate_outcome_optimized_calibration(bad_zero_notes, dictionary)
    assert any("zero reductions are express" in error["error"] for error in zero_errors)
    print("Outcome validation rules test passed")


def test_outcome_merge_and_aggregation():
    """Test additive merge and aggregation output shape."""
    from copy import deepcopy
    from .outcome_aggregation import aggregate_outcome_optimized_cases
    from .outcome_runner import merge_outcome_optimization, repair_outcome_optimization

    dict_path = Path("/home/hello/Projects/Calibrator/input/dictionary/WS_Controlled_Theme_Dictionary_v1_2_final.json")
    dictionary = load_dictionary(dict_path)
    calibration = _valid_outcome_calibration()
    base = {
        key: value
        for key, value in calibration.items()
        if key != "outcome_optimization"
    }
    patch = {
        "outcome_optimization": calibration["outcome_optimization"],
        "quality_control": {
            key: value
            for key, value in calibration["quality_control"].items()
            if key not in {"ws_rewrite_performed", "new_allegations_created", "new_themes_created"}
        }
    }
    merged = merge_outcome_optimization(base, patch)
    assert "outcome_optimization" not in base
    assert merged["case_metadata"] == base["case_metadata"]
    assert merged["outcome_optimization"] == calibration["outcome_optimization"]

    aggregation = aggregate_outcome_optimized_cases([calibration], dictionary)
    assert aggregation["aggregation_metadata"]["case_count"] == 1
    assert aggregation["theme_strength_matrix"]
    assert aggregation["case_shortlist"][0]["claimant_usefulness_score"] > 0
    assert aggregation["case_shortlist"][0]["liability_usefulness_score"] > 0
    assert aggregation["case_shortlist"][0]["liability_usefulness_band"] in {"LOW", "MEDIUM", "MEDIUM_HIGH", "HIGH"}
    assert "NO_SIGNAL" in aggregation["ranked_theme_optimization_table"]
    assert aggregation["ranked_theme_optimization_table"]["NO_SIGNAL"]
    assert all(
        row["supporting_case_count"] > 0 or row["total_negative_penalty"] != 0
        for row in aggregation["ranked_theme_optimization_table"]["REFRAME"]
    )
    assert aggregation["ws_optimization_mapping"]["status"] == "blocked_until_ws_theme_anchor_map"

    risk_case = deepcopy(calibration)
    risk_case["judgment_signals"].append({
        "signal_id": "JS03",
        "mapped_theme_id": "T20_RISK_CONTROL",
        "theme_priority": 20,
        "recommended_action": "DO_NOT_USE",
        "case_effect": "NON_TRANSFERABLE",
        "ws_presence": "ABSENT",
        "dictionary_match_confidence": "MEDIUM",
        "signal_summary": "The remedy finding is adverse.",
        "judgment_references": ["3"],
        "relevance_to_ws": "This should be quarantined.",
        "cross_reference_theme_ids": []
    })
    risk_case["outcome_optimization"]["signal_causal_weights"].append({
        "signal_id": "JS03",
        "causal_weight": "DECISIVE",
        "causal_weight_reason": "The Tribunal treated the remedy issue as decisive."
    })
    risk_case["outcome_optimization"]["negative_theme_flags"].append({
        "theme_id": "T20_RISK_CONTROL",
        "negative_pattern": "HIGH_POLKEY_REDUCTION",
        "severity": "HIGH",
        "reason": "The Tribunal made a high Polkey reduction."
    })
    risk_aggregation = aggregate_outcome_optimized_cases([risk_case], dictionary)
    risk_t20 = next(row for row in risk_aggregation["theme_strength_matrix"] if row["theme_id"] == "T20_RISK_CONTROL")
    assert risk_t20["recommendation"] == "RISK_CONTROL"
    assert risk_t20["total_positive_score"] == 0
    assert risk_aggregation["risk_control_summary"]["polkey_risk_cases"] == 1

    invalid = _valid_outcome_calibration()
    invalid["outcome_optimization"]["signal_causal_weights"] = invalid["outcome_optimization"]["signal_causal_weights"][:1]

    class FakeRepairClient:
        def complete_json(self, system_prompt, user_payload):
            assert "VALIDATION_ERRORS" in user_payload
            return {
                "outcome_optimization": calibration["outcome_optimization"],
                "quality_control": calibration["quality_control"],
            }

    repaired = repair_outcome_optimization(
        invalid,
        [{"path": "outcome_optimization.signal_causal_weights", "error": "Missing signal_id entries"}],
        "repair prompt",
        FakeRepairClient(),
    )
    assert len(repaired["outcome_optimization"]["signal_causal_weights"]) == 2
    print("Outcome merge and aggregation test passed")


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


def test_theme_store_builds_batch_review_outputs():
    """Test deterministic theme store exports from aggregation keys and outcome items."""
    import tempfile
    from .theme_store import build_theme_store, write_theme_store_outputs

    aggregation = {
        "theme_strength_matrix": [
            {
                "theme_id": "T11_SHORT_NOTICE_PROCEDURAL_PREJUDICE",
                "theme_name": "Short notice",
                "net_theme_score": 1.2,
                "recommendation": "REFRAME",
            }
        ]
    }
    outcome_case = {
        "case_metadata": {
            "case_name": "Test Case",
            "case_number": "123",
        },
        "judgment_signals": [
            {
                "signal_id": "JS01",
                "mapped_theme_id": "T11_SHORT_NOTICE_PROCEDURAL_PREJUDICE",
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "dictionary_match_confidence": "HIGH",
                "signal_summary": "Invitation lacked specific allegations.",
                "judgment_references": ["27", "28"],
                "relevance_to_ws": "Supports preparation prejudice.",
                "subtheme": "missing_particulars",
                "factual_hooks": ["no specific allegations"],
                "legal_functions": ["procedural unfairness"],
            },
            {
                "signal_id": "JS02",
                "mapped_theme_id": "T11_SHORT_NOTICE_PROCEDURAL_PREJUDICE",
                "recommended_action": "REINFORCE",
                "case_effect": "WIN_DRIVER",
                "dictionary_match_confidence": "HIGH",
                "signal_summary": "Invitation lacked specific allegations.",
                "judgment_references": ["27", "28"],
                "relevance_to_ws": "Duplicate should not become an active second match.",
                "subtheme": "missing_particulars",
                "factual_hooks": ["no specific allegations"],
                "legal_functions": ["procedural unfairness"],
            },
            {
                "signal_id": "JS03",
                "mapped_theme_id": "T11_SHORT_NOTICE_PROCEDURAL_PREJUDICE",
                "recommended_action": "REVIEW_MANUALLY",
                "case_effect": "NEUTRAL_CONTEXT",
                "dictionary_match_confidence": "HIGH",
                "signal_summary": "Invitation lacked specific allegations.",
                "judgment_references": ["27", "28"],
                "relevance_to_ws": "Same factual point but different action lane should remain separate.",
                "subtheme": "missing_particulars",
                "factual_hooks": ["no specific allegations"],
                "legal_functions": ["procedural unfairness"],
            },
        ],
        "outcome_optimization": {
            "signal_causal_weights": [
                {
                    "signal_id": "JS01",
                    "causal_weight": "CONTRIBUTING",
                    "causal_weight_reason": "Material procedural defect.",
                },
                {
                    "signal_id": "JS02",
                    "causal_weight": "CONTRIBUTING",
                    "causal_weight_reason": "Duplicate material procedural defect.",
                },
                {
                    "signal_id": "JS03",
                    "causal_weight": "PERIPHERAL",
                    "causal_weight_reason": "Manual review version of the same material.",
                },
            ]
        },
    }

    bundle = build_theme_store(aggregation, [outcome_case], {0: "case_outcome_optimized.json"})
    theme = bundle["theme_store"]["T11_SHORT_NOTICE_PROCEDURAL_PREJUDICE"]
    reinforce_matches = theme["groups"]["REINFORCE|WIN_DRIVER|HIGH"]["matches"]
    review_matches = theme["groups"]["REVIEW_MANUALLY|NEUTRAL_CONTEXT|HIGH"]["matches"]

    assert theme["n_matches"] == 3
    assert reinforce_matches[0]["rank_score"] == 1.0
    assert reinforce_matches[0]["source_pointer"] == "judgment paragraphs: 27, 28"
    assert review_matches[0]["effect"] == "REVIEW_MANUALLY"
    assert bundle["theme_summary"][0]["number_of_matches"] == 3
    assert bundle["theme_summary"][0]["number_of_groups"] == 2
    assert len(bundle["review_queue"]) == 3
    assert len(bundle["top_matches_per_theme"]) == 3

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = write_theme_store_outputs(bundle, Path(temp_dir))
        for path in paths.values():
            assert path.exists()
    print("Theme store export test passed")


if __name__ == "__main__":
    test_dictionary()
    test_compact_dictionary_for_llm()
    test_fake_calibration()
    test_invalid_theme_id()
    test_invalid_priority()
    test_invalid_action()
    test_invalid_cross_ref()
    test_json_reload()
    test_run_scope_slug_preserves_single_case_name()
    test_prepare_ws_tagging_loads_existing_summary()
    test_prepare_ws_tagging_reuses_existing_summary_before_llm()
    test_prepare_ws_tagging_runs_and_writes_summary()
    test_record_case_failure_writes_review_artifact()
    test_llm_cache()
    test_llm_timeout_configuration_does_not_affect_cache_key()
    test_judgment_path_selection()
    test_default_require_temperature_support()
    test_openai_request_omits_temperature_when_not_required()
    test_ws_tagging_summary()
    test_ws_tagging_summary_sanitizes_reinforcement_actions()
    test_reinforcement_cluster_count()
    test_outcome_scoring_functions()
    test_outcome_validation_rules()
    test_outcome_merge_and_aggregation()
    test_ws_baseline_validation_rules()
    test_theme_store_builds_batch_review_outputs()
    print("All tests passed!")
