"""Configuration strictness and writable data-root regression tests."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sol_fable import config
from sol_fable.analysis_backend import DeterministicAssessmentBackend
from sol_fable.config import Settings, load_settings
from sol_fable.errors import ConfigurationError
from sol_fable.orchestrator import PipelineOrchestrator


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def _raw_settings() -> dict[str, object]:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update({"typo_backend": "deterministic-v1"}),
        lambda raw: raw["thresholds"].update({"high_confidnce": 0.9}),
        lambda raw: raw["paths"].update({"reports_dir": "elsewhere"}),
    ],
)
def test_settings_forbid_unknown_keys_at_every_level(mutation) -> None:
    raw = _raw_settings()
    mutation(raw)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings.model_validate(raw)


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_ranking_weights_require_exact_supported_keys(change: str) -> None:
    raw = _raw_settings()
    weights = raw["ranking_weights"]
    if change == "missing":
        weights.pop("ws_support")
    else:
        weights["new_unreviewed_factor"] = 0.0

    with pytest.raises(ValidationError, match="exactly the six supported keys"):
        Settings.model_validate(raw)


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [
        (math.nan, "must be finite"),
        (math.inf, "must be finite"),
        (-0.01, "must be between 0 and 1"),
        (1.01, "must be between 0 and 1"),
        (True, "must be numeric"),
        ("0.30", "must be numeric"),
    ],
)
def test_ranking_weights_require_finite_bounded_numbers(
    invalid_value: object,
    message: str,
) -> None:
    raw = _raw_settings()
    raw["ranking_weights"]["outcome_materiality"] = invalid_value

    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(raw)


def test_ranking_weights_must_sum_to_one() -> None:
    raw = _raw_settings()
    raw["ranking_weights"]["outcome_materiality"] = 0.29

    with pytest.raises(ValidationError, match="must sum to 1.0"):
        Settings.model_validate(raw)


def test_load_settings_wraps_strict_validation_as_configuration_error(
    tmp_path: Path,
) -> None:
    raw = _raw_settings()
    raw["paths"]["unknown_path"] = "case-two"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown_path"):
        load_settings(path)


def test_default_data_root_honours_explicit_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "explicit-data"
    monkeypatch.setenv("SOL_FABLE_DATA_DIR", str(selected))

    assert config.default_data_root() == selected.resolve()


@pytest.mark.skipif(os.name == "nt", reason="XDG is a POSIX data-directory convention")
def test_installed_default_data_root_uses_xdg_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xdg_root = tmp_path / "xdg-data"
    monkeypatch.delenv("SOL_FABLE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_root))
    monkeypatch.setattr(config, "source_checkout_root", lambda: None)
    monkeypatch.setattr(config.sys, "platform", "linux")

    assert config.default_data_root() == (xdg_root / "sol-fable").resolve()


def test_source_checkout_retains_repository_local_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOL_FABLE_DATA_DIR", raising=False)

    assert config.source_checkout_root() == PROJECT_ROOT.resolve()
    assert config.default_data_root() == PROJECT_ROOT.resolve()


def test_default_settings_prefer_editable_checkout_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    config_dir = checkout / "config"
    config_dir.mkdir(parents=True)
    raw = _raw_settings()
    raw["n_rounds"] = 4
    (config_dir / "settings.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(config, "source_checkout_root", lambda: checkout)

    assert load_settings().n_rounds == 4


def test_code_default_prompt_version_matches_bundled_contract() -> None:
    raw = _raw_settings()
    raw.pop("prompt_version")

    assert Settings.model_validate(raw).prompt_version == "0.3.0"


@pytest.mark.parametrize("case_dir", ["", "../outside", "/tmp/outside"])
def test_case_directory_cannot_escape_data_root(case_dir: str) -> None:
    raw = _raw_settings()
    raw["paths"]["case_dir"] = case_dir

    with pytest.raises(ValidationError, match="relative path"):
        Settings.model_validate(raw)


def test_unknown_backend_is_rejected_during_config_load() -> None:
    raw = _raw_settings()
    raw["analysis_backend"] = "open-ai-typo"

    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_ollama_gpu_layer_request_cannot_be_negative() -> None:
    raw = _raw_settings()
    raw["ollama_num_gpu"] = -1

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Settings.model_validate(raw)


def test_orchestrator_uses_environment_data_root_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "orchestrator-data"
    monkeypatch.setenv("SOL_FABLE_DATA_DIR", str(selected))

    orchestrator = PipelineOrchestrator(backend=DeterministicAssessmentBackend())

    assert orchestrator.project_dir == selected.resolve()
    assert orchestrator.store.path == selected / "case" / "database" / "case.sqlite"
    assert orchestrator.store.path.is_file()
