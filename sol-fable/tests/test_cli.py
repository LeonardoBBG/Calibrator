"""Focused tests for invocation-wide CLI options and backend overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sol_fable.cli import _parser, main
from sol_fable.models import RunRecord
from sol_fable.storage import CaseStore


FIXTURES = Path(__file__).parent / "fixtures"
DEFAULT_CONFIG = Path(__file__).parents[1] / "config" / "settings.yaml"


@pytest.mark.parametrize(
    ("option", "value", "destination", "expected"),
    [
        ("--project-root", "/tmp/sol-fable-data", "project_root", Path("/tmp/sol-fable-data")),
        ("--config", "/tmp/settings.yaml", "config", Path("/tmp/settings.yaml")),
        ("--run-id", "RUN-TEST", "run_id", "RUN-TEST"),
        ("--analysis-backend", "dual-api", "analysis_backend", "dual-api"),
        ("--ollama-model", "gemma3:test", "ollama_model", "gemma3:test"),
        ("--ollama-host", "http://ollama.test:11434", "ollama_host", "http://ollama.test:11434"),
        ("--ollama-num-gpu", "999", "ollama_num_gpu", 999),
        ("--ollama-argument-cap", "7", "ollama_argument_cap", 7),
        ("--live-argument-cap", "9", "live_argument_cap", 9),
        ("--openai-model", "gpt-test", "openai_model", "gpt-test"),
        ("--anthropic-model", "claude-test", "anthropic_model", "claude-test"),
    ],
)
def test_global_value_options_work_before_and_after_subcommand(
    option: str,
    value: str,
    destination: str,
    expected: object,
) -> None:
    before = _parser().parse_args([option, value, "status"])
    after = _parser().parse_args(["status", option, value])

    assert getattr(before, destination) == expected
    assert getattr(after, destination) == expected


def test_verbose_works_before_and_after_subcommand() -> None:
    assert _parser().parse_args(["--verbose", "status"]).verbose is True
    assert _parser().parse_args(["status", "--verbose"]).verbose is True


def test_later_duplicate_global_option_wins() -> None:
    args = _parser().parse_args(
        ["--run-id", "RUN-BEFORE", "status", "--run-id", "RUN-AFTER"]
    )
    assert args.run_id == "RUN-AFTER"


def test_negative_live_cap_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["status", "--live-argument-cap", "-1"])


def test_deterministic_cli_override_beats_ollama_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["analysis_backend"] = "ollama"
    config_path = tmp_path / "ollama-settings.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    project_root = tmp_path / "case-project"

    exit_code = main(
        [
            "ingest",
            "--et1",
            str(FIXTURES / "ET1.txt"),
            "--et3",
            str(FIXTURES / "ET3.txt"),
            "--ws",
            str(FIXTURES / "WS.txt"),
            "--project-root",
            str(project_root),
            "--config",
            str(config_path),
            "--analysis-backend",
            "deterministic-v1",
        ]
    )

    assert exit_code == 0
    store = CaseStore(project_root / "case" / "database" / "case.sqlite")
    run_id = store.latest_run_id()
    assert run_id is not None
    run = store.load_run(run_id)
    assert run.backend == "deterministic-v1"
    assert run.config_snapshot["analysis_backend"] == "deterministic-v1"


def test_status_does_not_require_configured_cloud_key(tmp_path: Path, monkeypatch) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["analysis_backend"] = "openai"
    config_path = tmp_path / "openai-settings.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    project_root = tmp_path / "case-project"
    store = CaseStore(project_root / "case" / "database" / "case.sqlite")
    store.create_run(
        RunRecord(
            run_id="RUN-STATUS",
            started_at="2026-01-01T00:00:00+00:00",
            status="RUNNING",
            backend="openai",
            prompt_version="0.3.0",
            config_snapshot={},
        )
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main(
        [
            "status",
            "--project-root",
            str(project_root),
            "--config",
            str(config_path),
            "--run-id",
            "RUN-STATUS",
        ]
    ) == 0
