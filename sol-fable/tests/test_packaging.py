"""Packaging metadata and resource-copy regression tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

from sol_fable.config import default_data_root
from sol_fable.streamlit_runner import _app_path


PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_RESOURCES = PROJECT_ROOT / "src" / "sol_fable" / "resources"


def test_packaged_defaults_and_prompts_match_checkout_sources() -> None:
    source_prompts = sorted((PROJECT_ROOT / "prompts").glob("*.txt"))
    packaged_prompts = sorted((PACKAGE_RESOURCES / "prompts").glob("*.txt"))
    assert {item.name for item in packaged_prompts} == {item.name for item in source_prompts}

    pairs = [
        (PROJECT_ROOT / "config" / "settings.yaml", PACKAGE_RESOURCES / "default_settings.yaml"),
        *[
            (source, PACKAGE_RESOURCES / "prompts" / source.name)
            for source in source_prompts
        ],
    ]

    for source, packaged in pairs:
        assert packaged.read_text(encoding="utf-8").strip() == source.read_text(
            encoding="utf-8"
        ).strip()


def test_wheel_metadata_includes_resources_and_streamlit_runner() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"]["sol-fable-ui"] == (
        "sol_fable.streamlit_runner:main"
    )
    assert "resources/default_settings.yaml" in (
        metadata["tool"]["setuptools"]["package-data"]["sol_fable"]
    )
    root_data = metadata["tool"]["setuptools"]["data-files"]["."]
    assert "streamlit_app.py" in root_data
    assert "run_streamlit.sh" in root_data


def test_streamlit_runner_finds_checkout_application() -> None:
    assert _app_path() == (PROJECT_ROOT / "streamlit_app.py").resolve()


def test_packaged_runner_and_cli_share_environment_data_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "wheel-safe-data"
    monkeypatch.setenv("SOL_FABLE_DATA_DIR", str(selected))

    assert default_data_root() == selected.resolve()
    app_source = _app_path().read_text(encoding="utf-8")
    assert "PipelineOrchestrator(backend=" in app_source
    assert "PipelineOrchestrator(PROJECT_ROOT" not in app_source
