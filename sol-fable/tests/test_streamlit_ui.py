from pathlib import Path

import pytest
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture(autouse=True)
def _isolate_app_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOL_FABLE_DATA_DIR", str(tmp_path / "app-data"))


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=20).run()


def test_fresh_session_hides_prior_runs_and_masks_cloud_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    app = _run_app()

    assert not app.exception
    assert any("Previous cases are never opened automatically" in item.value for item in app.info)
    assert not app.subheader
    inputs = {item.label: item for item in app.text_input}
    assert inputs["OpenAI API key"].proto.type == TextInput.PASSWORD
    assert inputs["Anthropic API key"].proto.type == TextInput.PASSWORD
    assert inputs["OpenAI API key"].value == ""
    assert inputs["Anthropic API key"].value == ""


def test_environment_key_is_used_without_becoming_a_widget_value(monkeypatch) -> None:
    secret = "openai-test-secret-that-must-not-render"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    app = _run_app()
    app.selectbox[0].select("openai").run()

    inputs = {item.label: item for item in app.text_input}
    assert inputs["OpenAI API key"].value == ""
    assert any("configured from OPENAI_API_KEY" in item.value for item in app.caption)
    assert secret not in str(app)


def test_remote_ollama_host_is_blocked_and_replaced_with_deterministic() -> None:
    app = _run_app()
    app.selectbox[0].select("ollama").run()
    app.selectbox[1].select("ollama").run()
    host = next(item for item in app.text_input if item.label == "Ollama host")
    host.set_value("http://models.example.test:11434").run()

    assert any("not local" in item.value for item in app.error)
    assert any("Active backend: deterministic-v1" in item.value for item in app.success)


def test_unreachable_loopback_ollama_really_falls_back_to_deterministic() -> None:
    app = _run_app()
    app.selectbox[0].select("ollama").run()
    app.selectbox[1].select("ollama").run()
    host = next(item for item in app.text_input if item.label == "Ollama host")
    host.set_value("http://127.0.0.1:1").run()

    assert any("Local Ollama is unavailable" in item.value for item in app.warning)
    assert any("Active backend: deterministic-v1" in item.value for item in app.success)


def test_streamlit_honours_writable_data_root_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "streamlit-data"
    monkeypatch.setenv("SOL_FABLE_DATA_DIR", str(selected))

    app = _run_app()

    assert not app.exception
    assert (selected / "case" / "database" / "case.sqlite").is_file()
