"""Live, opt-in end-to-end test against a local Ollama instance.

Runs only when ``SOL_FABLE_RUN_LIVE_OLLAMA=1`` is explicitly set and the requested
model is already installed. A normal test run never starts a model workload.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ollama = pytest.importorskip("ollama")


MODEL = os.environ.get("SOL_FABLE_TEST_OLLAMA_MODEL", "gemma3:27b")


def _ollama_available(host: str = "http://localhost:11434") -> bool:
    if os.environ.get("SOL_FABLE_RUN_LIVE_OLLAMA") != "1":
        return False
    try:
        response = ollama.Client(host=host).list()
        return any(item.model == MODEL for item in response.models)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="set SOL_FABLE_RUN_LIVE_OLLAMA=1 and install the requested model",
)

from sol_fable.llm_backend import OllamaAssessmentBackend  # noqa: E402
from sol_fable.orchestrator import PipelineOrchestrator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def test_ollama_backend_end_to_end(tmp_path: Path) -> None:
    backend = OllamaAssessmentBackend(model=MODEL)
    orchestrator = PipelineOrchestrator(project_dir=tmp_path / "standalone", backend=backend)
    run_id = orchestrator.run_all(
        FIXTURES / "ET1.txt",
        FIXTURES / "ET3.txt",
        FIXTURES / "WS.txt",
        n_rounds=2,
        live_argument_cap=1,
    )
    run = orchestrator.store.load_run(run_id)
    assert run.status == "COMPLETED"
    assert run.backend == backend.name
    arguments = orchestrator.store.load_arguments(run_id)
    assert arguments
    for argument in arguments:
        allowed = set(argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs)
        assert argument.sol_assessment is not None
        assert argument.fable_assessment is not None
        assert argument.debate_summary is not None
        assert set(argument.sol_assessment.paragraph_citations) <= allowed
        assert set(argument.fable_assessment.paragraph_citations) <= allowed
        assert len(argument.debate_summary.turns) == 4  # n_rounds=2
        for turn in argument.debate_summary.turns:
            assert set(turn.paragraph_citations) <= allowed
