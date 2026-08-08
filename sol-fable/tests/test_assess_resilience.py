"""Resumable checkpointing and the local-model argument cap, no live Ollama needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from sol_fable.agents import assess_fable, assess_sol
from sol_fable.debate import deterministic_debate_summary, deterministic_debate_turn
from sol_fable.errors import LLMAssessmentError, PipelineStateError
from sol_fable.orchestrator import PipelineOrchestrator

FIXTURES = Path(__file__).parent / "fixtures"


class _CountingBackend:
    """Deterministic-backed stub that counts calls and carries a configurable name."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def sol(self, argument, prompt_version, paragraph_texts=None):
        self.calls += 1
        return assess_sol(argument, self.name, prompt_version, paragraph_texts)

    def fable(self, argument, prompt_version, paragraph_texts=None):
        return assess_fable(argument, self.name, prompt_version, paragraph_texts)

    def debate_turn(
        self,
        argument,
        round_number,
        barrister,
        side,
        opponent_turn,
        own_previous_turn,
        prompt_version,
        paragraph_texts=None,
    ):
        return deterministic_debate_turn(
            argument,
            round_number,
            barrister,
            side,
            opponent_turn,
            own_previous_turn,
            self.name,
            prompt_version,
            paragraph_texts,
        )

    def summarize_debate(
        self, argument, n_rounds, claimant_barrister, respondent_barrister, turns, paragraph_texts=None
    ):
        return deterministic_debate_summary(
            argument, n_rounds, claimant_barrister, respondent_barrister, turns, paragraph_texts
        )


class _FailingLiveBackend(_CountingBackend):
    is_live = True
    supports_argument_cap = True

    def metadata(self):
        return {"type": "test-live", "name": self.name, "live": True}

    def sol(self, argument, prompt_version, paragraph_texts=None):
        self.calls += 1
        raise LLMAssessmentError("Unknown paragraph citations: ['ET3-P999']")


def _orchestrator_with_arguments(tmp_path: Path, backend) -> tuple[PipelineOrchestrator, str]:
    orchestrator = PipelineOrchestrator(project_dir=tmp_path / "standalone", backend=backend)
    run_id = orchestrator.start_run()
    orchestrator.ingest(FIXTURES / "ET1.txt", FIXTURES / "ET3.txt", FIXTURES / "WS.txt", run_id)
    orchestrator.parse_references(run_id)
    orchestrator.build_issues(run_id)
    orchestrator.extract_propositions(run_id)
    orchestrator.match_pleadings(run_id)
    orchestrator.build_arguments(run_id)
    return orchestrator, run_id


def test_assess_resumes_without_redoing_completed_arguments(tmp_path: Path) -> None:
    backend = _CountingBackend("stub-v1")
    orchestrator, run_id = _orchestrator_with_arguments(tmp_path, backend)

    orchestrator.assess(run_id, n_rounds=1)
    total_arguments = len(orchestrator.store.load_arguments(run_id))
    assert total_arguments > 0
    assert backend.calls == total_arguments
    assert all(item.debate_summary is not None for item in orchestrator.store.load_arguments(run_id))

    orchestrator.assess(run_id, n_rounds=1)
    assert backend.calls == total_arguments  # no re-processing on the resumed call


def test_assess_caps_ollama_backend_and_falls_back_deterministically(tmp_path: Path) -> None:
    backend = _CountingBackend("ollama-stub")
    orchestrator, run_id = _orchestrator_with_arguments(tmp_path, backend)

    total_arguments = len(orchestrator.store.load_arguments(run_id))
    assert total_arguments > 1  # otherwise the cap can't meaningfully split the batch

    orchestrator.assess(run_id, n_rounds=1, ollama_argument_cap=1)

    assert backend.calls == 1
    arguments = orchestrator.store.load_arguments(run_id)
    backends_used = [item.sol_assessment.backend for item in arguments]
    assert backends_used.count("ollama-stub") == 1
    assert backends_used.count("deterministic-v1") == total_arguments - 1
    assert all(item.debate_summary is not None for item in arguments)


def test_live_validation_failure_falls_back_for_one_argument_and_continues(
    tmp_path: Path,
) -> None:
    backend = _FailingLiveBackend("ollama-failing-stub")
    orchestrator, run_id = _orchestrator_with_arguments(tmp_path, backend)

    orchestrator.assess(run_id, n_rounds=1, live_argument_cap=1)

    arguments = orchestrator.store.load_arguments(run_id)
    assert all(item.debate_summary is not None for item in arguments)
    assert all(item.assessment_mode == "DETERMINISTIC_FALLBACK" for item in arguments)
    failed = next(
        item
        for item in arguments
        if any(
            "could not produce a grounded assessment" in reason
            for reason in item.debate_summary.human_review_reasons
        )
    )
    assert failed.sol_assessment.backend == "deterministic-v1"
    run = orchestrator.store.load_run(run_id)
    assert run.config_snapshot["assessment_failures"] == [
        {
            "argument_id": failed.argument_id,
            "backend": "ollama-failing-stub",
            "error_type": "LLMAssessmentError",
            "error": "Unknown paragraph citations: ['ET3-P999']",
        }
    ]


def test_changed_debate_configuration_reassesses_and_invalidates_downstream(tmp_path: Path) -> None:
    backend = _CountingBackend("stub-v1")
    orchestrator, run_id = _orchestrator_with_arguments(tmp_path, backend)
    orchestrator.assess(run_id, n_rounds=1)
    total_arguments = len(orchestrator.store.load_arguments(run_id))
    orchestrator.rank(run_id)
    orchestrator.generate_search_packages(run_id)
    orchestrator.report(run_id)
    assert orchestrator.store.load_run(run_id).status == "COMPLETED"

    orchestrator.assess(run_id, n_rounds=2)

    assert backend.calls == total_arguments * 2
    changed = orchestrator.store.load_arguments(run_id)
    assert all(item.debate_summary.n_rounds == 2 for item in changed)
    assert all(item.ranking_score is None and not item.treatment for item in changed)
    assert orchestrator.store.load_search_packages(run_id) == []
    assert orchestrator.store.load_run(run_id).status == "RUNNING"
    with pytest.raises(PipelineStateError, match="assessed and ranked"):
        orchestrator.report(run_id)

    orchestrator.rank(run_id)
    orchestrator.generate_search_packages(run_id)
    orchestrator.report(run_id)
    assert orchestrator.store.load_run(run_id).status == "COMPLETED"
