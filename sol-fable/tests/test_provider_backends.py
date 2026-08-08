"""Provider-adapter contract tests without making paid or network API calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import sol_fable.llm_backend as llm_backend
from sol_fable.analysis_backend import DeterministicAssessmentBackend
from sol_fable.debate import conduct_debate, deterministic_debate_summary
from sol_fable.llm_backend import (
    AnthropicAssessmentBackend,
    OllamaAssessmentBackend,
    OpenAIAssessmentBackend,
    PersonaRouterBackend,
)
from sol_fable.errors import LLMAssessmentError
from sol_fable.errors import ConfigurationError
from sol_fable.models import (
    Argument,
    ArgumentCategory,
    BarristerPersona,
    Importance,
    PartySide,
    PleadingStatus,
    SourceOrigin,
)


def _argument() -> Argument:
    return Argument(
        argument_id="ARG-MIX-001",
        title="Source-bound proposition",
        proposition="The meeting was not understood as a warning.",
        category=ArgumentCategory.MIXED,
        importance=Importance.HIGH,
        issue_ids=["ISSUE-001"],
        ws_paragraphs=["WS-P001"],
        et1_paragraphs=["ET1-P001"],
        et3_paragraphs=["ET3-P001"],
        source_types=[SourceOrigin.SELF_ACCOUNT, SourceOrigin.ET1, SourceOrigin.ET3],
        document_placeholders=[],
        pleading_statuses=[PleadingStatus.ET1_DETAIL_OR_ELABORATION],
        proposition_ids=["WS-P001-C01"],
        human_review_status="NOT_REQUIRED",
    )


def _structured_payload(model_name: str) -> dict:
    evidence = [
        {
            "claim": "The record contains the proposition.",
            "paragraph_ids": ["WS-P001"],
            "relationship": "SUPPORTS",
        }
    ]
    common = {
        "paragraph_citations": ["WS-P001"],
        "evidence_links": evidence,
        "confidence": 0.75,
    }
    if model_name == "_SolLLMOutput":
        return {
            "best_formulation": "Keep the proposition tied to the witness account.",
            "pleading_basis": "ET1-P001",
            "claimant_relevance": "Potentially helpful if accepted.",
            "required_findings": ["The witness did not understand a warning."],
            "hidden_assumptions": ["The recollection is reliable."],
            "strength": "MEDIUM",
            "new_case_risk": "LOW",
            "protect_points": ["Preserve the narrow formulation."],
            "risks": ["The ET3 disputes the account."],
            **common,
        }
    if model_name == "_FableLLMOutput":
        return {
            "respondent_case": "The meeting can be characterised differently.",
            "pleading_attack": "Test consistency with the pleaded formulation.",
            "evidence_attack": "The account is not independent corroboration.",
            "alternative_explanation": "A warning may have been conveyed informally.",
            "damaging_concessions": ["A meeting occurred."],
            "cross_examination_risk": "MEDIUM",
            "defend_points": ["Prepare the chronology."],
            "risk_level": "MEDIUM",
            "hidden_assumptions": ["Informal words were sufficiently clear."],
            **common,
        }
    if model_name == "_DebateTurnLLMOutput":
        return {
            "position": "A source-bound position.",
            "supporting_points": ["The cited witness paragraph records the account."],
            "responses_to_opponent": ["The prior position is answered within the same source map."],
            "challenges_for_opponent": ["Identify the evidential basis."],
            "concessions_or_risks": ["The point remains disputed."],
            "hidden_assumptions": ["The recollection is accurate."],
            "evidence_gaps": ["No independent document is ingested."],
            "protect_or_defend_points": ["Stay within the cited text."],
            **common,
        }
    if model_name == "_DebateSummaryLLMOutput":
        return {
            "methodological_agreements": ["Use the same source map."],
            "remaining_disputes": ["How the meeting should be understood."],
            "hidden_assumptions": ["The recollection is accurate."],
            "evidence_gaps": ["No independent document is ingested."],
            "human_review_reasons": [],
            "claimant_strength": "MEDIUM",
            "respondent_strength": "MEDIUM",
            "recommended_treatment": "BOTH",
            "treatment_rationale": "Both positions remain materially arguable on the cited record.",
            "paragraph_citations": ["WS-P001"],
            "evidence_links": evidence,
        }
    raise AssertionError(f"Unexpected structured model: {model_name}")


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        output_model = kwargs["text_format"]
        return SimpleNamespace(
            id="resp_test",
            status="completed",
            model="resolved-openai-snapshot",
            output_parsed=output_model.model_validate(
                _structured_payload(output_model.__name__)
            ),
            usage=SimpleNamespace(input_tokens=100, output_tokens=25, total_tokens=125),
        )


def test_openai_adapter_uses_responses_structured_output_and_records_usage(monkeypatch) -> None:
    responses = _FakeResponses()
    constructor: dict = {}

    def fake_openai(**kwargs):
        constructor.update(kwargs)
        return SimpleNamespace(responses=responses)

    monkeypatch.setattr(llm_backend, "OpenAI", fake_openai)
    backend = OpenAIAssessmentBackend(api_key="session-secret", model="test-openai")
    assessment = backend.sol(_argument(), "0.3.0", {"WS-P001": "Meeting text."})

    assert constructor["api_key"] == "session-secret"
    assert responses.calls[0]["model"] == "test-openai"
    assert responses.calls[0]["store"] is False
    assert assessment.paragraph_citations == ["WS-P001"]
    assert assessment.usage.provider == "openai"
    assert assessment.usage.requested_model == "test-openai"
    assert assessment.usage.model == "resolved-openai-snapshot"
    assert assessment.usage.total_tokens == 125
    assert "session-secret" not in repr(backend.metadata())


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        output_model = kwargs["output_format"]
        return SimpleNamespace(
            id="msg_test",
            stop_reason="end_turn",
            model="resolved-claude-snapshot",
            parsed_output=output_model.model_validate(
                _structured_payload(output_model.__name__)
            ),
            usage=SimpleNamespace(input_tokens=90, output_tokens=30),
        )


def test_anthropic_adapter_uses_messages_parse_and_records_usage(monkeypatch) -> None:
    messages = _FakeMessages()
    constructor: dict = {}

    def fake_anthropic(**kwargs):
        constructor.update(kwargs)
        return SimpleNamespace(messages=messages)

    monkeypatch.setattr(llm_backend, "Anthropic", fake_anthropic)
    backend = AnthropicAssessmentBackend(api_key="session-secret", model="test-claude")
    assessment = backend.fable(_argument(), "0.3.0", {"WS-P001": "Meeting text."})

    assert constructor["api_key"] == "session-secret"
    assert messages.calls[0]["model"] == "test-claude"
    assert assessment.paragraph_citations == ["WS-P001"]
    assert assessment.usage.provider == "anthropic"
    assert assessment.usage.requested_model == "test-claude"
    assert assessment.usage.model == "resolved-claude-snapshot"
    assert assessment.usage.total_tokens == 120
    assert "session-secret" not in repr(backend.metadata())


def test_ollama_adapter_explicitly_requests_full_gpu_offload(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(_structured_payload("_SolLLMOutput"))
                ),
                prompt_eval_count=100,
                eval_count=25,
            )

    monkeypatch.setattr(
        llm_backend,
        "ollama",
        SimpleNamespace(Client=lambda **kwargs: FakeClient()),
    )
    backend = OllamaAssessmentBackend(model="test-ollama", num_gpu=999)

    assessment = backend.sol(_argument(), "0.3.0", {"WS-P001": "Meeting text."})

    assert calls[0]["options"] == {
        "temperature": 0,
        "num_predict": 4096,
        "num_gpu": 999,
    }
    assert assessment.usage.provider == "ollama"
    assert assessment.usage.total_tokens == 125
    assert backend.metadata()["num_gpu"] == 999


def test_ollama_adapter_can_leave_gpu_selection_automatic(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(_structured_payload("_SolLLMOutput"))
                ),
                prompt_eval_count=None,
                eval_count=None,
            )

    monkeypatch.setattr(
        llm_backend,
        "ollama",
        SimpleNamespace(Client=lambda **kwargs: FakeClient()),
    )
    backend = OllamaAssessmentBackend(model="test-ollama", num_gpu=None)

    backend.sol(_argument(), "0.3.0")

    assert "num_gpu" not in calls[0]["options"]


def test_live_debate_retries_an_empty_opponent_response(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            payload = _structured_payload("_DebateTurnLLMOutput")
            if len(calls) == 1:
                payload["responses_to_opponent"] = []
            return SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)),
                prompt_eval_count=100,
                eval_count=25,
            )

    monkeypatch.setattr(
        llm_backend,
        "ollama",
        SimpleNamespace(Client=lambda **kwargs: FakeClient()),
    )
    backend = OllamaAssessmentBackend(model="test-ollama", max_retries=1)
    opponent_turn = DeterministicAssessmentBackend().debate_turn(
        _argument(),
        round_number=1,
        barrister=BarristerPersona.FABLE,
        side=PartySide.RESPONDENT,
        opponent_turn=None,
        own_previous_turn=None,
        prompt_version="0.3.0",
    )

    turn = backend.debate_turn(
        _argument(),
        round_number=2,
        barrister=BarristerPersona.SOL,
        side=PartySide.CLAIMANT,
        opponent_turn=opponent_turn,
        own_previous_turn=None,
        prompt_version="0.3.0",
    )

    assert len(calls) == 2
    assert turn.responses_to_opponent
    assert "responses_to_opponent must contain" in calls[1]["messages"][1]["content"]


class _TaggedDeterministic(DeterministicAssessmentBackend):
    def __init__(self, name: str):
        self.name = name

    def summarize_debate(
        self,
        argument,
        n_rounds,
        claimant_barrister,
        respondent_barrister,
        turns,
        paragraph_texts=None,
    ):
        return deterministic_debate_summary(
            argument,
            n_rounds,
            claimant_barrister,
            respondent_barrister,
            turns,
            paragraph_texts,
        )


class _LiveTaggedDeterministic(_TaggedDeterministic):
    is_live = True

    def metadata(self):
        return {"type": "test-live", "name": self.name, "live": True}


@pytest.mark.parametrize(
    ("claimant", "expected_backends"),
    [
        (BarristerPersona.SOL, ["sol-provider", "fable-provider"]),
        (BarristerPersona.FABLE, ["fable-provider", "sol-provider"]),
    ],
)
def test_persona_router_is_independent_of_party_assignment(claimant, expected_backends) -> None:
    router = PersonaRouterBackend(
        sol_backend=_TaggedDeterministic("sol-provider"),
        fable_backend=_TaggedDeterministic("fable-provider"),
    )

    summary = conduct_debate(
        _argument(),
        router,
        "0.3.0",
        n_rounds=1,
        claimant_barrister=claimant,
    )

    assert [turn.side for turn in summary.turns] == [
        PartySide.CLAIMANT,
        PartySide.RESPONDENT,
    ]
    assert [turn.backend for turn in summary.turns] == expected_backends
    assert summary.stop_reason == "CONFIGURED_ROUND_LIMIT_REACHED:1"


def test_mixed_router_uses_dual_neutral_consensus_summary() -> None:
    router = PersonaRouterBackend(
        sol_backend=_LiveTaggedDeterministic("live-sol"),
        fable_backend=_TaggedDeterministic("deterministic-fable"),
    )

    summary = conduct_debate(
        _argument(),
        router,
        "0.3.0",
        n_rounds=1,
        claimant_barrister=BarristerPersona.SOL,
    )

    assert router.is_live is True
    assert summary.backend.endswith(":dual-neutral-consensus")
    assert summary.recommended_treatment in {"PROTECT", "DEFEND", "BOTH"}
    assert set(summary.paragraph_citations) == {
        paragraph_id
        for link in summary.evidence_links
        for paragraph_id in link.paragraph_ids
    }


def test_live_summary_is_grounded_and_supplies_bounded_verdict(monkeypatch) -> None:
    responses = _FakeResponses()
    monkeypatch.setattr(
        llm_backend,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=responses),
    )
    backend = OpenAIAssessmentBackend(api_key="session-secret", model="test-openai")

    summary = conduct_debate(
        _argument(),
        backend,
        "0.3.0",
        n_rounds=1,
        claimant_barrister=BarristerPersona.SOL,
    )

    assert summary.recommended_treatment == "BOTH"
    assert summary.claimant_strength == "MEDIUM"
    assert summary.respondent_strength == "MEDIUM"
    assert summary.paragraph_citations == ["WS-P001"]
    assert len(responses.calls) == 3  # two turns plus one neutral summary


def test_grounding_retry_accumulates_observed_usage(monkeypatch) -> None:
    class RetryResponses:
        def __init__(self):
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            output_model = kwargs["text_format"]
            payload = _structured_payload(output_model.__name__)
            if self.calls == 1:
                payload["paragraph_citations"] = ["WS-P999"]
                payload["evidence_links"][0]["paragraph_ids"] = ["WS-P999"]
            return SimpleNamespace(
                id=f"resp_{self.calls}",
                status="completed",
                model="resolved-openai-snapshot",
                output_parsed=output_model.model_validate(payload),
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=25,
                    total_tokens=125,
                ),
            )

    responses = RetryResponses()
    monkeypatch.setattr(
        llm_backend,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=responses),
    )
    backend = OpenAIAssessmentBackend(
        api_key="session-secret", model="test-openai", max_retries=1
    )

    assessment = backend.sol(_argument(), "0.3.0")

    assert responses.calls == 2
    assert assessment.usage.observed_attempts == 2
    assert assessment.usage.total_tokens == 250
    assert assessment.usage.request_ids == ["resp_1", "resp_2"]


def test_live_baseline_rejects_pleading_labelled_as_support(monkeypatch) -> None:
    class PleadingAsProofResponses:
        def parse(self, **kwargs):
            output_model = kwargs["text_format"]
            payload = _structured_payload(output_model.__name__)
            payload["paragraph_citations"] = ["ET3-P001"]
            payload["evidence_links"] = [
                {
                    "claim": "The pleaded allegation proves the event.",
                    "paragraph_ids": ["ET3-P001"],
                    "relationship": "SUPPORTS",
                }
            ]
            return SimpleNamespace(
                id="pleading_as_proof",
                status="completed",
                model="resolved-openai-snapshot",
                output_parsed=output_model.model_validate(payload),
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            )

    monkeypatch.setattr(
        llm_backend,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=PleadingAsProofResponses()),
    )
    backend = OpenAIAssessmentBackend(
        api_key="session-secret", model="test-openai", max_retries=0
    )

    with pytest.raises(LLMAssessmentError, match="CONTEXT only"):
        backend.sol(_argument(), "0.3.0")


def test_unbounded_provider_risk_label_is_rejected(monkeypatch) -> None:
    class InvalidLabelResponses:
        def parse(self, **kwargs):
            output_model = kwargs["text_format"]
            payload = _structured_payload(output_model.__name__)
            payload["strength"] = "moderately strong"
            return SimpleNamespace(
                id="invalid",
                status="completed",
                output_parsed=output_model.model_validate(payload),
                usage=None,
            )

    monkeypatch.setattr(
        llm_backend,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=InvalidLabelResponses()),
    )
    backend = OpenAIAssessmentBackend(
        api_key="session-secret", model="test-openai", max_retries=0
    )

    with pytest.raises(LLMAssessmentError, match="failed to produce grounded"):
        backend.sol(_argument(), "0.3.0")


@pytest.mark.parametrize(
    "host",
    [
        "http://user:password@localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434?token=secret",
        "http://localhost:bad-port",
    ],
)
def test_ollama_origin_rejects_persistable_secrets_and_paths(host: str) -> None:
    with pytest.raises(ConfigurationError):
        llm_backend._validate_ollama_host(host, allow_remote_host=False)
