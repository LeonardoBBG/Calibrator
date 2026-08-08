"""Structured assessment backends for local, OpenAI and Anthropic models.

All providers share one evidence contract. Provider SDKs are optional: importing
this module is safe, and a clear configuration error is raised only when the
corresponding backend is constructed.
"""

from __future__ import annotations

import ipaddress
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from pydantic import Field, ValidationError

from .analysis_backend import AssessmentBackend, DeterministicAssessmentBackend
from .config import Settings, project_root
from .debate import citations as debate_citations
from .debate import deterministic_debate_summary, validate_pleading_evidence_links
from .errors import ConfigurationError, LLMAssessmentError
from .models import (
    Argument,
    BarristerPersona,
    DebateSummary,
    DebateTurn,
    EvidenceLink,
    FableAssessment,
    LLMUsage,
    PartySide,
    SolAssessment,
    StrictModel,
)

LOGGER = logging.getLogger("sol_fable.llm_backend")

try:  # Optional dependency.
    import ollama
except ImportError:  # pragma: no cover - environment dependent
    ollama = None  # type: ignore[assignment]

try:  # Optional dependency.
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependent
    OpenAI = None  # type: ignore[assignment,misc]

try:  # Optional dependency.
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - environment dependent
    Anthropic = None  # type: ignore[assignment,misc]


_PROMPT_STEMS = {
    "sol": "sol",
    "fable": "fable",
    "debate": "barrister_debate",
    "summary": "debate_summary",
}


def _prompt_filename(name: str, prompt_version: str) -> str:
    if name not in _PROMPT_STEMS:
        raise ConfigurationError(f"Unknown prompt role: {name}")
    if not re.fullmatch(r"\d+\.\d+\.\d+", prompt_version):
        raise ConfigurationError("prompt_version must use semantic version form, for example 0.3.0")
    return f"{_PROMPT_STEMS[name]}_v{prompt_version}.txt"


@lru_cache(maxsize=16)
def _load_prompt(name: str, prompt_version: str) -> str:
    """Load the exact configured prompt, preferring editable checkout files."""

    filename = _prompt_filename(name, prompt_version)
    checkout_path = project_root() / "prompts" / filename
    if checkout_path.is_file():
        return checkout_path.read_text(encoding="utf-8").strip()
    try:
        packaged = resources.files("sol_fable.resources.prompts").joinpath(filename)
        return packaged.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ModuleNotFoundError, TypeError) as exc:
        raise ConfigurationError(
            f"Prompt {filename} is not bundled; refusing to label a different prompt as {prompt_version}"
        ) from exc


def prompt_bundle_sha256(prompt_version: str) -> str:
    """Hash the exact prompt texts used by a live assessment contract."""

    # Freeze a coherent bundle for the upcoming assessment call while allowing
    # an in-place checkout edit to be detected on the next invocation.
    _load_prompt.cache_clear()
    digest = hashlib.sha256()
    for name in sorted(_PROMPT_STEMS):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_load_prompt(name, prompt_version).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _paragraph_records(
    paragraph_ids: list[str], paragraph_texts: dict[str, str] | None
) -> list[dict[str, str]]:
    return [
        {
            "paragraph_id": paragraph_id,
            "text": (paragraph_texts or {}).get(paragraph_id, "[paragraph text unavailable]"),
        }
        for paragraph_id in paragraph_ids
    ]


def _argument_payload(
    argument: Argument, paragraph_texts: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "argument_id": argument.argument_id,
        "title": argument.title,
        "proposition": argument.proposition,
        "category": str(argument.category),
        "importance": str(argument.importance),
        "pleading_statuses": [str(value) for value in argument.pleading_statuses],
        "document_placeholders": argument.document_placeholders,
        "ws_paragraphs": _paragraph_records(argument.ws_paragraphs, paragraph_texts),
        "et1_paragraphs": _paragraph_records(argument.et1_paragraphs, paragraph_texts),
        "et3_paragraphs": _paragraph_records(argument.et3_paragraphs, paragraph_texts),
        "permitted_paragraph_ids": debate_citations(argument),
    }


def _turn_payload(turn: DebateTurn | None) -> dict[str, Any] | None:
    if turn is None:
        return None
    return turn.model_dump(mode="json", exclude={"usage"})


def _persona_contract(barrister: BarristerPersona, side: PartySide, round_number: int) -> str:
    persona = BarristerPersona(barrister)
    party = PartySide(side)
    persona_rule = (
        "SOL is a constructive case strategist. Build the clearest source-bound theory, "
        "identify the findings needed for it to succeed, preserve genuine helpful points, "
        "and acknowledge weaknesses that cannot responsibly be avoided."
        if persona == BarristerPersona.SOL
        else
        "FABLE is a forensic stress-tester. Probe pleading overreach, proof gaps, internal "
        "tension, alternative inferences and concessions, while acknowledging any point the "
        "record genuinely supports."
    )
    side_rule = (
        "For the CLAIMANT, establish or responsibly narrow the proposition and explain why "
        "the fact-finder could accept it."
        if party == PartySide.CLAIMANT
        else
        "For the RESPONDENT, defeat, narrow or expose the proposition without treating the "
        "ET3 as proof or inventing a positive account."
    )
    round_rule = (
        "This is the opening round: advance the strongest complete position from the record."
        if round_number == 1
        else
        "This is a rebuttal round: answer every material opponent challenge supplied, state "
        "what is conceded, and identify exactly what remains unresolved."
    )
    return f"{persona_rule}\n\n{side_rule}\n\n{round_rule}"


_StrengthLabel = Literal["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "MINIMAL", "NONE"]


class _SolLLMOutput(StrictModel):
    best_formulation: str = Field(min_length=1)
    pleading_basis: str = Field(min_length=1)
    claimant_relevance: str = Field(min_length=1)
    required_findings: list[str] = Field(min_length=1)
    hidden_assumptions: list[str]
    strength: _StrengthLabel
    new_case_risk: _StrengthLabel
    protect_points: list[str] = Field(min_length=1)
    risks: list[str]
    paragraph_citations: list[str] = Field(min_length=1)
    evidence_links: list[EvidenceLink] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class _FableLLMOutput(StrictModel):
    respondent_case: str = Field(min_length=1)
    pleading_attack: str = Field(min_length=1)
    evidence_attack: str = Field(min_length=1)
    alternative_explanation: str = Field(min_length=1)
    damaging_concessions: list[str]
    cross_examination_risk: str = Field(min_length=1)
    defend_points: list[str] = Field(min_length=1)
    risk_level: _StrengthLabel
    hidden_assumptions: list[str]
    paragraph_citations: list[str] = Field(min_length=1)
    evidence_links: list[EvidenceLink] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class _DebateTurnLLMOutput(StrictModel):
    position: str = Field(min_length=1)
    supporting_points: list[str] = Field(min_length=1)
    responses_to_opponent: list[str]
    challenges_for_opponent: list[str] = Field(min_length=1)
    concessions_or_risks: list[str]
    hidden_assumptions: list[str]
    evidence_gaps: list[str]
    protect_or_defend_points: list[str] = Field(min_length=1)
    paragraph_citations: list[str] = Field(min_length=1)
    evidence_links: list[EvidenceLink] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class _DebateSummaryLLMOutput(StrictModel):
    methodological_agreements: list[str] = Field(min_length=1)
    remaining_disputes: list[str] = Field(min_length=1)
    hidden_assumptions: list[str]
    evidence_gaps: list[str]
    human_review_reasons: list[str]
    claimant_strength: _StrengthLabel
    respondent_strength: _StrengthLabel
    recommended_treatment: Literal["PROTECT", "DEFEND", "BOTH"]
    treatment_rationale: str = Field(min_length=1)
    paragraph_citations: list[str] = Field(min_length=1)
    evidence_links: list[EvidenceLink] = Field(min_length=1)


@dataclass(frozen=True)
class _StructuredResult:
    value: StrictModel
    usage: LLMUsage


def _combine_usage(
    values: list[LLMUsage], *, provider: str | None = None, model: str | None = None
) -> LLMUsage:
    """Combine every observable response; provider-internal retries are disabled."""

    if not values:
        raise ValueError("At least one usage record is required")
    request_ids = [
        request_id
        for value in values
        for request_id in (value.request_ids or ([value.response_id] if value.response_id else []))
    ]
    input_known = all(value.input_tokens is not None for value in values)
    output_known = all(value.output_tokens is not None for value in values)
    total_known = all(value.total_tokens is not None for value in values)
    requested_models = list(dict.fromkeys(value.requested_model for value in values if value.requested_model))
    resolved_models = list(dict.fromkeys(value.model for value in values))
    return LLMUsage(
        provider=provider or values[-1].provider,
        requested_model=(", ".join(requested_models) if requested_models else None),
        model=model or ", ".join(resolved_models),
        input_tokens=sum(value.input_tokens or 0 for value in values) if input_known else None,
        output_tokens=sum(value.output_tokens or 0 for value in values) if output_known else None,
        total_tokens=sum(value.total_tokens or 0 for value in values) if total_known else None,
        response_id=request_ids[-1] if request_ids else None,
        request_ids=request_ids,
        observed_attempts=sum(value.observed_attempts for value in values),
    )


class _StructuredAssessmentBackend:
    """Provider-neutral prompt construction and semantic grounding validation."""

    provider: str
    model: str
    name: str
    is_live = True
    supports_argument_cap = True

    def __init__(self, *, max_retries: int = 1):
        self.max_retries = max_retries

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.provider,
            "model": self.model,
            "name": self.name,
            "live": True,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
        }

    def _provider_call(
        self, system_prompt: str, user_prompt: str, output_model: type[StrictModel]
    ) -> _StructuredResult:
        raise NotImplementedError

    @staticmethod
    def _validate_grounding(
        value: StrictModel,
        allowed_citations: list[str],
        pleading_paragraph_ids: set[str],
    ) -> None:
        citations = list(getattr(value, "paragraph_citations", []))
        evidence_links = list(getattr(value, "evidence_links", []))
        allowed = set(allowed_citations)
        unknown = set(citations) - allowed
        if unknown:
            raise ValueError(f"Unknown paragraph citations: {sorted(unknown)}")
        linked_ids = {
            paragraph_id
            for link in evidence_links
            for paragraph_id in link.paragraph_ids
        }
        unknown_links = linked_ids - allowed
        if unknown_links:
            raise ValueError(f"Unknown evidence-link citations: {sorted(unknown_links)}")
        if allowed and not evidence_links:
            raise ValueError("At least one claim-to-paragraph evidence link is required")
        if set(citations) != linked_ids:
            raise ValueError(
                "paragraph_citations must exactly equal the paragraph IDs used in evidence_links"
            )
        validate_pleading_evidence_links(
            evidence_links,
            pleading_paragraph_ids,
            type(value).__name__,
        )

    def _call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: type[StrictModel],
        allowed_citations: list[str],
        pleading_paragraph_ids: set[str],
        semantic_validator: Callable[[StrictModel], None] | None = None,
    ) -> _StructuredResult:
        prompt = user_prompt
        last_error: Exception | None = None
        observed_usage: list[LLMUsage] = []
        for attempt in range(self.max_retries + 1):
            try:
                result = self._provider_call(system_prompt, prompt, output_model)
                observed_usage.append(result.usage)
                self._validate_grounding(
                    result.value,
                    allowed_citations,
                    pleading_paragraph_ids,
                )
                if semantic_validator is not None:
                    semantic_validator(result.value)
                return _StructuredResult(
                    value=result.value,
                    usage=_combine_usage(observed_usage),
                )
            except (ValidationError, ValueError, json.JSONDecodeError, LLMAssessmentError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                prompt = (
                    f"{user_prompt}\n\n<correction_request>\n"
                    f"The prior output failed validation: {exc}. Correct that specific failure "
                    "and return a complete result. "
                    "The permitted paragraph IDs are exactly "
                    f"{json.dumps(allowed_citations, ensure_ascii=False)}. "
                    "Copy IDs verbatim from that list only; remove every other ID from both "
                    "paragraph_citations and evidence_links. Make paragraph_citations exactly "
                    "match the union of evidence_links[].paragraph_ids. ET1 and ET3 are "
                    "pleadings, not proof: label every link to them as CONTEXT."
                    "\n</correction_request>"
                )
        raise LLMAssessmentError(
            f"{self.name} failed to produce grounded {output_model.__name__} after "
            f"{self.max_retries + 1} attempt(s): {last_error}"
        )

    def sol(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> SolAssessment:
        allowed = debate_citations(argument)
        prompt = json.dumps(
            {"task": "Produce SOL's independent claimant-side baseline assessment.",
             "case_material": _argument_payload(argument, paragraph_texts)},
            ensure_ascii=False,
        )
        result = self._call_structured(
            _load_prompt("sol", prompt_version),
            prompt,
            _SolLLMOutput,
            allowed,
            set(argument.et1_paragraphs + argument.et3_paragraphs),
        )
        return SolAssessment(
            argument_id=argument.argument_id,
            backend=self.name,
            prompt_version=prompt_version,
            usage=result.usage,
            **result.value.model_dump(),
        )

    def fable(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> FableAssessment:
        allowed = debate_citations(argument)
        prompt = json.dumps(
            {"task": "Produce FABLE's independent respondent-side baseline assessment.",
             "case_material": _argument_payload(argument, paragraph_texts)},
            ensure_ascii=False,
        )
        result = self._call_structured(
            _load_prompt("fable", prompt_version),
            prompt,
            _FableLLMOutput,
            allowed,
            set(argument.et1_paragraphs + argument.et3_paragraphs),
        )
        return FableAssessment(
            argument_id=argument.argument_id,
            backend=self.name,
            prompt_version=prompt_version,
            usage=result.usage,
            **result.value.model_dump(),
        )

    def debate_turn(
        self,
        argument: Argument,
        round_number: int,
        barrister: BarristerPersona,
        side: PartySide,
        opponent_turn: DebateTurn | None,
        own_previous_turn: DebateTurn | None,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateTurn:
        allowed = debate_citations(argument)
        payload = {
            "task": "Produce the next source-bound barrister turn.",
            "persona_and_party_contract": _persona_contract(barrister, side, round_number),
            "round_number": round_number,
            "barrister": BarristerPersona(barrister).value,
            "side": PartySide(side).value,
            "case_material": _argument_payload(argument, paragraph_texts),
            "opponent_turn": _turn_payload(opponent_turn),
            "own_previous_turn": _turn_payload(own_previous_turn),
        }

        def validate_turn_semantics(value: StrictModel) -> None:
            if (
                opponent_turn is not None
                and isinstance(value, _DebateTurnLLMOutput)
                and not value.responses_to_opponent
            ):
                raise ValueError(
                    "responses_to_opponent must contain at least one direct response "
                    "when an opponent turn is supplied"
                )

        result = self._call_structured(
            _load_prompt("debate", prompt_version),
            json.dumps(payload, ensure_ascii=False),
            _DebateTurnLLMOutput,
            allowed,
            set(argument.et1_paragraphs + argument.et3_paragraphs),
            semantic_validator=validate_turn_semantics,
        )
        return DebateTurn(
            turn_id=f"{argument.argument_id}-R{round_number:02d}-{PartySide(side).value}",
            argument_id=argument.argument_id,
            round_number=round_number,
            barrister=BarristerPersona(barrister),
            side=PartySide(side),
            opponent_turn_id=opponent_turn.turn_id if opponent_turn else None,
            backend=self.name,
            prompt_version=prompt_version,
            usage=result.usage,
            **result.value.model_dump(),
        )

    def summarize_debate(
        self,
        argument: Argument,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        respondent_barrister: BarristerPersona,
        turns: list[DebateTurn],
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateSummary:
        payload = {
            "task": "Act as a neutral clerk and synthesise only the validated turn record.",
            "argument": _argument_payload(argument, paragraph_texts),
            "turns": [_turn_payload(turn) for turn in turns],
        }
        prompt_version = turns[0].prompt_version if turns else "unknown"
        result = self._call_structured(
            _load_prompt("summary", prompt_version),
            json.dumps(payload, ensure_ascii=False),
            _DebateSummaryLLMOutput,
            debate_citations(argument),
            set(argument.et1_paragraphs + argument.et3_paragraphs),
        )
        return DebateSummary(
            argument_id=argument.argument_id,
            n_rounds=n_rounds,
            claimant_barrister=BarristerPersona(claimant_barrister),
            respondent_barrister=BarristerPersona(respondent_barrister),
            turns=turns,
            stop_reason=f"CONFIGURED_ROUND_LIMIT_REACHED:{n_rounds}",
            backend=self.name,
            prompt_version=prompt_version,
            usage=result.usage,
            **result.value.model_dump(),
        )


class OpenAIAssessmentBackend(_StructuredAssessmentBackend):
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.6-sol",
        max_retries: int = 1,
        timeout_seconds: float = 180,
        max_output_tokens: int = 4096,
    ):
        if OpenAI is None:
            raise ConfigurationError("Install the OpenAI adapter with: pip install -e '.[api]'")
        if not api_key.strip():
            raise ConfigurationError("An OpenAI API key is required for the OpenAI backend")
        super().__init__(max_retries=max_retries)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.name = f"openai-{model}"
        # One retry budget is owned here so billed calls cannot multiply through
        # opaque SDK retries and our semantic-correction loop.
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    def _provider_call(
        self, system_prompt: str, user_prompt: str, output_model: type[StrictModel]
    ) -> _StructuredResult:
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=output_model,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except Exception as exc:  # SDK exposes a provider-specific exception hierarchy.
            raise LLMAssessmentError(f"OpenAI request failed: {exc}") from exc
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            status = getattr(response, "status", "unknown")
            raise LLMAssessmentError(f"OpenAI returned no parsed output (status={status})")
        usage = getattr(response, "usage", None)
        response_id = getattr(response, "id", None)
        return _StructuredResult(
            value=parsed,
            usage=LLMUsage(
                provider=self.provider,
                requested_model=self.model,
                model=getattr(response, "model", None) or self.model,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                response_id=response_id,
                request_ids=[response_id] if response_id else [],
            ),
        )


class AnthropicAssessmentBackend(_StructuredAssessmentBackend):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_retries: int = 1,
        timeout_seconds: float = 180,
        max_output_tokens: int = 4096,
    ):
        if Anthropic is None:
            raise ConfigurationError("Install the Anthropic adapter with: pip install -e '.[api]'")
        if not api_key.strip():
            raise ConfigurationError("An Anthropic API key is required for the Claude backend")
        super().__init__(max_retries=max_retries)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.name = f"anthropic-{model}"
        self._client = Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    def _provider_call(
        self, system_prompt: str, user_prompt: str, output_model: type[StrictModel]
    ) -> _StructuredResult:
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                output_format=output_model,
            )
        except Exception as exc:
            raise LLMAssessmentError(f"Anthropic request failed: {exc}") from exc
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason in {"max_tokens", "refusal"}:
            raise LLMAssessmentError(f"Anthropic stopped without a usable result: {stop_reason}")
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMAssessmentError("Anthropic returned no parsed structured output")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        response_id = getattr(response, "id", None)
        return _StructuredResult(
            value=parsed,
            usage=LLMUsage(
                provider=self.provider,
                requested_model=self.model,
                model=getattr(response, "model", None) or self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                response_id=response_id,
                request_ids=[response_id] if response_id else [],
            ),
        )


def _validate_ollama_host(host: str, allow_remote_host: bool) -> str:
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("Ollama host must be an http(s) URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError("Ollama host contains an invalid port") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError("Ollama host must not contain credentials, a query, or a fragment")
    if parsed.path not in {"", "/"}:
        raise ConfigurationError("Ollama host must be an origin URL without a path")
    hostname_for_url = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port is not None else ""
    safe_origin = f"{parsed.scheme}://{hostname_for_url}{port}"
    if allow_remote_host:
        return safe_origin
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if not is_loopback:
        raise ConfigurationError(
            "Remote Ollama hosts are disabled by default; use a loopback address or "
            "explicitly enable remote hosts in trusted CLI deployments"
        )
    return safe_origin


def list_ollama_models(
    host: str = "http://localhost:11434", *, allow_remote_host: bool = False
) -> list[dict[str, str]]:
    """Return pulled local models, or an empty list when Ollama is unavailable."""

    if ollama is None:
        return []
    try:
        safe_host = _validate_ollama_host(host, allow_remote_host)
        response = ollama.Client(host=safe_host, timeout=10).list()
    except Exception:  # Availability probes intentionally do not break the UI.
        return []
    return [
        {
            "name": item.model,
            "parameter_size": getattr(item.details, "parameter_size", "") or "",
            "family": getattr(item.details, "family", "") or "",
        }
        for item in response.models
    ]


class OllamaAssessmentBackend(_StructuredAssessmentBackend):
    provider = "ollama"
    supports_argument_cap = True

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        max_retries: int = 1,
        name: str | None = None,
        timeout_seconds: float = 180,
        max_output_tokens: int = 4096,
        num_gpu: int | None = 999,
        allow_remote_host: bool = False,
    ):
        if ollama is None:
            raise ConfigurationError("Install the Ollama adapter with: pip install -e '.[llm]'")
        super().__init__(max_retries=max_retries)
        self.model = model
        self.host = _validate_ollama_host(host, allow_remote_host)
        self.max_output_tokens = max_output_tokens
        if num_gpu is not None and num_gpu < 0:
            raise ConfigurationError("Ollama num_gpu must be zero or a positive integer")
        self.num_gpu = num_gpu
        self.name = name or f"ollama-{model}"
        self._client = ollama.Client(host=self.host, timeout=timeout_seconds)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.provider,
            "model": self.model,
            "name": self.name,
            "live": True,
            "host": self.host,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
            "num_gpu": self.num_gpu,
        }

    def _provider_call(
        self, system_prompt: str, user_prompt: str, output_model: type[StrictModel]
    ) -> _StructuredResult:
        schema = output_model.model_json_schema()
        options: dict[str, int] = {
            "temperature": 0,
            "num_predict": self.max_output_tokens,
        }
        if self.num_gpu is not None:
            # Ollama's automatic GPU-layer selection can incorrectly choose the
            # CPU even when a supported accelerator has enough free VRAM.  A
            # deliberately high request value is clamped to the model's actual
            # layer count and forces the scheduler to attempt full offload.
            options["num_gpu"] = self.num_gpu
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema,
                options=options,
            )
            parsed = output_model.model_validate_json(response.message.content or "")
        except Exception as exc:
            raise LLMAssessmentError(f"Ollama request failed: {exc}") from exc
        input_tokens = getattr(response, "prompt_eval_count", None)
        output_tokens = getattr(response, "eval_count", None)
        total = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return _StructuredResult(
            value=parsed,
            usage=LLMUsage(
                provider=self.provider,
                requested_model=self.model,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
            ),
        )


class PersonaRouterBackend:
    """Route turns by persona while keeping party assignment independent."""

    def __init__(
        self,
        sol_backend: AssessmentBackend,
        fable_backend: AssessmentBackend,
        summary_backend: AssessmentBackend | None = None,
    ):
        self.sol_backend = sol_backend
        self.fable_backend = fable_backend
        self.summary_backend = summary_backend
        summary_name = summary_backend.name if summary_backend else "dual-neutral-consensus"
        self.name = (
            f"persona-router[SOL={sol_backend.name};FABLE={fable_backend.name};"
            f"SUMMARY={summary_name}]"
        )
        self.is_live = bool(
            getattr(sol_backend, "is_live", sol_backend.name != "deterministic-v1")
            or getattr(fable_backend, "is_live", fable_backend.name != "deterministic-v1")
            or (
                summary_backend is not None
                and getattr(summary_backend, "is_live", summary_backend.name != "deterministic-v1")
            )
        )
        self.supports_argument_cap = bool(
            getattr(sol_backend, "supports_argument_cap", False)
            or getattr(fable_backend, "supports_argument_cap", False)
            or (
                summary_backend is not None
                and getattr(summary_backend, "supports_argument_cap", False)
            )
        )

    def metadata(self) -> dict[str, Any]:
        def describe(backend: AssessmentBackend) -> dict[str, Any]:
            method = getattr(backend, "metadata", None)
            return method() if callable(method) else {"name": backend.name}

        return {
            "type": "persona-router",
            "name": self.name,
            "live": self.is_live,
            "personas": {
                "SOL": describe(self.sol_backend),
                "FABLE": describe(self.fable_backend),
            },
            "summary": (
                describe(self.summary_backend)
                if self.summary_backend
                else "dual-neutral-consensus"
                if self.is_live
                else "neutral-python"
            ),
        }

    def sol(self, argument: Argument, prompt_version: str, paragraph_texts=None) -> SolAssessment:
        return self.sol_backend.sol(argument, prompt_version, paragraph_texts)

    def fable(self, argument: Argument, prompt_version: str, paragraph_texts=None) -> FableAssessment:
        return self.fable_backend.fable(argument, prompt_version, paragraph_texts)

    def debate_turn(
        self,
        argument: Argument,
        round_number: int,
        barrister: BarristerPersona,
        side: PartySide,
        opponent_turn: DebateTurn | None,
        own_previous_turn: DebateTurn | None,
        prompt_version: str,
        paragraph_texts=None,
    ) -> DebateTurn:
        backend = (
            self.sol_backend
            if BarristerPersona(barrister) == BarristerPersona.SOL
            else self.fable_backend
        )
        return backend.debate_turn(
            argument,
            round_number,
            barrister,
            side,
            opponent_turn,
            own_previous_turn,
            prompt_version,
            paragraph_texts,
        )

    def summarize_debate(
        self,
        argument: Argument,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        respondent_barrister: BarristerPersona,
        turns: list[DebateTurn],
        paragraph_texts=None,
    ) -> DebateSummary:
        if self.summary_backend is not None:
            return self.summary_backend.summarize_debate(
                argument,
                n_rounds,
                claimant_barrister,
                respondent_barrister,
                turns,
                paragraph_texts,
            )
        routed = [self.sol_backend, self.fable_backend]
        if not self.is_live:
            summary = deterministic_debate_summary(
                argument,
                n_rounds,
                claimant_barrister,
                respondent_barrister,
                turns,
                paragraph_texts,
            )
            return summary.model_copy(
                update={
                    "backend": f"{self.name}:neutral-python",
                    "prompt_version": turns[0].prompt_version if turns else "unknown",
                }
            )

        # Each configured persona independently performs the same neutral-clerk
        # synthesis. Agreement is preserved; disagreement becomes BOTH rather
        # than allowing either advocate/provider to decide the final direction.
        summaries = [
            backend.summarize_debate(
                argument,
                n_rounds,
                claimant_barrister,
                respondent_barrister,
                turns,
                paragraph_texts,
            )
            for backend in routed
        ]

        def ordered(values):
            return list(dict.fromkeys(item for value in values for item in value))

        band_score = {
            "VERY_HIGH": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "MINIMAL": 1,
            "NONE": 0,
            "UNKNOWN": 3,
        }
        score_band = {value: key for key, value in band_score.items() if key != "UNKNOWN"}

        def consensus_band(field: str) -> str:
            average = round(
                sum(band_score[getattr(summary, field)] for summary in summaries)
                / len(summaries)
            )
            return score_band[max(0, min(5, average))]

        recommendations = {summary.recommended_treatment for summary in summaries}
        recommendation = recommendations.pop() if len(recommendations) == 1 else "BOTH"
        evidence_links = []
        seen_links: set[str] = set()
        for summary in summaries:
            for link in summary.evidence_links:
                key = link.model_dump_json()
                if key not in seen_links:
                    evidence_links.append(link)
                    seen_links.add(key)
        usage_values = [summary.usage for summary in summaries if summary.usage is not None]
        return DebateSummary(
            argument_id=argument.argument_id,
            n_rounds=n_rounds,
            claimant_barrister=claimant_barrister,
            respondent_barrister=respondent_barrister,
            turns=turns,
            methodological_agreements=ordered(
                [summary.methodological_agreements for summary in summaries]
            ),
            remaining_disputes=ordered([summary.remaining_disputes for summary in summaries]),
            hidden_assumptions=ordered([summary.hidden_assumptions for summary in summaries]),
            evidence_gaps=ordered([summary.evidence_gaps for summary in summaries]),
            human_review_reasons=ordered(
                [summary.human_review_reasons for summary in summaries]
            ),
            stop_reason=f"CONFIGURED_ROUND_LIMIT_REACHED:{n_rounds}",
            claimant_strength=consensus_band("claimant_strength"),
            respondent_strength=consensus_band("respondent_strength"),
            recommended_treatment=recommendation,
            treatment_rationale=" | ".join(
                f"{summary.backend}: {summary.treatment_rationale}" for summary in summaries
            ),
            paragraph_citations=ordered(
                [summary.paragraph_citations for summary in summaries]
            ),
            evidence_links=evidence_links,
            backend=f"{self.name}:dual-neutral-consensus",
            prompt_version=turns[0].prompt_version if turns else "unknown",
            usage=(
                _combine_usage(
                    usage_values,
                    provider="multi-provider",
                    model=" + ".join(value.model for value in usage_values),
                )
                if usage_values
                else None
            ),
        )


def build_backend(settings: Settings) -> AssessmentBackend:
    """Construct the configured backend using environment credentials for CLI use."""

    backend = settings.analysis_backend
    common = {
        "max_retries": settings.llm_max_retries,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_output_tokens": settings.llm_max_output_tokens,
    }
    if backend == "deterministic-v1":
        return DeterministicAssessmentBackend()
    if backend == "ollama" or backend.startswith("ollama"):
        return OllamaAssessmentBackend(
            model=settings.ollama_model,
            host=settings.ollama_host,
            num_gpu=settings.ollama_num_gpu,
            allow_remote_host=settings.ollama_allow_remote_host,
            **common,
        )
    if backend == "openai":
        return OpenAIAssessmentBackend(
            api_key=os.getenv("OPENAI_API_KEY", ""), model=settings.openai_model, **common
        )
    if backend == "anthropic":
        return AnthropicAssessmentBackend(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""), model=settings.anthropic_model, **common
        )
    if backend == "dual-api":
        return PersonaRouterBackend(
            sol_backend=OpenAIAssessmentBackend(
                api_key=os.getenv("OPENAI_API_KEY", ""), model=settings.openai_model, **common
            ),
            fable_backend=AnthropicAssessmentBackend(
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                model=settings.anthropic_model,
                **common,
            ),
        )
    raise ConfigurationError(f"Unknown analysis_backend: {backend}")
