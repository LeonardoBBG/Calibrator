"""Replaceable structured assessment interface.

The MVP ships with a deterministic backend. A future LLM adapter must return the
same Pydantic models, preserve citations and record its model name.
"""

from __future__ import annotations

from typing import Protocol

from .agents import assess_fable, assess_sol
from .debate import deterministic_debate_summary, deterministic_debate_turn
from .models import (
    Argument,
    BarristerPersona,
    DebateSummary,
    DebateTurn,
    FableAssessment,
    PartySide,
    SolAssessment,
)


class AssessmentBackend(Protocol):
    name: str

    def sol(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> SolAssessment: ...

    def fable(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> FableAssessment: ...

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
    ) -> DebateTurn: ...

    def summarize_debate(
        self,
        argument: Argument,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        respondent_barrister: BarristerPersona,
        turns: list[DebateTurn],
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateSummary: ...


class DeterministicAssessmentBackend:
    name = "deterministic-v1"
    is_live = False
    supports_argument_cap = False

    def metadata(self) -> dict[str, object]:
        return {
            "type": "deterministic",
            "name": self.name,
            "implementation_version": "2",
            "live": False,
        }

    def sol(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> SolAssessment:
        return assess_sol(argument, self.name, prompt_version, paragraph_texts)

    def fable(
        self,
        argument: Argument,
        prompt_version: str,
        paragraph_texts: dict[str, str] | None = None,
    ) -> FableAssessment:
        return assess_fable(argument, self.name, prompt_version, paragraph_texts)

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
        self,
        argument: Argument,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        respondent_barrister: BarristerPersona,
        turns: list[DebateTurn],
        paragraph_texts: dict[str, str] | None = None,
    ) -> DebateSummary:
        return deterministic_debate_summary(
            argument,
            n_rounds,
            claimant_barrister,
            respondent_barrister,
            turns,
            paragraph_texts,
        )


def assess_arguments(
    arguments: list[Argument],
    backend: AssessmentBackend,
    prompt_version: str,
    paragraph_texts: dict[str, str] | None = None,
) -> list[Argument]:
    return [
        argument.model_copy(
            update={
                "sol_assessment": backend.sol(argument, prompt_version, paragraph_texts),
                "fable_assessment": backend.fable(argument, prompt_version, paragraph_texts),
            }
        )
        for argument in arguments
    ]
