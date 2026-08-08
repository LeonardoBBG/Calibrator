"""Validated domain models used by storage, pipeline stages and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class DocumentType(str, Enum):
    ET1 = "ET1"
    ET3 = "ET3"
    WS = "WS"


class ArgumentCategory(str, Enum):
    SUBSTANTIVE = "SUBSTANTIVE"
    PROCEDURAL = "PROCEDURAL"
    MIXED = "MIXED"


class ArgumentTreatment(str, Enum):
    PROTECT = "PROTECT"
    DEFEND = "DEFEND"
    BOTH = "BOTH"
    SECONDARY = "SECONDARY"
    DISTRACTING = "DISTRACTING"
    UNSUPPORTED = "UNSUPPORTED"


class BarristerPersona(str, Enum):
    SOL = "SOL"
    FABLE = "FABLE"


class PartySide(str, Enum):
    CLAIMANT = "CLAIMANT"
    RESPONDENT = "RESPONDENT"


class Importance(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SourceOrigin(str, Enum):
    ET1 = "ET1"
    ET3 = "ET3"
    DOCUMENT_PLACEHOLDER = "DOCUMENT_PLACEHOLDER"
    SELF_ACCOUNT = "SELF_ACCOUNT"
    OTHER_WITNESS = "OTHER_WITNESS"
    AGREED_FACT = "AGREED_FACT"
    INFERENCE = "INFERENCE"
    LEGAL_SUBMISSION = "LEGAL_SUBMISSION"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class PleadingStatus(str, Enum):
    PLEADED_ET1_EXPLICIT = "PLEADED_ET1_EXPLICIT"
    PLEADED_ET1_IMPLICIT = "PLEADED_ET1_IMPLICIT"
    ET1_DETAIL_OR_ELABORATION = "ET1_DETAIL_OR_ELABORATION"
    RESPONDS_TO_ET3 = "RESPONDS_TO_ET3"
    ET3_ONLY_ALLEGATION = "ET3_ONLY_ALLEGATION"
    COMMON_GROUND = "COMMON_GROUND"
    POTENTIALLY_NEW_CASE = "POTENTIALLY_NEW_CASE"
    NOT_MATERIAL_TO_PLEADINGS = "NOT_MATERIAL_TO_PLEADINGS"
    UNRESOLVED = "UNRESOLVED"


class RespondentPosition(str, Enum):
    ADMITTED = "ADMITTED"
    DENIED = "DENIED"
    NOT_ADMITTED = "NOT_ADMITTED"
    PARTLY_ADMITTED = "PARTLY_ADMITTED"
    ALTERNATIVE_ACCOUNT = "ALTERNATIVE_ACCOUNT"
    NOT_ADDRESSED = "NOT_ADDRESSED"
    RESPONDENT_ALLEGATION = "RESPONDENT_ALLEGATION"
    UNCLEAR = "UNCLEAR"


class EvidentialCharacter(str, Enum):
    PERSONAL_RECOLLECTION = "PERSONAL_RECOLLECTION"
    CONTEMPORANEOUS_DOCUMENT_PLACEHOLDER = "CONTEMPORANEOUS_DOCUMENT_PLACEHOLDER"
    LATER_DOCUMENT_PLACEHOLDER = "LATER_DOCUMENT_PLACEHOLDER"
    ADMISSION = "ADMISSION"
    INFERENCE_FROM_FACTS = "INFERENCE_FROM_FACTS"
    HEARSAY_ACCOUNT = "HEARSAY_ACCOUNT"
    LEGAL_CHARACTERISATION = "LEGAL_CHARACTERISATION"
    UNSUPPORTED_ASSERTION = "UNSUPPORTED_ASSERTION"
    EVIDENCE_NOT_YET_INGESTED = "EVIDENCE_NOT_YET_INGESTED"


class ReferenceType(str, Enum):
    PLEADING_REFERENCE = "PLEADING_REFERENCE"
    EVIDENCE_REFERENCE = "EVIDENCE_REFERENCE"
    INTERNAL_WS_REFERENCE = "INTERNAL_WS_REFERENCE"
    EDITORIAL_NOTE = "EDITORIAL_NOTE"
    PLACEHOLDER = "PLACEHOLDER"
    UNKNOWN = "UNKNOWN"


class MatchType(str, Enum):
    EXACT_REFERENCE = "EXACT_REFERENCE"
    DIRECT_SEMANTIC = "DIRECT_SEMANTIC"
    CONTEXTUAL = "CONTEXTUAL"
    ISSUE_LEVEL = "ISSUE_LEVEL"
    NO_MATCH = "NO_MATCH"
    UNRESOLVED = "UNRESOLVED"


class Document(StrictModel):
    document_id: str
    document_type: DocumentType
    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ingested_at: str
    parser_version: str


class Paragraph(StrictModel):
    paragraph_id: str
    document_id: str
    document_type: DocumentType
    original_number: str | None = None
    text: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    section_heading: str | None = None
    sequence: int = Field(ge=1)


class ReferencePlaceholder(StrictModel):
    reference_id: str
    ws_paragraph_id: str
    raw_text: str
    reference_type: ReferenceType
    resolved: bool = False
    resolved_document_id: str | None = None


class ParagraphMatch(StrictModel):
    paragraph_id: str
    match_type: MatchType
    confidence: float = Field(ge=0, le=1)
    respondent_position: RespondentPosition | None = None


class AtomicProposition(StrictModel):
    proposition_id: str
    ws_paragraph_id: str
    text: str
    source_origins: list[SourceOrigin]
    pleading_status: PleadingStatus = PleadingStatus.UNRESOLVED
    et1_matches: list[ParagraphMatch] = Field(default_factory=list)
    et3_matches: list[ParagraphMatch] = Field(default_factory=list)
    evidential_character: list[EvidentialCharacter]
    materiality: Importance = Importance.MEDIUM
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class Issue(StrictModel):
    issue_id: str
    title: str
    claimant_position: str
    respondent_position: str
    et1_paragraphs: list[str]
    et3_paragraphs: list[str]
    category: ArgumentCategory
    status: str


class EvidenceLink(StrictModel):
    """A substantive claim tied to the exact paragraphs said to support it."""

    claim: str = Field(min_length=1)
    paragraph_ids: list[str] = Field(min_length=1)
    relationship: Literal["SUPPORTS", "CONTRADICTS", "CONTEXT"] = "SUPPORTS"


class LLMUsage(StrictModel):
    """Provider provenance without persisting credentials or unstable price data."""

    provider: str
    requested_model: str | None = None
    model: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    response_id: str | None = None
    request_ids: list[str] = Field(default_factory=list)
    observed_attempts: int = Field(default=1, ge=1)


class SolAssessment(StrictModel):
    agent: str = "SOL"
    argument_id: str
    best_formulation: str
    pleading_basis: str
    claimant_relevance: str
    required_findings: list[str]
    hidden_assumptions: list[str]
    strength: str
    new_case_risk: str
    protect_points: list[str]
    risks: list[str]
    paragraph_citations: list[str]
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    backend: str
    prompt_version: str
    usage: LLMUsage | None = None


class FableAssessment(StrictModel):
    agent: str = "FABLE"
    argument_id: str
    respondent_case: str
    pleading_attack: str
    evidence_attack: str
    alternative_explanation: str
    damaging_concessions: list[str]
    cross_examination_risk: str
    defend_points: list[str]
    risk_level: str
    hidden_assumptions: list[str]
    paragraph_citations: list[str]
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    backend: str
    prompt_version: str
    usage: LLMUsage | None = None


class DebateTurn(StrictModel):
    turn_id: str
    argument_id: str
    round_number: int = Field(ge=1)
    barrister: BarristerPersona
    side: PartySide
    opponent_turn_id: str | None = None
    position: str
    supporting_points: list[str]
    responses_to_opponent: list[str]
    challenges_for_opponent: list[str]
    concessions_or_risks: list[str]
    hidden_assumptions: list[str]
    evidence_gaps: list[str]
    protect_or_defend_points: list[str]
    paragraph_citations: list[str]
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    backend: str
    prompt_version: str
    usage: LLMUsage | None = None


class DebateSummary(StrictModel):
    argument_id: str
    n_rounds: int = Field(ge=1)
    claimant_barrister: BarristerPersona
    respondent_barrister: BarristerPersona
    turns: list[DebateTurn]
    methodological_agreements: list[str]
    remaining_disputes: list[str]
    hidden_assumptions: list[str]
    evidence_gaps: list[str]
    human_review_reasons: list[str]
    stop_reason: str
    claimant_strength: Literal["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "MINIMAL", "NONE", "UNKNOWN"] = "UNKNOWN"
    respondent_strength: Literal["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "MINIMAL", "NONE", "UNKNOWN"] = "UNKNOWN"
    recommended_treatment: Literal["PROTECT", "DEFEND", "BOTH"] | None = None
    treatment_rationale: str = ""
    paragraph_citations: list[str] = Field(default_factory=list)
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    backend: str = "deterministic-v1"
    prompt_version: str = "unknown"
    usage: LLMUsage | None = None


class RankingComponents(StrictModel):
    outcome_materiality: float = Field(ge=0, le=1)
    pleading_alignment: float = Field(ge=0, le=1)
    ws_support: float = Field(ge=0, le=1)
    respondent_attack_strength: float = Field(ge=0, le=1)
    cross_examination_vulnerability: float = Field(ge=0, le=1)
    case_law_calibration_value: float = Field(ge=0, le=1)


class Argument(StrictModel):
    argument_id: str
    title: str
    proposition: str
    category: ArgumentCategory
    treatment: list[ArgumentTreatment] = Field(default_factory=list)
    importance: Importance
    issue_ids: list[str]
    ws_paragraphs: list[str]
    et1_paragraphs: list[str]
    et3_paragraphs: list[str]
    source_types: list[SourceOrigin]
    document_placeholders: list[str]
    pleading_statuses: list[PleadingStatus]
    proposition_ids: list[str]
    sol_assessment: SolAssessment | None = None
    fable_assessment: FableAssessment | None = None
    debate_summary: DebateSummary | None = None
    search_package: dict[str, Any] | None = None
    ranking_score: float | None = Field(default=None, ge=0, le=100)
    ranking_components: RankingComponents | None = None
    human_review_status: str = "PENDING"
    assessment_fingerprint: str | None = None
    assessment_mode: Literal["LIVE", "MIXED", "DETERMINISTIC", "DETERMINISTIC_FALLBACK"] | None = None


class CaseLawSearchPackage(StrictModel):
    argument_id: str
    core_proposition: str
    legal_dimensions: list[str]
    factual_analogues: list[str]
    search_terms: list[str]
    broad_queries: list[str]
    targeted_queries: list[str]
    authority_queries: list[str]
    questions_for_cases: list[str]
    do_not_assume: list[str]


class HumanReviewItem(StrictModel):
    review_id: str
    run_id: str
    gate: str
    entity_type: str
    entity_id: str
    reason: str
    confidence: float | None = None
    status: str = "PENDING"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunRecord(StrictModel):
    run_id: str
    started_at: str
    completed_at: str | None = None
    status: str
    backend: str
    prompt_version: str
    config_snapshot: dict[str, Any]
