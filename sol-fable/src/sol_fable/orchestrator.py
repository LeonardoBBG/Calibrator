"""Fixed-order orchestration, validation, persistence and artifact export."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .analysis_backend import AssessmentBackend, DeterministicAssessmentBackend
from .argument_builder import build_arguments
from .config import Settings, default_data_root, load_settings
from .debate import conduct_debate, other_barrister
from .errors import LLMAssessmentError, PipelineRunError, PipelineStateError
from .ingest import ingest_document
from .io_utils import write_csv, write_json, write_jsonl, write_text
from .issue_builder import build_issues, pleadings_map_markdown
from .llm_backend import build_backend, prompt_bundle_sha256
from .models import (
    ArgumentTreatment,
    BarristerPersona,
    DocumentType,
    HumanReviewItem,
    Importance,
    RunRecord,
)
from .pleading_matcher import match_propositions
from .proposition_extractor import extract_propositions
from .ranker import rank_arguments, top_arguments_markdown
from .reference_parser import parse_references
from .report_builder import (
    barrister_debate_markdown,
    build_json_report,
    build_markdown_report,
    review_csv_rows,
)
from .search_package_builder import build_search_packages, research_plan_markdown
from .storage import CaseStore

LOGGER = logging.getLogger("sol_fable")


class PipelineOrchestrator:
    def __init__(
        self,
        project_dir: Path | None = None,
        config_path: Path | None = None,
        backend: AssessmentBackend | None = None,
    ):
        self.project_dir = (project_dir or default_data_root()).expanduser().resolve()
        self.settings: Settings = load_settings(config_path)
        self.case_dir = self.project_dir / self.settings.paths.case_dir
        self.case_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.case_dir.chmod(0o700)
        except OSError:
            pass
        self._harden_case_permissions()
        self.extracted_dir = self.case_dir / "extracted"
        self.reports_dir = self.case_dir / "reports"
        self.store = CaseStore(self.case_dir / "database" / "case.sqlite")
        self._backend = backend

    def _harden_case_permissions(self) -> None:
        """Best-effort privacy for existing local case artifacts; never follow links."""

        for path in [self.case_dir, *self.case_dir.rglob("*")]:
            try:
                if path.is_symlink():
                    continue
                path.chmod(0o700 if path.is_dir() else 0o600)
            except OSError:
                continue

    @property
    def backend(self) -> AssessmentBackend:
        """Construct an optional live provider only when assessment actually needs it."""

        if self._backend is None:
            self._backend = build_backend(self.settings)
        return self._backend

    @backend.setter
    def backend(self, value: AssessmentBackend) -> None:
        self._backend = value

    @staticmethod
    def _metadata(backend: AssessmentBackend) -> dict[str, object]:
        method = getattr(backend, "metadata", None)
        return method() if callable(method) else {"name": backend.name}

    @classmethod
    def _backend_mode(cls, backend: AssessmentBackend) -> str:
        metadata = cls._metadata(backend)
        personas = metadata.get("personas")
        if isinstance(personas, dict):
            live_values = [
                bool(value.get("live", False))
                for value in personas.values()
                if isinstance(value, dict)
            ]
            summary = metadata.get("summary")
            if isinstance(summary, dict):
                live_values.append(bool(summary.get("live", False)))
            elif bool(metadata.get("live", False)) and live_values and not any(live_values):
                live_values.append(True)
            if live_values and any(live_values) and not all(live_values):
                return "MIXED"
            if live_values and all(live_values):
                return "LIVE"
            return "DETERMINISTIC"
        live = metadata.get("live", getattr(backend, "is_live", False))
        return "LIVE" if live else "DETERMINISTIC"

    def extracted_for(self, run_id: str) -> Path:
        return self.extracted_dir / run_id

    def reports_for(self, run_id: str) -> Path:
        return self.reports_dir / run_id

    @staticmethod
    def _barrister(value: BarristerPersona | str) -> BarristerPersona:
        if isinstance(value, BarristerPersona):
            return value
        return BarristerPersona(str(value).upper())

    def start_run(
        self,
        n_rounds: int | None = None,
        claimant_barrister: BarristerPersona | str | None = None,
    ) -> str:
        effective_rounds = n_rounds if n_rounds is not None else self.settings.n_rounds
        if not 1 <= effective_rounds <= 10:
            raise ValueError("n_rounds must be between 1 and 10")
        effective_claimant = self._barrister(
            claimant_barrister or self.settings.claimant_barrister
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"RUN-{timestamp}-{uuid.uuid4().hex[:8].upper()}"
        snapshot = self.settings.model_dump(mode="json")
        if self._backend is not None:
            backend_name = self._backend.name
            backend_metadata = self._metadata(self._backend)
        else:
            backend_name = self.settings.analysis_backend
            backend_metadata = {
                "type": self.settings.analysis_backend,
                "name": self.settings.analysis_backend,
                "live": self.settings.analysis_backend != "deterministic-v1",
                "deferred": True,
            }
        snapshot["analysis_backend"] = backend_name
        snapshot["backend_configuration"] = backend_metadata
        snapshot["n_rounds"] = effective_rounds
        snapshot["claimant_barrister"] = effective_claimant.value
        snapshot["respondent_barrister"] = other_barrister(effective_claimant).value
        self.store.create_run(
            RunRecord(
                run_id=run_id,
                started_at=datetime.now(timezone.utc).isoformat(),
                status="RUNNING",
                backend=backend_name,
                prompt_version=self.settings.prompt_version,
                config_snapshot=snapshot,
            )
        )
        return run_id

    def _require_run(self, run_id: str | None) -> str:
        selected = run_id or self.store.latest_run_id()
        if not selected:
            raise PipelineStateError("No pipeline run exists. Start with ingest or run-all.")
        self.store.load_run(selected)
        return selected

    def _record(
        self,
        run_id: str,
        stage: int,
        name: str,
        count: int,
        details: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {"records": count}
        payload.update(details or {})
        self.store.record_stage(run_id, stage, name, "COMPLETED", payload)
        LOGGER.info("stage_completed", extra={"run_id": run_id, "stage": stage, "records": count})

    def _invalidate_argument_outputs(
        self, run_id: str, *, review_gates: tuple[str, ...]
    ) -> None:
        """Cascade an upstream mutation so stale analysis can never be reported."""

        self.store.save_arguments(run_id, [])
        self.store.save_search_packages(run_id, [])
        for gate in review_gates:
            self.store.delete_reviews_for_gate(run_id, gate)
        run = self.store.load_run(run_id)
        snapshot = dict(run.config_snapshot)
        for key in (
            "assessment_backend_counts",
            "turn_backend_counts",
            "assessment_mode_counts",
            "token_usage",
            "planned_live_arguments",
            "planned_provider_calls",
            "planned_call_scope",
            "assessment_failures",
        ):
            snapshot.pop(key, None)
        self.store.update_run_config(run_id, snapshot)
        self.store.update_run(run_id, "RUNNING")
        extracted = self.extracted_for(run_id)
        for filename in (
            "arguments_draft.jsonl",
            "sol_assessments.jsonl",
            "fable_assessments.jsonl",
            "debate_summaries.jsonl",
            "debate_rounds.jsonl",
            "ranked_arguments.jsonl",
            "case_law_search_packages.jsonl",
        ):
            write_jsonl(extracted / filename, [])
        reports = self.reports_for(run_id)
        stale_message = (
            "# Stale output\n\nAn upstream stage was rerun. Complete assessment, ranking, "
            "search-package generation and reporting before using this artifact."
        )
        for filename in (
            "ws_calibration_report.md",
            "barrister_debate.md",
            "top_arguments.md",
            "case_law_research_plan.md",
        ):
            path = reports / filename
            if path.exists():
                write_text(path, stale_message)
        json_report = reports / "ws_calibration_report.json"
        if json_report.exists():
            write_json(
                json_report,
                {"status": "STALE", "reason": "UPSTREAM_STAGE_RERUN", "run_id": run_id},
            )
        review_csv = reports / "human_review_queue.csv"
        if review_csv.exists():
            fields = [
                "review_id",
                "run_id",
                "gate",
                "entity_type",
                "entity_id",
                "reason",
                "confidence",
                "status",
                "created_at",
            ]
            write_csv(
                review_csv,
                review_csv_rows(self.store.load_reviews(run_id)),
                fields,
            )

    def _mark_final_report_stale(self, run_id: str, reason: str) -> None:
        reports = self.reports_for(run_id)
        message = (
            "# Stale output\n\nA downstream stage changed. Re-run the remaining stages "
            "before using this report."
        )
        for filename in ("ws_calibration_report.md", "barrister_debate.md"):
            path = reports / filename
            if path.exists():
                write_text(path, message)
        json_report = reports / "ws_calibration_report.json"
        if json_report.exists():
            write_json(
                json_report,
                {"status": "STALE", "reason": reason, "run_id": run_id},
            )
        review_csv = reports / "human_review_queue.csv"
        if review_csv.exists():
            write_csv(
                review_csv,
                review_csv_rows(self.store.load_reviews(run_id)),
                [
                    "review_id",
                    "run_id",
                    "gate",
                    "entity_type",
                    "entity_id",
                    "reason",
                    "confidence",
                    "status",
                    "created_at",
                ],
            )

    def ingest(self, et1: Path, et3: Path, ws: Path, run_id: str | None = None) -> str:
        selected = self._require_run(run_id) if run_id else self.start_run()
        if self.store.load_documents(selected):
            raise PipelineStateError(
                "Run inputs are immutable after ingestion; start a new run for different documents."
            )
        all_paragraphs = []
        documents = []
        for document_type, path in (
            (DocumentType.ET1, et1),
            (DocumentType.ET3, et3),
            (DocumentType.WS, ws),
        ):
            document, paragraphs = ingest_document(path.resolve(), document_type, self.settings.parser_version)
            stored = self.store.save_document(selected, document, paragraphs)
            documents.append(stored)
            all_paragraphs.extend(paragraphs)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)
            preserved = self.case_dir / "source" / document_type.value.lower() / f"{document.sha256[:12]}_{safe_name}"
            preserved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                preserved.parent.chmod(0o700)
            except OSError:
                pass
            if not preserved.exists():
                shutil.copyfile(path, preserved)
                try:
                    preserved.chmod(0o600)
                except OSError:
                    pass
        all_paragraphs = self.store.load_paragraphs(selected)
        output_dir = self.extracted_for(selected)
        write_json(output_dir / "documents.json", documents)
        write_jsonl(output_dir / "paragraphs.jsonl", all_paragraphs)
        self._record(selected, 1, "ingest_documents", len(all_paragraphs))
        return selected

    def parse_references(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        references = parse_references(self.store.load_paragraphs(selected, "WS"))
        self.store.save_references(selected, references)
        self._invalidate_argument_outputs(selected, review_gates=("C", "D", "E"))
        write_jsonl(self.extracted_for(selected) / "reference_placeholders.jsonl", references)
        self._record(selected, 2, "parse_references", len(references))
        return selected

    def build_issues(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        et1 = self.store.load_paragraphs(selected, "ET1")
        et3 = self.store.load_paragraphs(selected, "ET3")
        if not et1 or not et3:
            raise PipelineStateError("Issue building requires ingested ET1 and ET3 paragraphs.")
        issues = build_issues(et1, et3)
        self.store.save_issues(selected, issues)
        self._invalidate_argument_outputs(selected, review_gates=("C", "D", "E"))
        gate_reviews = [
            HumanReviewItem(
                review_id=f"REV-A-{index:04d}",
                run_id=selected,
                gate="A",
                entity_type="ISSUE",
                entity_id=issue.issue_id,
                reason="Confirm that this neutral issue map does not overstate either party's pleaded position.",
            )
            for index, issue in enumerate(issues, start=1)
        ]
        self.store.delete_reviews_for_gate(selected, "A")
        self.store.save_reviews(selected, gate_reviews, append=True)
        write_jsonl(self.extracted_for(selected) / "issues.jsonl", issues)
        write_text(self.reports_for(selected) / "pleadings_map.md", pleadings_map_markdown(issues))
        self._record(selected, 3, "build_pleadings_issue_registry", len(issues))
        return selected

    def extract_propositions(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        ws = self.store.load_paragraphs(selected, "WS")
        if not ws:
            raise PipelineStateError("Proposition extraction requires an ingested WS.")
        propositions = extract_propositions(ws)
        self.store.save_propositions(selected, propositions)
        self._invalidate_argument_outputs(selected, review_gates=("B", "C", "D", "E"))
        write_jsonl(self.extracted_for(selected) / "ws_propositions.jsonl", propositions)
        self._record(selected, 4, "extract_atomic_ws_propositions", len(propositions))
        return selected

    def match_pleadings(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        propositions = self.store.load_propositions(selected)
        if not propositions:
            raise PipelineStateError("Pleading matching requires extracted propositions.")
        matched, reviews = match_propositions(
            selected,
            propositions,
            self.store.load_paragraphs(selected, "ET1"),
            self.store.load_paragraphs(selected, "ET3"),
            self.settings.thresholds,
        )
        self.store.save_propositions(selected, matched)
        self._invalidate_argument_outputs(selected, review_gates=("C", "D", "E"))
        self.store.delete_reviews_for_gate(selected, "B")
        self.store.save_reviews(selected, reviews, append=True)
        write_jsonl(self.extracted_for(selected) / "pleading_matches.jsonl", matched)
        write_jsonl(self.extracted_for(selected) / "human_review_queue.jsonl", self.store.load_reviews(selected))
        self._record(selected, 5, "match_propositions_to_pleadings", len(matched))
        return selected

    def build_arguments(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        propositions = self.store.load_propositions(selected)
        issues = self.store.load_issues(selected)
        if not propositions or not issues:
            raise PipelineStateError("Argument building requires matched propositions and an issue registry.")
        arguments = build_arguments(propositions, issues, self.store.load_references(selected))
        self._invalidate_argument_outputs(selected, review_gates=("C", "D", "E"))
        self.store.save_arguments(selected, arguments)
        write_jsonl(self.extracted_for(selected) / "arguments_draft.jsonl", arguments)
        self._record(selected, 6, "extract_and_deduplicate_arguments", len(arguments))
        return selected

    _IMPORTANCE_RANK = {
        Importance.CRITICAL: 0,
        Importance.HIGH: 1,
        Importance.MEDIUM: 2,
        Importance.LOW: 3,
    }

    @staticmethod
    def _assessment_fingerprint(
        argument,
        backend: AssessmentBackend,
        prompt_version: str,
        n_rounds: int,
        claimant_barrister: BarristerPersona,
        paragraph_texts: dict[str, str],
        prompt_hash: str | None,
    ) -> str:
        argument_payload = argument.model_dump(
            mode="json",
            exclude={
                "sol_assessment",
                "fable_assessment",
                "debate_summary",
                "search_package",
                "ranking_score",
                "ranking_components",
                "treatment",
                "assessment_fingerprint",
                "assessment_mode",
            },
        )
        source_ids = sorted(
            set(argument.et1_paragraphs + argument.et3_paragraphs + argument.ws_paragraphs)
        )
        backend_metadata = PipelineOrchestrator._metadata(backend)
        backend_mode = PipelineOrchestrator._backend_mode(backend)
        payload = {
            "argument": argument_payload,
            "paragraphs": {item: paragraph_texts.get(item, "") for item in source_ids},
            "backend": backend_metadata,
            "prompt_version": prompt_version,
            "prompt_bundle_sha256": (
                prompt_hash
                if backend_mode in {"LIVE", "MIXED"}
                else None
            ),
            "n_rounds": n_rounds,
            "claimant_barrister": claimant_barrister.value,
            "respondent_barrister": other_barrister(claimant_barrister).value,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _select_capped_arguments(self, arguments: list, cap: int) -> list:
        """Select important arguments while preserving issue diversity."""

        ordered = sorted(
            arguments,
            key=lambda item: (
                self._IMPORTANCE_RANK.get(Importance(item.importance), 4),
                0 if item.human_review_status == "PENDING" else 1,
                0 if item.document_placeholders else 1,
                item.argument_id,
            ),
        )
        selected = []
        seen_issues: set[str] = set()
        for argument in ordered:
            issue_key = argument.issue_ids[0] if argument.issue_ids else argument.argument_id
            if issue_key in seen_issues:
                continue
            selected.append(argument)
            seen_issues.add(issue_key)
            if len(selected) == cap:
                return selected
        selected_ids = {item.argument_id for item in selected}
        selected.extend(item for item in ordered if item.argument_id not in selected_ids)
        return selected[:cap]

    def assess(
        self,
        run_id: str | None = None,
        n_rounds: int | None = None,
        claimant_barrister: BarristerPersona | str | None = None,
        ollama_argument_cap: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        live_argument_cap: int | None = None,
    ) -> str:
        selected = self._require_run(run_id)
        run = self.store.load_run(selected)
        active_backend = self.backend
        effective_rounds = (
            n_rounds
            if n_rounds is not None
            else int(run.config_snapshot.get("n_rounds", self.settings.n_rounds))
        )
        if not 1 <= effective_rounds <= 10:
            raise ValueError("n_rounds must be between 1 and 10")
        effective_claimant = self._barrister(
            claimant_barrister
            or run.config_snapshot.get("claimant_barrister")
            or self.settings.claimant_barrister
        )
        effective_respondent = other_barrister(effective_claimant)
        configured_cap = run.config_snapshot.get(
            "live_argument_cap", self.settings.live_argument_cap
        )
        if live_argument_cap is not None:
            effective_cap = live_argument_cap
        elif ollama_argument_cap is not None:
            effective_cap = ollama_argument_cap
        elif active_backend.name.startswith("ollama") and run.config_snapshot.get(
            "ollama_argument_cap"
        ) is not None:
            effective_cap = run.config_snapshot.get("ollama_argument_cap")
        else:
            effective_cap = configured_cap
        if effective_cap is not None:
            effective_cap = int(effective_cap)
            if effective_cap < 0:
                raise ValueError("live_argument_cap must be zero (uncapped) or a positive integer")
            if effective_cap == 0:
                effective_cap = None
        config_snapshot = dict(run.config_snapshot)
        backend_metadata = self._metadata(active_backend)
        backend_mode = self._backend_mode(active_backend)
        config_snapshot["analysis_backend"] = active_backend.name
        config_snapshot["backend_configuration"] = backend_metadata
        config_snapshot["prompt_version"] = self.settings.prompt_version
        config_snapshot["prompt_bundle_sha256"] = (
            prompt_bundle_sha256(self.settings.prompt_version)
            if backend_mode in {"LIVE", "MIXED"}
            else None
        )
        config_snapshot["n_rounds"] = effective_rounds
        config_snapshot["claimant_barrister"] = effective_claimant.value
        config_snapshot["respondent_barrister"] = effective_respondent.value
        config_snapshot["ollama_argument_cap"] = effective_cap
        config_snapshot["live_argument_cap"] = effective_cap
        arguments = self.store.load_arguments(selected)
        if not arguments:
            raise PipelineStateError("Assessment requires draft arguments.")
        paragraph_texts = {
            paragraph.paragraph_id: paragraph.text
            for paragraph in (
                self.store.load_paragraphs(selected, "ET1")
                + self.store.load_paragraphs(selected, "ET3")
                + self.store.load_paragraphs(selected, "WS")
            )
        }

        # Assign the cap against the complete set so interrupted/resumed calls do
        # not move the boundary and silently change which backend owns an argument.
        supports_cap = bool(
            getattr(active_backend, "supports_argument_cap", False)
            or active_backend.name.startswith("ollama")
        )
        fallback_backend = None
        if supports_cap and effective_cap:
            live_ids = {
                item.argument_id
                for item in self._select_capped_arguments(arguments, int(effective_cap))
            }
            fallback_backend = DeterministicAssessmentBackend()
        else:
            live_ids = {item.argument_id for item in arguments}

        by_id = {argument.argument_id: argument for argument in arguments}
        target_backend = {
            argument.argument_id: (
                active_backend if argument.argument_id in live_ids else fallback_backend
            )
            for argument in arguments
        }
        target_fingerprints = {
            argument.argument_id: self._assessment_fingerprint(
                argument,
                target_backend[argument.argument_id],
                self.settings.prompt_version,
                effective_rounds,
                effective_claimant,
                paragraph_texts,
                config_snapshot["prompt_bundle_sha256"],
            )
            for argument in arguments
        }
        pending = [
            argument
            for argument in arguments
            if argument.debate_summary is None
            or argument.assessment_fingerprint != target_fingerprints[argument.argument_id]
        ]
        if pending:
            # Any changed assessment makes all downstream comparative scores and
            # research packages stale, including those attached to otherwise
            # compatible arguments. Preserve assessments, but invalidate those
            # downstream products atomically before the first live model call.
            for argument_id, argument in list(by_id.items()):
                reset = argument.model_copy(
                    update={
                        "ranking_score": None,
                        "ranking_components": None,
                        "treatment": [],
                        "search_package": None,
                    }
                )
                by_id[argument_id] = reset
                self.store.upsert_argument(selected, reset)
            self.store.save_search_packages(selected, [])
            self.store.delete_reviews_for_gate(selected, "C")
            self.store.delete_reviews_for_gate(selected, "D")
            self.store.update_run(selected, "RUNNING")
            write_jsonl(self.extracted_for(selected) / "ranked_arguments.jsonl", [])
            write_jsonl(
                self.extracted_for(selected) / "case_law_search_packages.jsonl", []
            )
            self._mark_final_report_stale(selected, "ASSESSMENT_CHANGED")
        total = len(arguments)
        completed = total - len(pending)
        if progress_callback:
            progress_callback(completed, total)
        if not pending:
            return selected

        live_argument_count = (
            sum(item.argument_id in live_ids for item in pending)
            if backend_mode in {"LIVE", "MIXED"}
            else 0
        )
        summary_calls = 2 if backend_metadata.get("type") == "persona-router" else 1
        calls_per_live_argument = (2 * effective_rounds) + 2 + summary_calls
        config_snapshot["planned_live_arguments"] = live_argument_count
        config_snapshot["planned_provider_calls"] = live_argument_count * calls_per_live_argument
        config_snapshot["planned_call_scope"] = "CONSERVATIVE_UPPER_BOUND_BEFORE_RETRIES"
        self.store.update_run_provenance(
            selected,
            backend=active_backend.name,
            prompt_version=self.settings.prompt_version,
            config_snapshot=config_snapshot,
        )

        llm_batch = [item for item in pending if item.argument_id in live_ids]
        fallback_batch = [item for item in pending if item.argument_id not in live_ids]
        error_fallback_backend = fallback_backend or DeterministicAssessmentBackend()
        pending_ids = {item.argument_id for item in pending}
        assessment_failures = [
            failure
            for failure in config_snapshot.get("assessment_failures", [])
            if isinstance(failure, dict) and failure.get("argument_id") not in pending_ids
        ]

        def evaluate_argument(argument, assessment_backend):
            debate_summary = conduct_debate(
                argument,
                assessment_backend,
                self.settings.prompt_version,
                effective_rounds,
                effective_claimant,
                paragraph_texts,
            )
            sol_assessment = assessment_backend.sol(
                argument, self.settings.prompt_version, paragraph_texts
            )
            fable_assessment = assessment_backend.fable(
                argument, self.settings.prompt_version, paragraph_texts
            )
            return debate_summary, sol_assessment, fable_assessment

        newly_processed = []
        for backend, batch in ((active_backend, llm_batch), (fallback_backend, fallback_batch)):
            for argument in batch:
                assessment_backend = backend
                assessment_mode = (
                    "DETERMINISTIC_FALLBACK"
                    if backend is fallback_backend
                    else self._backend_mode(backend)
                )
                try:
                    debate_summary, sol_assessment, fable_assessment = evaluate_argument(
                        argument, assessment_backend
                    )
                except LLMAssessmentError as exc:
                    if backend is not active_backend or backend_mode not in {"LIVE", "MIXED"}:
                        raise
                    failure = {
                        "argument_id": argument.argument_id,
                        "backend": backend.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    assessment_failures.append(failure)
                    config_snapshot["assessment_failures"] = assessment_failures
                    # Persist this audit fact immediately in case a later argument
                    # encounters a non-recoverable pipeline failure.
                    self.store.update_run_config(selected, config_snapshot)
                    LOGGER.warning(
                        "live_assessment_fallback",
                        extra={
                            "run_id": selected,
                            "argument_id": argument.argument_id,
                            "backend": backend.name,
                            "error_type": type(exc).__name__,
                        },
                    )
                    assessment_backend = error_fallback_backend
                    assessment_mode = "DETERMINISTIC_FALLBACK"
                    debate_summary, sol_assessment, fable_assessment = evaluate_argument(
                        argument, assessment_backend
                    )
                    review_reason = (
                        f"Live backend {backend.name} could not produce a grounded assessment "
                        f"({type(exc).__name__}); deterministic fallback was used."
                    )
                    debate_summary = debate_summary.model_copy(
                        update={
                            "human_review_reasons": list(
                                dict.fromkeys(
                                    [review_reason, *debate_summary.human_review_reasons]
                                )
                            )
                        }
                    )
                updated = argument.model_copy(
                    update={
                        # Retained as independent baseline assessments for backwards
                        # compatibility; debate_summary is the active mediated workflow.
                        "sol_assessment": sol_assessment,
                        "fable_assessment": fable_assessment,
                        "debate_summary": debate_summary,
                        "assessment_fingerprint": target_fingerprints[argument.argument_id],
                        "assessment_mode": assessment_mode,
                        "ranking_score": None,
                        "ranking_components": None,
                        "treatment": [],
                        "search_package": None,
                    }
                )
                by_id[argument.argument_id] = updated
                newly_processed.append(updated)
                completed += 1
                # Checkpoint only this argument; this is O(N), not O(N²), over the run.
                self.store.upsert_argument(selected, updated)
                if progress_callback:
                    progress_callback(completed, total)

        assessed = list(by_id.values())
        output_dir = self.extracted_for(selected)
        write_jsonl(
            output_dir / "sol_assessments.jsonl",
            [item.sol_assessment for item in assessed if item.sol_assessment],
        )
        write_jsonl(
            output_dir / "fable_assessments.jsonl",
            [item.fable_assessment for item in assessed if item.fable_assessment],
        )
        debate_summaries = [item.debate_summary for item in assessed if item.debate_summary]
        # Review IDs are scoped to (argument_id, reason index) so a resumed call's
        # newly generated reviews never collide with (and silently overwrite) reviews
        # already stored from an earlier call on the same run.
        debate_reviews = [
            HumanReviewItem(
                review_id=f"REV-E-{argument.argument_id}-{index:02d}",
                run_id=selected,
                gate="E",
                entity_type="DEBATE",
                entity_id=argument.debate_summary.argument_id,
                reason=reason,
            )
            for argument in newly_processed
            if argument.debate_summary
            for index, reason in enumerate(argument.debate_summary.human_review_reasons, start=1)
        ]
        self.store.delete_reviews_for_entities(
            selected,
            "E",
            [item.argument_id for item in newly_processed],
        )
        self.store.save_reviews(selected, debate_reviews, append=True)
        self._mark_final_report_stale(selected, "ASSESSMENT_CHANGED")
        if newly_processed:
            self.store.save_search_packages(selected, [])
        write_jsonl(output_dir / "debate_summaries.jsonl", debate_summaries)
        write_jsonl(
            output_dir / "debate_rounds.jsonl",
            [turn for summary in debate_summaries for turn in summary.turns],
        )
        write_jsonl(output_dir / "human_review_queue.jsonl", self.store.load_reviews(selected))
        assessment_backend_counts = Counter(
            key
            for item in assessed
            for key in (
                f"SOL:{item.sol_assessment.backend}" if item.sol_assessment else None,
                f"FABLE:{item.fable_assessment.backend}" if item.fable_assessment else None,
            )
            if key is not None
        )
        turn_backend_counts = Counter(
            turn.backend
            for item in assessed
            if item.debate_summary is not None
            for turn in item.debate_summary.turns
        )
        mode_counts = Counter(item.assessment_mode or "UNKNOWN" for item in assessed)
        usages = [
            usage
            for item in assessed
            for usage in (
                item.sol_assessment.usage if item.sol_assessment else None,
                item.fable_assessment.usage if item.fable_assessment else None,
                item.debate_summary.usage if item.debate_summary else None,
                *(
                    [turn.usage for turn in item.debate_summary.turns]
                    if item.debate_summary
                    else []
                ),
            )
            if usage is not None
        ]
        token_usage = {
            "input_tokens": sum(item.input_tokens or 0 for item in usages),
            "output_tokens": sum(item.output_tokens or 0 for item in usages),
            "total_tokens": sum(item.total_tokens or 0 for item in usages),
            "observed_attempts": sum(item.observed_attempts for item in usages),
            "accounting_scope": "OBSERVED_RESPONSES_ONLY",
        }
        stage_details = {
            "n_rounds": effective_rounds,
            "claimant_barrister": effective_claimant.value,
            "respondent_barrister": effective_respondent.value,
            "live_argument_cap": effective_cap,
            "planned_live_arguments": live_argument_count,
            "planned_provider_calls": config_snapshot["planned_provider_calls"],
            "planned_call_scope": config_snapshot["planned_call_scope"],
            "newly_processed_arguments": len(newly_processed),
            "assessment_backend_counts": dict(assessment_backend_counts),
            "turn_backend_counts": dict(turn_backend_counts),
            "assessment_mode_counts": dict(mode_counts),
            "assessment_failures": assessment_failures,
            "token_usage": token_usage,
        }
        config_snapshot["assessment_backend_counts"] = dict(assessment_backend_counts)
        config_snapshot["turn_backend_counts"] = dict(turn_backend_counts)
        config_snapshot["assessment_mode_counts"] = dict(mode_counts)
        config_snapshot["assessment_failures"] = assessment_failures
        config_snapshot["token_usage"] = token_usage
        self.store.update_run_config(selected, config_snapshot)
        self._record(
            selected,
            7,
            "claimant_barrister_debate",
            len(assessed) * effective_rounds,
            stage_details,
        )
        self._record(
            selected,
            8,
            "respondent_barrister_debate",
            len(assessed) * effective_rounds,
            stage_details,
        )
        return selected

    def rank(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        arguments = self.store.load_arguments(selected)
        if not arguments or any(not item.debate_summary for item in arguments):
            raise PipelineStateError("Ranking requires a completed mediated debate for every argument.")
        ranked = [
            item.model_copy(update={"search_package": None})
            for item in rank_arguments(arguments, self.settings.ranking_weights)
        ]
        self.store.save_arguments(selected, ranked)
        self.store.save_search_packages(selected, [])
        self.store.delete_reviews_for_gate(selected, "C")
        self.store.delete_reviews_for_gate(selected, "D")
        self.store.update_run(selected, "RUNNING")
        reviews = [
            HumanReviewItem(
                review_id=f"REV-C-{index:04d}",
                run_id=selected,
                gate="C",
                entity_type="ARGUMENT",
                entity_id=argument.argument_id,
                reason="Confirm that PROTECT/DEFEND/BOTH priority reflects outcome relevance rather than repetition.",
                confidence=(argument.ranking_score or 0) / 100,
            )
            for index, argument in enumerate(ranked, start=1)
            if argument.importance in (Importance.CRITICAL, Importance.HIGH)
            or ArgumentTreatment.BOTH in argument.treatment
        ]
        self.store.save_reviews(selected, reviews, append=True)
        self._mark_final_report_stale(selected, "RANKING_CHANGED")
        write_jsonl(self.extracted_for(selected) / "ranked_arguments.jsonl", ranked)
        write_jsonl(
            self.extracted_for(selected) / "human_review_queue.jsonl",
            self.store.load_reviews(selected),
        )
        write_text(self.reports_for(selected) / "top_arguments.md", top_arguments_markdown(ranked))
        self._record(selected, 9, "rank_arguments", len(ranked))
        return selected

    def generate_search_packages(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        arguments = self.store.load_arguments(selected)
        if not arguments or any(item.ranking_score is None for item in arguments):
            raise PipelineStateError("Search-package generation requires ranked arguments.")
        packages = build_search_packages(arguments)
        by_argument = {item.argument_id: item for item in packages}
        attached = [
            argument.model_copy(
                update={
                    "search_package": by_argument[argument.argument_id].model_dump(mode="json")
                    if argument.argument_id in by_argument
                    else None
                }
            )
            for argument in arguments
        ]
        self.store.save_search_packages(selected, packages)
        self.store.save_arguments(selected, attached)
        self.store.delete_reviews_for_gate(selected, "D")
        self.store.update_run(selected, "RUNNING")
        reviews = [
            HumanReviewItem(
                review_id=f"REV-D-{index:04d}",
                run_id=selected,
                gate="D",
                entity_type="SEARCH_PACKAGE",
                entity_id=package.argument_id,
                reason="Confirm that the search concepts are neutral and reflect the actual factual pattern.",
            )
            for index, package in enumerate(packages, start=1)
        ]
        self.store.save_reviews(selected, reviews, append=True)
        self._mark_final_report_stale(selected, "SEARCH_PACKAGES_CHANGED")
        write_jsonl(self.extracted_for(selected) / "case_law_search_packages.jsonl", packages)
        write_jsonl(
            self.extracted_for(selected) / "human_review_queue.jsonl",
            self.store.load_reviews(selected),
        )
        write_text(self.reports_for(selected) / "case_law_research_plan.md", research_plan_markdown(packages))
        self._record(selected, 10, "generate_case_law_search_packages", len(packages))
        return selected

    def report(self, run_id: str | None = None) -> str:
        selected = self._require_run(run_id)
        documents = self.store.load_documents(selected)
        issues = self.store.load_issues(selected)
        arguments = self.store.load_arguments(selected)
        references = self.store.load_references(selected)
        packages = self.store.load_search_packages(selected)
        reviews = self.store.load_reviews(selected)
        if not arguments:
            raise PipelineStateError("Reporting requires built arguments.")
        if any(
            item.debate_summary is None
            or item.ranking_score is None
            or item.ranking_components is None
            or not item.treatment
            for item in arguments
        ):
            raise PipelineStateError(
                "Reporting requires every argument to be assessed and ranked after the latest change."
            )
        expected_packages = {
            item.argument_id
            for item in arguments
            if item.importance in (Importance.CRITICAL, Importance.HIGH)
        }
        stored_packages = {item.argument_id for item in packages}
        if stored_packages != expected_packages:
            raise PipelineStateError(
                "Reporting requires current search packages after the latest ranking."
            )
        self._record(selected, 11, "build_reports", len(arguments))
        completed_at = datetime.now(timezone.utc).isoformat()
        stage_events = self.store.stage_events(selected)
        run = self.store.load_run(selected).model_copy(
            update={"status": "COMPLETED", "completed_at": completed_at}
        )
        markdown = build_markdown_report(
            run, documents, issues, arguments, references, packages, reviews, stage_events
        )
        report_json = build_json_report(
            run, documents, issues, arguments, references, packages, reviews, stage_events
        )
        output_dir = self.reports_for(selected)
        write_text(output_dir / "ws_calibration_report.md", markdown)
        write_text(output_dir / "barrister_debate.md", barrister_debate_markdown(run, arguments))
        write_json(output_dir / "ws_calibration_report.json", report_json)
        fields = [
            "review_id", "run_id", "gate", "entity_type", "entity_id", "reason",
            "confidence", "status", "created_at",
        ]
        write_csv(output_dir / "human_review_queue.csv", review_csv_rows(reviews), fields)
        self.store.update_run(selected, "COMPLETED", completed_at)
        return selected

    def run_all(
        self,
        et1: Path,
        et3: Path,
        ws: Path,
        n_rounds: int | None = None,
        claimant_barrister: BarristerPersona | str | None = None,
        ollama_argument_cap: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        live_argument_cap: int | None = None,
    ) -> str:
        run_id = self.start_run(n_rounds, claimant_barrister)
        try:
            self.ingest(et1, et3, ws, run_id)
            self.parse_references(run_id)
            self.build_issues(run_id)
            self.extract_propositions(run_id)
            self.match_pleadings(run_id)
            self.build_arguments(run_id)
            self.assess(
                run_id,
                n_rounds,
                claimant_barrister,
                ollama_argument_cap,
                progress_callback,
                live_argument_cap,
            )
            self.rank(run_id)
            self.generate_search_packages(run_id)
            self.report(run_id)
        except Exception as exc:
            self.store.record_stage(run_id, 0, "pipeline", "FAILED", {"error": str(exc)})
            self.store.update_run(run_id, "FAILED", datetime.now(timezone.utc).isoformat())
            raise PipelineRunError(run_id, exc) from exc
        return run_id
