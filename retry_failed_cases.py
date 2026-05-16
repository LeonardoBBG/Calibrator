#!/usr/bin/env python3
"""One-off retry/resume tool for failed Calibrator cases.

Default mode is a dry run. Add --execute only after API quota is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compression_runner import count_reinforcement_clusters, run_compression
from src.config import Config, default_require_temperature_support, safe_model_output_name
from src.dictionary_loader import compact_dictionary_for_llm, load_dictionary, validate_dictionary
from src.io_utils import (
    ensure_dirs,
    make_run_id,
    make_source_slug,
    read_json,
    read_text_file,
    write_json,
    write_text,
)
from src.main import (
    _load_all_outcome_optimized_from_disk,
    build_llm_client,
    prepare_ws_tagging,
    process_judgment_case,
    record_case_failure,
    write_single_case_outcome_artifacts,
)
from src.outcome_aggregation import aggregate_outcome_optimized_cases
from src.outcome_runner import repair_outcome_optimization, run_outcome_optimization
from src.outcome_validators import validate_outcome_optimized_calibration
from src.run_inventory import (
    claim_judgment_run,
    get_judgment_run_status,
    refresh_judgment_index_statuses,
    release_judgment_run_claim,
)
from src.text_extract import load_text
from src.theme_store import build_theme_store, write_theme_store_outputs
from src.validators import CalibrationValidationContext, validate_calibration_output


def latest_case_artifact(output_root: Path, folder: str, case_slug: str, suffix: str) -> Optional[Path]:
    paths = sorted(
        (output_root / folder).glob(f"*_{case_slug}_{suffix}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


def load_failure_records(output_root: Path, source_run_id: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    failure_dir = output_root / "human_review_queue"
    for path in sorted(failure_dir.glob(f"{source_run_id}_*_case_failure.json")):
        record = read_json(path)
        record["_failure_path"] = str(path)
        judgment_path = Path(record.get("judgment_path") or "")
        if not record.get("case_slug"):
            record["case_slug"] = make_source_slug(judgment_path)
        records.append(record)
    return records


def classify_records(records: List[Dict[str, Any]], output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        judgment_path = Path(record.get("judgment_path") or "").expanduser()
        case_slug = record.get("case_slug") or make_source_slug(judgment_path)
        outcome_path = latest_case_artifact(output_root, "outcome_optimized", case_slug, "outcome_optimized")
        validated_path = latest_case_artifact(output_root, "calibration_validated", case_slug, "calibration_validated")
        compression_path = latest_case_artifact(output_root, "compression", case_slug, "reinforcement_plan")
        if outcome_path:
            resume_stage = "skip_complete"
        elif validated_path and compression_path:
            resume_stage = "outcome_optimization"
        elif validated_path:
            resume_stage = "compression"
        else:
            resume_stage = "calibration"
        rows.append({
            "record": record,
            "judgment_path": judgment_path,
            "case_slug": case_slug,
            "failed_stage": record.get("failed_stage") or "unknown",
            "resume_stage": resume_stage,
            "validated_path": validated_path,
            "compression_path": compression_path,
            "outcome_path": outcome_path,
        })
    return rows


def resume_from_validated(
    *,
    config: Config,
    dictionary: Dict[str, Any],
    validation_context: CalibrationValidationContext,
    run_id: str,
    judgment_path: Path,
    case_slug: str,
    compact_dict: Dict[str, Any],
    ws_tagging_summary: Dict[str, Any],
    prompts: Dict[str, str],
    validated_path: Path,
    compression_path: Optional[Path],
) -> Dict[str, Any]:
    """Resume a failed case from calibration_validated, preserving completed work."""
    sentinel_path = claim_judgment_run(judgment_path, config.output_root, run_id)
    if sentinel_path is None:
        return {
            "ok": False,
            "skipped": True,
            "case_slug": case_slug,
            "skip_status": get_judgment_run_status(judgment_path, config.output_root),
        }

    stage = "load_validated_calibration"
    llm_client = build_llm_client(config)
    try:
        complete_path = latest_case_artifact(config.output_root, "outcome_optimized", case_slug, "outcome_optimized")
        if complete_path:
            return {
                "ok": False,
                "skipped": True,
                "case_slug": case_slug,
                "skip_status": get_judgment_run_status(judgment_path, config.output_root),
            }

        validated_calibration = read_json(validated_path)
        validation_errors = validate_calibration_output(
            validated_calibration,
            context=validation_context,
            ws_tagging_summary=ws_tagging_summary,
        )
        if validation_errors:
            raise ValueError(
                f"Stored calibration_validated artifact is no longer validator-clean: {validated_path}"
            )

        if compression_path is not None and compression_path.exists():
            reinforcement_plan = read_json(compression_path)
        else:
            stage = "compression"
            reinforcement_plan = run_compression(
                validated_calibration,
                compact_dict,
                prompts["compression"],
                llm_client,
            )
            write_json(
                config.output_root / "compression" / f"{run_id}_{case_slug}_reinforcement_plan.json",
                reinforcement_plan,
                validate_reload=config.validate_json_writes,
            )

        stage = "outcome_optimization"
        outcome_optimized = run_outcome_optimization(
            validated_calibration,
            prompts["outcome_optimization"],
            llm_client,
        )
        stage = "outcome_validation"
        outcome_errors = validate_outcome_optimized_calibration(
            outcome_optimized,
            context=validation_context,
        )
        outcome_repair_attempts = 0
        while outcome_errors and outcome_repair_attempts < config.max_outcome_repair_attempts:
            outcome_repair_attempts += 1
            stage = "outcome_repair"
            outcome_optimized = repair_outcome_optimization(
                outcome_optimized,
                outcome_errors,
                prompts["outcome_repair"],
                llm_client,
            )
            stage = "outcome_validation"
            outcome_errors = validate_outcome_optimized_calibration(
                outcome_optimized,
                context=validation_context,
            )
        if outcome_errors:
            write_json(
                config.output_root / "human_review_queue" / f"{run_id}_{case_slug}_outcome_validation_errors.json",
                outcome_errors,
                validate_reload=config.validate_json_writes,
            )
            raise ValueError(f"Outcome optimization validation failed for {judgment_path.name}")

        outcome_filename = f"{run_id}_{case_slug}_outcome_optimized.json"
        write_json(
            config.output_root / "outcome_optimized" / outcome_filename,
            outcome_optimized,
            validate_reload=config.validate_json_writes,
        )
        case_artifacts = write_single_case_outcome_artifacts(
            config,
            dictionary,
            run_id,
            case_slug,
            outcome_optimized,
            outcome_filename,
        )
        summary = {
            "judgment": judgment_path.name,
            "case": validated_calibration.get("case_metadata", {}).get("case_name", "Unknown"),
            "signals": len(validated_calibration.get("judgment_signals", [])),
            "repair_attempts": 0,
            "outcome_repair_attempts": outcome_repair_attempts,
            "clusters": count_reinforcement_clusters(reinforcement_plan),
            "outcome_aggregation_path": case_artifacts["outcome_aggregation_path"],
            "theme_store_dir": case_artifacts["theme_store_dir"],
        }
        return {
            "ok": True,
            "case_slug": case_slug,
            "summary": summary,
            "outcome_optimized": outcome_optimized,
            "outcome_filename": outcome_filename,
            "case_artifacts": case_artifacts,
            "resumed_from": "compression" if compression_path is None else "outcome_optimization",
        }
    except Exception as exc:
        failure = record_case_failure(config, run_id, judgment_path, case_slug, stage, exc)
        print(f"Failed retry for {judgment_path.name} at {stage}: {exc}")
        if not config.continue_on_case_error:
            raise
        return {
            "ok": False,
            "case_slug": case_slug,
            "failure": failure,
        }
    finally:
        release_judgment_run_claim(sentinel_path)


def build_retry_config(args: argparse.Namespace, run_id: str) -> Config:
    config = Config.default(run_id)
    config.model_name = args.model
    config.require_temperature_support = default_require_temperature_support(config.model_name)
    config.output_root = config.project_root / "output" / safe_model_output_name(config.model_name)
    config.max_parallel_cases = max(1, min(5, args.workers))
    config.cache_enabled = not args.no_cache
    config.continue_on_case_error = True
    config.run_mode = "batch"
    return config


def run_retry(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.retry_run_id or f"{make_run_id()}_retry_{args.source_run_id}"
    config = build_retry_config(args, run_id)
    ensure_dirs(config)

    records = load_failure_records(config.output_root, args.source_run_id)
    if args.limit:
        records = records[: args.limit]
    classified = classify_records(records, config.output_root)

    if not records:
        return {
            "run_id": run_id,
            "source_run_id": args.source_run_id,
            "dry_run": not args.execute,
            "message": "No failed case artifacts found.",
        }

    stage_counts = Counter(item["resume_stage"] for item in classified)
    failed_stage_counts = Counter(item["failed_stage"] for item in classified)

    if not args.execute:
        return {
            "run_id": run_id,
            "source_run_id": args.source_run_id,
            "dry_run": True,
            "selected_case_count": len(classified),
            "resume_stage_counts": dict(stage_counts),
            "failed_stage_counts": dict(failed_stage_counts),
            "cases": [
                {
                    "judgment": item["judgment_path"].name,
                    "failed_stage": item["failed_stage"],
                    "resume_stage": item["resume_stage"],
                    "has_calibration_validated": item["validated_path"] is not None,
                    "has_compression": item["compression_path"] is not None,
                    "already_complete": item["outcome_path"] is not None,
                }
                for item in classified
            ],
        }

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add quota/key first, then rerun with --execute.")

    dictionary = load_dictionary(config.dictionary_path)
    validate_dictionary(dictionary)
    compact_dict = compact_dictionary_for_llm(dictionary)
    ws_text = load_text(config.ws_path, config.cache_root / "text", config.cache_enabled)
    write_text(config.output_root / "extracted_text" / f"{run_id}_ws_text.txt", ws_text)

    prompts = {
        "calibration": read_text_file(config.calibration_prompt_path),
        "compression": read_text_file(config.compression_prompt_path),
        "repair": read_text_file(config.repair_prompt_path),
        "outcome_optimization": read_text_file(config.outcome_optimization_prompt_path),
        "outcome_repair": read_text_file(config.outcome_repair_prompt_path),
    }
    ws_tagging_summary = prepare_ws_tagging(
        config,
        run_id,
        ws_text,
        compact_dict,
        read_text_file(config.ws_tagging_prompt_path),
        None,
    )
    validation_context = CalibrationValidationContext.from_dictionary(dictionary)

    def run_one(item: Dict[str, Any]) -> Dict[str, Any]:
        judgment_path = item["judgment_path"]
        case_slug = item["case_slug"]
        if item["resume_stage"] == "skip_complete":
            return {
                "ok": False,
                "skipped": True,
                "case_slug": case_slug,
                "skip_status": get_judgment_run_status(judgment_path, config.output_root),
            }
        if item["validated_path"] is not None:
            return resume_from_validated(
                config=config,
                dictionary=dictionary,
                validation_context=validation_context,
                run_id=run_id,
                judgment_path=judgment_path,
                case_slug=case_slug,
                compact_dict=compact_dict,
                ws_tagging_summary=ws_tagging_summary,
                prompts=prompts,
                validated_path=item["validated_path"],
                compression_path=item["compression_path"],
            )
        return process_judgment_case(
            config,
            dictionary,
            validation_context,
            run_id,
            judgment_path,
            ws_text,
            compact_dict,
            ws_tagging_summary,
            prompts,
        )

    results: List[Optional[Dict[str, Any]]] = [None] * len(classified)
    workers = min(config.max_parallel_cases, len(classified))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(run_one, item): index
                for index, item in enumerate(classified)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
    else:
        results = [run_one(item) for item in classified]

    successful_cases = []
    successful_slugs = []
    outcome_source_filenames = {}
    processed_summaries = []
    failed_cases = []
    skipped_cases = []
    per_case_artifacts = []

    for result in results:
        if not result:
            continue
        if result.get("skipped"):
            status = result.get("skip_status")
            skipped_cases.append({
                "judgment": status.pdf_path.name if status else result.get("case_slug"),
                "case_slug": result.get("case_slug"),
                "status": status.status if status else "skipped",
                "reason": status.reason if status else "Skipped by retry script.",
            })
        elif result.get("ok"):
            outcome_source_filenames[len(successful_cases)] = result["outcome_filename"]
            successful_cases.append(result["outcome_optimized"])
            successful_slugs.append(result["case_slug"])
            processed_summaries.append(result["summary"])
            per_case_artifacts.append(result["case_artifacts"])
        else:
            failed_cases.append(result.get("failure"))

    outcome_aggregation_path = None
    theme_store_dir = None
    if successful_cases:
        all_cases, all_filenames = _load_all_outcome_optimized_from_disk(config.output_root)
        if not all_cases:
            all_cases, all_filenames = successful_cases, outcome_source_filenames
        aggregate_slug = f"{run_id}_batch_{len(all_cases)}_cases"
        aggregation = aggregate_outcome_optimized_cases(all_cases, dictionary)
        outcome_aggregation_path = (
            config.output_root / "outcome_aggregation" / f"{aggregate_slug}_outcome_aggregation.json"
        )
        write_json(
            outcome_aggregation_path,
            aggregation,
            validate_reload=config.validate_json_writes,
        )
        theme_store_bundle = build_theme_store(aggregation, all_cases, all_filenames)
        theme_store_dir = config.output_root / "theme_store" / aggregate_slug
        write_theme_store_outputs(theme_store_bundle, theme_store_dir)

    index_refresh = None
    if config.judgment_index_path.exists():
        index_refresh = refresh_judgment_index_statuses(config.judgment_index_path, config.output_root)

    summary = {
        "run_id": run_id,
        "source_run_id": args.source_run_id,
        "dry_run": False,
        "selected_case_count": len(classified),
        "processed_case_count": len(processed_summaries),
        "failed_case_count": len(failed_cases),
        "skipped_case_count": len(skipped_cases),
        "resume_stage_counts": dict(stage_counts),
        "failed_stage_counts": dict(failed_stage_counts),
        "processed_cases": processed_summaries,
        "failed_cases": failed_cases,
        "skipped_cases": skipped_cases,
        "outcome_aggregation_path": str(outcome_aggregation_path) if outcome_aggregation_path else None,
        "theme_store_dir": str(theme_store_dir) if theme_store_dir else None,
        "per_case_outcome_aggregation_paths": [
            item["outcome_aggregation_path"]
            for item in per_case_artifacts
        ],
        "per_case_theme_store_dirs": [
            item["theme_store_dir"]
            for item in per_case_artifacts
        ],
        "judgment_index_refresh": index_refresh,
    }
    write_json(
        config.output_root / "human_review_queue" / f"{run_id}_retry_failed_cases_summary.json",
        summary,
        validate_reload=config.validate_json_writes,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed Calibrator cases from a previous run, reusing completed stage artifacts."
    )
    parser.add_argument("--source-run-id", default="20260514_120349")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-run-id", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the API. Omit for dry-run planning.",
    )
    return parser.parse_args()


def main() -> None:
    result = run_retry(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
