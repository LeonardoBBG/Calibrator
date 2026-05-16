import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback

from .config import Config
from .io_utils import ensure_dirs, read_json, read_text_file, write_json, write_text, make_run_id, make_run_scope_slug, make_source_slug
from .text_extract import load_text
from .dictionary_loader import load_dictionary, validate_dictionary, compact_dictionary_for_llm
from .dictionary_runner import build_ws_tagging_summary, run_ws_tagging
from .llm_client import LLMClient
from .calibration_runner import run_calibration
from .validators import CalibrationValidationContext, validate_calibration_output
from .repair_runner import repair_calibration_output
from .compression_runner import count_reinforcement_clusters, run_compression
from .outcome_aggregation import aggregate_outcome_optimized_cases
from .outcome_runner import repair_outcome_optimization, run_outcome_optimization
from .outcome_validators import validate_outcome_optimized_calibration
from .run_inventory import plan_judgment_run
from .theme_store import build_theme_store, write_theme_store_outputs

def build_llm_client(config):
    """Create an LLM client for one independent worker."""
    return LLMClient(
        provider=config.api_provider,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        require_temperature_support=config.require_temperature_support,
        cache_dir=config.cache_root / "llm",
        cache_enabled=config.cache_enabled,
        request_timeout_seconds=config.request_timeout_seconds,
        request_max_retries=config.request_max_retries
    )


def latest_ws_tagging_summary_path(output_root: Path):
    """Return the newest local WS tagging summary for this output root, if present."""
    ws_dir = output_root / "ws_tagging"
    if not ws_dir.exists():
        return None
    summaries = sorted(
        ws_dir.glob("*_ws_tagging_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def latest_ws_tagging_artifact_path(output_root: Path):
    """Return the newest full local WS tagging artifact for this output root, if present."""
    ws_dir = output_root / "ws_tagging"
    if not ws_dir.exists():
        return None
    artifacts = sorted(
        ws_dir.glob("*_ws_tagging.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return artifacts[0] if artifacts else None


def load_reusable_ws_tagging_summary(config, run_id):
    """Load a local WS summary, or derive one from a local full WS tagging artifact."""
    candidates = []
    latest_summary = latest_ws_tagging_summary_path(config.output_root)
    if latest_summary is not None:
        candidates.append(latest_summary)

    configured_summary = getattr(config, "ws_tagging_summary_path", None)
    if configured_summary:
        configured_summary = Path(configured_summary).expanduser()
        if configured_summary.exists():
            candidates.append(configured_summary)

    if candidates:
        seen = set()
        unique_candidates = []
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_candidates.append(path)
        summary_path = sorted(
            unique_candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[0]
        print(f"Using existing WS tagging summary: {summary_path}")
        return read_json(summary_path)

    full_artifact_path = latest_ws_tagging_artifact_path(config.output_root)
    if full_artifact_path is None:
        return None

    print(f"Using existing WS tagging artifact: {full_artifact_path}")
    ws_tagging_summary = build_ws_tagging_summary(read_json(full_artifact_path))
    write_json(
        config.output_root / "ws_tagging" / f"{run_id}_ws_tagging_summary.json",
        ws_tagging_summary,
        validate_reload=config.validate_json_writes
    )
    return ws_tagging_summary


def prepare_ws_tagging(config, run_id, ws_text, compact_dict, ws_tagging_prompt, llm_client):
    """Run WS tagging for this run, or load a previously saved WS summary."""
    if config.run_ws:
        if getattr(config, "reuse_existing_ws_tagging", True):
            reusable_summary = load_reusable_ws_tagging_summary(config, run_id)
            if reusable_summary is not None:
                return reusable_summary

        if llm_client is None:
            llm_client = build_llm_client(config)
        ws_tagging = run_ws_tagging(ws_text, compact_dict, ws_tagging_prompt, llm_client)
        ws_tagging_summary = build_ws_tagging_summary(ws_tagging)
        write_json(
            config.output_root / "ws_tagging" / f"{run_id}_ws_tagging.json",
            ws_tagging,
            validate_reload=config.validate_json_writes
        )
        write_json(
            config.output_root / "ws_tagging" / f"{run_id}_ws_tagging_summary.json",
            ws_tagging_summary,
            validate_reload=config.validate_json_writes
        )
        return ws_tagging_summary

    reusable_summary = load_reusable_ws_tagging_summary(config, run_id)
    if reusable_summary is not None:
        return reusable_summary

    summary_path = getattr(config, "ws_tagging_summary_path", None)
    if not summary_path:
        raise ValueError("ws_tagging_summary_path is required when run_ws=False")

    summary_path = Path(summary_path).expanduser()
    if not summary_path.exists():
        raise FileNotFoundError(
            f"WS tagging summary path does not exist: {summary_path}"
        )

    return read_json(summary_path)

def build_case_failure(run_id, judgment_path, case_slug, stage, exc):
    """Build a serializable failure record for a judgment that could not complete."""
    return {
        "run_id": run_id,
        "judgment": judgment_path.name,
        "judgment_path": str(judgment_path),
        "case_slug": case_slug,
        "failed_stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }

def record_case_failure(config, run_id, judgment_path, case_slug, stage, exc):
    """Write the per-case failure artifact and return its failure record."""
    failure = build_case_failure(run_id, judgment_path, case_slug, stage, exc)
    write_json(
        config.output_root / "human_review_queue" / f"{run_id}_{case_slug}_case_failure.json",
        failure,
        validate_reload=config.validate_json_writes
    )
    return failure


def write_single_case_outcome_artifacts(
    config,
    dictionary,
    run_id,
    case_slug,
    outcome_optimized,
    outcome_filename,
):
    """Write deterministic audit artifacts for one outcome-optimized case."""
    case_scope_slug = make_run_scope_slug(run_id, [case_slug])
    aggregation = aggregate_outcome_optimized_cases([outcome_optimized], dictionary)
    outcome_aggregation_path = (
        config.output_root
        / "outcome_aggregation"
        / f"{case_scope_slug}_outcome_aggregation.json"
    )
    write_json(
        outcome_aggregation_path,
        aggregation,
        validate_reload=config.validate_json_writes,
    )
    theme_store_bundle = build_theme_store(
        aggregation,
        [outcome_optimized],
        {0: outcome_filename},
    )
    theme_store_dir = config.output_root / "theme_store" / case_scope_slug
    write_theme_store_outputs(theme_store_bundle, theme_store_dir)
    return {
        "outcome_aggregation_path": str(outcome_aggregation_path),
        "theme_store_dir": str(theme_store_dir),
    }


def process_judgment_case(
    config,
    dictionary,
    validation_context,
    run_id,
    judgment_path,
    ws_text,
    compact_dict,
    ws_tagging_summary,
    prompts,
):
    """Run calibration, compression, and outcome optimization for one judgment."""
    case_slug = make_source_slug(judgment_path)
    stage = "load_judgment_text"
    llm_client = build_llm_client(config)
    try:
        judgment_text = load_text(judgment_path, config.cache_root / "text", config.cache_enabled)
        write_text(config.output_root / "extracted_text" / f"{run_id}_{case_slug}_text.txt", judgment_text)

        stage = "calibration"
        calibration = run_calibration(
            ws_text,
            judgment_text,
            compact_dict,
            ws_tagging_summary,
            prompts["calibration"],
            llm_client
        )
        write_json(
            config.output_root / "calibration_raw" / f"{run_id}_{case_slug}_calibration_raw.json",
            calibration,
            validate_reload=config.validate_json_writes
        )

        stage = "calibration_validation"
        errors = validate_calibration_output(
            calibration,
            context=validation_context,
            ws_tagging_summary=ws_tagging_summary
        )
        validated_calibration = calibration
        repair_attempts = 0

        while errors and repair_attempts < config.max_repair_attempts:
            repair_attempts += 1
            write_json(
                config.output_root / "calibration_repaired" / f"{run_id}_{case_slug}_repair_attempt_{repair_attempts}.json",
                validated_calibration,
                validate_reload=config.validate_json_writes
            )
            write_json(
                config.output_root / "calibration_repaired" / f"{run_id}_{case_slug}_validation_errors_attempt_{repair_attempts}.json",
                errors,
                validate_reload=config.validate_json_writes
            )
            stage = "calibration_repair"
            validated_calibration = repair_calibration_output(
                validated_calibration,
                errors,
                compact_dict,
                ws_tagging_summary,
                prompts["repair"],
                llm_client
            )
            stage = "calibration_validation"
            errors = validate_calibration_output(
                validated_calibration,
                context=validation_context,
                ws_tagging_summary=ws_tagging_summary
            )

        if errors:
            raise ValueError(
                f"Calibration validation failed for {judgment_path.name} "
                f"after {config.max_repair_attempts} attempts"
            )

        write_json(
            config.output_root / "calibration_validated" / f"{run_id}_{case_slug}_calibration_validated.json",
            validated_calibration,
            validate_reload=config.validate_json_writes
        )

        stage = "compression"
        reinforcement_plan = run_compression(
            validated_calibration,
            compact_dict,
            prompts["compression"],
            llm_client
        )
        write_json(
            config.output_root / "compression" / f"{run_id}_{case_slug}_reinforcement_plan.json",
            reinforcement_plan,
            validate_reload=config.validate_json_writes
        )

        stage = "outcome_optimization"
        outcome_optimized = run_outcome_optimization(
            validated_calibration,
            prompts["outcome_optimization"],
            llm_client
        )
        stage = "outcome_validation"
        outcome_errors = validate_outcome_optimized_calibration(
            outcome_optimized,
            context=validation_context
        )
        outcome_repair_attempts = 0
        while outcome_errors and outcome_repair_attempts < config.max_outcome_repair_attempts:
            outcome_repair_attempts += 1
            stage = "outcome_repair"
            outcome_optimized = repair_outcome_optimization(
                outcome_optimized,
                outcome_errors,
                prompts["outcome_repair"],
                llm_client
            )
            stage = "outcome_validation"
            outcome_errors = validate_outcome_optimized_calibration(
                outcome_optimized,
                context=validation_context
            )
        if outcome_errors:
            write_json(
                config.output_root / "human_review_queue" / f"{run_id}_{case_slug}_outcome_validation_errors.json",
                outcome_errors,
                validate_reload=config.validate_json_writes
            )
            raise ValueError(f"Outcome optimization validation failed for {judgment_path.name}")

        outcome_filename = f"{run_id}_{case_slug}_outcome_optimized.json"
        write_json(
            config.output_root / "outcome_optimized" / outcome_filename,
            outcome_optimized,
            validate_reload=config.validate_json_writes
        )
        stage = "per_case_outcome_artifacts"
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
            "repair_attempts": repair_attempts,
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
        }
    except Exception as exc:
        failure = record_case_failure(config, run_id, judgment_path, case_slug, stage, exc)
        print(f"Failed {judgment_path.name} at {stage}: {exc}")
        if not config.continue_on_case_error:
            raise
        return {
            "ok": False,
            "case_slug": case_slug,
            "failure": failure,
        }


def _parallel_worker_count(config, judgment_count):
    configured = int(getattr(config, "max_parallel_cases", 1) or 1)
    configured = max(1, min(5, configured))
    if config.run_mode != "batch":
        return 1
    return min(configured, judgment_count)


def run_calibrator(config: Config) -> dict:
    run_id = config.run_id
    ensure_dirs(config)

    selected_judgment_paths = config.selected_judgment_paths()
    if not selected_judgment_paths:
        raise ValueError(f"No judgment PDFs found for run_mode={config.run_mode}")

    judgment_paths, skipped_statuses = plan_judgment_run(
        selected_judgment_paths,
        config.output_root,
    )
    if not judgment_paths:
        return {
            "run_id": run_id,
            "run_mode": config.run_mode,
            "selected_case_count": len(selected_judgment_paths),
            "processed_case_count": 0,
            "failed_case_count": 0,
            "skipped_case_count": len(skipped_statuses),
            "skipped_cases": [
                {
                    "judgment": status.pdf_path.name,
                    "case_slug": status.case_slug,
                    "status": status.status,
                    "reason": status.reason,
                    "artifact_counts": status.artifact_counts,
                }
                for status in skipped_statuses
            ],
            "processed_cases": [],
            "failed_cases": [],
            "outcome_aggregation_path": None,
            "theme_store_dir": None,
            "per_case_outcome_aggregation_paths": [],
            "per_case_theme_store_dirs": [],
        }

    # Load shared inputs once. These do not vary across judgments in a batch.
    ws_text = load_text(config.ws_path, config.cache_root / "text", config.cache_enabled)
    write_text(config.output_root / "extracted_text" / f"{run_id}_ws_text.txt", ws_text)

    # Load dictionary and prompts
    dictionary = load_dictionary(config.dictionary_path)
    validate_dictionary(dictionary)
    compact_dict = compact_dictionary_for_llm(dictionary)
    ws_tagging_prompt = read_text_file(config.ws_tagging_prompt_path)
    calibration_prompt = read_text_file(config.calibration_prompt_path)
    compression_prompt = read_text_file(config.compression_prompt_path)
    repair_prompt = read_text_file(config.repair_prompt_path)
    outcome_optimization_prompt = read_text_file(config.outcome_optimization_prompt_path)
    outcome_repair_prompt = read_text_file(config.outcome_repair_prompt_path)

    ws_tagging_summary = prepare_ws_tagging(
        config,
        run_id,
        ws_text,
        compact_dict,
        ws_tagging_prompt,
        None
    )

    validation_context = CalibrationValidationContext.from_dictionary(dictionary)
    prompts = {
        "calibration": calibration_prompt,
        "compression": compression_prompt,
        "repair": repair_prompt,
        "outcome_optimization": outcome_optimization_prompt,
        "outcome_repair": outcome_repair_prompt,
    }
    summaries = []
    failed_cases = []
    outcome_optimized_cases = []
    outcome_source_filenames = {}
    successful_case_slugs = []
    per_case_artifacts = []
    case_worker_count = _parallel_worker_count(config, len(judgment_paths))
    print(f"Case workers: {case_worker_count}")

    if case_worker_count > 1:
        case_results = [None] * len(judgment_paths)
        with ThreadPoolExecutor(max_workers=case_worker_count) as executor:
            future_to_index = {
                executor.submit(
                    process_judgment_case,
                    config,
                    dictionary,
                    validation_context,
                    run_id,
                    judgment_path,
                    ws_text,
                    compact_dict,
                    ws_tagging_summary,
                    prompts,
                ): index
                for index, judgment_path in enumerate(judgment_paths)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    case_results[index] = future.result()
                except Exception:
                    for pending_future in future_to_index:
                        pending_future.cancel()
                    raise
    else:
        case_results = [
            process_judgment_case(
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
            for judgment_path in judgment_paths
        ]

    for result in case_results:
        if not result:
            continue
        if result["ok"]:
            per_case_artifacts.append(result["case_artifacts"])
            outcome_source_filenames[len(outcome_optimized_cases)] = result["outcome_filename"]
            outcome_optimized_cases.append(result["outcome_optimized"])
            successful_case_slugs.append(result["case_slug"])
            summaries.append(result["summary"])
        else:
            failed_cases.append(result["failure"])

    if outcome_optimized_cases:
        aggregate_slug = make_run_scope_slug(run_id, successful_case_slugs)
        aggregation = aggregate_outcome_optimized_cases(outcome_optimized_cases, dictionary)
        outcome_aggregation_path = config.output_root / "outcome_aggregation" / f"{aggregate_slug}_outcome_aggregation.json"
        write_json(
            outcome_aggregation_path,
            aggregation,
            validate_reload=config.validate_json_writes
        )
        theme_store_bundle = build_theme_store(aggregation, outcome_optimized_cases, outcome_source_filenames)
        theme_store_dir = config.output_root / "theme_store" / aggregate_slug
        write_theme_store_outputs(theme_store_bundle, theme_store_dir)
    else:
        outcome_aggregation_path = None
        theme_store_dir = None

    if failed_cases:
        write_json(
            config.output_root / "human_review_queue" / f"{run_id}_failed_cases_summary.json",
            {
                "run_id": run_id,
                "failed_case_count": len(failed_cases),
                "processed_case_count": len(summaries),
                "failures": failed_cases,
            },
            validate_reload=config.validate_json_writes
        )

    print(f"Run mode: {config.run_mode}")
    print(f"Judgments processed: {len(summaries)}")
    print(f"Judgments failed: {len(failed_cases)}")
    for summary in summaries:
        print(f"Judgment: {summary['judgment']}")
        print(f"  Case: {summary['case']}")
        print(f"  Judgment signals: {summary['signals']}")
        print(f"  Validation errors repaired: {summary['repair_attempts']}")
        print(f"  Compressed clusters: {summary['clusters']}")

    return {
        "run_id": run_id,
        "run_mode": config.run_mode,
        "selected_case_count": len(selected_judgment_paths),
        "processed_case_count": len(summaries),
        "failed_case_count": len(failed_cases),
        "skipped_case_count": len(skipped_statuses),
        "skipped_cases": [
            {
                "judgment": status.pdf_path.name,
                "case_slug": status.case_slug,
                "status": status.status,
                "reason": status.reason,
                "artifact_counts": status.artifact_counts,
            }
            for status in skipped_statuses
        ],
        "processed_cases": summaries,
        "failed_cases": failed_cases,
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
    }


def main():
    run_id = make_run_id()
    config = Config.default(run_id)
    run_calibrator(config)

if __name__ == "__main__":
    main()
