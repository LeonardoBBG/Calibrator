import sys
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
from .theme_store import build_theme_store, write_theme_store_outputs

def prepare_ws_tagging(config, run_id, ws_text, compact_dict, ws_tagging_prompt, llm_client):
    """Run WS tagging for this run, or load a previously saved WS summary."""
    if config.run_ws:
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

    if not config.ws_tagging_summary_path:
        raise ValueError("ws_tagging_summary_path is required when run_ws=False")

    if not config.ws_tagging_summary_path.exists():
        raise FileNotFoundError(
            f"WS tagging summary path does not exist: {config.ws_tagging_summary_path}"
        )

    return read_json(config.ws_tagging_summary_path)

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

def main():
    run_id = make_run_id()
    config = Config.default(run_id)

    ensure_dirs(config)

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

    # Instantiate LLM
    llm_client = LLMClient(
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

    ws_tagging_summary = prepare_ws_tagging(
        config,
        run_id,
        ws_text,
        compact_dict,
        ws_tagging_prompt,
        llm_client
    )

    validation_context = CalibrationValidationContext.from_dictionary(dictionary)
    judgment_paths = config.selected_judgment_paths()
    if not judgment_paths:
        raise ValueError(f"No judgment PDFs found for run_mode={config.run_mode}")

    summaries = []
    failed_cases = []
    outcome_optimized_cases = []
    outcome_source_filenames = {}
    successful_case_slugs = []
    for judgment_path in judgment_paths:
        case_slug = make_source_slug(judgment_path)
        stage = "load_judgment_text"
        try:
            judgment_text = load_text(judgment_path, config.cache_root / "text", config.cache_enabled)
            write_text(config.output_root / "extracted_text" / f"{run_id}_{case_slug}_text.txt", judgment_text)

            stage = "calibration"
            calibration = run_calibration(
                ws_text,
                judgment_text,
                compact_dict,
                ws_tagging_summary,
                calibration_prompt,
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
                    repair_prompt,
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
            reinforcement_plan = run_compression(validated_calibration, compact_dict, compression_prompt, llm_client)
            write_json(
                config.output_root / "compression" / f"{run_id}_{case_slug}_reinforcement_plan.json",
                reinforcement_plan,
                validate_reload=config.validate_json_writes
            )

            stage = "outcome_optimization"
            outcome_optimized = run_outcome_optimization(
                validated_calibration,
                outcome_optimization_prompt,
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
                    outcome_repair_prompt,
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
            write_json(
                config.output_root / "outcome_optimized" / f"{run_id}_{case_slug}_outcome_optimized.json",
                outcome_optimized,
                validate_reload=config.validate_json_writes
            )
            outcome_source_filenames[len(outcome_optimized_cases)] = f"{run_id}_{case_slug}_outcome_optimized.json"
            outcome_optimized_cases.append(outcome_optimized)
            successful_case_slugs.append(case_slug)

            summaries.append({
                "judgment": judgment_path.name,
                "case": validated_calibration.get("case_metadata", {}).get("case_name", "Unknown"),
                "signals": len(validated_calibration.get("judgment_signals", [])),
                "repair_attempts": repair_attempts,
                "clusters": count_reinforcement_clusters(reinforcement_plan)
            })
        except Exception as exc:
            failure = record_case_failure(config, run_id, judgment_path, case_slug, stage, exc)
            failed_cases.append(failure)
            print(f"Failed {judgment_path.name} at {stage}: {exc}")
            if not config.continue_on_case_error:
                raise

    if outcome_optimized_cases:
        aggregate_slug = make_run_scope_slug(run_id, successful_case_slugs)
        aggregation = aggregate_outcome_optimized_cases(outcome_optimized_cases, dictionary)
        write_json(
            config.output_root / "outcome_aggregation" / f"{aggregate_slug}_outcome_aggregation.json",
            aggregation,
            validate_reload=config.validate_json_writes
        )
        theme_store_bundle = build_theme_store(aggregation, outcome_optimized_cases, outcome_source_filenames)
        write_theme_store_outputs(theme_store_bundle, config.output_root / "theme_store" / aggregate_slug)

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

if __name__ == "__main__":
    main()
