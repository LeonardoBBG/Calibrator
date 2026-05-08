import sys
from pathlib import Path

from .config import Config
from .io_utils import ensure_dirs, read_json, read_text_file, write_json, write_text, make_run_id, make_source_slug
from .text_extract import load_text
from .dictionary_loader import load_dictionary, validate_dictionary
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

def prepare_ws_tagging(config, run_id, ws_text, dictionary, ws_tagging_prompt, llm_client):
    """Run WS tagging for this run, or load a previously saved WS summary."""
    if config.run_ws:
        ws_tagging = run_ws_tagging(ws_text, dictionary, ws_tagging_prompt, llm_client)
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
        cache_enabled=config.cache_enabled
    )

    ws_tagging_summary = prepare_ws_tagging(
        config,
        run_id,
        ws_text,
        dictionary,
        ws_tagging_prompt,
        llm_client
    )

    validation_context = CalibrationValidationContext.from_dictionary(dictionary)
    judgment_paths = config.selected_judgment_paths()
    if not judgment_paths:
        raise ValueError(f"No judgment PDFs found for run_mode={config.run_mode}")

    summaries = []
    outcome_optimized_cases = []
    outcome_source_filenames = {}
    for judgment_path in judgment_paths:
        judgment_text = load_text(judgment_path, config.cache_root / "text", config.cache_enabled)
        case_slug = make_source_slug(judgment_path)
        write_text(config.output_root / "extracted_text" / f"{run_id}_{case_slug}_text.txt", judgment_text)

        calibration = run_calibration(
            ws_text,
            judgment_text,
            dictionary,
            ws_tagging_summary,
            calibration_prompt,
            llm_client
        )
        write_json(
            config.output_root / "calibration_raw" / f"{run_id}_{case_slug}_calibration_raw.json",
            calibration,
            validate_reload=config.validate_json_writes
        )

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
            validated_calibration = repair_calibration_output(
                validated_calibration,
                errors,
                dictionary,
                ws_tagging_summary,
                repair_prompt,
                llm_client
            )
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

        reinforcement_plan = run_compression(validated_calibration, dictionary, compression_prompt, llm_client)
        write_json(
            config.output_root / "compression" / f"{run_id}_{case_slug}_reinforcement_plan.json",
            reinforcement_plan,
            validate_reload=config.validate_json_writes
        )

        outcome_optimized = run_outcome_optimization(
            validated_calibration,
            outcome_optimization_prompt,
            llm_client
        )
        outcome_errors = validate_outcome_optimized_calibration(
            outcome_optimized,
            context=validation_context
        )
        outcome_repair_attempts = 0
        while outcome_errors and outcome_repair_attempts < config.max_outcome_repair_attempts:
            outcome_repair_attempts += 1
            outcome_optimized = repair_outcome_optimization(
                outcome_optimized,
                outcome_errors,
                outcome_repair_prompt,
                llm_client
            )
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

        summaries.append({
            "judgment": judgment_path.name,
            "case": validated_calibration.get("case_metadata", {}).get("case_name", "Unknown"),
            "signals": len(validated_calibration.get("judgment_signals", [])),
            "repair_attempts": repair_attempts,
            "clusters": count_reinforcement_clusters(reinforcement_plan)
        })

    if outcome_optimized_cases:
        aggregation = aggregate_outcome_optimized_cases(outcome_optimized_cases, dictionary)
        write_json(
            config.output_root / "outcome_aggregation" / f"{run_id}_outcome_aggregation.json",
            aggregation,
            validate_reload=config.validate_json_writes
        )
        theme_store_bundle = build_theme_store(aggregation, outcome_optimized_cases, outcome_source_filenames)
        write_theme_store_outputs(theme_store_bundle, config.output_root / "theme_store" / run_id)

    print(f"Run mode: {config.run_mode}")
    print(f"Judgments processed: {len(summaries)}")
    for summary in summaries:
        print(f"Judgment: {summary['judgment']}")
        print(f"  Case: {summary['case']}")
        print(f"  Judgment signals: {summary['signals']}")
        print(f"  Validation errors repaired: {summary['repair_attempts']}")
        print(f"  Compressed clusters: {summary['clusters']}")

if __name__ == "__main__":
    main()
