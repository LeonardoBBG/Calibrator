import sys
from pathlib import Path

from .config import Config
from .io_utils import ensure_dirs, read_text_file, write_json, write_text, make_run_id, make_source_slug
from .text_extract import load_text
from .dictionary_loader import load_dictionary, validate_dictionary
from .dictionary_runner import build_ws_tagging_summary, run_ws_tagging
from .llm_client import LLMClient
from .calibration_runner import run_calibration
from .validators import CalibrationValidationContext, validate_calibration_output
from .repair_runner import repair_calibration_output
from .compression_runner import run_compression

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

    # Run the WS tagging prompt against the controlled dictionary.
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

    validation_context = CalibrationValidationContext.from_dictionary(dictionary)
    judgment_paths = config.selected_judgment_paths()
    if not judgment_paths:
        raise ValueError(f"No judgment PDFs found for run_mode={config.run_mode}")

    summaries = []
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

        summaries.append({
            "judgment": judgment_path.name,
            "case": validated_calibration.get("case_metadata", {}).get("case_name", "Unknown"),
            "signals": len(validated_calibration.get("judgment_signals", [])),
            "repair_attempts": repair_attempts,
            "clusters": len(reinforcement_plan.get("compressed_reinforcement_plan", []))
        })

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
