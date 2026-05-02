# Calibrator

Employment Tribunal witness-statement calibration pipeline.

## What This Repository Contains

- Python pipeline code in `src/`
- Notebook runner in `pipeline.ipynb`
- Controlled theme dictionary in `input/dictionary/`
- Prompts in `input/prompts/`

Local case PDFs, witness statements, generated outputs, and LLM caches are intentionally ignored by Git.

## Local Inputs

Place local case files at:

- `input/ws/witness_statement.pdf`
- `input/judgments/<judgment-file-name>.pdf`

The pipeline preserves the judgment PDF filename stem in output artifact names for auditability.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `pipeline.ipynb`, set the execution controls in the first code cell, and run from the top.

The notebook has explicit input bounds. Judgments above `WARN_JUDGMENT_CHARS` print a token-use warning; judgments above `MAX_JUDGMENT_CHARS` stop before making the calibration API call.

## Judgment Input Modes

The notebook supports two flows:

- `RUN_MODE = "debug"` processes one explicit judgment file from `DEBUG_JUDGMENT_PATH`.
- `RUN_MODE = "batch"` processes every `*.pdf` in `BATCH_JUDGMENTS_DIR`.

The WS statement and controlled dictionary do not vary across a batch. They are loaded once, and WS tagging is run once. The per-judgment loop only extracts each judgment, runs calibration/repair/compression, and saves outputs using the original judgment filename stem.

The notebook prompts for `OPENAI_API_KEY` when `RUN_LLM = True` and the key is not already set.

## WS Baseline

WS tagging is not just a review artifact. The pipeline saves the full WS tagging output and also derives a compact WS tagging summary. That summary is passed into every judgment calibration call as `WS_TAGGING_SUMMARY_JSON`, so each judgment is calibrated against the same fixed WS/theme baseline.

The summary's `theme_presence_by_id` is authoritative for calibration. `recommended_action_by_id` is advisory only; calibration and repair must still use the dictionary's permitted actions and the baseline coupling rules. Automated validation rejects reinforcement/add actions for `ABSENT` or `RISK_ONLY` baselines, rejects `PRESENT` WS presence for those baselines, and sends `LATENT` baselines to review rather than reinforcement.

## Repair

Repair runs only after calibration validation fails. It is a constructive LLM repair step: the invalid calibration JSON, validation errors, dictionary, and WS tagging summary are re-prompted through `input/prompts/repair_prompt.txt`. The repair step does not locally mutate or relax failed output; the repaired JSON must pass the same validator before it is saved as validated calibration.

## Cache And Outputs

Text extraction is cached by source file path, size, and modified time. LLM responses are cached by model, prompt, payload, temperature, and token settings, so a WS, dictionary, prompt, or judgment change creates a different cache key.

Outputs accumulate under `output/` by artifact type: `ws_tagging/`, `calibration_raw/`, `calibration_repaired/`, `calibration_validated/`, and `compression/`. Batch mode currently produces one validated calibration and one reinforcement plan per judgment; there is no cross-judgment aggregation layer yet.

## Model Temperature

The default configuration uses `temperature = 0.0` for models that support it. This is intentional for stable JSON calibration.

The notebook derives `REQUIRE_TEMPERATURE_SUPPORT` from `MODEL_NAME`. GPT-5.x models default to provider temperature because they may reject explicit temperature settings; other models require low-temperature support by default.
