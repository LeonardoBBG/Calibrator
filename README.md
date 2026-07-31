# Calibrator

Self-contained Employment Tribunal witness-statement calibration pipeline.

The application does not import code from the parent FATE repository. This
directory can be copied or cloned on its own and run independently; its code,
prompts, controlled dictionary, dependency manifest, and local input layout all
live here.

## What This Repository Contains

- Python pipeline code in `src/`
- Streamlit runner/review app in `streamlit_single_case_monitor.py`
- Controlled theme dictionary in `input/dictionary/` (`WS_Controlled_Theme_Dictionary_v1_3_3.json` is the active default)
- Prompts in `input/prompts/`

Local case PDFs, witness statements, generated outputs, and LLM caches are intentionally ignored by Git.

## Local Inputs

Place local case files at:

- `input/ws/witness_statement.pdf`
- `input/judgments/<judgment-file-name>.pdf`

The pipeline preserves the judgment PDF filename stem in output artifact names for auditability.

## Run

From this directory, create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run streamlit_single_case_monitor.py
```

The app writes generated artifacts beneath `output/` by default. It does not
need the parent FATE application to run.

## Standalone Data Locations

The default input and output paths are relative to this directory. For a fully
separate case workspace, set `CALIBRATOR_DATA_DIR`; code, prompts, and the
dictionary remain in this directory while case inputs and outputs live in the
external workspace.

```bash
export CALIBRATOR_DATA_DIR=/path/to/case-workspace
streamlit run streamlit_single_case_monitor.py
```

The workspace should contain `input/ws/` and `input/judgments/`. These optional
variables can override individual locations:

- `CALIBRATOR_WS_DIR` and `CALIBRATOR_WS_PATH`
- `CALIBRATOR_JUDGMENTS_DIR` and `CALIBRATOR_JUDGMENT_INDEX_PATH`
- `CALIBRATOR_OUTPUT_DIR` and `CALIBRATOR_CACHE_DIR`
- `CALIBRATOR_PROMPTS_DIR` and `CALIBRATOR_DICTIONARY_PATH`

Set `OPENAI_API_KEY` (or enter it in the Streamlit run settings) before making
LLM-backed runs.

## Verify

Run the self-contained test suite from this directory:

```bash
python -m pytest src/tests.py -q
```

## Judgment Input Modes

The Streamlit runner supports two flows:

- `Per doc` processes one runnable judgment.
- `Batch` processes runnable judgments from `input/judgments/moltie_judgment_index.csv` when present, otherwise every `*.pdf` in `input/judgments/`.

If `input/judgments/moltie_judgment_index.csv` exists, batch mode uses that
Moltie-generated index instead of requiring PDFs to be copied into
`input/judgments/`. The index stores the external PDF path, Moltie composite
rank/percentile columns, and Calibrator processing state. Before any API work,
the runner re-checks existing downstream artifacts and skips judgments whose
final output already exists.

The WS statement and controlled dictionary do not vary across a batch. They are loaded once, and WS tagging is run once. The per-judgment loop only extracts each judgment, runs calibration/repair/compression, and saves outputs using the original judgment filename stem. WS tagging summaries carry the active dictionary payload hash, so summaries from an older dictionary version are not reused silently.

The Streamlit runner accepts `OPENAI_API_KEY` in the run settings, or uses the key already set in the environment.

## WS Baseline

WS tagging is not just a review artifact. The pipeline saves the full WS tagging output and also derives a compact WS tagging summary. That summary is passed into every judgment calibration call as `WS_TAGGING_SUMMARY_JSON`, so each judgment is calibrated against the same fixed WS/theme baseline.

`Config.run_ws` controls how that baseline is prepared. With `run_ws=True`, the pipeline runs WS tagging for the current run and writes both `output/ws_tagging/{run_id}_ws_tagging.json` and `output/ws_tagging/{run_id}_ws_tagging_summary.json`. With `run_ws=False`, the pipeline does not create new WS tagging files; it loads the existing summary from `Config.ws_tagging_summary_path` and uses that for downstream calibration.

The summary's `theme_presence_by_id` is authoritative for calibration. `recommended_action_by_id` is advisory only; calibration and repair must still use the dictionary's permitted actions and the baseline coupling rules. Automated validation rejects reinforcement/add actions for `ABSENT` or `RISK_ONLY` baselines, rejects `PRESENT` WS presence for those baselines, and sends `LATENT` baselines to review rather than reinforcement.

## Repair

Repair runs only after calibration validation fails. It is a constructive LLM repair step: the invalid calibration JSON, validation errors, dictionary, and WS tagging summary are re-prompted through `input/prompts/repair_prompt.txt`. The repair step does not locally mutate or relax failed output; the repaired JSON must pass the same validator before it is saved as validated calibration.

## Cache And Outputs

Text extraction is cached by source file path, size, and modified time. LLM responses are cached by model, prompt, payload, temperature, and token settings, so a WS, dictionary, prompt, or judgment change creates a different cache key.

Outputs accumulate under `output/` by artifact type: `extracted_text/`, `ws_tagging/`, `calibration_raw/`, `calibration_repaired/`, `calibration_validated/`, `compression/`, `outcome_optimized/`, `outcome_aggregation/`, `theme_store/`, and `human_review_queue/`. Batch mode produces per-judgment calibration, compression, and outcome artifacts, plus batch-level outcome aggregation and theme-store review exports when outcome optimization succeeds.

`output/theme_store/{run_id}/` is deterministic. It uses `outcome_aggregation` as the theme index and `outcome_optimized` case files as the match items, then writes `theme_store.json`, `theme_summary.csv`, `review_queue.csv`, and `top_matches_per_theme.csv`. It groups each theme by (effect, case effect, confidence), so `T17_COMPARATOR_TREATMENT / REINFORCE / WIN_DRIVER / HIGH` and `T17_COMPARATOR_TREATMENT / REVIEW_MANUALLY / NEUTRAL_CONTEXT / HIGH` remain separate review buckets. It does not make any LLM calls.

## Model Temperature

The default configuration uses `temperature = 0.0` for models that support it. This is intentional for stable JSON calibration.

The runner derives `require_temperature_support` from the configured model name. GPT-5.x models default to provider temperature because they may reject explicit temperature settings; other models require low-temperature support by default.
