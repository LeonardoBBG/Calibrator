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

## Judgment Input Modes

The notebook supports two flows:

- `RUN_MODE = "debug"` processes one explicit judgment file from `DEBUG_JUDGMENT_PATH`.
- `RUN_MODE = "batch"` processes every `*.pdf` in `BATCH_JUDGMENTS_DIR`.

The WS statement and controlled dictionary do not vary across a batch. They are loaded once, and WS tagging is run once. The per-judgment loop only extracts each judgment, runs calibration/repair/compression, and saves outputs using the original judgment filename stem.

The notebook prompts for `OPENAI_API_KEY` when `RUN_LLM = True` and the key is not already set.

## Model Temperature

The default configuration uses `temperature = 0.0` and requires the selected model to support that setting. This is intentional for stable JSON calibration.

If a model only supports its default temperature, the pipeline stops with a clear error unless `REQUIRE_TEMPERATURE_SUPPORT = False` is set in the notebook.
