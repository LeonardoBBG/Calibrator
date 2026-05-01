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

The notebook prompts for `OPENAI_API_KEY` when `RUN_LLM = True` and the key is not already set.
