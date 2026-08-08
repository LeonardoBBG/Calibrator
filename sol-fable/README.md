# Sol-Fable WS Argument Calibrator

Sol-Fable is a standalone, auditable MVP for comparing a Claimant witness
statement with the pleaded positions in an ET1 and ET3. It indexes exact source
paragraphs, extracts atomic WS propositions, builds a neutral issue map, runs a
controlled SOL/FABLE barrister exchange, prioritises points to protect or defend,
and produces focused concepts for later case-law research.

It is intentionally separate from the parent Calibrator: everything it imports,
stores and runs lives inside this `sol-fable/` directory. It does not call the
existing Calibrator code.

## Important limits

- Pleadings are stored as party positions, not evidence that an event occurred.
- Bracketed documentary references remain unresolved until a future bundle stage.
- The default analysis backend is deterministic and inspectable. Optional live
  backends generate analysis but do not retrieve cases, coach witness answers,
  rewrite the WS or make final legal findings.
- Weak matches, possible new-case points and key rankings are sent to a human-review
  queue.
- Text PDFs are supported. Scanned PDFs require OCR before ingestion.

## Quick start

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_streamlit.sh
```

The equivalent direct command is:

```bash
streamlit run streamlit_app.py
```

After installing the package, the portable UI command is:

```bash
sol-fable-ui
```

Upload one ET1, one ET3 and one Claimant WS (`.txt`, `.docx` or text `.pdf`),
then click **Run full calibration**. The app provides the ranked argument map,
full structured detail, round-by-round barrister debate, review queue, audit trail
and report downloads.

The **Barrister setup** section exposes two run variables:

- `n_rounds` defaults to `2`. Every round contains a Claimant turn followed by a
  Respondent turn.
- `claimant_barrister` defaults to `SOL`. Selecting `FABLE` automatically assigns
  `SOL` to the Respondent; selecting `SOL` assigns `FABLE` to the Respondent.

The orchestrator mediates all turns. A round-two Claimant turn receives the
validated round-one Respondent turn; the next Respondent turn receives that new
Claimant turn. The personas never call each other directly, cannot introduce
unknown paragraph IDs, and stop at the configured round limit.
The validated exchanges are then synthesised into bounded Claimant/Respondent
strength bands and a `PROTECT`, `DEFEND` or `BOTH` recommendation. Ranking gives
that mediated recommendation limited influence; paragraph-led pleading and source
signals remain the primary safeguard.

The **Analysis backend** section in the sidebar selects how SOL and FABLE reason:

- `deterministic-v1` (default): fast, reproducible, rule-based templating. No
  network calls. This is what the test suite exercises.
- `ollama`: real reasoning via a local model served by [Ollama](https://ollama.com).
  The sidebar shows a live dropdown of the models already pulled on the selected
  Ollama host (`ollama pull gemma3:27b`, etc.) — nothing is hardcoded. `gemma3:27b`
  is selected by default when present. If Ollama isn't reachable at the given
  host, the app falls back to the deterministic backend and shows an error.
  `ollama_num_gpu` defaults to `999`, which asks Ollama to offload every model
  layer and is clamped to the model's real layer count. Set it to `null` for
  Ollama's automatic placement or `0` to force CPU execution.
- `openai`: SOL and FABLE use the configured OpenAI model. The key comes from
  `OPENAI_API_KEY` or the masked Streamlit credential field.
- `anthropic`: SOL and FABLE use the configured Anthropic model. The key comes
  from `ANTHROPIC_API_KEY` or the masked Streamlit credential field.
- `dual-api`: SOL uses OpenAI (`gpt-5.6-sol` by default) and FABLE uses Anthropic
  (`claude-sonnet-5` by default). Party assignment remains independent: either
  persona can be the Claimant barrister.

Every prompt sent to a live model uses that provider's structured-output mechanism
and is validated against the exact `SolAssessment`/`FableAssessment`/`DebateTurn`/
`DebateSummary` schema. `paragraph_citations` is further constrained to only
the paragraph IDs actually linked to that argument. A failed or invalid response
is retried once with the validation error fed back to the model before the run
fails — see `config/settings.yaml`'s `llm_max_retries`. The orchestrator's
citation-bound validation (`debate.py:_validate_turn`) still runs on every turn
regardless of backend, as a final backstop.

The exact prompt bundle is selected from `prompt_version` and SHA-256 hashed into
the run. Requested and provider-resolved model names, response IDs, observable
token usage and retry attempts are recorded; API credentials are not.

`config/settings.yaml` controls the backend, provider model names, Ollama host,
GPU-layer request, retry limit and provider-neutral live argument cap. Install live-backend clients
with `pip install -e '.[llm]'`, or only the hosted-provider clients with
`pip install -e '.[api]'`. API keys are read at runtime and are never stored in
that file.

### Real cases are large — the live-analysis cap

A real case can produce hundreds of bounded argument units after paragraph-led
consolidation. Each live argument costs several *sequential* calls: two baseline
assessments, `2 × n_rounds` turns, and one neutral summary (two summaries for the
dual-provider consensus). `live_argument_cap` therefore defaults to `20` for
Ollama, OpenAI, Anthropic and mixed routes. The highest-priority arguments use the
selected model providers; every remaining argument is completed deterministically,
so the final list is still complete. The sidebar shows the per-argument call
estimate and requires explicit cloud-data/cost confirmation. Pass
`--live-argument-cap 0` only when an intentionally uncapped run is required.

### Resumable, checkpointed assessment

`assess()` checkpoints after **every** completed argument. If it is interrupted,
re-run `assess --run-id <id>` with the same backend, model, prompt, round count
and barrister assignment to continue the compatible run. `run-all` always creates
a new run; it does not resume an old one. Progress (`assessed N/M arguments`) is
logged at `--verbose` on the CLI and shown as a live progress bar in Streamlit.
If a full UI run fails, its run ID is displayed. When argument assessment had
already started, the selected run can be finished with **Resume assessment and
finish selected run**.

## CLI

Install the package in editable mode to expose the `sol-fable` command:

```bash
pip install -e '.[ui,dev]'
sol-fable run-all --et1 path/to/ET1.docx --et3 path/to/ET3.docx --ws path/to/WS.docx \
  --n-rounds 2 --claimant-barrister SOL
```

Without installation, use `PYTHONPATH=src python -m sol_fable.cli` in place of
`sol-fable`. Every stage can also run separately against the latest run (or a
specific `--run-id`):

```bash
sol-fable ingest --et1 ET1.txt --et3 ET3.txt --ws WS.txt
sol-fable parse-references
sol-fable build-issues
sol-fable extract-propositions
sol-fable match-pleadings
sol-fable build-arguments
sol-fable assess --n-rounds 2 --claimant-barrister SOL
sol-fable rank
sol-fable generate-search-packages
sol-fable report
sol-fable status
```

Invocation-wide options can appear either before or after the subcommand.
`--analysis-backend {deterministic-v1,ollama,openai,anthropic,dual-api}` genuinely
overrides the YAML backend for that invocation. Provider options include
`--ollama-model`, `--ollama-host`, `--live-argument-cap`, `--openai-model` and
`--anthropic-model`; API keys are deliberately not accepted as CLI arguments.

```bash
pip install -e '.[llm]'
sol-fable list-models  # lists models pulled in the local Ollama instance
sol-fable run-all --et1 ET1.txt --et3 ET3.txt --ws WS.txt \
  --analysis-backend ollama --ollama-model gemma3:27b --live-argument-cap 20 --n-rounds 2

# Interrupted assessment? Continue the same, configuration-compatible run:
sol-fable --verbose --analysis-backend ollama --ollama-model gemma3:27b \
  assess --run-id RUN-... --n-rounds 2
```

## Outputs and auditability

SQLite at `case/database/case.sqlite` is authoritative. In a source checkout this
is repository-local; an installed wheel uses `SOL_FABLE_DATA_DIR` or the current
user's platform data directory, never `site-packages`. Case directories/files are
created with private user-only permissions where the OS supports them. Each run has a unique ID;
source hashes, parser version, backend name, prompt version, timestamps and every
stage event are recorded. Original uploads are preserved under `case/source/`
with hash-prefixed names. Derived files are deterministic where no timestamp is
part of the required metadata.

Key outputs are run-scoped beneath `case/extracted/<run-id>/` and
`case/reports/<run-id>/`, so a later run cannot overwrite earlier artifacts:

- `case/extracted/<run-id>/documents.json` and `paragraphs.jsonl`
- `case/extracted/<run-id>/reference_placeholders.jsonl`
- `case/extracted/<run-id>/issues.jsonl` and `ws_propositions.jsonl`
- `case/extracted/<run-id>/pleading_matches.jsonl` and `arguments_draft.jsonl`
- `case/extracted/<run-id>/sol_assessments.jsonl` and `fable_assessments.jsonl`
- `case/extracted/<run-id>/debate_rounds.jsonl` and `debate_summaries.jsonl`
- `case/extracted/<run-id>/ranked_arguments.jsonl`
- `case/extracted/<run-id>/case_law_search_packages.jsonl`
- `case/reports/<run-id>/ws_calibration_report.md` and `.json`
- `case/reports/<run-id>/human_review_queue.csv`
- `case/reports/<run-id>/barrister_debate.md`

The JSON report includes the run metadata, input hashes, source-linked arguments,
every mediated debate turn, ranking components, search packages, review items and
audit trail.

## Architecture

Each stage has one narrow module under `src/sol_fable/`. Pydantic models reject
unknown fields and enforce the structured contracts. `storage.py` owns SQLite;
`orchestrator.py` fixes stage ordering and exports artifacts; `analysis_backend.py`
defines the replaceable Sol/Fable backend interface. Deterministic, local Ollama,
OpenAI, Anthropic and dual-provider adapters return the same validated models,
preserve exact citations and identify their model in the run record.
Atomic propositions remain individually traceable, while the argument builder
consolidates compatible propositions into stable, bounded units of at most five;
opposite meanings and conflicting pleading profiles remain separate.

Thresholds and transparent ranking weights are in `config/settings.yaml`. Versioned
agent prompt contracts live in `prompts/`. Pydantic JSON Schema snapshots can be
generated with:

```bash
PYTHONPATH=src python -m sol_fable.schema_export
```

## Verify

```bash
pytest
```

The tests cover all eight acceptance criteria from the build specification plus
the mediated-debate controls, round ordering, opponent handoffs, citation bounds,
role reversal and credential masking.

Small synthetic inputs for a harmless trial run are included in `tests/fixtures/`.
