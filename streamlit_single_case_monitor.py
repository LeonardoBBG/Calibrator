import json
import os
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT
DATA_ROOT = Path(os.getenv("CALIBRATOR_DATA_DIR", str(CODE_ROOT))).expanduser().resolve()

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.config import Config, default_require_temperature_support, safe_model_output_name
from src.dictionary_loader import load_dictionary
from src.io_utils import make_run_id
from src.main import latest_ws_tagging_artifact_path, run_calibrator
from src.outcome_aggregation import aggregate_outcome_optimized_cases
from src.run_inventory import (
    get_index_band_summary,
    load_judgment_index_paths_with_band_caps,
    plan_judgment_run,
    scan_judgment_run_statuses,
)

OUTPUT_BASE_ROOT = Path(
    os.getenv("CALIBRATOR_OUTPUT_DIR", str(DATA_ROOT / "output"))
).expanduser().resolve()
DEFAULT_CACHE_ROOT = Path(
    os.getenv("CALIBRATOR_CACHE_DIR", str(OUTPUT_BASE_ROOT / "cache"))
).expanduser().resolve()
DEFAULT_WS_DIR = Path(
    os.getenv("CALIBRATOR_WS_DIR", str(DATA_ROOT / "input" / "ws"))
).expanduser().resolve()
DEFAULT_WS_PATH = Path(
    os.getenv("CALIBRATOR_WS_PATH", str(DEFAULT_WS_DIR / "witness_statement.pdf"))
).expanduser().resolve()
DEFAULT_PROMPTS_DIR = Path(
    os.getenv("CALIBRATOR_PROMPTS_DIR", str(CODE_ROOT / "input" / "prompts"))
).expanduser().resolve()
DEFAULT_DICTIONARY_PATH = Path(
    os.getenv(
        "CALIBRATOR_DICTIONARY_PATH",
        str(CODE_ROOT / "input" / "dictionary" / "WS_Controlled_Theme_Dictionary_v1_2_final.json"),
    )
).expanduser().resolve()
DEFAULT_RESULT_MODEL_NAME = "gpt-5.5"
DEFAULT_MODEL_OUTPUT_ROOT = OUTPUT_BASE_ROOT / safe_model_output_name(DEFAULT_RESULT_MODEL_NAME)
DEFAULT_CASE_PATH = DEFAULT_MODEL_OUTPUT_ROOT / (
    "outcome_optimized/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1_outcome_optimized.json"
)
DEFAULT_AGGREGATION_PATH = DEFAULT_MODEL_OUTPUT_ROOT / (
    "outcome_aggregation/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1_outcome_aggregation.json"
)
DEFAULT_THEME_STORE_PATH = DEFAULT_MODEL_OUTPUT_ROOT / (
    "theme_store/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1/"
    "theme_store.json"
)
DEFAULT_CASE_DIR = DEFAULT_MODEL_OUTPUT_ROOT / "outcome_optimized"
DEFAULT_AGGREGATION_DIR = DEFAULT_MODEL_OUTPUT_ROOT / "outcome_aggregation"
DEFAULT_THEME_STORE_DIR = DEFAULT_MODEL_OUTPUT_ROOT / "theme_store"
DEFAULT_JUDGMENTS_DIR = Path(
    os.getenv("CALIBRATOR_JUDGMENTS_DIR", str(DATA_ROOT / "input" / "judgments"))
).expanduser().resolve()
DEFAULT_JUDGMENT_INDEX_PATH = Path(
    os.getenv("CALIBRATOR_JUDGMENT_INDEX_PATH", str(DEFAULT_JUDGMENTS_DIR / "moltie_judgment_index.csv"))
).expanduser().resolve()
DEFAULT_WS_TAGGING_DIR = DEFAULT_MODEL_OUTPUT_ROOT / "ws_tagging"

RECOMMENDATION_ORDER = [
    "REINFORCE_PRIMARY",
    "REINFORCE_SUPPORTING",
    "REFRAME",
    "MONITOR",
    "AVOID",
    "RISK_CONTROL",
    "NO_SIGNAL",
]

GUIDANCE_LABELS = {
    "REINFORCE_PRIMARY": "Maximise",
    "REINFORCE_SUPPORTING": "Maximise",
    "REFRAME": "Refine",
    "MONITOR": "Minimise",
    "AVOID": "Avoid",
    "RISK_CONTROL": "Quarantine",
    "NO_SIGNAL": "Ignore",
}

OUTCOME_LABELS = {
    "WIN": "Claimant won",
    "LOSS": "Claimant lost",
    "MIXED": "Mixed outcome",
    "MODERATE_WIN": "Moderate liability win",
    "STRONG_WIN": "Strong liability win",
    "WEAK_WIN": "Limited liability win",
    "COMPLETED": "Remedy decided",
    "NOT_DETERMINED": "Remedy not decided",
    "DETERMINED": "Reductions decided",
    "ANALOGY_ONLY": "Analogy only",
    "DIRECT": "Direct fit",
    "PARTIAL": "Partial fit",
    "MEDIUM_HIGH": "Medium-high",
    "ADVERSE": "Adverse",
    "LIABILITY_ONLY_PROCEDURAL_ANALOGY": "Use for liability/procedure only",
}

CASE_EFFECT_RANK = {
    "WIN_DRIVER": 0,
    "STRONG_SUPPORT": 1,
    "MODERATE_SUPPORT": 2,
    "WEAK_SUPPORT": 3,
    "NEUTRAL": 4,
    "NEUTRAL_CONTEXT": 5,
    "ADVERSE": 6,
}

EFFECT_RANK = {
    "REINFORCE": 0,
    "ADD FACT": 1,
    "ADD EVIDENCE ANCHOR": 2,
    "DISTINGUISH": 3,
    "REVIEW_MANUALLY": 4,
    "NEUTRAL": 5,
    "UNDERMINE": 6,
}

CONFIDENCE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json_from_path(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_from_upload(uploaded_file) -> Dict[str, Any]:
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    return path


def list_json_files(folder_text: str) -> List[Path]:
    folder = resolve_project_path(folder_text)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def list_theme_store_files(folder_text: str) -> List[Path]:
    folder = resolve_project_path(folder_text)
    if not folder.exists():
        return []
    return sorted(folder.rglob("theme_store.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        pass
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def artifact_scope_key(path: Path) -> str:
    if path.name == "theme_store.json":
        return path.parent.name
    suffixes = [
        "_outcome_optimized",
        "_outcome_aggregation",
        "_calibration_validated",
        "_calibration_raw",
        "_reinforcement_plan",
    ]
    stem = path.stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def outcome_timestamp(path: Path) -> str:
    name = path.name
    parts = name.split("_", 2)
    if len(parts) < 2:
        return ""
    return f"{parts[0]}_{parts[1]}"


def matched_aggregation_index(case_path: Path, aggregation_paths: List[Path]) -> int:
    case_scope = artifact_scope_key(case_path)
    for i, p in enumerate(aggregation_paths):
        if case_scope and artifact_scope_key(p) == case_scope:
            return i
    case_key = outcome_timestamp(case_path)
    for i, p in enumerate(aggregation_paths):
        if case_key and outcome_timestamp(p) == case_key:
            return i
    return 0


def matched_theme_store_index(case_path: Path, theme_store_paths: List[Path]) -> int:
    case_scope = artifact_scope_key(case_path)
    for i, p in enumerate(theme_store_paths):
        if case_scope and artifact_scope_key(p) == case_scope:
            return i
    case_key = outcome_timestamp(case_path)
    for i, p in enumerate(theme_store_paths):
        if case_key and outcome_timestamp(p.parent) == case_key:
            return i
    return 0


def _display_project_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        pass
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _artifact_counts_text(artifact_counts: Dict[str, int]) -> str:
    if not artifact_counts:
        return ""
    return ", ".join(f"{name}: {count}" for name, count in sorted(artifact_counts.items()))


def _runner_status_rows(statuses) -> List[Dict[str, str]]:
    rows = []
    for status in statuses:
        rows.append({
            "judgment": status.pdf_path.name,
            "status": status.status,
            "runnable": "yes" if status.runnable else "no",
            "reason": status.reason,
            "artifacts": _artifact_counts_text(status.artifact_counts),
            "latest_artifact": _display_project_path(status.latest_artifact),
        })
    return rows


def _latest_ws_summary_path() -> Optional[Path]:
    return _latest_ws_summary_path_for_output(DEFAULT_MODEL_OUTPUT_ROOT)


def _model_output_root(model_name: str) -> Path:
    return OUTPUT_BASE_ROOT / safe_model_output_name(model_name)


def _latest_ws_summary_path_for_output(output_root: Path) -> Optional[Path]:
    summaries = sorted(
        (output_root / "ws_tagging").glob("*_ws_tagging_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def _build_run_config(
    run_mode: str,
    selected_judgment: Optional[Path],
    model_name: str,
    max_tokens: int,
    run_ws: bool,
    reuse_existing_ws_tagging: bool,
    max_parallel_cases: int,
    ws_summary_path: Optional[Path],
    cache_enabled: bool,
    per_band_caps: Optional[Dict] = None,
) -> Config:
    run_id = make_run_id()
    config = Config.default(run_id)
    config.run_mode = run_mode
    config.ws_path = DEFAULT_WS_PATH
    config.judgments_dir = DEFAULT_JUDGMENTS_DIR
    config.judgment_index_path = DEFAULT_JUDGMENT_INDEX_PATH
    config.dictionary_path = DEFAULT_DICTIONARY_PATH
    config.ws_tagging_prompt_path = DEFAULT_PROMPTS_DIR / "ws_tagging_prompt.txt"
    config.calibration_prompt_path = DEFAULT_PROMPTS_DIR / "calibration_prompt.txt"
    config.compression_prompt_path = DEFAULT_PROMPTS_DIR / "compression_prompt.txt"
    config.repair_prompt_path = DEFAULT_PROMPTS_DIR / "repair_prompt.txt"
    config.outcome_optimization_prompt_path = DEFAULT_PROMPTS_DIR / "outcome_optimization_prompt.txt"
    config.outcome_repair_prompt_path = DEFAULT_PROMPTS_DIR / "outcome_repair_prompt.txt"
    if selected_judgment is not None:
        config.judgment_path = selected_judgment
    config.model_name = model_name.strip()
    config.require_temperature_support = default_require_temperature_support(config.model_name)
    config.output_root = _model_output_root(config.model_name)
    config.cache_root = DEFAULT_CACHE_ROOT
    config.max_tokens = int(max_tokens)
    config.run_ws = run_ws
    config.reuse_existing_ws_tagging = reuse_existing_ws_tagging
    config.max_parallel_cases = max(1, min(5, int(max_parallel_cases)))
    if ws_summary_path is not None:
        config.ws_tagging_summary_path = ws_summary_path
    config.cache_enabled = cache_enabled
    config.per_band_caps = per_band_caps or None
    return config


def render_runner_tab() -> None:
    model_name = st.text_input("OpenAI model", value=DEFAULT_RESULT_MODEL_NAME, key="calibrator_runner_model")
    model_output_root = _model_output_root(model_name)
    st.caption(f"Model output folder: {model_output_root}")

    use_index = DEFAULT_JUDGMENT_INDEX_PATH.exists()
    statuses = scan_judgment_run_statuses(
        DEFAULT_JUDGMENTS_DIR,
        model_output_root,
        DEFAULT_JUDGMENT_INDEX_PATH if use_index else None,
    )
    runnable_statuses = [s for s in statuses if s.runnable]
    complete_statuses = [s for s in statuses if s.status == "complete"]
    blocked_statuses = [s for s in statuses if s.status == "blocked_partial"]
    in_progress_statuses = [s for s in statuses if s.status == "in_progress"]

    # ── Inventory overview ───────────────────────────────────
    st.markdown("<div class='section-title'>Batch inventory</div>", unsafe_allow_html=True)
    source_label = "Indexed judgments" if use_index else "Input PDFs"
    source_detail = DEFAULT_JUDGMENT_INDEX_PATH.name if use_index else f"Found in {DEFAULT_JUDGMENTS_DIR.name}"
    if in_progress_statuses:
        mc = st.columns(5)
        mc[4].markdown(mini_card_html("In progress", str(len(in_progress_statuses)), "Running now — will be skipped", "info"), unsafe_allow_html=True)
    else:
        mc = st.columns(4)
    mc[0].markdown(mini_card_html(source_label, str(len(statuses)), source_detail), unsafe_allow_html=True)
    mc[1].markdown(mini_card_html("Ready to run", str(len(runnable_statuses)), "Pending — no output yet", "good" if runnable_statuses else ""), unsafe_allow_html=True)
    mc[2].markdown(mini_card_html("Complete", str(len(complete_statuses)), "Full output exists — will be skipped", "info"), unsafe_allow_html=True)
    mc[3].markdown(mini_card_html("Blocked partial", str(len(blocked_statuses)), "Partial output — will be skipped", "warn" if blocked_statuses else ""), unsafe_allow_html=True)

    if not statuses:
        st.warning(f"No indexed judgments or PDFs found in {DEFAULT_JUDGMENTS_DIR}")
        return

    if blocked_statuses:
        with st.expander(f"Blocked cases ({len(blocked_statuses)}) — review before running", expanded=False):
            for s in blocked_statuses:
                st.caption(f"**{s.pdf_path.name}** — {s.reason}")

    st.divider()

    # ── Mode selector ────────────────────────────────────────
    run_mode_label = st.radio("Mode", ["Per doc", "Batch"], horizontal=True)

    # ── Per-doc selector ──────────────────────────────────────
    selected_judgment = None
    if run_mode_label == "Per doc":
        if runnable_statuses:
            selected_status = st.selectbox(
                "Select document",
                runnable_statuses,
                format_func=lambda s: s.pdf_path.name,
                label_visibility="collapsed",
            )
            selected_judgment = selected_status.pdf_path
            st.caption(f"Status: {selected_status.reason}")
        else:
            st.info("No runnable documents — all PDFs are complete or blocked.")

    # ── Advanced settings (batch only) ────────────────────────
    per_band_caps = None
    if run_mode_label == "Batch":
        with st.expander("Advanced settings", expanded=False):
            if not DEFAULT_JUDGMENT_INDEX_PATH.exists():
                st.caption("No judgment index found — advanced settings require a CSV index.")
            else:
                band_summary_full = get_index_band_summary(DEFAULT_JUDGMENT_INDEX_PATH, max_per_band=0)
                not_yet_by_band = {e["band"]: e["runnable_selected"] for e in band_summary_full}

                st.markdown(
                    "<div class='section-title' style='font-size:1rem;margin-bottom:0.5rem;'>"
                    "Cases per percentile band</div>",
                    unsafe_allow_html=True,
                )

                if band_summary_full:
                    df_init = pd.DataFrame([
                        {
                            "Band": e["band"],
                            "Total cases": e["total"],
                            "Not yet run": e["runnable_selected"],
                            "Run": e["runnable_selected"],
                        }
                        for e in band_summary_full
                    ])

                    edited_df = st.data_editor(
                        df_init,
                        column_config={
                            "Band": st.column_config.TextColumn(disabled=True),
                            "Total cases": st.column_config.NumberColumn(disabled=True),
                            "Not yet run": st.column_config.NumberColumn(disabled=True),
                            "Run": st.column_config.NumberColumn(
                                "Run",
                                min_value=0,
                                step=1,
                                help="Cases to queue from this band. 0 = run all not-yet-run.",
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="calibrator_band_editor",
                    )

                    # Build per-band caps: only include bands where user reduced below not-yet-run
                    raw_caps = {
                        str(row["Band"]): int(row["Run"] or 0)
                        for _, row in edited_df.iterrows()
                    }
                    effective_caps = {
                        band: cap
                        for band, cap in raw_caps.items()
                        if 0 < cap < not_yet_by_band.get(band, 0)
                    }
                    per_band_caps = effective_caps if effective_caps else None

                    def requested_band_total(band: str, not_yet_run: int) -> int:
                        requested = raw_caps.get(band, not_yet_run)
                        if requested <= 0:
                            return not_yet_run
                        return min(requested, not_yet_run)

                    # Total line
                    total_selected = sum(
                        requested_band_total(band, nyr)
                        for band, nyr in not_yet_by_band.items()
                    )
                    cap_label = "custom per-band caps" if per_band_caps else "no cap — all pending"
                    st.markdown(
                        f"<div style='background:rgba(59,130,246,0.10);border:1px solid rgba(59,130,246,0.28);"
                        f"border-radius:8px;padding:0.6rem 0.9rem;margin-top:0.5rem;'>"
                        f"<span style='font-weight:800;font-size:1rem;'>{total_selected}</span>"
                        f"<span style='color:#94a3b8;font-size:0.88rem;margin-left:0.5rem;'>"
                        f"PDFs queued for this run &nbsp;·&nbsp; {cap_label}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Effective run plan ────────────────────────────────────
    batch_selected_paths: List[Path] = []
    batch_queued_paths: List[Path] = []
    batch_skipped_statuses = []
    if run_mode_label == "Batch":
        if per_band_caps and DEFAULT_JUDGMENT_INDEX_PATH.exists():
            batch_selected_paths = load_judgment_index_paths_with_band_caps(
                DEFAULT_JUDGMENT_INDEX_PATH,
                per_band_caps,
            )
        else:
            batch_selected_paths = [s.pdf_path for s in statuses]

        batch_queued_paths, batch_skipped_statuses = plan_judgment_run(
            batch_selected_paths,
            model_output_root,
        )
        n_selected = len(batch_selected_paths)
        n_run = len(batch_queued_paths)
        n_skipped_selected = len(batch_skipped_statuses)
        cap_state = "custom per-band caps applied" if per_band_caps else "no per-band cap"
        if n_run == 0:
            st.info("Nothing to run — all selected PDFs are either complete or blocked.")
        else:
            st.markdown(
                f"<div style='background:rgba(22,163,74,0.12);border:1px solid rgba(34,197,94,0.35);"
                f"border-radius:10px;padding:0.85rem 1.1rem;margin:0.5rem 0 0.75rem 0;'>"
                f"<div style='font-weight:800;font-size:1rem;margin-bottom:0.35rem;'>"
                f"Run plan: {n_run} document{'s' if n_run != 1 else ''}</div>"
                f"<div style='color:#cbd5e1;font-size:0.9rem;line-height:1.7;'>"
                f"{len(statuses)} PDFs indexed &nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{len(runnable_statuses)} pending before caps &nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{n_selected} selected by current settings &nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{n_skipped_selected} selected skipped &nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{cap_state} &nbsp;&nbsp;·&nbsp;&nbsp;"
                f"<b style='color:#86efac;'>{n_run} will run</b></div></div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"Documents queued for this run ({n_run})", expanded=False):
                for path in batch_queued_paths:
                    st.caption(f"· {path.name}")

    # ── Run settings ──────────────────────────────────────────
    with st.expander("Run settings", expanded=False):
        api_key = st.text_input("OPENAI_API_KEY", type="password", value="")
        max_tokens = st.number_input("Max completion tokens", min_value=1000, max_value=50000, value=12000, step=1000)
        latest_summary = _latest_ws_summary_path_for_output(model_output_root)
        latest_full_ws_artifact = latest_ws_tagging_artifact_path(model_output_root)
        reuse_existing_ws_tagging = st.checkbox(
            "Reuse local WS tagging artifact first",
            value=True,
            help="Checks this model's ws_tagging output folder before making a WS tagging LLM call.",
        )
        if latest_summary is not None:
            st.caption(f"Latest local WS summary: {display_path(latest_summary)}")
        elif latest_full_ws_artifact is not None:
            st.caption(f"Latest local WS artifact: {display_path(latest_full_ws_artifact)}")
        else:
            st.caption("No local WS artifact found for this model output folder.")
        run_ws = st.checkbox("Run WS tagging if no reusable artifact exists", value=True)
        max_parallel_cases = st.number_input(
            "Parallel case workers",
            min_value=1,
            max_value=5,
            value=5 if run_mode_label == "Batch" else 1,
            step=1,
            disabled=run_mode_label != "Batch",
        )
        if run_mode_label != "Batch":
            max_parallel_cases = 1
        cache_enabled = st.checkbox("Use LLM cache", value=True)
        dry_run = st.checkbox(
            "Dry run — validate inputs without calling the API",
            value=False,
            help="Shows which PDFs would be queued and which would be skipped. No API calls are made.",
        )

        ws_summary_path = None
        if not run_ws:
            summary_default = str(latest_summary) if latest_summary else ""
            summary_text = st.text_input("Existing WS tagging summary", value=summary_default)
            ws_summary_path = Path(summary_text).expanduser() if summary_text.strip() else None

    # ── Run button ────────────────────────────────────────────
    run_count = 1 if run_mode_label == "Per doc" and selected_judgment else len(batch_queued_paths)
    can_run = run_count > 0
    if run_mode_label == "Per doc":
        base_label = f"· {selected_judgment.name}" if selected_judgment else "No document selected"
    else:
        base_label = f"batch · {run_count} document{'s' if run_count != 1 else ''}"
    button_label = f"{'Dry run' if dry_run else 'Run'} {base_label}"

    if st.button(button_label, type="primary", disabled=not can_run, use_container_width=True):
        run_mode = "debug" if run_mode_label == "Per doc" else "batch"
        config = _build_run_config(
            run_mode=run_mode,
            selected_judgment=selected_judgment,
            model_name=model_name,
            max_tokens=max_tokens,
            run_ws=run_ws,
            reuse_existing_ws_tagging=reuse_existing_ws_tagging,
            max_parallel_cases=max_parallel_cases,
            ws_summary_path=ws_summary_path,
            cache_enabled=cache_enabled,
            per_band_caps=per_band_caps,
        )

        if dry_run:
            selected = config.selected_judgment_paths()
            queued, skipped_statuses_dr = plan_judgment_run(selected, config.output_root)
            st.success(
                f"Dry run complete — {len(queued)} would be queued, "
                f"{len(skipped_statuses_dr)} would be skipped."
            )
            if queued:
                with st.expander(f"Would run ({len(queued)})", expanded=True):
                    for p in queued:
                        st.caption(f"· {p.name}")
            if skipped_statuses_dr:
                with st.expander(f"Would skip ({len(skipped_statuses_dr)})", expanded=False):
                    for s in skipped_statuses_dr:
                        st.caption(f"· {s.pdf_path.name} — {s.reason}")
            return

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is required to run the calibration.")
            return
        has_reusable_ws_artifact = latest_summary is not None or latest_full_ws_artifact is not None
        has_selected_ws_summary = ws_summary_path is not None and ws_summary_path.exists()
        if not run_ws and not has_selected_ws_summary and not has_reusable_ws_artifact:
            st.error("A valid existing WS tagging summary is required when WS tagging is disabled.")
            return

        doc_label = selected_judgment.name if run_mode == "debug" and selected_judgment else f"{run_count} documents"
        with st.spinner(f"Running calibrator on {doc_label}. This may take several minutes per document…"):
            try:
                result = run_calibrator(config)
            except Exception as exc:
                st.exception(exc)
                return

        processed = result.get("processed_case_count", 0)
        failed = result.get("failed_case_count", 0)
        skipped_r = result.get("skipped_case_count", 0)
        if failed == 0:
            st.success(f"Run {result['run_id']} complete — {processed} processed, {skipped_r} skipped.")
        else:
            st.warning(f"Run {result['run_id']} finished with errors — {processed} processed, {failed} failed, {skipped_r} skipped.")
        with st.expander("Run details", expanded=False):
            st.json(result)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_case_label(case_json: Dict[str, Any]) -> str:
    metadata = case_json.get("case_metadata", {})
    name = metadata.get("case_name") or "Unknown case"
    number = metadata.get("case_number")
    return f"{name} ({number})" if number else name


def get_theme_rows(aggregation_json: Dict[str, Any], recommendation: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = aggregation_json.get("theme_strength_matrix", [])
    if recommendation is None:
        return rows
    return [row for row in rows if row.get("recommendation") == recommendation]


def humanize(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return OUTCOME_LABELS.get(text, text.replace("_", " ").title())


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def percent_label(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{value}%"


def recommendation_counts(aggregation_json: Dict[str, Any]) -> Dict[str, int]:
    rows = aggregation_json.get("theme_strength_matrix", [])
    counts = {key: 0 for key in RECOMMENDATION_ORDER}
    for row in rows:
        rec = row.get("recommendation", "UNKNOWN")
        counts[rec] = counts.get(rec, 0) + 1
    return counts


def guidance_from_aggregation(aggregation_json: Dict[str, Any]) -> Dict[str, List[str]]:
    guidance = {"Maximise": [], "Refine": [], "Minimise": [], "Quarantine": [], "Ignore": []}
    for row in aggregation_json.get("theme_strength_matrix", []):
        theme = f"{row.get('theme_id')} - {row.get('theme_name')}"
        rec = row.get("recommendation")
        if rec in {"REINFORCE_PRIMARY", "REINFORCE_SUPPORTING"}:
            guidance["Maximise"].append(theme)
        elif rec == "REFRAME":
            guidance["Refine"].append(theme)
        elif rec in {"MONITOR", "AVOID"}:
            guidance["Minimise"].append(theme)
        elif rec == "RISK_CONTROL":
            guidance["Quarantine"].append(theme)
        elif rec == "NO_SIGNAL":
            guidance["Ignore"].append(theme)
    return guidance


def signal_weight_map(case_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item.get("signal_id"): item
        for item in case_json.get("outcome_optimization", {}).get("signal_causal_weights", [])
        if isinstance(item, dict)
    }


def validate_pair(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> List[str]:
    warnings = []
    case_name = case_json.get("case_metadata", {}).get("case_name")
    shortlist = aggregation_json.get("case_shortlist", [])
    if aggregation_json.get("aggregation_metadata", {}).get("case_count") != 1:
        warnings.append("Aggregation case_count is not 1. This app is currently single-case mode.")
    if shortlist:
        shortlist_name = shortlist[0].get("case_name")
        if case_name and shortlist_name and case_name != shortlist_name:
            warnings.append("Case name in outcome JSON does not match aggregation case_shortlist.")
    return warnings


def compact_theme_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Theme": row.get("theme_name"),
        "Theme ID": row.get("theme_id"),
        "Supporting cases": row.get("supporting_case_count"),
        "Decisive signals": row.get("decisive_signal_count"),
        "Supporting signals": row.get("contributing_signal_count"),
        "Net signal": row.get("net_theme_score"),
        "Guidance": GUIDANCE_LABELS.get(row.get("recommendation"), humanize(row.get("recommendation"))),
    }


def render_rows(rows: List[Dict[str, Any]], empty_text: str) -> None:
    if not rows:
        st.caption(empty_text)
        return
    st.dataframe([compact_theme_row(row) for row in rows], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Theme-store extraction
# ---------------------------------------------------------------------------

def extract_all_matches(theme_store_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract every match from a theme_store.json."""
    matches = []
    for theme_id, theme_data in theme_store_json.items():
        if not isinstance(theme_data, dict):
            continue
        theme_label = theme_data.get("theme_label") or theme_id
        for group_data in (theme_data.get("groups") or {}).values():
            for match in (group_data.get("matches") or []):
                if isinstance(match, dict):
                    m = dict(match)
                    m["theme_id"] = theme_id
                    m["theme_label"] = theme_label
                    matches.append(m)
    return matches


def flatten_theme_store_matches(theme_store_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for m in extract_all_matches(theme_store_json):
        theme_id = m.get("theme_id") or m.get("theme", "")
        theme_label = m.get("theme_label") or theme_id
        rows.append({
            "Theme ID": theme_id,
            "Theme": theme_label,
            "Effect": humanize(m.get("effect")),
            "Case effect": humanize(m.get("case_effect")),
            "Confidence": humanize(m.get("confidence")),
            "Rank": m.get("rank_score"),
            "Review priority": m.get("review_priority_score"),
            "Review status": humanize(m.get("review_status")),
            "Case": m.get("case_name"),
            "Source": m.get("source_pointer") or m.get("paragraph_reference"),
            "Summary": m.get("summary"),
            "Relevance to WS": m.get("relevance_to_ws"),
            "Causal weight reason": m.get("causal_weight_reason"),
            "Factual hooks": ", ".join(m.get("factual_hooks") or []),
            "Legal functions": ", ".join(m.get("legal_functions") or []),
        })
    return sorted(rows, key=lambda r: (
        str(r.get("Theme ID") or ""),
        EFFECT_RANK.get(str(r.get("Effect", "")).upper().replace(" ", "_"), 99),
        CASE_EFFECT_RANK.get(str(r.get("Case effect", "")).upper().replace(" ", "_"), 99),
        -(r.get("Review priority") or r.get("Rank") or 0),
    ))


def summarize_theme_store(theme_store_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for theme_id, theme_data in theme_store_json.items():
        if not isinstance(theme_data, dict):
            continue
        rank_data = theme_data.get("theme_rank_data") or {}
        groups = theme_data.get("groups") or {}
        rows.append({
            "Theme ID": theme_id,
            "Theme": theme_data.get("theme_label") or theme_id,
            "Matches": theme_data.get("n_matches", 0),
            "High confidence": theme_data.get("n_high_confidence", 0),
            "Win drivers": theme_data.get("n_win_drivers", 0),
            "Groups": ", ".join(groups.keys()) or "-",
            "Recommendation": humanize(rank_data.get("recommendation")),
            "Net score": rank_data.get("net_theme_score"),
        })
    return sorted(rows, key=lambda r: (-(r.get("Matches") or 0), str(r.get("Theme ID") or "")))


# ---------------------------------------------------------------------------
# Corpus aggregation helpers
# ---------------------------------------------------------------------------

WINNER_RANK = {"Employee": 0, "Employer": 1, "Mixed": 2, "Unknown": 3}
STRENGTH_RANK = {"Strong": 0, "Medium": 1, "Weak": 2, "Unknown": 3}
CORPUS_PAGE_SIZE = 15
CORPUS_ARGUMENT_PAGE_SIZE = 25


def _corpus_dir_signature(theme_store_dir: Path, outcome_dir: Path) -> tuple:
    """Fingerprint of corpus files — changes when a new batch run writes output."""
    files = []
    for d in (theme_store_dir, outcome_dir):
        if d.exists():
            for p in sorted(d.rglob("*.json")):
                try:
                    s = p.stat()
                    files.append((str(p), s.st_mtime, s.st_size))
                except OSError:
                    pass
    return tuple(files)


@st.cache_data(show_spinner="Loading argument library…")
def _load_corpus_cached(
    theme_store_dir: Path, outcome_dir: Path, corpus_sig: tuple
) -> Dict[str, Any]:
    return aggregate_all_theme_stores(theme_store_dir, outcome_dir)


def _theme_numeric_sort_key(tid: str) -> tuple:
    try:
        return (int(tid.lstrip("Tt").split("_")[0]), tid)
    except (ValueError, IndexError):
        return (9999, tid)


def _corpus_paginator(page: int, n_pages: int, *, key_prefix: str = "corpus_pg") -> None:
    """Render Google-style page navigation for the argument library."""
    def _page_window(current: int, total: int, max_shown: int = 7) -> List[int]:
        if total <= max_shown:
            return list(range(1, total + 1))
        half = max_shown // 2
        start = max(1, min(current - half, total - max_shown + 1))
        return list(range(start, start + max_shown))

    nums = _page_window(page, n_pages)
    show_first_gap = nums[0] > 1
    show_last_gap = nums[-1] < n_pages

    parts: List[tuple] = [("prev", "←")]
    if show_first_gap:
        parts += [("page", 1), ("gap", "…")]
    for p in nums:
        parts.append(("page", p))
    if show_last_gap:
        parts += [("gap", "…"), ("page", n_pages)]
    parts.append(("next", "→"))

    cols = st.columns(len(parts))
    for col, (kind, val) in zip(cols, parts):
        with col:
            if kind == "prev":
                if st.button("←", disabled=(page == 1), key=f"{key_prefix}_prev", use_container_width=True):
                    st.session_state["_corpus_page"] = page - 1
                    st.rerun()
            elif kind == "next":
                if st.button("→", disabled=(page == n_pages), key=f"{key_prefix}_next", use_container_width=True):
                    st.session_state["_corpus_page"] = page + 1
                    st.rerun()
            elif kind == "gap":
                st.markdown(
                    "<div style='text-align:center;padding-top:0.4rem;color:#94a3b8;'>…</div>",
                    unsafe_allow_html=True,
                )
            else:
                is_current = (val == page)
                if st.button(
                    str(val),
                    key=f"{key_prefix}_{val}",
                    type="primary" if is_current else "secondary",
                    use_container_width=True,
                ):
                    if not is_current:
                        st.session_state["_corpus_page"] = val
                        st.rerun()


def _group_sort_key(key: Tuple[str, str, str, str]) -> Tuple:
    _, effect, case_effect, confidence = key
    return (
        CASE_EFFECT_RANK.get(case_effect, 99),
        CONFIDENCE_RANK.get(confidence, 99),
        EFFECT_RANK.get(effect, 99),
    )


def _truthy_match_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "strong"}


def _is_strong_argument(match: Dict[str, Any], key: Tuple[str, str, str, str]) -> bool:
    """Prefer an explicit strong flag when present, otherwise derive from local scoring fields."""
    for field in ("strong_match", "strong_hit", "is_strong_match"):
        if field in match:
            return _truthy_match_flag(match.get(field))
    _theme_id, effect, case_effect, confidence = key
    rank_score = match.get("rank_score") or 0
    try:
        rank_score = float(rank_score)
    except (TypeError, ValueError):
        rank_score = 0
    return (
        effect == "REINFORCE"
        and case_effect == "WIN_DRIVER"
        and confidence == "HIGH"
        and rank_score >= 1.0
    )


def _argument_sort_key(item: Tuple[Tuple[str, str, str, str], Dict[str, Any]]) -> Tuple:
    key, match = item
    theme_id, effect, case_effect, confidence = key
    try:
        rank_score = float(match.get("rank_score") or 0)
    except (TypeError, ValueError):
        rank_score = 0
    try:
        review_priority = float(match.get("review_priority_score") or 0)
    except (TypeError, ValueError):
        review_priority = 0
    return (
        0 if _is_strong_argument(match, key) else 1,
        CASE_EFFECT_RANK.get(case_effect, 99),
        CONFIDENCE_RANK.get(confidence, 99),
        EFFECT_RANK.get(effect, 99),
        -rank_score,
        -review_priority,
        _theme_numeric_sort_key(theme_id),
        str(match.get("case_name") or ""),
    )


def _flatten_argument_items(
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]
) -> List[Tuple[Tuple[str, str, str, str], Dict[str, Any]]]:
    return sorted(
        [(key, match) for key, matches in grouped.items() for match in matches],
        key=_argument_sort_key,
    )


def _open_argument_library_for_theme(theme_id: str) -> None:
    _open_argument_library_for_section([theme_id], winner=[], effects=[], confidence=[])


def _open_argument_library_for_section(
    theme_ids: List[str],
    winner: List[str],
    effects: List[str],
    confidence: List[str],
) -> None:
    st.session_state["_corpus_jump_themes"] = theme_ids
    st.session_state["_corpus_jump_winner"] = winner
    st.session_state["_corpus_jump_effects"] = effects
    st.session_state["_corpus_jump_confidence"] = confidence
    st.session_state["_next_main_view"] = "Argument Library"
    st.rerun()


def _key_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _build_case_outcome_map(outcome_dir: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not outcome_dir.exists():
        return result
    for f in outcome_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("case_metadata", {})
        outcome = data.get("outcome_optimization", {})
        case_name = meta.get("case_name")
        if not case_name:
            continue
        liability = (outcome.get("claimant_liability_outcome") or "").upper()
        winner = "Employee" if liability == "WIN" else "Employer" if liability == "LOSS" else "Mixed" if liability == "MIXED" else "Unknown"
        band = (outcome.get("liability_outcome_strength_band") or "").upper()
        strength = "Strong" if "STRONG" in band else "Medium" if "MODERATE" in band else "Weak" if "WEAK" in band else "Unknown"
        result[case_name] = {"winner": winner, "strength": strength}
    return result


def aggregate_all_theme_stores(theme_store_dir: Path, outcome_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load every theme_store.json and aggregate by (theme, effect, case_effect, confidence).
    Attaches _winner and _strength to each match from the corresponding outcome_optimized file."""
    outcome_dir = outcome_dir or DEFAULT_CASE_DIR
    case_outcome_map = _build_case_outcome_map(outcome_dir)

    all_paths = sorted(theme_store_dir.rglob("theme_store.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not all_paths:
        return {"paths": [], "grouped": {}, "theme_labels": {}, "cases": set(), "n_matches": 0, "case_outcome_map": case_outcome_map}

    loaded = []
    for path in all_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        loaded.append((path, data))

    per_case_names = set()
    batch_entries = []
    selected_entries = []
    for path, data in loaded:
        case_names = {
            m["case_name"]
            for m in extract_all_matches(data)
            if m.get("case_name")
        }
        if _is_batch_theme_store_path(path):
            batch_entries.append((path, data, case_names))
            continue
        # Paths are sorted newest-first; skip older stores for already-covered cases.
        if case_names and case_names.issubset(per_case_names):
            continue
        selected_entries.append((path, data, case_names))
        per_case_names.update(case_names)

    # Batch stores: tag them so we can skip matches for already-covered cases at append time.
    batch_selected = []
    for path, data, case_names in batch_entries:
        gap = case_names - per_case_names
        if not gap:
            continue
        batch_selected.append((path, data, gap))

    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    theme_labels: Dict[str, str] = {}
    cases: set = set()

    for path, data, _case_names in selected_entries:
        for theme_id, theme_data in data.items():
            if not isinstance(theme_data, dict):
                continue
            theme_labels[theme_id] = theme_data.get("theme_label") or theme_id

            for m in extract_all_matches({theme_id: theme_data}):
                effect = m.get("effect") or "UNKNOWN"
                case_effect = m.get("case_effect") or "UNKNOWN"
                confidence = m.get("confidence") or "UNKNOWN"
                key = (theme_id, effect.upper(), case_effect.upper(), confidence.upper())
                case_outcome = case_outcome_map.get(m.get("case_name") or "")
                m["_winner"] = case_outcome["winner"] if case_outcome else "Unknown"
                m["_strength"] = case_outcome["strength"] if case_outcome else "Unknown"
                grouped[key].append(m)
                if m.get("case_name"):
                    cases.add(m["case_name"])

    # Add only the gap cases from batch stores (cases with no per-case store).
    for path, data, gap_cases in batch_selected:
        for theme_id, theme_data in data.items():
            if not isinstance(theme_data, dict):
                continue
            theme_labels[theme_id] = theme_data.get("theme_label") or theme_id

            for m in extract_all_matches({theme_id: theme_data}):
                if m.get("case_name") not in gap_cases:
                    continue
                effect = m.get("effect") or "UNKNOWN"
                case_effect = m.get("case_effect") or "UNKNOWN"
                confidence = m.get("confidence") or "UNKNOWN"
                key = (theme_id, effect.upper(), case_effect.upper(), confidence.upper())
                case_outcome = case_outcome_map.get(m.get("case_name") or "")
                m["_winner"] = case_outcome["winner"] if case_outcome else "Unknown"
                m["_strength"] = case_outcome["strength"] if case_outcome else "Unknown"
                grouped[key].append(m)
                if m.get("case_name"):
                    cases.add(m["case_name"])

    return {
        "paths": [path for path, _data, _case_names in selected_entries],
        "grouped": dict(grouped),
        "theme_labels": theme_labels,
        "cases": cases,
        "n_matches": sum(len(v) for v in grouped.values()),
        "case_outcome_map": case_outcome_map,
    }


def _is_batch_theme_store_path(path: Path) -> bool:
    """Detect run-level batch theme stores created as <run_id>_batch_<n>_cases."""
    parts = path.parent.name.rsplit("_batch_", 1)
    if len(parts) != 2 or not parts[1].endswith("_cases"):
        return False
    return parts[1][:-len("_cases")].isdigit()


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #0b1020 0%, #10182b 100%);
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2rem;
            max-width: none;
        }
        [data-testid="stTabs"] {
            padding-top: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            padding-bottom: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            font-size: 0.95rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        .app-shell {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem 1.1rem 1.1rem;
            box-shadow: 0 14px 40px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(16,185,129,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .eyebrow {
            color: #93c5fd;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0.15rem 0 0.45rem 0;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 0;
        }
        .mini-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 92px;
        }
        .mini-card.good { border-left: 4px solid #16a34a; }
        .mini-card.warn { border-left: 4px solid #d97706; }
        .mini-card.bad  { border-left: 4px solid #dc2626; }
        .mini-card.info { border-left: 4px solid #2563eb; }
        .mini-kicker {
            color: #94a3b8;
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .mini-value {
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .mini-note {
            color: #cbd5e1;
            font-size: 0.88rem;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .section-sub {
            color: #94a3b8;
            margin-bottom: 0.65rem;
        }
        .badge-good {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(16,185,129,0.18);
            color: #86efac;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-warn {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(245,158,11,0.18);
            color: #fcd34d;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-info {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-bad {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(239,68,68,0.18);
            color: #fca5a5;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge-good, .badge-warn, .badge-info, .badge-bad {
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
        }
        .arg-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 0.75rem 0.82rem;
            margin: 0.5rem 0;
        }
        .arg-title {
            font-weight: 800;
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }
        .arg-body {
            color: #dbeafe;
            font-size: 0.9rem;
            line-height: 1.4;
            margin: 0.35rem 0 0.45rem 0;
        }
        .arg-muted {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.35;
        }
        .arg-chip {
            display: inline-block;
            margin: 0.1rem 0.14rem 0.1rem 0;
            padding: 0.14rem 0.38rem;
            border-radius: 999px;
            background: rgba(148,163,184,0.14);
            color: #cbd5e1;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .guidance-empty {
            color: #94a3b8;
            font-size: 0.9rem;
            padding-top: 0.2rem;
        }
        .guidance-row {
            display: grid;
            grid-template-columns: minmax(150px, 210px) 1fr;
            gap: 0.9rem;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
        }
        .guidance-row.lead { border-left: 4px solid #16a34a; }
        .guidance-row.refine { border-left: 4px solid #2563eb; }
        .guidance-row.watch { border-left: 4px solid #f59e0b; }
        .guidance-row.quarantine { border-left: 4px solid #dc2626; }
        .guidance-row.park { border-left: 4px solid #64748b; }
        .guidance-row-head {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            align-items: flex-start;
            min-width: 0;
        }
        .guidance-row-title {
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.25;
        }
        .guidance-row-count {
            flex: 0 0 auto;
            border-radius: 999px;
            background: rgba(148,163,184,0.16);
            color: #e2e8f0;
            font-weight: 800;
            font-size: 0.8rem;
            padding: 0.16rem 0.48rem;
        }
        .guidance-row-body {
            min-width: 0;
        }
        .guidance-row-item {
            color: #dbeafe;
            font-size: 0.9rem;
            line-height: 1.35;
            padding: 0.38rem 0;
            border-bottom: 1px solid rgba(148,163,184,0.16);
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .guidance-row-item:last-child {
            border-bottom: 0;
        }
        @media (max-width: 900px) {
            .guidance-row {
                grid-template-columns: 1fr;
            }
        }
        [data-testid="stExpander"] details {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary {
            color: #e2e8f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def mini_card_html(kicker: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        f"<div class='mini-card{tone_class}'>"
        f"<div class='mini-kicker'>{escape(kicker)}</div>"
        f"<div class='mini-value'>{escape(value)}</div>"
        f"<div class='mini-note'>{escape(note)}</div>"
        "</div>"
    )


def badge(label: str, kind: str = "info") -> str:
    return f"<span class='badge-{kind}'>{escape(label)}</span>"


def arg_card_html(
    case_name: str,
    summary: str,
    relevance: str,
    causal_reason: str,
    refs: str,
    winner: str = "",
    strength: str = "",
    context: str = "",
    badges_html: str = "",
) -> str:
    chips = "".join(
        f"<span class='arg-chip'>{escape(c)}</span>"
        for c in [f"para {refs}"] if c.strip()
    )
    outcome_bits = []
    if winner and winner != "Unknown":
        winner_colour = "#86efac" if winner == "Employee" else "#fca5a5" if winner == "Employer" else "#fcd34d"
        outcome_bits.append(f"<span style='color:{winner_colour};font-weight:700;font-size:0.78rem;'>{escape(winner)} won</span>")
    if strength and strength != "Unknown":
        outcome_bits.append(f"<span class='arg-chip'>{escape(strength)}</span>")
    outcome_html = f"<div style='margin-bottom:0.3rem;'>{'&nbsp;'.join(outcome_bits)}</div>" if outcome_bits else ""
    context_html = f"<div class='arg-muted' style='margin-bottom:0.3rem;'>{escape(context)}</div>" if context else ""
    badge_html = f"<div style='margin-bottom:0.45rem;'>{badges_html}</div>" if badges_html else ""
    return (
        f"<div class='arg-card'>"
        f"{context_html}"
        f"{badge_html}"
        f"{outcome_html}"
        f"<div class='arg-title'>{escape(case_name)}</div>"
        f"<div class='arg-body'>{escape(summary or '-')}</div>"
        f"<div class='arg-muted'><b>WS relevance:</b> {escape(relevance or '-')}</div>"
        f"<div class='arg-muted'><b>Causal reason:</b> {escape(causal_reason or '-')}</div>"
        f"<div style='margin-top:0.35rem'>{chips}</div>"
        "</div>"
    )


def guidance_row_html(title: str, items: List[str], tone: str) -> str:
    shown_items = items[:12]
    if shown_items:
        body = "".join(f"<div class='guidance-row-item'>{escape(item)}</div>" for item in shown_items)
        if len(items) > len(shown_items):
            body += f"<div class='guidance-row-item'>+ {len(items) - len(shown_items)} more</div>"
    else:
        body = "<div class='guidance-empty'>None</div>"
    return (
        f"<div class='guidance-row {tone}'>"
        "<div class='guidance-row-head'>"
        f"<div class='guidance-row-title'>{escape(title)}</div>"
        f"<div class='guidance-row-count'>{len(items)}</div>"
        "</div>"
        f"<div class='guidance-row-body'>{body}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Single-case sections
# ---------------------------------------------------------------------------

def render_header(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> None:
    metadata = case_json.get("case_metadata", {})
    outcome = case_json.get("outcome_optimization", {})
    shortlist = (aggregation_json.get("case_shortlist") or [{}])[0]
    risk_summary = aggregation_json.get("risk_control_summary", {})
    case_title = escape(get_case_label(case_json))
    outcome_summary = escape(metadata.get("outcome") or "No outcome summary found.")

    st.markdown(
        f"""
        <div class='hero'>
          <div class='eyebrow'>Single-case outcome monitor</div>
          <div class='hero-title'>{case_title}</div>
          <p class='hero-sub'>{outcome_summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    liability = outcome.get("claimant_liability_outcome")
    polkey = outcome.get("polkey_reduction_pct")
    contrib = outcome.get("contributory_fault_pct")
    use_mode = shortlist.get("overall_use_mode")
    risk_use = risk_summary.get("recommended_use")

    cols = st.columns(5)
    cols[0].markdown(
        mini_card_html(
            "Liability result",
            humanize(liability),
            humanize(outcome.get("liability_outcome_strength_band")),
            "good" if liability == "WIN" else "warn",
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        mini_card_html(
            "Compensation risk",
            f"Polkey {percent_label(polkey)}",
            "Compensatory award risk from fair-dismissal inevitability.",
            "bad" if polkey else "info",
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        mini_card_html(
            "Conduct reduction",
            percent_label(contrib),
            "Basic/compensatory reduction risk from claimant conduct.",
            "bad" if contrib else "info",
        ),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        mini_card_html(
            "Best use",
            humanize(use_mode),
            f"Liability usefulness: {humanize(shortlist.get('liability_usefulness_band'))}",
            "info",
        ),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        mini_card_html(
            "Risk handling",
            humanize(risk_use),
            f"{risk_summary.get('risk_control_signal_count', 0)} risk-control signals",
            "bad" if risk_use == "QUARANTINE" else "info",
        ),
        unsafe_allow_html=True,
    )

    claims = metadata.get("claims", [])
    claims_str = ", ".join(claims) if isinstance(claims, list) else str(claims or "-")
    lbl = "color:#94a3b8;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;font-weight:700;margin-right:0.4rem"
    val = "font-size:0.9rem;color:#e2e8f0"
    st.markdown(
        f"<div style='display:flex;gap:2.5rem;padding:0.5rem 0 0.25rem 0;flex-wrap:wrap;'>"
        f"<span><span style='{lbl}'>Date</span><span style='{val}'>{escape(str(metadata.get('judgment_date', '-')))}</span></span>"
        f"<span><span style='{lbl}'>Claims</span><span style='{val}'>{escape(claims_str)}</span></span>"
        f"<span><span style='{lbl}'>Transferability</span><span style='{val}'>{escape(humanize(outcome.get('transferability_rating')))}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_source_panel(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> None:
    warnings = validate_pair(case_json, aggregation_json)
    for w in warnings:
        st.warning(w)
    st.caption("Read-only monitor. The Witness Statement is not rewritten or changed by this app.")


def render_case_intelligence(case_json: Dict[str, Any]) -> None:
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Case Intelligence</div>", unsafe_allow_html=True)

    outcome = case_json.get("outcome_optimization", {})
    relevance = case_json.get("case_relevance_to_ws", {})

    with st.expander("Why this case matters", expanded=True):
        st.write(relevance.get("overall_relevance_to_ws") or relevance.get("overall_similarity_to_ws") or "No relevance narrative found.")
        limits = relevance.get("transferability_limits", [])
        if limits:
            st.write("**Limits on use**")
            for item in limits:
                st.write(f"- {item}")

    left, right = st.columns(2)
    with left:
        st.subheader("Use profile")
        st.markdown("\n".join([
            f"- **Factual fit:** {humanize(outcome.get('factual_proximity'))}",
            f"- **Transferability:** {humanize(outcome.get('transferability_rating'))}",
            f"- **Liability outcome:** {humanize(outcome.get('claimant_liability_outcome'))}",
            f"- **Remedy status:** {humanize(outcome.get('remedy_status'))}",
        ]))
    with right:
        st.subheader("Remedy and risk notes")
        st.write(outcome.get("award_reduction_notes", "No remedy note found."))
        st.subheader("Use instruction")
        st.write(outcome.get("optimization_notes", "No optimization note found."))

    st.markdown("</div>", unsafe_allow_html=True)


def render_judgment_signals(case_json: Dict[str, Any]) -> None:
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Judgment Signals</div>", unsafe_allow_html=True)

    outcome = case_json.get("outcome_optimization", {})
    weights = signal_weight_map(case_json)
    signals = case_json.get("judgment_signals", [])

    if not signals:
        st.caption("No judgment signals recorded.")
    for signal in signals:
        signal_id = signal.get("signal_id")
        weight = weights.get(signal_id, {})
        causal = weight.get("causal_weight", "")
        causal_label = humanize(causal) if causal else "—"
        summary_short = (signal.get("signal_summary") or "Signal")[:75]
        title = f"{signal_id}  ·  {causal_label}  ·  {summary_short}"
        with st.expander(title):
            causal_tone = "good" if causal == "DECISIVE" else "warn" if causal == "CONTRIBUTING" else "info"
            st.markdown(
                f"{badge(causal_label, causal_tone)} "
                f"{badge(humanize(signal.get('recommended_action', '-')), 'info')} "
                f"{badge(humanize(signal.get('case_effect', '-')), 'info')} "
                f"{badge(humanize(signal.get('dictionary_match_confidence', '-')), 'info')}",
                unsafe_allow_html=True,
            )
            st.write(f"**Theme:** {signal.get('mapped_theme_id', '-')}")
            st.write(f"**Summary:** {signal.get('signal_summary', '-')}")
            st.write(f"**Relevance to WS:** {signal.get('relevance_to_ws', '-')}")
            st.write(f"**Causal weight reason:** {weight.get('causal_weight_reason', '-')}")
            refs = signal.get("judgment_references", [])
            st.caption(f"Judgment references: {', '.join(map(str, refs)) if refs else '-'}")

    st.markdown("<div class='section-title' style='margin-top:1rem'>Adverse Signals</div>", unsafe_allow_html=True)
    flags = outcome.get("negative_theme_flags", [])
    if flags:
        st.dataframe([{
            "Theme": f.get("theme_id"),
            "Pattern": humanize(f.get("negative_pattern")),
            "Severity": humanize(f.get("severity")),
            "Reason": f.get("reason"),
        } for f in flags], use_container_width=True, hide_index=True)
    else:
        st.caption("No adverse signals recorded.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_optimization_interpretation(aggregation_json: Dict[str, Any]) -> None:
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Theme Guidance</div>", unsafe_allow_html=True)

    metadata = aggregation_json.get("aggregation_metadata", {})
    counts = recommendation_counts(aggregation_json)

    cols = st.columns(5)
    cols[0].markdown(mini_card_html("Maximise", str(counts.get("REINFORCE_PRIMARY", 0) + counts.get("REINFORCE_SUPPORTING", 0)), "Themes to lead with", "good"), unsafe_allow_html=True)
    cols[1].markdown(mini_card_html("Refine", str(counts.get("REFRAME", 0)), "Useful, needs framing", "info"), unsafe_allow_html=True)
    cols[2].markdown(mini_card_html("Minimise", str(counts.get("MONITOR", 0) + counts.get("AVOID", 0)), "Use carefully", "warn"), unsafe_allow_html=True)
    cols[3].markdown(mini_card_html("Quarantine", str(counts.get("RISK_CONTROL", 0)), "Legal review only", "bad"), unsafe_allow_html=True)
    cols[4].markdown(mini_card_html("Ignore", str(counts.get("NO_SIGNAL", 0)), "No signal from this case"), unsafe_allow_html=True)

    with st.expander("Technical scoring details", expanded=False):
        st.write(f"Scoring profile: {metadata.get('scoring_profile_version', '-')}")
        st.write(f"Threshold profile: {metadata.get('threshold_profile', '-')}")
        st.write(f"Case count: {metadata.get('case_count', '-')}")
        st.write(f"Min cases for primary reinforcement: {metadata.get('min_primary_cases', '-')}")

    tabs = st.tabs(["Maximise", "Refine", "Minimise", "Quarantine", "Ignore"])
    with tabs[0]:
        st.markdown('<div class="section-sub">Strong themes to lead with. In single-case pilot mode this may be empty — primary reinforcement needs more than one case.</div>', unsafe_allow_html=True)
        render_rows(
            get_theme_rows(aggregation_json, "REINFORCE_PRIMARY") + get_theme_rows(aggregation_json, "REINFORCE_SUPPORTING"),
            "No themes currently meet reinforce thresholds for this case.",
        )
    with tabs[1]:
        st.markdown('<div class="section-sub">Useful positive analogies, not strong enough to treat as proven reinforcement on their own.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "REFRAME"), "No themes require reframing.")
    with tabs[2]:
        st.markdown('<div class="section-sub">Themes with adverse or qualified signals. Use carefully or avoid broad claims.</div>', unsafe_allow_html=True)
        render_rows(
            get_theme_rows(aggregation_json, "MONITOR") + get_theme_rows(aggregation_json, "AVOID"),
            "No minimise/avoid themes identified.",
        )
    with tabs[3]:
        st.markdown('<div class="section-sub">Risk-control material belongs in legal review, not in the Witness Statement narrative.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "RISK_CONTROL"), "No risk-control theme identified.")
        st.subheader("Risk summary")
        risk = aggregation_json.get("risk_control_summary", {})
        polkey_n = risk.get("polkey_risk_cases", 0)
        contrib_n = risk.get("contribution_risk_cases", 0)
        low_rem_n = risk.get("low_remedy_win_cases", 0)
        rec_use = humanize(risk.get("recommended_use"))
        rc = st.columns(4)
        rc[0].markdown(mini_card_html("Polkey risk cases", str(polkey_n), "Cases with Polkey reduction", "bad" if polkey_n else "info"), unsafe_allow_html=True)
        rc[1].markdown(mini_card_html("Contribution risk cases", str(contrib_n), "Cases with conduct finding", "bad" if contrib_n else "info"), unsafe_allow_html=True)
        rc[2].markdown(mini_card_html("Low remedy cases", str(low_rem_n), "Low compensation outcomes", "warn" if low_rem_n else "info"), unsafe_allow_html=True)
        rc[3].markdown(mini_card_html("Recommended use", rec_use, "Risk handling guidance", "bad" if risk.get("recommended_use") == "QUARANTINE" else "info"), unsafe_allow_html=True)
        patterns = risk.get("negative_pattern_counts", {})
        if patterns:
            st.dataframe([{"Pattern": humanize(p), "Count": c} for p, c in patterns.items()], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.markdown('<div class="section-sub">Dictionary themes unaffected by this case — not positive or negative points from this judgment.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "NO_SIGNAL"), "No unaffected dictionary themes.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_guidance(aggregation_json: Dict[str, Any]) -> None:
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Strategy Brief</div>", unsafe_allow_html=True)

    guidance = guidance_from_aggregation(aggregation_json)
    cards = [
        ("Lead With", guidance.get("Maximise", []), "lead"),
        ("Refine", guidance.get("Refine", []), "refine"),
        ("Watch", guidance.get("Minimise", []), "watch"),
        ("Quarantine", guidance.get("Quarantine", []), "quarantine"),
        ("Ignore", guidance.get("Ignore", []), "park"),
    ]
    for title, items, tone in cards:
        st.markdown(guidance_row_html(title, items, tone), unsafe_allow_html=True)

    st.subheader("Review Queues")
    left, right = st.columns(2)
    with left:
        st.write("**Other negative pattern review**")
        queue = aggregation_json.get("other_negative_pattern_review_queue", [])
        if queue:
            st.dataframe(queue, use_container_width=True, hide_index=True)
        else:
            st.caption("No OTHER negative patterns.")
    with right:
        st.write("**Pilot review points**")
        report = aggregation_json.get("pilot_review_report", {})
        review_points = report.get("review_points", [])
        if review_points:
            for point in review_points:
                st.write(f"- {point}")
        else:
            st.caption("No pilot review points.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_theme_store(theme_store_json: Optional[Dict[str, Any]]) -> None:
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Case Review Queue</div>", unsafe_allow_html=True)

    if not theme_store_json:
        st.info("No theme_store.json loaded. Select one in the sidebar to inspect the deterministic review layer.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    theme_rows = summarize_theme_store(theme_store_json)
    match_rows = flatten_theme_store_matches(theme_store_json)
    review_rows = [r for r in match_rows if str(r.get("Review status", "")).lower() in {"unreviewed", "review manually"}]

    effect_counts: Dict[str, int] = {}
    for row in match_rows:
        eff = str(row.get("Effect") or "UNKNOWN")
        effect_counts[eff] = effect_counts.get(eff, 0) + 1

    n_reinforce = effect_counts.get("Reinforce", 0)
    n_review = effect_counts.get("Review Manually", 0)
    n_risk = sum(1 for r in match_rows if "T20" in str(r.get("Theme ID", "")))
    mc = st.columns(5)
    mc[0].markdown(mini_card_html("Themes", str(len(theme_rows)), "With at least one match"), unsafe_allow_html=True)
    mc[1].markdown(mini_card_html("Total matches", str(len(match_rows)), "Across all themes"), unsafe_allow_html=True)
    mc[2].markdown(mini_card_html("Reinforce", str(n_reinforce), "Direct reinforcement signals", "good"), unsafe_allow_html=True)
    mc[3].markdown(mini_card_html("Review manually", str(n_review), "Needs manual check", "warn"), unsafe_allow_html=True)
    mc[4].markdown(mini_card_html("Risk control", str(n_risk), "T20 risk signals", "bad" if n_risk else "info"), unsafe_allow_html=True)

    tabs = st.tabs(["Summary", "Review Queue", "Theme Buckets"])

    with tabs[0]:
        st.markdown('<div class="section-sub">Themes are the routing buckets; the group key (effect · case effect · confidence) keeps reinforce material separate from manual-review or risk material.</div>', unsafe_allow_html=True)
        st.dataframe(theme_rows, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown('<div class="section-sub">Flat working queue sorted by review priority. Filter by effect and theme to focus your review.</div>', unsafe_allow_html=True)
        st.caption(f"{len(review_rows)} unreviewed rows before filtering.")

        effect_filter = st.multiselect(
            "Effect",
            sorted({str(r.get("Effect")) for r in match_rows if r.get("Effect")}),
            default=sorted({str(r.get("Effect")) for r in match_rows if str(r.get("Effect", "")).lower() == "review manually"}),
        )
        theme_filter = st.selectbox(
            "Theme",
            ["All"] + [f"{r['Theme ID']} - {r['Theme']}" for r in theme_rows],
        )
        search_text = st.text_input("Search summaries", value="")

        filtered = match_rows
        if effect_filter:
            filtered = [r for r in filtered if r.get("Effect") in effect_filter]
        if theme_filter != "All":
            tid = theme_filter.split(" - ", 1)[0]
            filtered = [r for r in filtered if r.get("Theme ID") == tid]
        if search_text.strip():
            needle = search_text.strip().lower()
            filtered = [r for r in filtered if
                needle in str(r.get("Summary") or "").lower()
                or needle in str(r.get("Theme") or "").lower()
                or needle in str(r.get("Source") or "").lower()]

        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tabs[2]:
        for theme_id, theme_data in theme_store_json.items():
            if not isinstance(theme_data, dict):
                continue
            title = f"{theme_id}: {theme_data.get('theme_label') or theme_id}"
            with st.expander(title):
                rank_data = theme_data.get("theme_rank_data") or {}
                st.write(
                    f"**Matches:** {theme_data.get('n_matches', 0)} | "
                    f"**Recommendation:** {humanize(rank_data.get('recommendation'))} | "
                    f"**Net score:** {format_number(rank_data.get('net_theme_score'))}"
                )
                groups = theme_data.get("groups") or {}
                for gkey, gdata in groups.items():
                    st.subheader(gkey)
                    rows = [{"Case": m.get("case_name"), "Effect": humanize(m.get("effect")), "Case effect": humanize(m.get("case_effect")), "Confidence": humanize(m.get("confidence")), "Rank": m.get("rank_score"), "Source": m.get("source_pointer"), "Summary": m.get("summary")} for m in (gdata.get("matches") or [])]
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Corpus tab
# ---------------------------------------------------------------------------

def render_corpus_theme_store() -> None:
    sig = _corpus_dir_signature(DEFAULT_THEME_STORE_DIR, DEFAULT_CASE_DIR)
    corpus = _load_corpus_cached(DEFAULT_THEME_STORE_DIR, DEFAULT_CASE_DIR, sig)
    paths = corpus["paths"]
    grouped = corpus["grouped"]
    theme_labels = corpus["theme_labels"]
    cases = corpus["cases"]
    n_matches = corpus["n_matches"]

    n_themes = len({k[0] for k in grouped})
    n_reinforce = sum(len(v) for k, v in grouped.items() if k[1] == "REINFORCE")
    n_win_drivers = sum(len(v) for k, v in grouped.items() if k[2] == "WIN_DRIVER")

    st.markdown(
        f"""
        <div class='hero'>
          <div class='eyebrow'>Argument Library</div>
          <div class='hero-title'>Cross-case argument library</div>
          <p class='hero-sub'>Arguments from all processed judgments grouped by (theme · effect · case effect · confidence). No deduplication — every argument is kept.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mc = st.columns(4)
    mc[0].markdown(mini_card_html("Cases", str(len(cases)), f"{len(paths)} theme-store files"), unsafe_allow_html=True)
    mc[1].markdown(mini_card_html("Total arguments", str(n_matches), "Across all cases and themes"), unsafe_allow_html=True)
    mc[2].markdown(mini_card_html("Themes covered", str(n_themes), "Distinct themes with at least one match"), unsafe_allow_html=True)
    mc[3].markdown(mini_card_html("Win-driver arguments", str(n_win_drivers), f"{n_reinforce} reinforce-lane arguments", "good"), unsafe_allow_html=True)

    if not grouped:
        st.info("No theme_store.json files found in output/theme_store/. Run the pipeline to generate them.")
        return

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)

    all_themes = sorted({k[0] for k in grouped}, key=_theme_numeric_sort_key)
    all_effects = sorted({k[1] for k in grouped}, key=lambda x: EFFECT_RANK.get(x, 99))
    all_case_effects = sorted({k[2] for k in grouped}, key=lambda x: CASE_EFFECT_RANK.get(x, 99))
    all_confidences = sorted({k[3] for k in grouped}, key=lambda x: CONFIDENCE_RANK.get(x, 99))
    all_winners = sorted({m["_winner"] for v in grouped.values() for m in v}, key=lambda x: WINNER_RANK.get(x, 99))
    all_strengths = sorted({m["_strength"] for v in grouped.values() for m in v}, key=lambda x: STRENGTH_RANK.get(x, 99))

    # Consume section jump (multi-theme with presets) or legacy single-theme jump
    jump_themes = st.session_state.pop("_corpus_jump_themes", None)
    jump_winner = st.session_state.pop("_corpus_jump_winner", None)
    jump_effects = st.session_state.pop("_corpus_jump_effects", None)
    jump_confidence = st.session_state.pop("_corpus_jump_confidence", None)
    legacy_theme = st.session_state.pop("_corpus_jump_theme", None)
    if legacy_theme and not jump_themes:
        jump_themes = [legacy_theme]
    if jump_themes:
        valid = [t for t in jump_themes if t in all_themes]
        if valid:
            st.session_state["corpus_theme_filter"] = valid
            st.session_state["corpus_winner_filter"] = [w for w in (jump_winner or []) if w in all_winners]
            st.session_state["corpus_effect_filter"] = [e for e in (jump_effects or []) if e in all_effects]
            st.session_state["corpus_confidence_filter"] = [c for c in (jump_confidence or []) if c in all_confidences]
            st.session_state["corpus_case_effect_filter"] = []
            st.session_state["corpus_strength_filter"] = []
            st.session_state["corpus_strong_only"] = False
            st.session_state["corpus_search"] = ""
            st.session_state["_corpus_page"] = 1
    else:
        st.session_state.setdefault("corpus_theme_filter", [all_themes[0]] if all_themes else [])
        st.session_state.setdefault("corpus_effect_filter", [v for v in ["REINFORCE"] if v in all_effects])
        st.session_state.setdefault("corpus_case_effect_filter", [v for v in ["WIN_DRIVER", "STRONG_SUPPORT"] if v in all_case_effects])
        st.session_state.setdefault("corpus_confidence_filter", [v for v in ["HIGH"] if v in all_confidences])
        st.session_state.setdefault("corpus_winner_filter", [v for v in ["Employee"] if v in all_winners])
        st.session_state.setdefault("corpus_strength_filter", [])
        st.session_state.setdefault("corpus_strong_only", False)
        st.session_state.setdefault("corpus_search", "")
        st.session_state.setdefault("corpus_results_per_page", CORPUS_ARGUMENT_PAGE_SIZE)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        theme_filter = st.multiselect("Theme", all_themes, key="corpus_theme_filter")
    with fc2:
        effect_filter = st.multiselect("Effect", all_effects, key="corpus_effect_filter")
    with fc3:
        case_effect_filter = st.multiselect("Case effect", all_case_effects, key="corpus_case_effect_filter")
    with fc4:
        confidence_filter = st.multiselect("Confidence", all_confidences, key="corpus_confidence_filter")

    fw1, fw2, fw3, fw4 = st.columns([1, 1, 1, 2])
    with fw1:
        winner_filter = st.multiselect("Who won", all_winners, key="corpus_winner_filter")
    with fw2:
        strength_filter = st.multiselect("Win strength", all_strengths, key="corpus_strength_filter")
    with fw3:
        strong_only = st.checkbox(
            "Strong matches only",
            key="corpus_strong_only",
            help="Uses an explicit strong flag when available; otherwise REINFORCE + WIN_DRIVER + HIGH + top rank.",
        )
    with fw4:
        search_corpus = st.text_input("Search argument summaries", key="corpus_search")
    page_size = st.selectbox(
        "Results per page",
        [10, CORPUS_ARGUMENT_PAGE_SIZE, 50, 100],
        key="corpus_results_per_page",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply group-level filters (theme, effect, case_effect, confidence)
    filtered_groups = {
        k: v for k, v in grouped.items()
        if (not theme_filter or k[0] in theme_filter)
        and (not effect_filter or k[1] in effect_filter)
        and (not case_effect_filter or k[2] in case_effect_filter)
        and (not confidence_filter or k[3] in confidence_filter)
    }

    # Apply per-match filters (winner, strength, search) — keeps groups that have at least one passing match
    if winner_filter or strength_filter or search_corpus.strip():
        needle = search_corpus.strip().lower()
        filtered_groups = {
            k: [
                m for m in v
                if (not winner_filter or m.get("_winner") in winner_filter)
                and (not strength_filter or m.get("_strength") in strength_filter)
                and (not needle or needle in str(m.get("summary") or "").lower() or needle in str(m.get("relevance_to_ws") or "").lower())
            ]
            for k, v in filtered_groups.items()
        }
        filtered_groups = {k: v for k, v in filtered_groups.items() if v}

    if strong_only:
        filtered_groups = {
            k: [m for m in v if _is_strong_argument(m, k)]
            for k, v in filtered_groups.items()
        }
        filtered_groups = {k: v for k, v in filtered_groups.items() if v}

    if not filtered_groups:
        st.info("No arguments match the current filters.")
        return

    argument_items = _flatten_argument_items(filtered_groups)
    n_groups = len(filtered_groups)
    total_shown = len(argument_items)
    n_strong = sum(1 for key, match in argument_items if _is_strong_argument(match, key))
    n_pages = max(1, (total_shown + int(page_size) - 1) // int(page_size))

    # Reset to page 1 when filters change
    filter_state = str((sorted(theme_filter), sorted(effect_filter), sorted(case_effect_filter),
                        sorted(confidence_filter), sorted(winner_filter), sorted(strength_filter),
                        strong_only, search_corpus, page_size))
    if st.session_state.get("_corpus_filter_state") != filter_state:
        st.session_state["_corpus_filter_state"] = filter_state
        st.session_state["_corpus_page"] = 1

    page = max(1, min(int(st.session_state.get("_corpus_page", 1)), n_pages))
    page_start = (page - 1) * int(page_size)
    page_items = argument_items[page_start: page_start + int(page_size)]

    st.caption(
        f"{total_shown} arguments · {n_strong} strong · {n_groups} groups · "
        f"page {page} of {n_pages}"
    )

    if n_pages > 1:
        _corpus_paginator(page, n_pages, key_prefix="corpus_pg_top")
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    for key, m in page_items:
        theme_id, effect, case_effect, confidence = key
        theme_label = theme_labels.get(theme_id) or theme_id

        effect_badge_kind = "good" if effect == "REINFORCE" else "warn" if effect == "REVIEW_MANUALLY" else "info"
        ce_badge_kind = "good" if case_effect == "WIN_DRIVER" else "warn" if case_effect in {"MODERATE_SUPPORT", "WEAK_SUPPORT"} else "bad" if case_effect == "ADVERSE" else "info"
        conf_badge_kind = "good" if confidence == "HIGH" else "warn" if confidence == "MEDIUM" else "info"

        strong_badge = badge("STRONG MATCH", "good") if _is_strong_argument(m, key) else ""
        rank_label = f"rank {m.get('rank_score')}" if m.get("rank_score") is not None else ""
        priority_label = f"priority {m.get('review_priority_score')}" if m.get("review_priority_score") is not None else ""
        score_badges = "".join(
            badge(label, "info")
            for label in [rank_label, priority_label]
            if label
        )
        badges_html = (
            f"{strong_badge}"
            f"{badge(effect, effect_badge_kind)} "
            f"{badge(case_effect, ce_badge_kind)} "
            f"{badge(confidence, conf_badge_kind)}"
            f"{score_badges}"
        )

        st.markdown(
            arg_card_html(
                m.get("case_name") or "Unknown",
                m.get("summary") or "",
                m.get("relevance_to_ws") or "",
                m.get("causal_weight_reason") or "",
                m.get("paragraph_reference") or m.get("source_pointer") or "",
                winner=m.get("_winner", ""),
                strength=m.get("_strength", ""),
                context=f"{theme_id} — {theme_label}",
                badges_html=badges_html,
            ),
            unsafe_allow_html=True,
        )

    if n_pages > 1:
        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        _corpus_paginator(page, n_pages, key_prefix="corpus_pg_bottom")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# WS Strategy tab
# ---------------------------------------------------------------------------

def _outcome_dir_sig(outcome_dir: Path) -> Tuple:
    """Cheap signature for the outcome_optimized directory (count + newest mtime)."""
    files = sorted(outcome_dir.glob("*.json")) if outcome_dir.exists() else []
    if not files:
        return (0,)
    return (len(files), max(f.stat().st_mtime for f in files))


def _theme_store_dir_sig(ts_dir: Path) -> Tuple:
    """Cheap signature for the theme_store directory (file count + newest mtime)."""
    files = list(ts_dir.rglob("theme_store.json")) if ts_dir.exists() else []
    if not files:
        return (0,)
    return (len(files), max(f.stat().st_mtime for f in files))


@st.cache_data(show_spinner=False)
def _compute_unified_corpus_cached(outcome_dir_sig: Tuple, dict_path_str: str) -> Dict[str, Any]:
    """Aggregate ALL outcome_optimized files into one corpus. Invalidates when files change."""
    files = sorted(DEFAULT_CASE_DIR.glob("*.json")) if DEFAULT_CASE_DIR.exists() else []
    cases, filenames = [], {}
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        filenames[len(cases)] = path.name
        cases.append(data)
    if not cases:
        return {}
    dictionary = load_dictionary(dict_path_str)
    return aggregate_outcome_optimized_cases(cases, dictionary)


@st.cache_data(show_spinner=False)
def _build_watch_items_cached(ts_dir_sig: Tuple, outcome_dir_sig: Tuple) -> Dict[str, Any]:
    """Per-theme employer-won counts from ALL theme_store files + outcome_optimized files."""
    loss_cases: set = set()
    win_cases: set = set()
    if DEFAULT_CASE_DIR.exists():
        for f in DEFAULT_CASE_DIR.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = d.get("case_metadata", {})
            oo = d.get("outcome_optimization", {})
            name = meta.get("case_name")
            if not name:
                continue
            lib = (oo.get("claimant_liability_outcome") or "").upper()
            if lib == "LOSS":
                loss_cases.add(name)
            elif lib == "WIN":
                win_cases.add(name)

    employer_cases: Dict[str, set] = defaultdict(set)
    employee_cases: Dict[str, set] = defaultdict(set)
    theme_labels: Dict[str, str] = {}

    for ts_file in DEFAULT_THEME_STORE_DIR.rglob("theme_store.json"):
        try:
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for theme_id, theme_data in ts.items():
            if not isinstance(theme_data, dict):
                continue
            theme_labels[theme_id] = theme_data.get("theme_label") or theme_id
            for group in theme_data.get("groups", {}).values():
                for m in group.get("matches", []):
                    cn = m.get("case_name") or ""
                    if cn in loss_cases:
                        employer_cases[theme_id].add(cn)
                    elif cn in win_cases:
                        employee_cases[theme_id].add(cn)

    items = [
        {
            "theme_id": tid,
            "theme_label": theme_labels.get(tid, tid),
            "employer_won": len(employer_cases[tid]),
            "employee_won": len(employee_cases[tid]),
        }
        for tid in set(list(employer_cases.keys()) + list(employee_cases.keys()))
        if len(employer_cases[tid]) >= 2
    ]
    items.sort(key=lambda x: -x["employer_won"])

    return {"items": items, "n_loss": len(loss_cases), "n_win": len(win_cases)}


def _render_strategy_section(
    title: str,
    themes: List[Dict],
    color: str,
    instruction: str,
    total_cases: int,
    winner_preset: List[str],
    effects_preset: List[str],
    confidence_preset: List[str],
) -> None:
    """Render one strategy section: header + one Explore button + compact dataframe."""
    theme_ids = [t["theme_id"] for t in themes]
    hdr_col, btn_col = st.columns([8, 2])
    with hdr_col:
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:0.5rem 0 0.4rem 0.85rem;"
            f"margin:0.6rem 0 0.4rem 0;'>"
            f"<span style='font-weight:800;font-size:1.05rem;'>{title}</span>"
            f"<span style='font-weight:400;font-size:0.82rem;color:#94a3b8;margin-left:0.55rem;'>"
            f"{len(themes)} theme{'s' if len(themes) != 1 else ''}</span>"
            f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:0.15rem;'>{instruction}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        winner_hint = f" · winner: {', '.join(winner_preset)}" if winner_preset else ""
        if st.button(
            "Explore →",
            key=f"str_explore_{_key_part(title)}",
            use_container_width=True,
            help=f"Open Argument Library with all {len(themes)} theme(s) selected{winner_hint}.",
        ):
            _open_argument_library_for_section(theme_ids, winner_preset, effects_preset, confidence_preset)

    rows = [
        {
            "Theme": f"{t['theme_id']} — {t['theme_name']}",
            "Cases": t["supporting_case_count"],
            "Coverage": f"{round(t['supporting_case_count'] / total_cases * 100)}%" if total_cases else "—",
            "Score": round(t["net_theme_score"], 2),
            "Decisive": t["decisive_signal_count"],
            "Contributing": t["contributing_signal_count"],
            "High conf.": t["high_confidence_case_count"],
        }
        for t in themes
    ]
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Theme": st.column_config.TextColumn(width="large"),
            "Cases": st.column_config.NumberColumn(
                width="small",
                help="Number of corpus cases where this theme appeared with at least one judgment signal.",
            ),
            "Coverage": st.column_config.TextColumn(
                width="small",
                help="Cases / total corpus size — how broadly this theme appears across all judgments.",
            ),
            "Score": st.column_config.NumberColumn(
                width="small",
                format="%.2f",
                help="Net theme score = total positive signal weight minus negative penalties across all cases. Higher is stronger.",
            ),
            "Decisive": st.column_config.NumberColumn(
                width="small",
                help="Tribunal findings where this theme directly drove the outcome. A single case can have 3–5 separate findings (liability, wrongful dismissal, Polkey, contribution…), so this will often exceed Cases.",
            ),
            "Contributing": st.column_config.NumberColumn(
                width="small",
                help="Tribunal findings where this theme supported the outcome but wasn't the deciding factor. Can exceed Cases for the same reason as Decisive.",
            ),
            "High conf.": st.column_config.NumberColumn(
                width="small",
                help="Cases with at least one HIGH confidence match — strong dictionary alignment with a judgment passage.",
            ),
        },
    )


def render_ws_strategy_tab() -> None:
    od_sig = _outcome_dir_sig(DEFAULT_CASE_DIR)
    total_cases = od_sig[0] if od_sig[0] else 0

    if not total_cases:
        st.info("No processed cases found. Run a batch to generate cross-case WS intelligence.")
        return

    dict_path = PROJECT_ROOT / "input" / "dictionary" / "WS_Controlled_Theme_Dictionary_v1_2_final.json"
    if not dict_path.exists():
        st.error(f"Dictionary not found at {dict_path}")
        return

    with st.spinner(f"Building unified corpus from {total_cases} cases…"):
        agg = _compute_unified_corpus_cached(od_sig, str(dict_path))

    if not agg:
        st.warning("Could not build corpus aggregation.")
        return

    tsm = agg.get("theme_strength_matrix", [])
    risk = agg.get("risk_control_summary", {})

    # Pre-group so we only iterate tsm once
    by_rec: Dict[str, List[Dict]] = {}
    for t in tsm:
        by_rec.setdefault(t["recommendation"], []).append(t)

    n_primary   = len(by_rec.get("REINFORCE_PRIMARY", []))
    n_supporting = len(by_rec.get("REINFORCE_SUPPORTING", []))
    n_reframe   = len(by_rec.get("REFRAME", []))
    n_monitor   = len(by_rec.get("MONITOR", []))
    n_adverse   = len(by_rec.get("AVOID", [])) + len(by_rec.get("RISK_CONTROL", []))

    # Watch items — compute early so we can show count in metrics
    watch_items: List[Dict] = []
    n_loss = n_win = 0
    if DEFAULT_THEME_STORE_DIR.exists():
        ts_sig = _theme_store_dir_sig(DEFAULT_THEME_STORE_DIR)
        watch_data = _build_watch_items_cached(ts_sig, od_sig)
        watch_items = watch_data.get("items", [])
        n_loss = watch_data.get("n_loss", 0)
        n_win  = watch_data.get("n_win", 0)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='hero'>"
        f"<div class='eyebrow'>WS Strategy Brief</div>"
        f"<div class='hero-title'>Cross-case calibration · {total_cases} judgments</div>"
        f"<p class='hero-sub'>All processed judgments in one unified corpus — updated automatically as new cases complete.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Metrics (4 cards) ─────────────────────────────────────────────────────
    mc = st.columns(4)
    mc[0].markdown(mini_card_html("Lead & support", str(n_primary), f"+ {n_supporting} supporting", "good"),   unsafe_allow_html=True)
    mc[1].markdown(mini_card_html("Frame carefully", str(n_reframe), f"+ {n_monitor} to monitor", "warn"),    unsafe_allow_html=True)
    mc[2].markdown(mini_card_html("Adverse", str(n_adverse), "Avoid + Quarantine signals", "bad" if n_adverse else "info"), unsafe_allow_html=True)
    mc[3].markdown(mini_card_html("Watch items", str(len(watch_items)), "Employer-won precedents", "warn" if watch_items else "info"), unsafe_allow_html=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.2rem 0 0 0;'>",
        unsafe_allow_html=True,
    )

    # ── Theme sections ────────────────────────────────────────────────────────
    # Columns: title, recs, color, instruction, collapsed, winner_preset, effects_preset, confidence_preset
    # winner/effects/confidence presets are applied when "Explore →" is clicked.
    SECTIONS = [
        ("Lead with",  ["REINFORCE_PRIMARY"],   "#16a34a", "Strong cross-case evidence — argue these confidently.",                  False, ["Employee"], ["REINFORCE"], ["HIGH"]),
        ("Reinforce",  ["REINFORCE_SUPPORTING"], "#2563eb", "Useful supporting evidence — include but don't lead.",                  False, ["Employee"], ["REINFORCE"], []),
        ("Refine",     ["REFRAME"],              "#f59e0b", "Mixed or qualified signals — frame carefully or distinguish.",          False, [],           [],            []),
        ("Monitor",    ["MONITOR"],              "#eab308", "Use only with strong document support; watch for employer rebuttal.",   True,  [],           [],            []),
        ("Avoid",      ["AVOID"],                "#dc2626", "Adverse signals — do not rely on these themes.",                       True,  ["Employer"], [],            []),
        ("Quarantine", ["RISK_CONTROL"],         "#b91c1c", "Risk-control material — legal review only, keep out of WS narrative.", True,  [],           [],            []),
        ("No signal",  ["NO_SIGNAL"],            "#64748b", "No cross-case evidence either way from this corpus.",                  True,  [],           [],            []),
    ]

    for title, recs, color, instruction, collapsed, w_pre, e_pre, c_pre in SECTIONS:
        themes = sorted(
            [t for t in tsm if t["recommendation"] in recs],
            key=lambda x: -x["net_theme_score"],
        )
        if not themes:
            continue
        if collapsed:
            with st.expander(
                f"{title}  ·  {len(themes)} theme{'s' if len(themes) != 1 else ''}",
                expanded=False,
            ):
                _render_strategy_section(title, themes, color, instruction, total_cases, w_pre, e_pre, c_pre)
        else:
            _render_strategy_section(title, themes, color, instruction, total_cases, w_pre, e_pre, c_pre)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.5rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── Risk profile (collapsed by default) ───────────────────────────────────
    if risk and any(risk.get(k) for k in ("polkey_risk_cases", "contribution_risk_cases", "negative_pattern_counts")):
        polkey_n = risk.get("polkey_risk_cases", 0)
        contrib_n = risk.get("contribution_risk_cases", 0)
        with st.expander(
            f"Risk profile  ·  Polkey {polkey_n}  ·  Conduct {contrib_n}",
            expanded=False,
        ):
            rc = st.columns(3)
            rec_use = humanize(risk.get("recommended_use"))
            rc[0].markdown(mini_card_html("Polkey risk", str(polkey_n), "Cases with Polkey reduction", "bad" if polkey_n else "info"), unsafe_allow_html=True)
            rc[1].markdown(mini_card_html("Conduct risk", str(contrib_n), "Cases with conduct finding", "bad" if contrib_n else "info"), unsafe_allow_html=True)
            rc[2].markdown(mini_card_html("Recommended use", rec_use, "Corpus-level risk handling", "bad" if risk.get("recommended_use") == "QUARANTINE" else "info"), unsafe_allow_html=True)
            patterns = risk.get("negative_pattern_counts") or {}
            if patterns:
                st.dataframe(
                    [{"Pattern": humanize(p), "Cases": c} for p, c in sorted(patterns.items(), key=lambda x: -x[1])],
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Watch items ───────────────────────────────────────────────────────────
    if DEFAULT_THEME_STORE_DIR.exists():
        hdr_col, btn_col = st.columns([8, 2])
        with hdr_col:
            st.markdown(
                "<div style='border-left:4px solid #f59e0b;padding:0.5rem 0 0.4rem 0.85rem;"
                "margin:0.6rem 0 0.4rem 0;'>"
                "<span style='font-weight:800;font-size:1.05rem;'>Watch</span>"
                f"<span style='font-weight:400;font-size:0.82rem;color:#94a3b8;margin-left:0.55rem;'>"
                f"{len(watch_items)} theme{'s' if len(watch_items) != 1 else ''} · employer-won precedents"
                f"  ·  {n_loss} employer-won cases / {n_win} employee-won</span>"
                "<div style='color:#94a3b8;font-size:0.85rem;margin-top:0.15rem;'>"
                "These themes appeared in cases the employer won. Prepare robust evidence before relying on them.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with btn_col:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if watch_items and st.button(
                "Explore →",
                key="str_explore_watch",
                use_container_width=True,
                help=f"Open Argument Library with all {len(watch_items)} watch theme(s) selected · winner: Employer.",
            ):
                _open_argument_library_for_section(
                    [w["theme_id"] for w in watch_items],
                    winner=["Employer"],
                    effects=[],
                    confidence=[],
                )

        if watch_items:
            watch_rows = [
                {
                    "Theme": f"{w['theme_id']} — {w['theme_label']}",
                    "Employer-won": w["employer_won"],
                    "Employee-won": w["employee_won"],
                    "Rate": f"{round(w['employer_won'] / n_loss * 100)}%" if n_loss else "—",
                }
                for w in watch_items
            ]
            st.dataframe(
                watch_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Theme": st.column_config.TextColumn(width="large"),
                    "Employer-won": st.column_config.NumberColumn(
                        width="small",
                        help=f"How many of the {n_loss} employer-won cases in the corpus featured this theme.",
                    ),
                    "Employee-won": st.column_config.NumberColumn(
                        width="small",
                        help=f"How many of the {n_win} employee-won cases in the corpus featured this theme.",
                    ),
                    "Rate": st.column_config.TextColumn(
                        width="small",
                        help=f"Employer-won appearances / {n_loss} total employer-won cases. Higher = the employer has stronger precedent to rebut this theme.",
                    ),
                },
            )
        else:
            st.info("No themes found in 2+ employer-won cases for this corpus.")

    # ── Download calibration brief ────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.5rem 0 1rem 0;'>",
        unsafe_allow_html=True,
    )

    REC_SECTION_KEY = {
        "REINFORCE_PRIMARY":   "lead_with",
        "REINFORCE_SUPPORTING":"reinforce",
        "REFRAME":             "refine",
        "MONITOR":             "monitor",
        "AVOID":               "avoid",
        "RISK_CONTROL":        "quarantine",
        "NO_SIGNAL":           "no_signal",
    }
    REC_GUIDANCE = {
        "REINFORCE_PRIMARY":   "Strong cross-case evidence — argue these confidently.",
        "REINFORCE_SUPPORTING":"Useful supporting evidence — include but don't lead.",
        "REFRAME":             "Mixed or qualified signals — frame carefully or distinguish.",
        "MONITOR":             "Use only with strong document support; watch for employer rebuttal.",
        "AVOID":               "Adverse signals — do not rely on these themes.",
        "RISK_CONTROL":        "Risk-control material — legal review only, keep out of WS narrative.",
        "NO_SIGNAL":           "No cross-case evidence either way from this corpus.",
    }

    brief: Dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "corpus_size": total_cases,
        "outcome_breakdown": {
            "employee_won": n_win,
            "employer_won": n_loss,
            "other": total_cases - n_win - n_loss,
        },
        "theme_recommendations": {v: [] for v in dict.fromkeys(REC_SECTION_KEY.values())},
        "watch_items": [],
        "risk_profile": {},
    }

    for t in sorted(tsm, key=lambda x: -x["net_theme_score"]):
        rec = t.get("recommendation", "")
        section_key = REC_SECTION_KEY.get(rec)
        if not section_key:
            continue
        brief["theme_recommendations"][section_key].append({
            "theme_id":             t["theme_id"],
            "theme_name":           t["theme_name"],
            "recommendation":       rec,
            "guidance":             REC_GUIDANCE.get(rec, ""),
            "supporting_cases":     t["supporting_case_count"],
            "coverage_pct":         round(t["supporting_case_count"] / total_cases * 100) if total_cases else 0,
            "net_score":            round(t["net_theme_score"], 4),
            "decisive_findings":    t["decisive_signal_count"],
            "contributing_findings":t["contributing_signal_count"],
            "high_confidence_cases":t["high_confidence_case_count"],
        })

    for w in watch_items:
        brief["watch_items"].append({
            "theme_id":             w["theme_id"],
            "theme_name":           w["theme_label"],
            "employer_won_cases":   w["employer_won"],
            "employee_won_cases":   w["employee_won"],
            "employer_won_rate_pct":round(w["employer_won"] / n_loss * 100) if n_loss else 0,
            "note": "Appeared in employer-won cases — prepare robust evidence before relying on this theme.",
        })

    if risk:
        brief["risk_profile"] = {
            "polkey_risk_cases":       risk.get("polkey_risk_cases", 0),
            "contribution_risk_cases": risk.get("contribution_risk_cases", 0),
            "recommended_use":         risk.get("recommended_use", ""),
            "negative_patterns":       risk.get("negative_pattern_counts") or {},
        }

    dl_col, _, _ = st.columns([2, 4, 4])
    with dl_col:
        st.download_button(
            label="Download calibration brief (JSON)",
            data=json.dumps(brief, indent=2, ensure_ascii=False),
            file_name=f"ws_calibration_brief_{total_cases}_cases.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_inputs() -> Optional[tuple]:
    st.sidebar.header("Inputs")
    input_mode = st.sidebar.radio("Input mode", ["Folder", "Paths", "Upload"], horizontal=True)

    if input_mode == "Folder":
        case_folder = st.sidebar.text_input("Outcome JSON folder", value=str(DEFAULT_CASE_DIR))
        aggregation_folder = st.sidebar.text_input("Aggregation JSON folder", value=str(DEFAULT_AGGREGATION_DIR))
        theme_store_folder = st.sidebar.text_input("Theme store folder", value=str(DEFAULT_THEME_STORE_DIR))

        case_paths = list_json_files(case_folder)
        aggregation_paths = list_json_files(aggregation_folder)
        theme_store_paths = list_theme_store_files(theme_store_folder)

        if not case_paths:
            st.error(f"No JSON files found in {case_folder}")
            return None
        if not aggregation_paths:
            st.error(f"No JSON files found in {aggregation_folder}")
            return None

        case_path = st.sidebar.selectbox("Case outcome JSON", case_paths, format_func=lambda p: p.name)
        aggregation_default = matched_aggregation_index(case_path, aggregation_paths)
        aggregation_path = st.sidebar.selectbox("Aggregation JSON", aggregation_paths, index=aggregation_default, format_func=lambda p: p.name)

        theme_store_path = None
        if theme_store_paths:
            ts_default = matched_theme_store_index(case_path, theme_store_paths)
            theme_store_path = st.sidebar.selectbox("Theme store JSON", theme_store_paths, index=ts_default, format_func=display_path)
        else:
            st.sidebar.warning("No theme_store.json files found.")

        if artifact_scope_key(case_path) != artifact_scope_key(aggregation_path):
            st.sidebar.warning("Selected case and aggregation files do not share the same run/case scope.")
        if theme_store_path and artifact_scope_key(case_path) != artifact_scope_key(theme_store_path):
            st.sidebar.warning("Selected case and theme store files do not share the same run/case scope.")

        st.sidebar.caption(f"Case: {display_path(case_path)}")
        st.sidebar.caption(f"Aggregation: {display_path(aggregation_path)}")
        if theme_store_path:
            st.sidebar.caption(f"Theme store: {display_path(theme_store_path)}")

        try:
            ts_json = load_json_from_path(str(theme_store_path)) if theme_store_path else None
            return load_json_from_path(str(case_path)), load_json_from_path(str(aggregation_path)), ts_json
        except Exception as exc:
            st.error(f"Could not load selected JSON input: {exc}")
            return None

    if input_mode == "Upload":
        case_upload = st.sidebar.file_uploader("case_outcome_optimized.json", type="json")
        aggregation_upload = st.sidebar.file_uploader("outcome_aggregation.json", type="json")
        theme_store_upload = st.sidebar.file_uploader("theme_store.json", type="json")
        if not case_upload or not aggregation_upload:
            st.info("Upload both JSON files to begin.")
            return None
        ts_json = load_json_from_upload(theme_store_upload) if theme_store_upload else None
        return load_json_from_upload(case_upload), load_json_from_upload(aggregation_upload), ts_json

    case_path = st.sidebar.text_input("Case outcome JSON path", value=str(DEFAULT_CASE_PATH))
    aggregation_path = st.sidebar.text_input("Aggregation JSON path", value=str(DEFAULT_AGGREGATION_PATH))
    theme_store_path = st.sidebar.text_input("Theme store JSON path", value=str(DEFAULT_THEME_STORE_PATH))
    try:
        ts_json = load_json_from_path(theme_store_path) if theme_store_path.strip() else None
        return load_json_from_path(case_path), load_json_from_path(aggregation_path), ts_json
    except Exception as exc:
        st.error(f"Could not load JSON input: {exc}")
        return None


# ---------------------------------------------------------------------------
# Single case tab
# ---------------------------------------------------------------------------

def _case_label_from_path(path: Path) -> str:
    """Parse a readable label from a filename without loading JSON."""
    stem = path.stem
    for suffix in ["_outcome_optimized", "_outcome_aggregation", "_calibration_validated", "_calibration_raw"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ").replace("-", " ")
    return stem


def _case_search_text(path: Path) -> str:
    return f"{_case_label_from_path(path)} {path.name}".lower()


@st.cache_data(show_spinner=False)
def _load_case_json_cached(path_str: str) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def render_single_case_tab() -> None:
    case_paths = list_json_files(str(DEFAULT_CASE_DIR))
    aggregation_paths = list_json_files(str(DEFAULT_AGGREGATION_DIR))
    theme_store_paths = list_theme_store_files(str(DEFAULT_THEME_STORE_DIR))

    if not case_paths:
        st.error(f"No JSON files found in {DEFAULT_CASE_DIR}")
        return
    if not aggregation_paths:
        st.error(f"No JSON files found in {DEFAULT_AGGREGATION_DIR}")
        return

    case_filter = st.text_input(
        "Filter by case name",
        value="",
        placeholder="Start typing a party, filename, or case keyword...",
    ).strip().lower()
    filtered_case_paths = [
        path for path in case_paths
        if not case_filter or case_filter in _case_search_text(path)
    ]

    if not filtered_case_paths:
        st.info(f"No cases match '{case_filter}'.")
        return

    st.caption(f"Showing {len(filtered_case_paths)} of {len(case_paths)} case files.")

    case_idx = st.selectbox(
        "Select case",
        range(len(filtered_case_paths)),
        format_func=lambda i: _case_label_from_path(filtered_case_paths[i]),
        label_visibility="collapsed",
    )
    case_path = filtered_case_paths[case_idx]

    agg_idx = matched_aggregation_index(case_path, aggregation_paths)
    aggregation_path = aggregation_paths[agg_idx]

    ts_path: Optional[Path] = None
    if theme_store_paths:
        ts_idx = matched_theme_store_index(case_path, theme_store_paths)
        ts_path = theme_store_paths[ts_idx]

    try:
        case_json = _load_case_json_cached(str(case_path))
        aggregation_json = _load_case_json_cached(str(aggregation_path))
        theme_store_json = _load_case_json_cached(str(ts_path)) if ts_path else None
    except Exception as exc:
        st.error(f"Could not load case files: {exc}")
        return

    render_header(case_json, aggregation_json)
    render_source_panel(case_json, aggregation_json)

    sec_tabs = st.tabs(["Case Intelligence", "Judgment Signals", "Theme Guidance", "Case Review Queue", "Strategy Brief"])
    with sec_tabs[0]:
        render_case_intelligence(case_json)
    with sec_tabs[1]:
        render_judgment_signals(case_json)
    with sec_tabs[2]:
        render_optimization_interpretation(aggregation_json)
    with sec_tabs[3]:
        render_theme_store(theme_store_json)
    with sec_tabs[4]:
        render_guidance(aggregation_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(
            page_title="Calibrator Monitor",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    inject_styles()
    if DATA_ROOT != CODE_ROOT:
        st.caption(f"Calibrator workspace: {DATA_ROOT}")

    main_views = ["WS Strategy", "Argument Library", "Single Case", "Runner"]
    next_view = st.session_state.pop("_next_main_view", None)
    if next_view in main_views:
        st.session_state["_active_main_view"] = next_view
    st.session_state.setdefault("_active_main_view", "WS Strategy")

    active_view = st.radio(
        "View",
        main_views,
        horizontal=True,
        key="_active_main_view",
        label_visibility="collapsed",
    )

    if active_view == "WS Strategy":
        render_ws_strategy_tab()
    elif active_view == "Argument Library":
        render_corpus_theme_store()
    elif active_view == "Single Case":
        render_single_case_tab()
    elif active_view == "Runner":
        render_runner_tab()


if __name__ == "__main__":
    main()
