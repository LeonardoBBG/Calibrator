"""Quick upload, run and review interface for the standalone Sol-Fable project."""

from __future__ import annotations

import ipaddress
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sol_fable.analysis_backend import DeterministicAssessmentBackend  # noqa: E402
from sol_fable.credentials import ProviderCredentials  # noqa: E402
from sol_fable.errors import PipelineRunError  # noqa: E402
from sol_fable.llm_backend import OllamaAssessmentBackend, list_ollama_models  # noqa: E402
from sol_fable.orchestrator import PipelineOrchestrator  # noqa: E402

try:  # Cloud providers remain optional installation features.
    from sol_fable.llm_backend import (  # noqa: E402
        AnthropicAssessmentBackend,
        OpenAIAssessmentBackend,
        PersonaRouterBackend,
    )
except ImportError:  # pragma: no cover - supports deterministic-only installations
    AnthropicAssessmentBackend = None  # type: ignore[assignment,misc]
    OpenAIAssessmentBackend = None  # type: ignore[assignment,misc]
    PersonaRouterBackend = None  # type: ignore[assignment,misc]


_PROVIDER_OPTIONS = ["deterministic", "ollama", "openai", "anthropic"]
_PROVIDER_LABELS = {
    "deterministic": "Deterministic (offline)",
    "ollama": "Ollama (local only)",
    "openai": "OpenAI",
    "anthropic": "Anthropic / Claude",
}

_TREATMENT_EFFECT = {
    "PROTECT": "HELPS WS",
    "DEFEND": "DAMAGES / EXPOSES WS",
    "BOTH": "MIXED",
    "SECONDARY": "SECONDARY",
    "DISTRACTING": "DISTRACTING / DILUTIVE",
    "UNSUPPORTED": "DAMAGES / UNSUPPORTED",
}


def _is_local_ollama_host(value: str) -> bool:
    """Accept only an HTTP(S) localhost or literal loopback Ollama endpoint."""
    candidate = value.strip()
    if not candidate or re.search(r"\s", candidate):
        return False
    try:
        parsed = urlsplit(candidate)
        # Accessing port validates malformed/non-numeric ports.
        _ = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if parsed.path not in {"", "/"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _stage_upload(upload, document_type: str, staging_root: Path) -> Path:
    """Write an upload temporarily; ingest creates the sole persistent source copy."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(upload.name).name).strip("._")
    safe_name = safe_name or f"{document_type.lower()}_upload"
    destination = staging_root / document_type.lower() / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(upload.getvalue())
    return destination


def _secret_value(credentials: ProviderCredentials, provider: str) -> str | None:
    secret = (
        credentials.openai_api_key
        if provider == "openai"
        else credentials.anthropic_api_key
    )
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def _argument_rows(arguments: list) -> list[dict[str, object]]:
    rows = []
    for item in arguments:
        treatments = [value.value if hasattr(value, "value") else str(value) for value in item.treatment]
        effects = [_TREATMENT_EFFECT.get(value, value) for value in treatments]
        rows.append({
            "ID": item.argument_id,
            "Score": item.ranking_score,
            "Importance": item.importance,
            "Category": item.category,
            "Effect on WS": ", ".join(effects) or "NOT YET RANKED",
            "Treatment": ", ".join(treatments),
            "Analysis mode": item.assessment_mode or "NOT ASSESSED",
            "Title": item.title,
            "Review": item.human_review_status,
        })
    return rows


# The UI performs its own credential and endpoint validation before replacing
# this safe offline default; never instantiate a configured live backend eagerly.
orchestrator = PipelineOrchestrator(backend=DeterministicAssessmentBackend())


st.set_page_config(page_title="Sol-Fable", page_icon="⚖️", layout="wide")
st.title("Sol-Fable WS Argument Calibrator")
st.caption("Standalone, auditable ET1 / ET3 / witness-statement analysis")
st.warning(
    "Assistive legal-preparation tool only. Pleadings are not proof, documentary brackets remain unresolved, "
    "and the output is not a final legal determination."
)
st.caption(
    "Deployment note: source documents, extracted text, and the case database are stored locally. "
    "Run this on a single-user or access-controlled system and protect the case directory."
)

with st.sidebar:
    st.header("Analysis providers")
    st.caption(
        "Assign a provider to each barrister persona. Party assignment remains separate below."
    )
    configured_backend = orchestrator.settings.analysis_backend
    if configured_backend.startswith("ollama"):
        default_sol_provider, default_fable_provider = "ollama", "ollama"
    elif configured_backend == "openai":
        default_sol_provider, default_fable_provider = "openai", "openai"
    elif configured_backend == "anthropic":
        default_sol_provider, default_fable_provider = "anthropic", "anthropic"
    elif configured_backend == "dual-api":
        default_sol_provider, default_fable_provider = "openai", "anthropic"
    else:
        default_sol_provider, default_fable_provider = "deterministic", "deterministic"
    sol_provider = st.selectbox(
        "SOL provider",
        _PROVIDER_OPTIONS,
        index=_PROVIDER_OPTIONS.index(default_sol_provider),
        format_func=_PROVIDER_LABELS.get,
    )
    fable_provider = st.selectbox(
        "FABLE provider",
        _PROVIDER_OPTIONS,
        index=_PROVIDER_OPTIONS.index(default_fable_provider),
        format_func=_PROVIDER_LABELS.get,
    )
    selected_providers = {sol_provider, fable_provider}
    if {"openai", "anthropic"} & selected_providers:
        st.warning(
            "Selected cloud providers will receive the cited case text needed for each assessment. "
            "Use them only when you are authorised to send the case material and have reviewed the "
            "provider's data controls."
        )

    with st.expander(
        "Cloud API credentials",
        expanded=bool({"openai", "anthropic"} & selected_providers),
    ):
        st.caption(
            "Keys are masked, held only in this Streamlit session, and never written to the case database, "
            "reports, or configuration files. Leave blank to use the named environment variable."
        )
        openai_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            value="",
            placeholder="Uses OPENAI_API_KEY when blank",
            key="sol_fable_openai_api_key",
        )
        anthropic_key_input = st.text_input(
            "Anthropic API key",
            type="password",
            value="",
            placeholder="Uses ANTHROPIC_API_KEY when blank",
            key="sol_fable_anthropic_api_key",
        )
        openai_env_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        anthropic_env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        if openai_key_input.strip():
            st.caption("OpenAI: using the session-only key entered above.")
        elif openai_env_key:
            st.caption("OpenAI: configured from OPENAI_API_KEY.")
        if anthropic_key_input.strip():
            st.caption("Anthropic: using the session-only key entered above.")
        elif anthropic_env_key:
            st.caption("Anthropic: configured from ANTHROPIC_API_KEY.")

    credentials = ProviderCredentials(
        openai_api_key=openai_key_input.strip() or openai_env_key,
        anthropic_api_key=anthropic_key_input.strip() or anthropic_env_key,
    )

    openai_model = (
        st.text_input(
            "OpenAI model",
            value=os.environ.get(
                "SOL_FABLE_OPENAI_MODEL",
                getattr(orchestrator.settings, "openai_model", "gpt-5.6-sol"),
            ),
        ).strip()
        if "openai" in selected_providers
        else "gpt-5.6-sol"
    )
    anthropic_model = (
        st.text_input(
            "Anthropic model",
            value=os.environ.get(
                "SOL_FABLE_ANTHROPIC_MODEL",
                getattr(orchestrator.settings, "anthropic_model", "claude-sonnet-5"),
            ),
        ).strip()
        if "anthropic" in selected_providers
        else "claude-sonnet-5"
    )

    deterministic_backend = DeterministicAssessmentBackend()
    ollama_backend = None
    chosen_backend = None
    backend_ready = True
    if "ollama" in selected_providers:
        st.caption(
            "Ollama is local-only: only localhost or a literal loopback address is permitted. "
            "Remote plain-HTTP model endpoints are blocked."
        )
        ollama_host = st.text_input("Ollama host", value=orchestrator.settings.ollama_host)
        if not _is_local_ollama_host(ollama_host):
            available_models = []
            st.error(
                "That Ollama address is not local. The request was blocked; selected Ollama "
                "routes will use the deterministic backend."
            )
        else:
            available_models = list_ollama_models(ollama_host)
        if not available_models and _is_local_ollama_host(ollama_host):
            st.warning(
                "Local Ollama is unavailable or has no models. Selected Ollama routes will "
                "use the deterministic backend."
            )
        elif available_models:
            model_names = [item["name"] for item in available_models]
            default_model = orchestrator.settings.ollama_model
            default_index = model_names.index(default_model) if default_model in model_names else 0
            ollama_model = st.selectbox(
                "Ollama model",
                model_names,
                index=default_index,
                format_func=lambda name: (
                    f"{name} ({next(m['parameter_size'] for m in available_models if m['name'] == name)})"
                ),
            )
            try:
                ollama_backend = OllamaAssessmentBackend(
                    model=ollama_model,
                    host=ollama_host,
                    max_retries=orchestrator.settings.llm_max_retries,
                    timeout_seconds=orchestrator.settings.llm_timeout_seconds,
                    max_output_tokens=orchestrator.settings.llm_max_output_tokens,
                    num_gpu=orchestrator.settings.ollama_num_gpu,
                )
            except Exception as exc:  # noqa: BLE001 - keep credentials/transport details private
                st.error(
                    f"The local Ollama backend could not be initialized ({type(exc).__name__}); "
                    "selected Ollama routes will use deterministic analysis."
                )

    provider_backends: dict[str, object] = {"deterministic": deterministic_backend}
    if "ollama" in selected_providers:
        # This is an actual fallback, not merely a warning in the UI.
        provider_backends["ollama"] = ollama_backend or deterministic_backend

    if "openai" in selected_providers:
        openai_key = _secret_value(credentials, "openai")
        if not openai_key:
            st.error("OpenAI is selected but no API key is configured.")
            backend_ready = False
        elif not openai_model:
            st.error("OpenAI is selected but its model name is empty.")
            backend_ready = False
        elif OpenAIAssessmentBackend is None:
            st.error("The OpenAI backend is not available in this installation.")
            backend_ready = False
        else:
            try:
                provider_backends["openai"] = OpenAIAssessmentBackend(
                    api_key=openai_key,
                    model=openai_model,
                    max_retries=orchestrator.settings.llm_max_retries,
                    timeout_seconds=orchestrator.settings.llm_timeout_seconds,
                    max_output_tokens=orchestrator.settings.llm_max_output_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - do not render exception values containing secrets
                st.error(f"The OpenAI backend could not be initialized ({type(exc).__name__}).")
                backend_ready = False

    if "anthropic" in selected_providers:
        anthropic_key = _secret_value(credentials, "anthropic")
        if not anthropic_key:
            st.error("Anthropic is selected but no API key is configured.")
            backend_ready = False
        elif not anthropic_model:
            st.error("Anthropic is selected but its model name is empty.")
            backend_ready = False
        elif AnthropicAssessmentBackend is None:
            st.error("The Anthropic backend is not available in this installation.")
            backend_ready = False
        else:
            try:
                provider_backends["anthropic"] = AnthropicAssessmentBackend(
                    api_key=anthropic_key,
                    model=anthropic_model,
                    max_retries=orchestrator.settings.llm_max_retries,
                    timeout_seconds=orchestrator.settings.llm_timeout_seconds,
                    max_output_tokens=orchestrator.settings.llm_max_output_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - do not render exception values containing secrets
                st.error(f"The Anthropic backend could not be initialized ({type(exc).__name__}).")
                backend_ready = False

    if backend_ready:
        sol_backend = provider_backends[sol_provider]
        fable_backend = provider_backends[fable_provider]
        if sol_backend is fable_backend:
            chosen_backend = sol_backend
        elif PersonaRouterBackend is None:
            st.error("Barrister-specific provider routing is unavailable in this installation.")
            backend_ready = False
        else:
            try:
                chosen_backend = PersonaRouterBackend(
                    sol_backend=sol_backend,
                    fable_backend=fable_backend,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Provider routing could not be initialized ({type(exc).__name__}).")
                backend_ready = False
    if backend_ready and chosen_backend is not None:
        st.success(f"Active backend: {chosen_backend.name}")

    st.divider()
    st.header("Barrister setup")
    n_rounds = int(
        st.number_input(
            "Number of rounds",
            min_value=1,
            max_value=10,
            value=orchestrator.settings.n_rounds,
            step=1,
            help="Each round contains one Claimant turn followed by one Respondent turn.",
        )
    )
    claimant_barrister = st.selectbox(
        "Claimant barrister",
        options=["SOL", "FABLE"],
        index=0 if orchestrator.settings.claimant_barrister.value == "SOL" else 1,
    )
    respondent_barrister = "FABLE" if claimant_barrister == "SOL" else "SOL"
    st.caption(f"Respondent barrister: {respondent_barrister} (assigned automatically)")
    model_analysis_selected = bool(selected_providers - {"deterministic"})
    if model_analysis_selected:
        cap_enabled = st.checkbox(
            "Limit live analysis to top arguments",
            value=True,
            help=(
                "Arguments outside the cap still receive complete deterministic analysis. "
                "This prevents accidental large local or paid-provider runs."
            ),
        )
        live_argument_cap = (
            int(
                st.number_input(
                    "Live argument cap",
                    min_value=1,
                    value=orchestrator.settings.live_argument_cap or 20,
                    step=1,
                )
            )
            if cap_enabled
            else 0
        )
        summary_calls = 2 if sol_provider != fable_provider else 1
        st.caption(
            f"Each live argument schedules about {(2 * n_rounds) + 2 + summary_calls} "
            "provider calls before any validation retry."
        )
    else:
        live_argument_cap = None
    cloud_selected = bool({"openai", "anthropic"} & selected_providers)
    cloud_confirmed = (
        st.checkbox(
            "I am authorised to send the cited case text and accept provider usage charges",
            value=False,
        )
        if cloud_selected
        else True
    )
    st.divider()
    st.header("Case documents")
    et1_upload = st.file_uploader("ET1", type=["txt", "docx", "pdf"], key="et1")
    et3_upload = st.file_uploader("ET3", type=["txt", "docx", "pdf"], key="et3")
    ws_upload = st.file_uploader("Claimant witness statement", type=["txt", "docx", "pdf"], key="ws")
    run_clicked = st.button(
        "Run full calibration",
        type="primary",
        use_container_width=True,
        disabled=(
            not all((et1_upload, et3_upload, ws_upload))
            or not backend_ready
            or not cloud_confirmed
        ),
    )

    st.divider()
    st.header("Open an existing run")
    st.caption("No previous case is shown unless you explicitly select its run ID in this session.")
    existing_run_id = st.text_input(
        "Run ID",
        value="",
        placeholder="RUN-YYYYMMDDTHHMMSSZ-XXXXXXXX",
        key="sol_fable_existing_run_id",
    )
    if st.button("Open selected run", use_container_width=True):
        candidate_run_id = existing_run_id.strip().upper()
        if not candidate_run_id:
            st.error("Enter a run ID first.")
        else:
            try:
                orchestrator.store.load_run(candidate_run_id)
            except Exception:  # noqa: BLE001 - do not expose database internals
                st.error("That run ID was not found.")
            else:
                st.session_state["sol_fable_run_id"] = candidate_run_id
                st.success("Run selected for this session.")
    if st.session_state.get("sol_fable_run_id") and st.button(
        "Close current run",
        use_container_width=True,
    ):
        st.session_state.pop("sol_fable_run_id", None)
        st.rerun()
    resume_clicked = bool(st.session_state.get("sol_fable_run_id")) and st.button(
        "Resume assessment and finish selected run",
        use_container_width=True,
        disabled=not backend_ready or not cloud_confirmed,
    )

if chosen_backend is not None:
    orchestrator.backend = chosen_backend

if run_clicked:
    try:
        with st.status("Running the 11-stage audit pipeline…", expanded=True) as status:
            with TemporaryDirectory(prefix="sol-fable-uploads-") as staging_directory:
                staging_root = Path(staging_directory)
                et1_path = _stage_upload(et1_upload, "ET1", staging_root)
                et3_path = _stage_upload(et3_upload, "ET3", staging_root)
                ws_path = _stage_upload(ws_upload, "WS", staging_root)
                st.write("Uploads staged temporarily; ingestion preserves one hashed source copy.")
                st.write(f"Backend: {orchestrator.backend.name}")
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def _update_progress(done: int, total: int) -> None:
                    progress_bar.progress(done / total if total else 1.0)
                    progress_text.text(f"Assessed {done}/{total} arguments")

                run_id = orchestrator.run_all(
                    et1_path,
                    et3_path,
                    ws_path,
                    n_rounds=n_rounds,
                    claimant_barrister=claimant_barrister,
                    progress_callback=_update_progress,
                    live_argument_cap=live_argument_cap,
                )
            st.session_state["sol_fable_run_id"] = run_id
            status.update(label="Calibration complete", state="complete")
    except PipelineRunError as exc:
        st.session_state["sol_fable_run_id"] = exc.run_id
        st.error(
            f"Calibration failed ({type(exc.cause).__name__}); run ID {exc.run_id} was preserved. "
            "If argument assessment had started, it can be resumed below. No API credential "
            "was displayed or persisted."
        )
    except Exception as exc:
        st.error(
            f"Calibration failed ({type(exc).__name__}). No API credential was displayed or persisted."
        )

if resume_clicked:
    resume_run_id = st.session_state["sol_fable_run_id"]
    try:
        with st.status("Resuming assessment…", expanded=True) as status:
            orchestrator.assess(
                resume_run_id,
                n_rounds=n_rounds,
                claimant_barrister=claimant_barrister,
                live_argument_cap=live_argument_cap,
            )
            orchestrator.rank(resume_run_id)
            orchestrator.generate_search_packages(resume_run_id)
            orchestrator.report(resume_run_id)
            status.update(label="Calibration complete", state="complete")
    except Exception as exc:
        st.error(
            f"Resume failed ({type(exc).__name__}) for {resume_run_id}. "
            "No API credential was displayed or persisted."
        )

run_id = st.session_state.get("sol_fable_run_id")
if not run_id:
    st.info(
        "Upload ET1, ET3 and WS files to start a run, or explicitly open an existing run ID. "
        "Previous cases are never opened automatically."
    )
    st.stop()

try:
    run = orchestrator.store.load_run(run_id)
except Exception:  # noqa: BLE001 - stale session selection
    st.session_state.pop("sol_fable_run_id", None)
    st.error("The run selected in this session is no longer available.")
    st.stop()
arguments = orchestrator.store.load_arguments(run_id)
reviews = orchestrator.store.load_reviews(run_id)
references = orchestrator.store.load_references(run_id)
packages = orchestrator.store.load_search_packages(run_id)

st.subheader(f"Run {run_id}")
st.caption(
    f"{run.config_snapshot.get('n_rounds', 2)} rounds · "
    f"Claimant: {run.config_snapshot.get('claimant_barrister', 'SOL')} · "
    f"Respondent: {run.config_snapshot.get('respondent_barrister', 'FABLE')} · "
    f"Backend: {run.backend}"
)
metric_cols = st.columns(5)
metric_cols[0].metric("Status", run.status)
metric_cols[1].metric("Arguments", len(arguments))
metric_cols[2].metric("Human reviews", len(reviews))
metric_cols[3].metric("Placeholders", len(references))
metric_cols[4].metric("Search packages", len(packages))

overview, debate_tab, detail, review_tab, audit_tab, downloads = st.tabs(
    ["Argument map", "Barrister debate", "Argument detail", "Human review", "Audit", "Downloads"]
)
with overview:
    st.dataframe(_argument_rows(arguments), use_container_width=True, hide_index=True)
with debate_tab:
    if arguments:
        debate_argument_id = st.selectbox(
            "Debated argument",
            [item.argument_id for item in arguments],
            key="debate_argument_id",
        )
        debated_argument = next(item for item in arguments if item.argument_id == debate_argument_id)
        summary = debated_argument.debate_summary
        if summary:
            st.markdown(f"### {debated_argument.title}")
            st.caption(
                f"Claimant: {summary.claimant_barrister} · Respondent: {summary.respondent_barrister} · "
                f"{summary.n_rounds} rounds · Mode: {debated_argument.assessment_mode or 'unknown'}"
            )
            st.info(
                f"Mediated recommendation: {summary.recommended_treatment or 'not recorded'} · "
                f"Claimant strength: {summary.claimant_strength} · "
                f"Respondent strength: {summary.respondent_strength}\n\n"
                f"{summary.treatment_rationale or ''}"
            )
            for round_number in range(1, summary.n_rounds + 1):
                round_turns = [
                    turn for turn in summary.turns if turn.round_number == round_number
                ]
                with st.expander(f"Round {round_number}", expanded=round_number == 1):
                    columns = st.columns(2)
                    for column, turn in zip(columns, round_turns):
                        with column:
                            st.markdown(f"#### {turn.side}: {turn.barrister}")
                            st.write(turn.position)
                            if turn.responses_to_opponent:
                                st.markdown("**Response to opponent**")
                                for item in turn.responses_to_opponent:
                                    st.write(f"- {item}")
                            st.markdown("**Challenges**")
                            for item in turn.challenges_for_opponent:
                                st.write(f"- {item}")
                            st.caption(
                                f"Provider: {turn.backend} · Citations: "
                                f"{', '.join(turn.paragraph_citations)}"
                            )
                            if turn.evidence_links:
                                with st.expander("Claim-to-source links"):
                                    for link in turn.evidence_links:
                                        st.write(
                                            f"- **{link.relationship}** — {link.claim} "
                                            f"[{', '.join(link.paragraph_ids)}]"
                                        )
            st.markdown("#### Remaining disputes")
            for item in summary.remaining_disputes:
                st.write(f"- {item}")
            if summary.human_review_reasons:
                st.warning("Human review: " + " ".join(summary.human_review_reasons))
        else:
            st.info("This argument has not completed the mediated debate stage.")
    else:
        st.info("No debated arguments are available for this run.")
with detail:
    if arguments:
        selected_id = st.selectbox("Argument", [item.argument_id for item in arguments])
        argument = next(item for item in arguments if item.argument_id == selected_id)
        st.markdown(f"### {argument.title}")
        st.write(argument.proposition)
        st.json(argument.model_dump(mode="json"), expanded=False)
    else:
        st.info("No arguments are available for this run.")
with review_tab:
    st.dataframe(
        [item.model_dump(mode="json") for item in reviews],
        use_container_width=True,
        hide_index=True,
    )
with audit_tab:
    st.json(
        {
            "run": run.model_dump(mode="json"),
            "stages": orchestrator.store.stage_events(run_id),
            "documents": [item.model_dump(mode="json") for item in orchestrator.store.load_documents(run_id)],
        },
        expanded=False,
    )
with downloads:
    run_reports = orchestrator.reports_for(run_id)
    report_md = run_reports / "ws_calibration_report.md"
    report_json = run_reports / "ws_calibration_report.json"
    review_csv = run_reports / "human_review_queue.csv"
    debate_report = run_reports / "barrister_debate.md"
    if run.status == "COMPLETED" and report_md.exists():
        st.download_button("Download Markdown report", report_md.read_bytes(), report_md.name, "text/markdown")
        st.download_button("Download JSON report", report_json.read_bytes(), report_json.name, "application/json")
        st.download_button("Download review queue", review_csv.read_bytes(), review_csv.name, "text/csv")
        if debate_report.exists():
            st.download_button("Download barrister debate", debate_report.read_bytes(), debate_report.name, "text/markdown")
    else:
        st.info("Final downloads appear after a completed report stage.")
