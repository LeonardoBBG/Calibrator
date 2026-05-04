import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_PATH = PROJECT_ROOT / (
    "output/outcome_optimized/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1_outcome_optimized.json"
)
DEFAULT_AGGREGATION_PATH = PROJECT_ROOT / "output/outcome_aggregation/20260504_124953_outcome_aggregation.json"
DEFAULT_CASE_DIR = PROJECT_ROOT / "output/outcome_optimized"
DEFAULT_AGGREGATION_DIR = PROJECT_ROOT / "output/outcome_aggregation"

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


def load_json_from_path(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_from_upload(uploaded_file) -> Dict[str, Any]:
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def list_json_files(folder_text: str) -> List[Path]:
    folder = resolve_project_path(folder_text)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def outcome_timestamp(path: Path) -> str:
    name = path.name
    parts = name.split("_", 2)
    if len(parts) < 2:
        return ""
    return f"{parts[0]}_{parts[1]}"


def matched_aggregation_index(case_path: Path, aggregation_paths: List[Path]) -> int:
    case_key = outcome_timestamp(case_path)
    for index, aggregation_path in enumerate(aggregation_paths):
        if case_key and outcome_timestamp(aggregation_path) == case_key:
            return index
    return 0


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


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def humanize(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return OUTCOME_LABELS.get(text, text.replace("_", " ").title())


def percent_label(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{value}%"


def card_html(label: str, value: Any, note: str = "", tone: str = "neutral") -> str:
    safe_label = escape(label)
    safe_value = escape(str(value))
    safe_note = escape(note)
    return (
        f'<div class="metric-card {tone}">'
        f'<div class="metric-label">{safe_label}</div>'
        f'<div class="metric-value">{safe_value}</div>'
        f'<div class="metric-note">{safe_note}</div>'
        "</div>"
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
        }
        .hero-panel {
            background: linear-gradient(110deg, #111827 0%, #132238 48%, #11251f 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.16);
        }
        .hero-eyebrow {
            color: #93c5fd;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 1.9rem;
            line-height: 1.15;
            font-weight: 800;
            margin: 0 0 0.45rem 0;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin: 0;
            max-width: 980px;
        }
        .metric-card {
            min-height: 120px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 12px;
            padding: 0.95rem 1rem;
            background: #ffffff;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.75rem;
        }
        .metric-card.good {
            border-left: 5px solid #16a34a;
        }
        .metric-card.warn {
            border-left: 5px solid #d97706;
        }
        .metric-card.bad {
            border-left: 5px solid #dc2626;
        }
        .metric-card.neutral {
            border-left: 5px solid #2563eb;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .metric-note {
            color: #475569;
            font-size: 0.86rem;
            margin-top: 0.45rem;
            line-height: 1.35;
        }
        .section-note {
            color: #475569;
            font-size: 0.94rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    st.dataframe([compact_theme_row(row) for row in rows], width="stretch", hide_index=True)


def signal_weight_map(case_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        item.get("signal_id"): item
        for item in case_json.get("outcome_optimization", {}).get("signal_causal_weights", [])
        if isinstance(item, dict)
    }


def recommendation_counts(aggregation_json: Dict[str, Any]) -> Dict[str, int]:
    rows = aggregation_json.get("theme_strength_matrix", [])
    counts = {key: 0 for key in RECOMMENDATION_ORDER}
    for row in rows:
        recommendation = row.get("recommendation", "UNKNOWN")
        counts[recommendation] = counts.get(recommendation, 0) + 1
    return counts


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


def guidance_from_aggregation(aggregation_json: Dict[str, Any]) -> Dict[str, List[str]]:
    guidance = {
        "Maximise": [],
        "Refine": [],
        "Minimise": [],
        "Quarantine": [],
        "Ignore": [],
    }
    for row in aggregation_json.get("theme_strength_matrix", []):
        theme = f"{row.get('theme_id')} - {row.get('theme_name')}"
        recommendation = row.get("recommendation")
        if recommendation in {"REINFORCE_PRIMARY", "REINFORCE_SUPPORTING"}:
            guidance["Maximise"].append(theme)
        elif recommendation == "REFRAME":
            guidance["Refine"].append(theme)
        elif recommendation in {"MONITOR", "AVOID"}:
            guidance["Minimise"].append(theme)
        elif recommendation == "RISK_CONTROL":
            guidance["Quarantine"].append(theme)
        elif recommendation == "NO_SIGNAL":
            guidance["Ignore"].append(theme)
    return guidance


def render_header(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> None:
    metadata = case_json.get("case_metadata", {})
    outcome = case_json.get("outcome_optimization", {})
    shortlist = (aggregation_json.get("case_shortlist") or [{}])[0]
    risk_summary = aggregation_json.get("risk_control_summary", {})
    case_title = escape(get_case_label(case_json))
    outcome_summary = metadata.get("outcome") or "No outcome summary found."

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">Single-case outcome monitor</div>
            <div class="hero-title">{case_title}</div>
            <p class="hero-subtitle">{escape(outcome_summary)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_columns = st.columns(5)
    top_columns[0].markdown(
        card_html(
            "Liability result",
            humanize(outcome.get("claimant_liability_outcome")),
            humanize(outcome.get("liability_outcome_strength_band")),
            "good" if outcome.get("claimant_liability_outcome") == "WIN" else "warn",
        ),
        unsafe_allow_html=True,
    )
    top_columns[1].markdown(
        card_html(
            "Compensation risk",
            f"Polkey {percent_label(outcome.get('polkey_reduction_pct'))}",
            "Compensatory award risk from fair-dismissal inevitability.",
            "bad" if outcome.get("polkey_reduction_pct") else "neutral",
        ),
        unsafe_allow_html=True,
    )
    top_columns[2].markdown(
        card_html(
            "Conduct reduction",
            percent_label(outcome.get("contributory_fault_pct")),
            "Basic/compensatory reduction risk from claimant conduct.",
            "bad" if outcome.get("contributory_fault_pct") else "neutral",
        ),
        unsafe_allow_html=True,
    )
    top_columns[3].markdown(
        card_html(
            "Best use",
            humanize(shortlist.get("overall_use_mode")),
            f"Liability usefulness: {humanize(shortlist.get('liability_usefulness_band'))}",
            "neutral",
        ),
        unsafe_allow_html=True,
    )
    top_columns[4].markdown(
        card_html(
            "Risk handling",
            humanize(risk_summary.get("recommended_use")),
            f"{risk_summary.get('risk_control_signal_count', 0)} risk-control signals",
            "bad" if risk_summary.get("recommended_use") == "QUARANTINE" else "neutral",
        ),
        unsafe_allow_html=True,
    )

    detail_columns = st.columns(3)
    detail_columns[0].write(f"**Judgment date:** {metadata.get('judgment_date', '-')}")
    detail_columns[1].write(f"**Claims:** {', '.join(metadata.get('claims', [])) if isinstance(metadata.get('claims'), list) else metadata.get('claims', '-')}")
    detail_columns[2].write(f"**Transferability:** {humanize(outcome.get('transferability_rating'))}")


def render_source_panel(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> None:
    warnings = validate_pair(case_json, aggregation_json)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    st.caption("Read-only monitor. The Witness Statement is not rewritten or changed by this app.")


def render_legal_intelligence(case_json: Dict[str, Any]) -> None:
    st.header("Case Intelligence")
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
        st.markdown(
            "\n".join(
                [
                    f"- **Factual fit:** {humanize(outcome.get('factual_proximity'))}",
                    f"- **Transferability:** {humanize(outcome.get('transferability_rating'))}",
                    f"- **Liability outcome:** {humanize(outcome.get('claimant_liability_outcome'))}",
                    f"- **Remedy status:** {humanize(outcome.get('remedy_status'))}",
                ]
            )
        )
    with right:
        st.subheader("Remedy and risk notes")
        st.write(outcome.get("award_reduction_notes", "No remedy note found."))
        st.subheader("Use instruction")
        st.write(outcome.get("optimization_notes", "No optimization note found."))

    weights = signal_weight_map(case_json)
    st.subheader("Judgment signals")
    for signal in case_json.get("judgment_signals", []):
        signal_id = signal.get("signal_id")
        weight = weights.get(signal_id, {})
        title = (
            f"{signal_id}: {signal.get('signal_summary', 'Signal')[:92]}"
            f" ({humanize(weight.get('causal_weight', 'NO_WEIGHT'))})"
        )
        with st.expander(title):
            st.write(f"**Theme:** {signal.get('mapped_theme_id', '-')} | {humanize(signal.get('recommended_action', '-'))}")
            st.write(f"**Case effect:** {humanize(signal.get('case_effect', '-'))}")
            st.write(f"**Confidence:** {humanize(signal.get('dictionary_match_confidence', '-'))}")
            st.write(f"**Summary:** {signal.get('signal_summary', '-')}")
            st.write(f"**Relevance to WS:** {signal.get('relevance_to_ws', '-')}")
            st.write(f"**Causal weight reason:** {weight.get('causal_weight_reason', '-')}")
            refs = signal.get("judgment_references", [])
            st.caption(f"Judgment references: {', '.join(map(str, refs)) if refs else '-'}")

    st.subheader("Adverse signals")
    flags = outcome.get("negative_theme_flags", [])
    if flags:
        rows = [
            {
                "Theme": flag.get("theme_id"),
                "Pattern": humanize(flag.get("negative_pattern")),
                "Severity": humanize(flag.get("severity")),
                "Reason": flag.get("reason"),
            }
            for flag in flags
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("No adverse signals recorded.")


def render_optimization_interpretation(aggregation_json: Dict[str, Any]) -> None:
    st.header("Theme Guidance")
    metadata = aggregation_json.get("aggregation_metadata", {})
    counts = recommendation_counts(aggregation_json)

    cols = st.columns(5)
    cols[0].metric("Maximise", counts.get("REINFORCE_PRIMARY", 0) + counts.get("REINFORCE_SUPPORTING", 0))
    cols[1].metric("Refine", counts.get("REFRAME", 0))
    cols[2].metric("Minimise", counts.get("MONITOR", 0) + counts.get("AVOID", 0))
    cols[3].metric("Quarantine", counts.get("RISK_CONTROL", 0))
    cols[4].metric("Ignore", counts.get("NO_SIGNAL", 0))

    with st.expander("Technical scoring details", expanded=False):
        st.write(f"Scoring profile: {metadata.get('scoring_profile_version', '-')}")
        st.write(f"Threshold profile: {metadata.get('threshold_profile', '-')}")
        st.write(f"Case count: {metadata.get('case_count', '-')}")
        st.write(f"Minimum cases for primary reinforcement: {metadata.get('min_primary_cases', '-')}")

    tabs = st.tabs(["Maximise", "Refine", "Minimise", "Quarantine", "Ignore"])
    with tabs[0]:
        st.markdown('<div class="section-note">Strong themes to lead with when case support is enough. In single-case pilot mode this may be empty because primary reinforcement needs more than one case.</div>', unsafe_allow_html=True)
        render_rows(
            get_theme_rows(aggregation_json, "REINFORCE_PRIMARY")
            + get_theme_rows(aggregation_json, "REINFORCE_SUPPORTING"),
            "No themes currently meet reinforce thresholds for this case.",
        )
    with tabs[1]:
        st.markdown('<div class="section-note">Useful positive analogies, but not strong enough to treat as proven reinforcement on their own.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "REFRAME"), "No themes require reframing.")
    with tabs[2]:
        st.markdown('<div class="section-note">Themes with adverse or qualified signals. Use carefully, narrow the wording, or avoid broad claims.</div>', unsafe_allow_html=True)
        render_rows(
            get_theme_rows(aggregation_json, "MONITOR") + get_theme_rows(aggregation_json, "AVOID"),
            "No minimise/avoid themes identified.",
        )
    with tabs[3]:
        st.markdown('<div class="section-note">Risk-control material belongs in legal review and remedy strategy, not in the Witness Statement narrative.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "RISK_CONTROL"), "No risk-control theme identified.")
        st.subheader("Risk summary")
        risk = aggregation_json.get("risk_control_summary", {})
        risk_cols = st.columns(4)
        risk_cols[0].metric("Polkey risk cases", risk.get("polkey_risk_cases", 0))
        risk_cols[1].metric("Contribution risk cases", risk.get("contribution_risk_cases", 0))
        risk_cols[2].metric("Low remedy win cases", risk.get("low_remedy_win_cases", 0))
        risk_cols[3].metric("Recommended use", humanize(risk.get("recommended_use")))
        patterns = risk.get("negative_pattern_counts", {})
        if patterns:
            st.dataframe(
                [{"Pattern": humanize(pattern), "Count": count} for pattern, count in patterns.items()],
                width="stretch",
                hide_index=True,
            )
    with tabs[4]:
        st.markdown('<div class="section-note">Dictionary themes unaffected by this case. They are not positive or negative points from this judgment.</div>', unsafe_allow_html=True)
        render_rows(get_theme_rows(aggregation_json, "NO_SIGNAL"), "No unaffected dictionary themes.")


def render_guidance(aggregation_json: Dict[str, Any]) -> None:
    st.header("Practical Guidance")
    guidance = guidance_from_aggregation(aggregation_json)
    columns = st.columns(5)
    for column, (label, items) in zip(columns, guidance.items()):
        with column:
            st.subheader(label)
            if not items:
                st.caption("None")
            for item in items[:12]:
                st.write(f"- {item}")
            if len(items) > 12:
                st.caption(f"+ {len(items) - 12} more")

    st.subheader("Review Queues")
    left, right = st.columns(2)
    with left:
        st.write("**Other negative pattern review**")
        queue = aggregation_json.get("other_negative_pattern_review_queue", [])
        if queue:
            st.dataframe(queue, width="stretch", hide_index=True)
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


def render_multi_case_placeholder() -> None:
    st.title("Multi-Case Monitor")
    st.info("Placeholder only. Multi-case corpus aggregation mode will be added after the single-case monitor is validated.")
    st.write("Planned additions:")
    st.write("- Corpus-level case selector")
    st.write("- Multi-case theme trend matrix")
    st.write("- Cross-case negative pattern promotion queue")
    st.write("- Pilot-to-scale threshold review controls")


def load_inputs() -> Optional[tuple]:
    st.sidebar.header("Inputs")
    input_mode = st.sidebar.radio("Input mode", ["Folder", "Paths", "Upload"], horizontal=True)

    if input_mode == "Folder":
        case_folder = st.sidebar.text_input("Outcome JSON folder", value=str(DEFAULT_CASE_DIR))
        aggregation_folder = st.sidebar.text_input("Aggregation JSON folder", value=str(DEFAULT_AGGREGATION_DIR))

        case_paths = list_json_files(case_folder)
        aggregation_paths = list_json_files(aggregation_folder)

        if not case_paths:
            st.error(f"No JSON files found in {case_folder}")
            return None
        if not aggregation_paths:
            st.error(f"No JSON files found in {aggregation_folder}")
            return None

        case_path = st.sidebar.selectbox(
            "Case outcome JSON",
            case_paths,
            format_func=lambda path: path.name,
        )
        aggregation_default = matched_aggregation_index(case_path, aggregation_paths)
        aggregation_path = st.sidebar.selectbox(
            "Aggregation JSON",
            aggregation_paths,
            index=aggregation_default,
            format_func=lambda path: path.name,
        )

        if outcome_timestamp(case_path) != outcome_timestamp(aggregation_path):
            st.sidebar.warning("Selected files do not share the same timestamp prefix.")

        st.sidebar.caption(f"Case: {display_path(case_path)}")
        st.sidebar.caption(f"Aggregation: {display_path(aggregation_path)}")

        try:
            return load_json_from_path(str(case_path)), load_json_from_path(str(aggregation_path))
        except Exception as exc:
            st.error(f"Could not load selected JSON input: {exc}")
            return None

    if input_mode == "Upload":
        case_upload = st.sidebar.file_uploader("case_outcome_optimized.json", type="json")
        aggregation_upload = st.sidebar.file_uploader("outcome_aggregation.json", type="json")
        if not case_upload or not aggregation_upload:
            st.info("Upload both JSON files to begin.")
            return None
        return load_json_from_upload(case_upload), load_json_from_upload(aggregation_upload)

    case_path = st.sidebar.text_input("Case outcome JSON path", value=str(DEFAULT_CASE_PATH))
    aggregation_path = st.sidebar.text_input("Aggregation JSON path", value=str(DEFAULT_AGGREGATION_PATH))
    try:
        return load_json_from_path(case_path), load_json_from_path(aggregation_path)
    except Exception as exc:
        st.error(f"Could not load JSON input: {exc}")
        return None


def main() -> None:
    st.set_page_config(
        page_title="Single-Case Outcome Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.sidebar.title("Calibrator Monitor")
    mode = st.sidebar.selectbox("Mode", ["Single case", "Multi-case placeholder"])

    if mode != "Single case":
        render_multi_case_placeholder()
        return

    loaded = load_inputs()
    if loaded is None:
        return

    case_json, aggregation_json = loaded
    render_header(case_json, aggregation_json)
    render_source_panel(case_json, aggregation_json)

    section = st.sidebar.radio(
        "Section",
        ["Overview", "Case intelligence", "Theme guidance", "Practical guidance", "Raw JSON"],
    )

    if section == "Overview":
        render_legal_intelligence(case_json)
        render_optimization_interpretation(aggregation_json)
        render_guidance(aggregation_json)
    elif section == "Case intelligence":
        render_legal_intelligence(case_json)
    elif section == "Theme guidance":
        render_optimization_interpretation(aggregation_json)
    elif section == "Practical guidance":
        render_guidance(aggregation_json)
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Case Outcome Optimized JSON")
            st.json(case_json)
        with right:
            st.subheader("Outcome Aggregation JSON")
            st.json(aggregation_json)


if __name__ == "__main__":
    main()
