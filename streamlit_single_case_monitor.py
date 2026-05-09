import json
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_PATH = PROJECT_ROOT / (
    "output/outcome_optimized/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1_outcome_optimized.json"
)
DEFAULT_AGGREGATION_PATH = PROJECT_ROOT / (
    "output/outcome_aggregation/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1_outcome_aggregation.json"
)
DEFAULT_THEME_STORE_PATH = PROJECT_ROOT / (
    "output/theme_store/"
    "20260504_124953_Mr_P_Pronzynski_v_3663_Transport_-_2413742_2018_-_Judgment_1/"
    "theme_store.json"
)
DEFAULT_CASE_DIR = PROJECT_ROOT / "output/outcome_optimized"
DEFAULT_AGGREGATION_DIR = PROJECT_ROOT / "output/outcome_aggregation"
DEFAULT_THEME_STORE_DIR = PROJECT_ROOT / "output/theme_store"

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
    return sorted(folder.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def list_theme_store_files(folder_text: str) -> List[Path]:
    folder = resolve_project_path(folder_text)
    if not folder.exists():
        return []
    return sorted(folder.rglob("theme_store.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def display_path(path: Path) -> str:
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
# Theme-store extraction — handles both old and new file structures
# ---------------------------------------------------------------------------

def extract_all_matches(theme_store_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract every match from a theme_store.json regardless of nesting structure."""
    matches = []
    for theme_id, theme_data in theme_store_json.items():
        if not isinstance(theme_data, dict):
            continue
        theme_label = theme_data.get("theme_label") or theme_id

        # New structure: groups keyed by "EFFECT|CASE_EFFECT|CONFIDENCE"
        if "groups" in theme_data:
            for group_data in (theme_data.get("groups") or {}).values():
                for match in (group_data.get("matches") or []):
                    if isinstance(match, dict):
                        m = dict(match)
                        m["theme_id"] = theme_id
                        m["theme_label"] = theme_label
                        matches.append(m)

        # Old structure: action_lanes → subthemes → matches
        elif "action_lanes" in theme_data:
            for lane_data in (theme_data.get("action_lanes") or {}).values():
                for sub_data in (lane_data.get("subthemes") or {}).values():
                    for match in (sub_data.get("matches") or []):
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
            "Effect": humanize(m.get("effect") or m.get("action_lane")),
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
        lanes = theme_data.get("action_lanes") or {}
        rows.append({
            "Theme ID": theme_id,
            "Theme": theme_data.get("theme_label") or theme_id,
            "Matches": theme_data.get("n_matches", 0),
            "High confidence": theme_data.get("n_high_confidence", 0),
            "Win drivers": theme_data.get("n_win_drivers", 0),
            "Groups": ", ".join(groups.keys() or lanes.keys()) or "-",
            "Recommendation": humanize(rank_data.get("recommendation")),
            "Net score": rank_data.get("net_theme_score"),
        })
    return sorted(rows, key=lambda r: (-(r.get("Matches") or 0), str(r.get("Theme ID") or "")))


# ---------------------------------------------------------------------------
# Corpus aggregation helpers
# ---------------------------------------------------------------------------

def _group_sort_key(key: Tuple[str, str, str, str]) -> Tuple:
    _, effect, case_effect, confidence = key
    return (
        CASE_EFFECT_RANK.get(case_effect, 99),
        CONFIDENCE_RANK.get(confidence, 99),
        EFFECT_RANK.get(effect, 99),
    )


def aggregate_all_theme_stores(theme_store_dir: Path) -> Dict[str, Any]:
    """Load every theme_store.json in the directory and aggregate by (theme, effect, case_effect, confidence)."""
    paths = sorted(theme_store_dir.rglob("theme_store.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not paths:
        return {"paths": [], "grouped": {}, "theme_labels": {}, "cases": set(), "n_matches": 0}

    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    theme_labels: Dict[str, str] = {}
    cases: set = set()

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for theme_id, theme_data in data.items():
            if not isinstance(theme_data, dict):
                continue
            theme_labels[theme_id] = theme_data.get("theme_label") or theme_id

            for m in extract_all_matches({theme_id: theme_data}):
                effect = m.get("effect") or m.get("action_lane") or "UNKNOWN"
                case_effect = m.get("case_effect") or "UNKNOWN"
                confidence = m.get("confidence") or "UNKNOWN"
                key = (theme_id, effect.upper(), case_effect.upper(), confidence.upper())
                grouped[key].append(m)
                if m.get("case_name"):
                    cases.add(m["case_name"])

    return {
        "paths": paths,
        "grouped": dict(grouped),
        "theme_labels": theme_labels,
        "cases": cases,
        "n_matches": sum(len(v) for v in grouped.values()),
    }


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
        .view-strip {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 0.65rem 0.75rem 0.2rem 0.75rem;
            margin: 0.8rem 0 1rem 0;
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


def arg_card_html(case_name: str, summary: str, relevance: str, causal_reason: str, refs: str) -> str:
    chips = "".join(
        f"<span class='arg-chip'>{escape(c)}</span>"
        for c in [f"para {refs}"] if c.strip()
    )
    return (
        f"<div class='arg-card'>"
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

    detail_cols = st.columns(3)
    detail_cols[0].write(f"**Judgment date:** {metadata.get('judgment_date', '-')}")
    claims = metadata.get("claims", [])
    claims_str = ", ".join(claims) if isinstance(claims, list) else str(claims or "-")
    detail_cols[1].write(f"**Claims:** {claims_str}")
    detail_cols[2].write(f"**Transferability:** {humanize(outcome.get('transferability_rating'))}")


def render_source_panel(case_json: Dict[str, Any], aggregation_json: Dict[str, Any]) -> None:
    warnings = validate_pair(case_json, aggregation_json)
    for w in warnings:
        st.warning(w)
    st.caption("Read-only monitor. The Witness Statement is not rewritten or changed by this app.")


def render_legal_intelligence(case_json: Dict[str, Any]) -> None:
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
    cols[0].metric("Maximise", counts.get("REINFORCE_PRIMARY", 0) + counts.get("REINFORCE_SUPPORTING", 0))
    cols[1].metric("Refine", counts.get("REFRAME", 0))
    cols[2].metric("Minimise", counts.get("MONITOR", 0) + counts.get("AVOID", 0))
    cols[3].metric("Quarantine", counts.get("RISK_CONTROL", 0))
    cols[4].metric("Ignore", counts.get("NO_SIGNAL", 0))

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
        rc = st.columns(4)
        rc[0].metric("Polkey risk cases", risk.get("polkey_risk_cases", 0))
        rc[1].metric("Contribution risk cases", risk.get("contribution_risk_cases", 0))
        rc[2].metric("Low remedy win cases", risk.get("low_remedy_win_cases", 0))
        rc[3].metric("Recommended use", humanize(risk.get("recommended_use")))
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
        ("Park", guidance.get("Ignore", []), "park"),
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

    mc = st.columns(5)
    mc[0].metric("Themes", len(theme_rows))
    mc[1].metric("Matches", len(match_rows))
    mc[2].metric("Reinforce", effect_counts.get("Reinforce", 0))
    mc[3].metric("Review manually", effect_counts.get("Review Manually", 0))
    mc[4].metric("Risk control", sum(1 for r in match_rows if "T20" in str(r.get("Theme ID", ""))))

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
                # Handle both new (groups) and old (action_lanes) structure
                groups = theme_data.get("groups") or {}
                lanes = theme_data.get("action_lanes") or {}
                if groups:
                    for gkey, gdata in groups.items():
                        st.subheader(gkey)
                        rows = [{"Case": m.get("case_name"), "Effect": humanize(m.get("effect")), "Case effect": humanize(m.get("case_effect")), "Confidence": humanize(m.get("confidence")), "Rank": m.get("rank_score"), "Source": m.get("source_pointer"), "Summary": m.get("summary")} for m in (gdata.get("matches") or [])]
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    for lane_key, lane_data in lanes.items():
                        st.subheader(humanize(lane_key))
                        for subtheme, sub_data in (lane_data.get("subthemes") or {}).items():
                            rows = [{"Case": m.get("case_name"), "Effect": humanize(m.get("effect")), "Case effect": humanize(m.get("case_effect")), "Confidence": humanize(m.get("confidence")), "Rank": m.get("rank_score"), "Source": m.get("source_pointer"), "Summary": m.get("summary")} for m in (sub_data.get("matches") or [])]
                            if rows:
                                st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Corpus tab
# ---------------------------------------------------------------------------

def render_corpus_theme_store() -> None:
    corpus = aggregate_all_theme_stores(DEFAULT_THEME_STORE_DIR)
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
    mc[0].markdown(mini_card_html("Cases", str(len(paths)), f"{len(cases)} named claimants"), unsafe_allow_html=True)
    mc[1].markdown(mini_card_html("Total arguments", str(n_matches), "Across all cases and themes"), unsafe_allow_html=True)
    mc[2].markdown(mini_card_html("Themes covered", str(n_themes), "Distinct themes with at least one match"), unsafe_allow_html=True)
    mc[3].markdown(mini_card_html("Win-driver arguments", str(n_win_drivers), f"{n_reinforce} reinforce-lane arguments", "good"), unsafe_allow_html=True)

    if not grouped:
        st.info("No theme_store.json files found in output/theme_store/. Run the pipeline to generate them.")
        return

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)

    all_themes = sorted({k[0] for k in grouped})
    all_effects = sorted({k[1] for k in grouped}, key=lambda x: EFFECT_RANK.get(x, 99))
    all_case_effects = sorted({k[2] for k in grouped}, key=lambda x: CASE_EFFECT_RANK.get(x, 99))
    all_confidences = sorted({k[3] for k in grouped}, key=lambda x: CONFIDENCE_RANK.get(x, 99))

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        theme_filter = st.multiselect("Theme", all_themes, default=[])
    with fc2:
        effect_default = [v for v in ["REINFORCE"] if v in all_effects]
        effect_filter = st.multiselect("Effect", all_effects, default=effect_default)
    with fc3:
        ce_default = [v for v in ["WIN_DRIVER", "STRONG_SUPPORT"] if v in all_case_effects]
        case_effect_filter = st.multiselect("Case effect", all_case_effects, default=ce_default)
    with fc4:
        conf_default = [v for v in ["HIGH", "MEDIUM"] if v in all_confidences]
        confidence_filter = st.multiselect("Confidence", all_confidences, default=conf_default)

    search_corpus = st.text_input("Search argument summaries", value="", key="corpus_search")
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered_groups = {
        k: v for k, v in grouped.items()
        if (not theme_filter or k[0] in theme_filter)
        and (not effect_filter or k[1] in effect_filter)
        and (not case_effect_filter or k[2] in case_effect_filter)
        and (not confidence_filter or k[3] in confidence_filter)
    }

    if search_corpus.strip():
        needle = search_corpus.strip().lower()
        filtered_groups = {
            k: [m for m in v if needle in str(m.get("summary") or "").lower() or needle in str(m.get("relevance_to_ws") or "").lower()]
            for k, v in filtered_groups.items()
        }
        filtered_groups = {k: v for k, v in filtered_groups.items() if v}

    total_shown = sum(len(v) for v in filtered_groups.values())
    st.caption(f"Showing {total_shown} arguments in {len(filtered_groups)} groups after filters.")

    if not filtered_groups:
        st.info("No arguments match the current filters.")
        return

    sorted_keys = sorted(filtered_groups.keys(), key=_group_sort_key)

    for key in sorted_keys:
        theme_id, effect, case_effect, confidence = key
        theme_label = theme_labels.get(theme_id) or theme_id
        matches = filtered_groups[key]
        n = len(matches)

        effect_badge_kind = "good" if effect == "REINFORCE" else "warn" if effect == "REVIEW_MANUALLY" else "info"
        ce_badge_kind = "good" if case_effect == "WIN_DRIVER" else "warn" if case_effect in {"MODERATE_SUPPORT", "WEAK_SUPPORT"} else "bad" if case_effect == "ADVERSE" else "info"
        conf_badge_kind = "good" if confidence == "HIGH" else "warn" if confidence == "MEDIUM" else "info"

        header_html = (
            f"{badge(effect, effect_badge_kind)} "
            f"{badge(case_effect, ce_badge_kind)} "
            f"{badge(confidence, conf_badge_kind)}"
        )

        with st.expander(f"{theme_id} — {theme_label}  ·  {n} argument{'s' if n != 1 else ''}", expanded=(case_effect == "WIN_DRIVER" and confidence == "HIGH")):
            st.markdown(header_html, unsafe_allow_html=True)
            for m in matches:
                st.markdown(
                    arg_card_html(
                        m.get("case_name") or "Unknown",
                        m.get("summary") or "",
                        m.get("relevance_to_ws") or "",
                        m.get("causal_weight_reason") or "",
                        m.get("paragraph_reference") or m.get("source_pointer") or "",
                    ),
                    unsafe_allow_html=True,
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

    # Case selector — single dropdown, canvas-level
    st.markdown(
        """
        <div class='hero'>
          <div class='eyebrow'>Single-case outcome monitor</div>
          <div class='hero-title'>Case review</div>
          <p class='hero-sub'>Select a case to inspect its outcome, theme guidance, and deterministic review layer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    case_idx = st.selectbox(
        "Select case",
        range(len(case_paths)),
        format_func=lambda i: _case_label_from_path(case_paths[i]),
        label_visibility="collapsed",
    )
    case_path = case_paths[case_idx]

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

    st.markdown("<div class='view-strip'>", unsafe_allow_html=True)
    section = st.radio(
        "Single-case view",
        ["Overview", "Case intelligence", "Theme guidance", "Case review queue", "Strategy brief"],
        horizontal=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if section == "Overview":
        render_legal_intelligence(case_json)
        render_optimization_interpretation(aggregation_json)
        render_guidance(aggregation_json)
    elif section == "Case intelligence":
        render_legal_intelligence(case_json)
    elif section == "Theme guidance":
        render_optimization_interpretation(aggregation_json)
    elif section == "Case review queue":
        render_theme_store(theme_store_json)
    elif section == "Strategy brief":
        render_guidance(aggregation_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Calibrator Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.sidebar.title("Calibrator")

    tab_single, tab_corpus = st.tabs(["Single Case", "Argument Library"])

    with tab_single:
        render_single_case_tab()

    with tab_corpus:
        render_corpus_theme_store()


if __name__ == "__main__":
    main()
