"""
theme_generator.py — automated WS Theme Dictionary draft generator.

Generates a case-specific WS Theme Dictionary from the FATE handoff
(ws_narrative + x_tests) and optionally a manually written WS text.

CRITICAL RULE: This module NEVER reads, modifies, or overwrites the hardened
WS_Controlled_Theme_Dictionary_v1_2_final.json.  All output is always written
to a NEW timestamped file in the caller-specified output directory.

Two-pass design:
  Draft pass  — LLM generates 10–20 themes in ONE call (so cross-references
                in exclude_when are coherent).
  Cap pass    — Second LLM call validates each draft theme against the
                manual WS text; marks CONFIRMED, MODIFIED, or REMOVED.

Public API:
    theme_dict_from_handoff(ws_narrative, x_tests, *, manual_ws_text, ...)
        -> dict   draft theme dictionary (same schema as hardened dict)
    save_theme_dict(theme_dict, out_dir, *, label="") -> Path
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_VALID_PERMITTED_ACTIONS = frozenset({
    "REINFORCE", "REFRAME", "REVIEW_MANUALLY",
    "ADD FACT", "ADD EVIDENCE ANCHOR", "SOFTEN", "REMOVE", "DO_NOT_USE",
})

_VALID_THEME_STATUSES = frozenset({"PRESENT", "DRAFT_CANDIDATE", "RISK_ONLY", "REMOVED"})

# ---------------------------------------------------------------------------
# LLM plumbing (self-contained — same pattern as doc_to_moltie.py)
# ---------------------------------------------------------------------------

def _ollama_chat(model: str, system: str, user: str, temperature: float) -> str:
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or ""},
            {"role": "user",   "content": user or ""},
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed (url={url}): {e}") from e
    try:
        obj = json.loads(raw)
        return (obj.get("message") or {}).get("content", "") or ""
    except Exception:
        return raw


def _openai_chat(model: str, system: str, user: str, temperature: float) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("FATE_OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")

    try:
        import openai
    except ImportError:
        return _openai_chat_http(model, system, user, temperature, api_key, base_url)

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "response_format" not in str(exc):
            raise
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
        )
    return resp.choices[0].message.content or ""


def _openai_chat_http(
    model: str,
    system: str,
    user: str,
    temperature: float,
    api_key: str,
    base_url: str = "",
) -> str:
    url = f"{base_url or 'https://api.openai.com/v1'}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
    }

    def _post(body: dict) -> str:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        outer = json.loads(raw)
        choice = (outer.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(
                f"OpenAI returned empty content (finish_reason={choice.get('finish_reason')!r})."
            )
        return content

    try:
        return _post(payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "response_format" in body:
            payload.pop("response_format", None)
            return _post(payload)
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {body[:800]}") from exc


def _call_llm(provider: str, model: str, system: str, user: str, temperature: float) -> str:
    norm = (provider or "ollama").strip().lower()
    if norm in {"openai", "api", "cloud"}:
        return _openai_chat(model, system, user, temperature)
    return _ollama_chat(model, system, user, temperature)


def _strip_raw(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", s).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].strip()
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        s = s[start : end + 1]
    return s


# ---------------------------------------------------------------------------
# Draft-pass prompts
# ---------------------------------------------------------------------------

_THEME_SCHEMA_BLOCK = """{
  "theme_id": "T01_UNIQUE_UPPER_SLUG",
  "theme_priority": 1,
  "theme_name": "Human-Readable Theme Name",
  "theme_status": "PRESENT",
  "definition": "One paragraph — what aspect of the claimant's defence does this theme cover.",
  "include_when": ["signal condition 1", "signal condition 2", "signal condition 3"],
  "exclude_when": ["condition that should map to T02 instead", "condition → map to T03"],
  "common_subthemes": ["concrete factual angle 1", "angle 2", "angle 3"],
  "preferred_ws_destination": ["WS Section Heading 1", "WS Section Heading 2"],
  "evidence_anchor_types": ["document type 1", "record type 2", "communication type 3"],
  "duplication_guardrail": "This theme is X. Do not merge with T02 which is Y.",
  "permitted_actions": ["REINFORCE", "REVIEW_MANUALLY"],
  "forbidden_uses": ["must not do this", "must not assert that"],
  "example_mapping_language": "Signal: 'example ET judgment phrase.' → Map to T01."
}"""


def _draft_system_prompt() -> str:
    return (
        "You are an employment law specialist building a WS Theme Dictionary for an unfair dismissal case.\n"
        "Task: given the claimant's WS narrative and X-tests (evidential gap propositions), "
        "generate 10–20 themes that represent the claimant's distinct defence pillars for the Witness Statement.\n"
        "\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY a single valid JSON object. No markdown fences. No preamble. No trailing text.\n"
        '- Top-level key: "ws_theme_dictionary" — a JSON array of theme objects.\n'
        "- Generate between 10 and 20 themes. Fewer is better than padding with duplicates.\n"
        "- All themes MUST be generated in ONE call so cross-references in exclude_when are coherent.\n"
        "\n"
        "SCHEMA FOR EACH THEME:\n"
        f"{_THEME_SCHEMA_BLOCK}\n"
        "\n"
        "FIELD RULES:\n"
        '- theme_id: "T{NN}_{UPPER_SNAKE}" where NN is zero-padded (T01, T02…). Must be globally unique.\n'
        "- theme_priority: unique integer 1–N. Represents conflict-resolution rank (1 = highest).\n"
        "  NOTE: theme_priority is NOT the same as the numeric part of theme_id. Assign priority by analytical importance.\n"
        '- theme_status: "PRESENT" if anchored to WS content; "DRAFT_CANDIDATE" if inferred from X-tests only.\n'
        '- include_when: 3–5 strings — each a distinct signal condition for this theme.\n'
        '- exclude_when: 2–4 strings — cross-reference adjacent themes by T-id ("→ map to T02").\n'
        '- common_subthemes: 3–5 concrete factual angles specific to this case.\n'
        '- preferred_ws_destination: 1–3 named WS section headings.\n'
        '- evidence_anchor_types: 2–5 evidence or document types that ground this theme.\n'
        '- duplication_guardrail: one sentence with T-id cross-references to adjacent themes.\n'
        '- permitted_actions: subset of '
        '["REINFORCE","REFRAME","REVIEW_MANUALLY","ADD FACT","ADD EVIDENCE ANCHOR","SOFTEN","REMOVE","DO_NOT_USE"].\n'
        '- forbidden_uses: 2–3 strings — things this theme must not do or assert.\n'
        '- example_mapping_language: one concrete mapping example ("Signal: \'…\' → Map to T0N.").\n'
        "\n"
        "THEME DESIGN PRINCIPLES:\n"
        "- Themes are DEFENCE PILLARS, not allegation summaries.\n"
        "- Themes must be mutually exclusive at their primary level — no overlapping scope.\n"
        "- theme_priority 1 should be the theme most critical for resolving conflicting signal mappings.\n"
        "- The highest-numbered theme should be a risk/quarantine theme for weak or speculative material.\n"
        "- Prioritise: chronology and consistency, role and expectation context, process failures, "
        "evidence gaps, management awareness, proportionality, comparators.\n"
        "- Do NOT invent facts not present in the WS narrative or X-tests.\n"
        "- Do NOT mirror legal tests directly — themes should be factual and narrative anchors.\n"
    )


def _draft_user_prompt(
    ws_narrative: str,
    x_tests: list[dict],
    manual_ws_text: str | None,
) -> str:
    narrative_snippet = (ws_narrative or "")[:5_000]
    trunc_note = (
        f"\n[Truncated — {len(ws_narrative):,} chars total, showing first 5,000]"
        if len(ws_narrative or "") > 5_000
        else ""
    )

    x_summary: list[str] = []
    for i, xt in enumerate(x_tests or [], start=1):
        name = str(xt.get("name", f"X{i}")).strip()
        defn = str(xt.get("definition", "")).strip()
        indicators = xt.get("positive_indicators", [])
        x_summary.append(f"X{i}: {name} — {defn}")
        if indicators:
            x_summary.append(f"   evidence signals: {', '.join(str(x) for x in indicators[:3])}")

    blocks = [
        f"WS NARRATIVE:\n<<<\n{narrative_snippet}{trunc_note}\n>>>",
        "X-TESTS (evidential gap propositions):\n<<<\n"
        + "\n".join(x_summary or ["(none)"])
        + "\n>>>",
    ]

    if manual_ws_text and (manual_ws_text or "").strip():
        ws_snippet = manual_ws_text[:4_000]
        ws_trunc = (
            f"\n[Truncated — {len(manual_ws_text):,} chars, showing first 4,000]"
            if len(manual_ws_text) > 4_000
            else ""
        )
        blocks.append(
            f"MANUAL WITNESS STATEMENT (use to ground themes in real WS content):\n"
            f"<<<\n{ws_snippet}{ws_trunc}\n>>>"
        )

    blocks.append(
        'Generate 10–20 themes. Output ONLY the JSON object: {"ws_theme_dictionary": [...]}'
    )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Draft-pass parser
# ---------------------------------------------------------------------------

def _list_of_str(val: object, default: list[str]) -> list[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    return default


def _parse_theme_draft(raw: str) -> list[dict]:
    s = _strip_raw(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    themes_raw = obj.get("ws_theme_dictionary")
    if not isinstance(themes_raw, list) or not themes_raw:
        raise ValueError('"ws_theme_dictionary" must be a non-empty list')

    ids_seen: set[str] = set()
    priorities_seen: set[int] = set()
    cleaned: list[dict] = []

    for i, t in enumerate(themes_raw):
        if not isinstance(t, dict):
            continue

        theme_id = str(t.get("theme_id", f"T{i+1:02d}_UNKNOWN")).strip().upper()
        # Deduplicate ids (shouldn't happen but guard against LLM errors)
        base_id = theme_id
        suffix = 0
        while theme_id in ids_seen:
            suffix += 1
            theme_id = f"{base_id}_{suffix}"
        ids_seen.add(theme_id)

        try:
            priority = int(t.get("theme_priority", i + 1))
        except (TypeError, ValueError):
            priority = i + 1
        while priority in priorities_seen:
            priority += 1
        priorities_seen.add(priority)

        status = str(t.get("theme_status", "DRAFT_CANDIDATE")).strip().upper()
        if status not in _VALID_THEME_STATUSES:
            status = "DRAFT_CANDIDATE"

        actions_raw = t.get("permitted_actions", ["REVIEW_MANUALLY"])
        if not isinstance(actions_raw, list):
            actions_raw = ["REVIEW_MANUALLY"]
        permitted_actions = [
            a for a in (str(x).strip().upper() for x in actions_raw)
            if a in _VALID_PERMITTED_ACTIONS
        ] or ["REVIEW_MANUALLY"]

        cleaned.append({
            "theme_id": theme_id,
            "theme_priority": priority,
            "theme_name": str(t.get("theme_name", "")).strip(),
            "theme_status": status,
            "definition": str(t.get("definition", "")).strip(),
            "include_when": _list_of_str(t.get("include_when"), []),
            "exclude_when": _list_of_str(t.get("exclude_when"), []),
            "common_subthemes": _list_of_str(t.get("common_subthemes"), []),
            "preferred_ws_destination": _list_of_str(t.get("preferred_ws_destination"), []),
            "evidence_anchor_types": _list_of_str(t.get("evidence_anchor_types"), []),
            "duplication_guardrail": str(t.get("duplication_guardrail", "")).strip(),
            "permitted_actions": permitted_actions,
            "forbidden_uses": _list_of_str(t.get("forbidden_uses"), []),
            "example_mapping_language": str(t.get("example_mapping_language", "")).strip(),
        })

    if not cleaned:
        raise ValueError("No valid themes found after parsing")

    cleaned.sort(key=lambda x: x["theme_priority"])
    return cleaned


# ---------------------------------------------------------------------------
# Cap-pass prompts and parser
# ---------------------------------------------------------------------------

def _cap_system_prompt() -> str:
    return (
        "You are an employment law specialist reviewing a draft WS Theme Dictionary.\n"
        "Task: for each draft theme, determine whether it is grounded in the Witness Statement text.\n"
        "\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY a single valid JSON object. No markdown fences. No commentary.\n"
        '- Top-level key: "cap_results" — a JSON array, one entry per input theme.\n'
        "- Each entry must have exactly these keys:\n"
        '  "theme_id"            — copied from the draft theme (string)\n'
        '  "cap_status"          — "CONFIRMED", "MODIFIED", or "REMOVED"\n'
        '  "notes"               — one sentence explaining the decision\n'
        '  "modified_definition" — revised definition if MODIFIED; null otherwise\n'
        "\n"
        "DECISION RULES:\n"
        "- CONFIRMED: theme is clearly anchored in WS content (a specific fact, event, or argument).\n"
        "- MODIFIED: theme is partially anchored but its definition needs narrowing to remove unsupported claims.\n"
        "- REMOVED: theme is speculative, has no WS anchor, or duplicates another confirmed theme.\n"
        "- Do NOT remove themes that are anchored even if they appear weak.\n"
        "- Do NOT confirm themes that require inventing facts not stated in the WS.\n"
    )


def _cap_user_prompt(draft_themes: list[dict], manual_ws_text: str) -> str:
    ws_snippet = manual_ws_text[:5_000]
    ws_trunc = (
        f"\n[Truncated — {len(manual_ws_text):,} chars, showing first 5,000]"
        if len(manual_ws_text) > 5_000
        else ""
    )
    themes_summary = [
        {
            "theme_id": t["theme_id"],
            "theme_name": t["theme_name"],
            "definition": t["definition"][:300],
        }
        for t in draft_themes
    ]
    return (
        f"WITNESS STATEMENT:\n<<<\n{ws_snippet}{ws_trunc}\n>>>\n\n"
        f"DRAFT THEMES:\n<<<\n"
        f"{json.dumps(themes_summary, ensure_ascii=False, indent=2)}\n>>>\n\n"
        'Review each theme. Output ONLY: {"cap_results": [...]}'
    )


def _parse_cap_results(raw: str, draft_themes: list[dict]) -> list[dict]:
    s = _strip_raw(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cap pass output is not valid JSON: {e}") from e

    results_raw = obj.get("cap_results", [])
    if not isinstance(results_raw, list):
        raise ValueError('"cap_results" must be a list')

    cap_map: dict[str, dict] = {
        str(r.get("theme_id", "")).strip().upper(): r
        for r in results_raw
        if isinstance(r, dict) and r.get("theme_id")
    }

    capped: list[dict] = []
    for theme in draft_themes:
        tid = theme["theme_id"]
        cap = cap_map.get(tid, {})
        status = str(cap.get("cap_status", "CONFIRMED")).strip().upper()

        if status == "REMOVED":
            updated = {**theme, "theme_status": "REMOVED"}
        elif status == "MODIFIED":
            new_def = cap.get("modified_definition")
            if new_def and str(new_def).strip():
                updated = {**theme, "definition": str(new_def).strip()}
            else:
                updated = dict(theme)
        else:
            updated = dict(theme)

        cap_note = str(cap.get("notes", "")).strip()
        if cap_note:
            updated["_cap_notes"] = cap_note

        capped.append(updated)

    return capped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def theme_dict_from_handoff(
    ws_narrative: str,
    x_tests: list[dict],
    *,
    manual_ws_text: str | None = None,
    provider: str = "ollama",
    model: str = "mistral-small3.2:latest",
    temperature: float = 0.2,
    attempts: int = 2,
    run_cap_pass: bool = True,
) -> dict:
    """
    Generate a case-specific WS Theme Dictionary from the FATE handoff.

    Args:
        ws_narrative:   WS narrative text from the handoff.
        x_tests:        List of x_test dicts from the handoff.
        manual_ws_text: Optional full text of a manually written WS.
                        If provided and run_cap_pass=True, validates themes against it.
        provider:       "ollama" (local) or "openai" / "api" / "cloud".
        model:          LLM model name.
        temperature:    Sampling temperature.
        attempts:       Retry attempts on validation failure.
        run_cap_pass:   If True AND manual_ws_text is provided, runs the cap pass.

    Returns:
        Dict with "dictionary_metadata" and "ws_theme_dictionary" keys,
        matching the schema of WS_Controlled_Theme_Dictionary_v1_2_final.json.

    CRITICAL: The hardened dict is NEVER read, modified, or overwritten.
    This function always produces a NEW draft dict.
    """
    if not (ws_narrative or "").strip() and not x_tests:
        raise ValueError("ws_narrative and x_tests are both empty — cannot generate themes")

    system = _draft_system_prompt()
    base_user = _draft_user_prompt(ws_narrative, x_tests, manual_ws_text)
    current_user = base_user
    last_err: Exception | None = None
    themes: list[dict] = []

    for attempt in range(1, attempts + 1):
        try:
            raw = _call_llm(provider, model, system, current_user, temperature)
            themes = _parse_theme_draft(raw)
            print(f"[theme_generator] Draft pass: {len(themes)} themes generated.")
            break
        except Exception as exc:
            last_err = exc
            if attempt < attempts:
                current_user = (
                    base_user
                    + f"\n\nPREVIOUS ATTEMPT {attempt} FAILED — reason: {exc}\n"
                    'Output ONLY valid JSON: {"ws_theme_dictionary": [...]}'
                )
    else:
        raise ValueError(
            f"Failed to generate theme dictionary after {attempts} attempts. "
            f"Last error: {last_err}"
        )

    if run_cap_pass and manual_ws_text and (manual_ws_text or "").strip():
        try:
            cap_sys = _cap_system_prompt()
            cap_user = _cap_user_prompt(themes, manual_ws_text)
            cap_raw = _call_llm(provider, model, cap_sys, cap_user, temperature)
            themes = _parse_cap_results(cap_raw, themes)
            n_removed = sum(1 for t in themes if t.get("theme_status") == "REMOVED")
            print(f"[theme_generator] Cap pass: {n_removed} theme(s) removed or flagged.")
        except Exception as exc:
            print(f"[theme_generator] Cap pass failed (non-fatal, draft returned): {exc}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_themes = len(themes)
    n_removed = sum(1 for t in themes if t.get("theme_status") == "REMOVED")

    return {
        "dictionary_metadata": {
            "dictionary_name": "WS_Draft_Theme_Dictionary",
            "version": f"draft_{ts}",
            "purpose": (
                "DRAFT — case-specific WS theme taxonomy generated automatically from the FATE handoff. "
                "Subject to human review before use in production mapping."
            ),
            "source": (
                "FATE handoff (ws_narrative + x_tests)"
                + (" + manual WS cap pass" if manual_ws_text and run_cap_pass else "")
            ),
            "created_for": "Employment Tribunal WS calibration",
            "strict_mapping_rule": (
                "DRAFT ONLY. Do not use for production mapping without human review. "
                "Themes with theme_status DRAFT_CANDIDATE or REMOVED require review before activation."
            ),
            "generator": "theme_generator.py",
            "generated_at": ts,
            "theme_count": n_themes,
            "removed_by_cap_pass": n_removed,
            "WARNING": (
                "This is an automatically generated draft. "
                "WS_Controlled_Theme_Dictionary_v1_2_final.json has NOT been modified."
            ),
        },
        "ws_theme_dictionary": themes,
        "global_mapping_rules": {
            "default_rule": (
                "DRAFT: Map each judgment signal to one primary theme. "
                "Use theme_priority to resolve conflicts."
            ),
            "new_theme_rule": "DRAFT: Human reviewers may add, remove, or merge themes.",
            "speculation_rule": (
                "Any theme with theme_status DRAFT_CANDIDATE or REMOVED must be reviewed "
                "and confirmed before production use."
            ),
        },
    }


def save_theme_dict(
    theme_dict: dict,
    out_dir: str | Path,
    *,
    label: str = "",
) -> Path:
    """
    Save the draft theme dict to a NEW timestamped file.

    NEVER overwrites existing files.  The hardened dict lives in
    Calibrator/input/dictionary/ — this function must never write there.

    Returns the path of the saved file.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9_]+", "_", (label or "").lower().strip())[:40]
    name_parts = [ts, "ws_theme_dict_draft"]
    if slug:
        name_parts.append(slug)
    filename = "__".join(name_parts) + ".json"
    path = out / filename

    if path.exists():
        path = out / f"{ts}_{os.getpid()}__{filename}"

    path.write_text(json.dumps(theme_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[theme_generator] Draft saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Handoff helpers
# ---------------------------------------------------------------------------

def x_tests_from_handoff(handoff: dict) -> list[dict]:
    """
    Extract a flat list of x_test dicts from a doc_to_moltie handoff JSON.

    Handles both single-row and multi-row handoffs.  Returns all x_tests
    found across all rows, deduplicated by name.
    """
    seen_names: set[str] = set()
    all_tests: list[dict] = []
    for row in (handoff.get("rows") or {}).values():
        y = row.get("y") or {}
        xt_map = y.get("x_tests") or {}
        for xt in (xt_map.values() if isinstance(xt_map, dict) else xt_map):
            if not isinstance(xt, dict):
                continue
            name = str(xt.get("name", "")).strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            all_tests.append(xt)
    return all_tests


def narrative_from_x_tests(x_tests: list[dict]) -> str:
    """
    Build a prose-like WS narrative summary from x_tests when the original
    WS text is not available.  Used as ws_narrative input to theme_dict_from_handoff.
    """
    lines: list[str] = []
    for i, xt in enumerate(x_tests, start=1):
        name = str(xt.get("name", f"Issue {i}")).strip()
        defn = str(xt.get("definition", "")).strip()
        pattern = str(xt.get("pattern", "")).strip()
        indicators = xt.get("positive_indicators", [])
        required = xt.get("required_elements", [])

        lines.append(f"Issue {i}: {name}")
        if defn:
            lines.append(f"  {defn}")
        if pattern:
            lines.append(f"  Evidence pattern: {pattern}")
        if required:
            lines.append(f"  Required: {'; '.join(str(r) for r in required)}")
        if indicators:
            lines.append(f"  Indicators: {'; '.join(str(ind) for ind in indicators[:4])}")
        lines.append("")
    return "\n".join(lines).strip()
