"""Strict configuration loading and writable application-data path selection."""

from __future__ import annotations

import math
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConfigurationError
from .models import BarristerPersona


class _StrictConfigModel(BaseModel):
    """Reject misspelled or obsolete configuration keys at every level."""

    model_config = ConfigDict(extra="forbid")


class Thresholds(_StrictConfigModel):
    high_confidence: float = Field(default=0.85, ge=0, le=1)
    medium_confidence: float = Field(default=0.65, ge=0, le=1)
    human_review_below: float = Field(default=0.65, ge=0, le=1)
    lexical_candidate_floor: float = Field(default=0.12, ge=0, le=1)
    possible_new_case_below: float = Field(default=0.22, ge=0, le=1)


class Paths(_StrictConfigModel):
    case_dir: str = "case"

    @field_validator("case_dir")
    @classmethod
    def validate_case_dir(cls, value: str) -> str:
        candidate = Path(value)
        if not value.strip() or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("paths.case_dir must be a non-empty relative path without '..'")
        return value


class Settings(_StrictConfigModel):
    RANKING_WEIGHT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "outcome_materiality",
            "pleading_alignment",
            "ws_support",
            "respondent_attack_strength",
            "cross_examination_vulnerability",
            "case_law_calibration_value",
        }
    )

    parser_version: str = "0.1.0"
    prompt_version: str = "0.3.0"
    analysis_backend: Literal[
        "deterministic-v1", "ollama", "openai", "anthropic", "dual-api"
    ] = "deterministic-v1"
    n_rounds: int = Field(default=2, ge=1, le=10)
    claimant_barrister: BarristerPersona = BarristerPersona.SOL
    thresholds: Thresholds = Field(default_factory=Thresholds)
    ranking_weights: dict[str, float]
    paths: Paths = Field(default_factory=Paths)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = Field(default="gemma3:27b", min_length=1)
    ollama_num_gpu: int | None = Field(default=999, ge=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_timeout_seconds: float = Field(default=180, gt=0, le=1800)
    llm_max_output_tokens: int = Field(default=4096, ge=256, le=131072)
    ollama_argument_cap: int | None = Field(default=None, ge=1)
    live_argument_cap: int | None = Field(default=20, ge=1)
    ollama_allow_remote_host: bool = False
    openai_model: str = Field(default="gpt-5.6-sol", min_length=1)
    anthropic_model: str = Field(default="claude-sonnet-5", min_length=1)

    @field_validator("ranking_weights", mode="before")
    @classmethod
    def validate_ranking_weights(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("ranking_weights must be a mapping")

        actual_keys = set(value)
        if actual_keys != cls.RANKING_WEIGHT_KEYS:
            missing = sorted(cls.RANKING_WEIGHT_KEYS - actual_keys)
            unexpected = sorted(str(key) for key in actual_keys - cls.RANKING_WEIGHT_KEYS)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError(
                "ranking_weights must contain exactly the six supported keys"
                + (f" ({'; '.join(details)})" if details else "")
            )

        normalized: dict[str, float] = {}
        for key in cls.RANKING_WEIGHT_KEYS:
            raw_weight = value[key]
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"ranking_weights.{key} must be numeric")
            weight = float(raw_weight)
            if not math.isfinite(weight):
                raise ValueError(f"ranking_weights.{key} must be finite")
            if not 0 <= weight <= 1:
                raise ValueError(f"ranking_weights.{key} must be between 0 and 1")
            normalized[key] = weight

        if not math.isclose(
            sum(normalized.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("ranking_weights must sum to 1.0")
        return normalized

    @property
    def weight_total(self) -> float:
        return sum(self.ranking_weights.values())


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_checkout_root() -> Path | None:
    """Return the repository root only when this module is running from ``src``."""

    candidate = project_root()
    if (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "src" / "sol_fable").is_dir()
        and (candidate / "config" / "settings.yaml").is_file()
    ):
        return candidate
    return None


def default_data_root() -> Path:
    """Select a writable project/data root for checkout and installed use.

    ``SOL_FABLE_DATA_DIR`` always wins. Source checkouts retain their historical
    repository-local ``case`` directory, while wheel installations use the
    current user's platform data directory instead of trying to write into
    ``site-packages`` or the installation prefix.
    """

    configured = os.environ.get("SOL_FABLE_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    checkout = source_checkout_root()
    if checkout is not None:
        return checkout.resolve()

    if os.name == "nt":
        windows_base = (
            os.environ.get("LOCALAPPDATA", "").strip()
            or os.environ.get("APPDATA", "").strip()
        )
        base = (
            Path(windows_base).expanduser()
            if windows_base
            else Path.home() / "AppData" / "Local"
        )
        return (base / "Sol-Fable").resolve()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "Sol-Fable").resolve()

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    xdg_base = Path(xdg_data_home).expanduser() if xdg_data_home else None
    if xdg_base is None or not xdg_base.is_absolute():
        xdg_base = Path.home() / ".local" / "share"
    return (xdg_base / "sol-fable").resolve()


def load_settings(path: Path | None = None) -> Settings:
    try:
        if path is not None:
            config_label = str(path)
            text = path.read_text(encoding="utf-8")
        else:
            checkout = source_checkout_root()
            if checkout is not None:
                checkout_config = checkout / "config" / "settings.yaml"
                text = checkout_config.read_text(encoding="utf-8")
                config_label = str(checkout_config)
            else:
                try:
                    packaged = resources.files("sol_fable.resources").joinpath(
                        "default_settings.yaml"
                    )
                    text = packaged.read_text(encoding="utf-8")
                    config_label = "packaged default_settings.yaml"
                except (FileNotFoundError, ModuleNotFoundError, TypeError):
                    fallback = project_root() / "config" / "settings.yaml"
                    text = fallback.read_text(encoding="utf-8")
                    config_label = str(fallback)
        raw: dict[str, Any] = yaml.safe_load(text)
        settings = Settings.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        label = locals().get("config_label", str(path) if path else "default settings")
        raise ConfigurationError(f"Unable to load configuration {label}: {exc}") from exc
    return settings
