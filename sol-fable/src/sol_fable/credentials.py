"""In-memory provider credentials that are safe to inspect and log."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr


class ProviderCredentials(BaseModel):
    """API credentials resolved by the UI without persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def claude_configured(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value().strip())

