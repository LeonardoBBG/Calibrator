from pydantic import SecretStr

from sol_fable.credentials import ProviderCredentials


def test_provider_credentials_are_masked_and_report_configuration() -> None:
    credentials = ProviderCredentials(
        openai_api_key="openai-secret-value",
        anthropic_api_key="anthropic-secret-value",
    )

    assert credentials.openai_configured
    assert credentials.claude_configured
    assert isinstance(credentials.openai_api_key, SecretStr)
    assert "openai-secret-value" not in repr(credentials)
    assert "anthropic-secret-value" not in repr(credentials)


def test_empty_provider_credentials_are_not_configured() -> None:
    credentials = ProviderCredentials(openai_api_key="", anthropic_api_key=None)

    assert not credentials.openai_configured
    assert not credentials.claude_configured

