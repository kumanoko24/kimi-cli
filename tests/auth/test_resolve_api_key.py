"""Tests for OAuthManager: resolve_api_key and ensure_fresh behavior."""

import base64
import json
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from pydantic import SecretStr

from kimi_cli.auth import OPENAI_CODEX_PLATFORM_ID
from kimi_cli.auth.codex_oauth import _codex_token_from_response, openai_codex_session_email
from kimi_cli.auth.oauth import (
    _REJECTED_REFRESH_TOKENS,
    OAuthManager,
    OAuthToken,
    OAuthUnauthorized,
    _save_to_file,
    load_tokens,
)
from kimi_cli.auth.platforms import managed_model_key, managed_provider_key
from kimi_cli.config import Config, LLMModel, LLMProvider, OAuthRef, Services
from kimi_cli.llm import LLM


@pytest.fixture(autouse=True)
def _clear_rejected_refresh_tokens():
    _REJECTED_REFRESH_TOKENS.clear()
    yield
    _REJECTED_REFRESH_TOKENS.clear()


def _make_config(*, with_oauth: bool = True, api_key: str = "") -> Config:
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr(api_key),
        oauth=OAuthRef(storage="file", key="oauth/kimi-code") if with_oauth else None,
    )
    model = LLMModel(provider="managed:kimi-code", model="test-model", max_context_size=100_000)
    return Config(
        default_model="managed:kimi-code/test-model",
        providers={"managed:kimi-code": provider},
        models={"managed:kimi-code/test-model": model},
        services=Services(),
    )


def _make_openai_codex_config() -> Config:
    provider_key = managed_provider_key(OPENAI_CODEX_PLATFORM_ID)
    model_key = managed_model_key(OPENAI_CODEX_PLATFORM_ID, "codex-mini-latest")
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key="oauth/openai-codex"),
    )
    model = LLMModel(provider=provider_key, model="codex-mini-latest", max_context_size=128_000)
    return Config(
        default_model=model_key,
        providers={provider_key: provider},
        models={model_key: model},
        services=Services(),
    )


def _make_oauth_manager(config: Config, initial_token: OAuthToken | None = None) -> OAuthManager:
    """Create an OAuthManager with mocked disk I/O."""
    with patch("kimi_cli.auth.oauth.load_tokens", return_value=initial_token):
        return OAuthManager(config)


def test_resolve_api_key_returns_oauth_token_when_available():
    config = _make_config(with_oauth=True)
    token = OAuthToken(
        access_token="oauth-access-123",
        refresh_token="refresh-123",
        expires_at=0.0,
        scope="",
        token_type="Bearer",
    )
    oauth = _make_oauth_manager(config, initial_token=token)

    ref = OAuthRef(storage="file", key="oauth/kimi-code")
    result = oauth.resolve_api_key(SecretStr(""), ref)

    assert result == "oauth-access-123"


def test_resolve_api_key_falls_back_to_api_key_when_no_token():
    config = _make_config(with_oauth=True)
    oauth = _make_oauth_manager(config, initial_token=None)
    ref = OAuthRef(storage="file", key="oauth/kimi-code")

    with patch("kimi_cli.auth.oauth.load_tokens", return_value=None):
        result = oauth.resolve_api_key(SecretStr("fallback-key"), ref)

    assert result == "fallback-key"


def test_resolve_api_key_no_warning_without_oauth_ref():
    """When oauth ref is None, no warning should be emitted."""
    config = _make_config(with_oauth=False)
    oauth = _make_oauth_manager(config)

    result = oauth.resolve_api_key(SecretStr("my-api-key"), None)

    assert result == "my-api-key"


def test_resolve_api_key_falls_back_when_token_has_empty_access_token():
    """Token loaded but access_token is empty should trigger fallback."""
    config = _make_config(with_oauth=True)
    empty_token = OAuthToken(
        access_token="",
        refresh_token="refresh-123",
        expires_at=0.0,
        scope="",
        token_type="Bearer",
    )
    oauth = _make_oauth_manager(config, initial_token=empty_token)
    ref = OAuthRef(storage="file", key="oauth/kimi-code")

    with patch("kimi_cli.auth.oauth.load_tokens", return_value=empty_token):
        result = oauth.resolve_api_key(SecretStr("fallback"), ref)

    assert result == "fallback"


@pytest.mark.asyncio
async def test_resolve_api_key_falls_back_after_rejected_refresh_token(tmp_path, monkeypatch):
    """After a confirmed refresh 401, keep the file but stop preferring the
    same persisted OAuth token over a configured static API key.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    config = _make_config(with_oauth=True, api_key="fallback-key")
    token = OAuthToken(
        access_token="oauth-access-123",
        refresh_token="refresh-123",
        expires_at=time.time() + 100,
        scope="",
        token_type="Bearer",
        expires_in=100,
    )
    _save_to_file("oauth/kimi-code", token)

    oauth = OAuthManager(config)
    ref = OAuthRef(storage="file", key="oauth/kimi-code")

    with (
        patch(
            "kimi_cli.auth.oauth.refresh_token",
            AsyncMock(side_effect=OAuthUnauthorized("revoked")),
        ),
        patch("kimi_cli.auth.oauth.asyncio.sleep", new=AsyncMock()),
        pytest.raises(OAuthUnauthorized, match="revoked"),
    ):
        await oauth.ensure_fresh(force=True)

    result = oauth.resolve_api_key(config.providers["managed:kimi-code"].api_key, ref)
    assert result == "fallback-key"


# ---------------------------------------------------------------------------
# ensure_fresh() with runtime=None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_fresh_without_runtime_caches_token():
    """ensure_fresh(runtime=None) should load and cache the token without
    requiring a Runtime — used by title generation and other lightweight callers.
    """
    config = _make_config(with_oauth=True)
    oauth = _make_oauth_manager(config, initial_token=None)

    fresh_token = OAuthToken(
        access_token="fresh-access-token",
        refresh_token="refresh-123",
        expires_at=time.time() + 3600,
        scope="",
        token_type="Bearer",
    )

    with patch("kimi_cli.auth.oauth.load_tokens", return_value=fresh_token):
        await oauth.ensure_fresh()  # no runtime

    # After ensure_fresh, resolve_api_key should return the cached token
    ref = OAuthRef(storage="file", key="oauth/kimi-code")
    result = oauth.resolve_api_key(SecretStr(""), ref)
    assert result == "fresh-access-token"


@pytest.mark.asyncio
async def test_ensure_fresh_without_runtime_refreshes_expired_token():
    """ensure_fresh(runtime=None) should refresh an expired token and update
    the internal cache, so the next resolve_api_key returns the new token.
    """
    config = _make_config(with_oauth=True)
    oauth = _make_oauth_manager(config, initial_token=None)

    expired_token = OAuthToken(
        access_token="expired-access",
        refresh_token="refresh-123",
        expires_at=time.time() - 100,  # expired
        scope="",
        token_type="Bearer",
    )
    refreshed_token = OAuthToken(
        access_token="refreshed-access",
        refresh_token="new-refresh",
        expires_at=time.time() + 3600,
        scope="",
        token_type="Bearer",
    )

    with (
        patch("kimi_cli.auth.oauth.load_tokens", return_value=expired_token),
        patch(
            "kimi_cli.auth.oauth.refresh_token",
            new_callable=AsyncMock,
            return_value=refreshed_token,
        ),
        patch("kimi_cli.auth.oauth.save_tokens"),
    ):
        await oauth.ensure_fresh()  # no runtime — should still refresh

    ref = OAuthRef(storage="file", key="oauth/kimi-code")
    result = oauth.resolve_api_key(SecretStr(""), ref)
    assert result == "refreshed-access"


def test_openai_codex_cli_auth_imported_before_initial_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "kimi"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "auth.json").write_text(
        """
        {
          "tokens": {
            "access_token": "codex-access",
            "refresh_token": "codex-refresh",
            "account_id": "chatgpt-account",
            "email": "codex-user@example.com"
          }
        }
        """,
        encoding="utf-8",
    )

    config = _make_openai_codex_config()
    oauth = OAuthManager(config)
    provider = config.providers[managed_provider_key(OPENAI_CODEX_PLATFORM_ID)]
    ref = OAuthRef(storage="file", key="oauth/openai-codex")

    assert provider.custom_headers == {"ChatGPT-Account-Id": "chatgpt-account"}
    assert oauth.resolve_api_key(SecretStr(""), ref) == "codex-access"
    imported_token = load_tokens(ref)
    assert imported_token is not None
    assert imported_token.metadata["email"] == "codex-user@example.com"
    assert openai_codex_session_email(config, config.models[config.default_model]) == (
        "codex-user@example.com"
    )


def test_openai_codex_existing_token_hydrates_email_from_codex_cli_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "kimi"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "auth.json").write_text(
        """
        {
          "tokens": {
            "access_token": "codex-cli-access",
            "refresh_token": "codex-cli-refresh",
            "account_id": "chatgpt-account",
            "email": "codex-user@example.com"
          }
        }
        """,
        encoding="utf-8",
    )
    _save_to_file(
        "oauth/openai-codex",
        OAuthToken(
            access_token="existing-access",
            refresh_token="existing-refresh",
            expires_at=time.time() + 3600,
            scope="",
            token_type="Bearer",
        ),
    )

    config = _make_openai_codex_config()
    provider = config.providers[managed_provider_key(OPENAI_CODEX_PLATFORM_ID)]
    provider.custom_headers = {"ChatGPT-Account-Id": "chatgpt-account"}
    OAuthManager(config)

    stored = load_tokens(OAuthRef(storage="file", key="oauth/openai-codex"))
    assert stored is not None
    assert stored.access_token == "existing-access"
    assert stored.metadata["email"] == "codex-user@example.com"
    assert openai_codex_session_email(config, config.models[config.default_model]) == (
        "codex-user@example.com"
    )


def test_openai_codex_session_email_requires_active_codex_model(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "kimi"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _save_to_file(
        "oauth/openai-codex",
        OAuthToken(
            access_token="codex-access",
            refresh_token="codex-refresh",
            expires_at=0.0,
            scope="",
            token_type="Bearer",
            metadata={"email": "codex-user@example.com"},
        ),
    )
    codex_config = _make_openai_codex_config()
    kimi_config = _make_config(with_oauth=True)

    assert openai_codex_session_email(codex_config, codex_config.models[codex_config.default_model])
    assert (
        openai_codex_session_email(kimi_config, kimi_config.models[kimi_config.default_model])
        is None
    )

    codex_config.providers[managed_provider_key(OPENAI_CODEX_PLATFORM_ID)].oauth = None
    assert (
        openai_codex_session_email(codex_config, codex_config.models[codex_config.default_model])
        is None
    )


def _unsigned_jwt(payload: dict[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"e30.{encoded}.signature"


def test_openai_codex_response_id_token_metadata_extracts_email():
    token = _codex_token_from_response(
        {
            "access_token": "codex-access",
            "refresh_token": "codex-refresh",
            "expires_in": 3600,
            "scope": "openid profile email offline_access",
            "token_type": "Bearer",
            "id_token": _unsigned_jwt(
                {
                    "email": "codex-user@example.com",
                    "chatgpt_account_id": "chatgpt-account",
                }
            ),
        }
    )

    assert token.metadata == {
        "account_id": "chatgpt-account",
        "email": "codex-user@example.com",
    }


def test_openai_codex_account_id_updates_live_openai_client(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "kimi"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    config = _make_openai_codex_config()
    oauth = OAuthManager(config)
    provider_key = managed_provider_key(OPENAI_CODEX_PLATFORM_ID)
    model_key = managed_model_key(OPENAI_CODEX_PLATFORM_ID, "codex-mini-latest")
    chat_provider = OpenAIResponses(
        model="codex-mini-latest",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="access-token",
    )
    llm = LLM(
        chat_provider=chat_provider,
        max_context_size=128_000,
        capabilities=set(),
        model_config=config.models[model_key],
        provider_config=config.providers[provider_key],
    )
    runtime = cast(Any, SimpleNamespace(config=config, llm=llm))

    oauth._apply_openai_codex_account_id(runtime, "live-account")

    assert chat_provider.client.default_headers["ChatGPT-Account-Id"] == "live-account"


@pytest.mark.asyncio
async def test_openai_codex_force_refresh_does_not_preemptively_import(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "kimi"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "auth.json").write_text(
        """
        {
          "tokens": {
            "access_token": "imported-access",
            "refresh_token": "imported-refresh",
            "account_id": "imported-account"
          }
        }
        """,
        encoding="utf-8",
    )
    existing = OAuthToken(
        access_token="existing-access",
        refresh_token="existing-refresh",
        expires_at=time.time() + 3600,
        scope="",
        token_type="Bearer",
        metadata={"email": "codex-user@example.com"},
    )
    refreshed = OAuthToken(
        access_token="refreshed-access",
        refresh_token="refreshed-refresh",
        expires_at=time.time() + 3600,
        scope="",
        token_type="Bearer",
    )
    _save_to_file("oauth/openai-codex", existing)

    config = _make_openai_codex_config()
    oauth = OAuthManager(config)

    refresh = AsyncMock(return_value=refreshed)
    with patch("kimi_cli.auth.codex_oauth.refresh_openai_codex_token", refresh):
        await oauth.ensure_fresh(force=True)

    refresh.assert_awaited_once_with("existing-refresh")
    stored = load_tokens(OAuthRef(storage="file", key="oauth/openai-codex"))
    assert stored is not None
    assert stored.access_token == "refreshed-access"
    assert stored.metadata["email"] == "codex-user@example.com"


@pytest.mark.asyncio
async def test_ensure_fresh_with_no_oauth_ref_is_noop():
    """ensure_fresh() should be a no-op when no OAuth ref is configured."""
    config = _make_config(with_oauth=False)
    oauth = _make_oauth_manager(config)

    # Should not raise or do anything
    await oauth.ensure_fresh()
