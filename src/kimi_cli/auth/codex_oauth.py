"""OpenAI Codex OAuth: PKCE browser flow + headless device code flow.

The CLIENT_ID is the public OAuth client ID used by OpenAI's own Codex CLI
(app_EMoamEEZ73f0CkXaXp7hrann). Per RFC 6749 §2.3.1, client IDs are not
secrets when PKCE is used; they identify the application, not the user.

Token exchange and refresh are performed directly against auth.openai.com
over TLS. JWT id_token claims are parsed for metadata (chatgpt_account_id)
only — signature verification is intentionally skipped because the token is
received over TLS from a trusted origin and used solely for header metadata,
not for authentication.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

import aiohttp

from kimi_cli.auth import OPENAI_CODEX_PLATFORM_ID
from kimi_cli.auth.oauth import (
    OAuthError,
    OAuthEvent,
    OAuthRef,
    OAuthToken,
    OAuthUnauthorized,
    delete_tokens,
    save_tokens,
)
from kimi_cli.auth.platforms import managed_model_key, managed_provider_key
from kimi_cli.config import Config, LLMModel, LLMProvider, save_config
from kimi_cli.utils.aiohttp import new_client_session
from kimi_cli.utils.logging import logger

# Public client ID from OpenAI's Codex CLI — not a secret under PKCE (RFC 6749 §2.3.1).
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_AUTH_HOST = "https://auth.openai.com"
# The OpenAI Python SDK appends "/responses" to base_url, yielding the ChatGPT Codex backend.
OPENAI_CODEX_API_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_OAUTH_KEY = "oauth/openai-codex"
OPENAI_CODEX_OAUTH_PORT = 1455
OPENAI_CODEX_REDIRECT_URI = f"http://127.0.0.1:{OPENAI_CODEX_OAUTH_PORT}/callback"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
# Model IDs served by the ChatGPT Codex backend (update as OpenAI releases new versions).
_DEFAULT_MODEL = "codex-mini-latest"
_DEFAULT_CONTEXT = 128_000
_CALLBACK_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class ImportedCodexAuth:
    account_id: str | None
    path: Path


def _codex_auth_candidates() -> list[Path]:
    candidates: list[Path] = []
    if codex_home := os.getenv("CODEX_HOME"):
        candidates.append(Path(codex_home).expanduser() / "auth.json")
    home = Path.home()
    candidates.extend(
        [
            home / ".codex" / "auth.json",
            home / ".codex-atk" / "auth.json",
            home / ".codex-leo" / "auth.json",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _load_codex_cli_auth(path: Path) -> tuple[OAuthToken, str | None] | None:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    payload = cast(dict[str, Any], raw_payload)
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None
    tokens = cast(dict[str, Any], tokens)
    access_token = str(tokens.get("access_token") or "")
    refresh_token_value = str(tokens.get("refresh_token") or "")
    if not access_token or not refresh_token_value:
        return None
    account_id = tokens.get("account_id")
    token = OAuthToken(
        access_token=access_token,
        refresh_token=refresh_token_value,
        # Codex auth.json does not carry access-token expiry. Force a refresh
        # soon, while still allowing the current access token as an immediate fallback.
        expires_at=0.0,
        scope=OPENAI_CODEX_SCOPE,
        token_type="Bearer",
        expires_in=0.0,
    )
    return token, str(account_id) if account_id else None


def import_openai_codex_cli_auth(oauth_ref: OAuthRef) -> ImportedCodexAuth | None:
    """Import OpenAI Codex CLI auth.json tokens into Kimi's OAuth store.

    This lets Kimi reuse Codex-family auth from CODEX_HOME, ~/.codex,
    ~/.codex-atk, or ~/.codex-leo when ~/.kimi/credentials/openai-codex.json
    is missing or stale.
    """
    for path in _codex_auth_candidates():
        loaded = _load_codex_cli_auth(path)
        if loaded is None:
            continue
        token, account_id = loaded
        save_tokens(oauth_ref, token)
        logger.info("Imported OpenAI Codex auth from {path}", path=str(path))
        return ImportedCodexAuth(account_id=account_id, path=path)
    return None


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_code_verifier() -> str:
    # 64 random bytes → 86-char verifier (512-bit entropy, matching industry standard).
    return secrets.token_urlsafe(64)


def _generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _generate_state() -> str:
    raw = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _parse_jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padding = "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else None
    except Exception:
        return None


def _extract_account_id(id_token: str | None) -> str | None:
    if not id_token:
        return None
    claims = _parse_jwt_claims(id_token)
    if not claims:
        return None
    auth_block_raw = claims.get("https://api.openai.com/auth")
    auth_block = cast(dict[str, Any], auth_block_raw) if isinstance(auth_block_raw, dict) else {}
    orgs_raw = claims.get("organizations")
    orgs: list[dict[str, Any]] = []
    if isinstance(orgs_raw, list):
        for org_raw in cast(list[Any], orgs_raw):
            if isinstance(org_raw, dict):
                orgs.append(cast(dict[str, Any], org_raw))
    first_org_id = orgs[0].get("id") if orgs else None
    return claims.get("chatgpt_account_id") or auth_block.get("chatgpt_account_id") or first_org_id


# ---------------------------------------------------------------------------
# Authorization URL
# ---------------------------------------------------------------------------


def _build_authorize_url(verifier: str, state: str) -> str:
    challenge = _generate_code_challenge(verifier)
    params = {
        "response_type": "code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        "scope": OPENAI_CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": "kimi-cli",
    }
    return f"{OPENAI_CODEX_AUTH_HOST}/oauth/authorize?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Local callback server (browser flow)
# ---------------------------------------------------------------------------


async def _wait_for_callback(
    state: str,
    timeout: float = _CALLBACK_TIMEOUT_SECONDS,
) -> str:
    """Start local HTTP server on the PKCE redirect port, return the auth code.

    The callback server is bound to 127.0.0.1 only (no LAN exposure per RFC 8252 §7.3).
    State is verified on receipt for CSRF protection (RFC 6749 §10.12).
    """
    from aiohttp import web

    loop = asyncio.get_event_loop()
    code_future: asyncio.Future[str] = loop.create_future()

    async def _handler(request: web.Request) -> web.Response:
        if code_future.done():
            return web.Response(text="Already handled.", status=400)
        received_state = request.rel_url.query.get("state")
        code = request.rel_url.query.get("code")
        error = request.rel_url.query.get("error")

        if error:
            code_future.set_exception(OAuthError(f"Authorization error: {error}"))
        elif received_state != state:
            code_future.set_exception(OAuthError("State mismatch — possible CSRF."))
        elif not code:
            code_future.set_exception(OAuthError("No authorization code received."))
        else:
            code_future.set_result(code)

        html = (
            "<html><body><h2>Authorization complete</h2>"
            "<p>You can close this window and return to kimi-cli.</p></body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/callback", _handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", OPENAI_CODEX_OAUTH_PORT)
    try:
        await site.start()
    except OSError as exc:
        await runner.cleanup()
        raise OAuthError(
            f"Could not start callback server on port {OPENAI_CODEX_OAUTH_PORT}: {exc}. "
            "Try --headless mode instead."
        ) from exc
    try:
        return await asyncio.wait_for(code_future, timeout=timeout)
    except TimeoutError as exc:
        raise OAuthError("Browser authorization timed out (5 minutes).") from exc
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------


def _assert_dict_response(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise OAuthError(f"Unexpected response shape in {context}: {type(data).__name__}")
    return cast(dict[str, Any], data)


async def _exchange_code(code: str, verifier: str) -> tuple[OAuthToken, str | None]:
    """Exchange auth code for tokens; returns (token, id_token)."""
    async with (
        new_client_session() as session,
        session.post(
            f"{OPENAI_CODEX_AUTH_HOST}/oauth/token",
            data=urlencode(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
                    "client_id": OPENAI_CODEX_CLIENT_ID,
                    "code_verifier": verifier,
                }
            ),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp,
    ):
        status = resp.status
        raw = await resp.json(content_type=None)

    data = _assert_dict_response(raw, "token exchange")
    if status != 200:
        raise OAuthError(f"Token exchange failed: {data.get('error_description') or data}")
    id_token: str | None = data.get("id_token")
    return OAuthToken.from_response(data), id_token


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


async def refresh_openai_codex_token(refresh_token_value: str) -> OAuthToken:
    async with (
        new_client_session() as session,
        session.post(
            f"{OPENAI_CODEX_AUTH_HOST}/oauth/token",
            data=urlencode(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token_value,
                    "client_id": OPENAI_CODEX_CLIENT_ID,
                }
            ),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp,
    ):
        status = resp.status
        raw = await resp.json(content_type=None)

    data = _assert_dict_response(raw, "token refresh")
    if status in (401, 403):
        raise OAuthUnauthorized(data.get("error_description") or "Token refresh unauthorized.")
    if status != 200:
        raise OAuthError(data.get("error_description") or f"Token refresh failed (HTTP {status}).")
    return OAuthToken.from_response(data)


# ---------------------------------------------------------------------------
# Device code flow (headless)
# ---------------------------------------------------------------------------


async def _request_device_code() -> dict[str, Any]:
    async with (
        new_client_session() as session,
        session.post(
            f"{OPENAI_CODEX_AUTH_HOST}/api/accounts/deviceauth/usercode",
            json={"client_id": OPENAI_CODEX_CLIENT_ID},
        ) as resp,
    ):
        status = resp.status
        raw = await resp.json(content_type=None)

    data = _assert_dict_response(raw, "device authorization")
    if status != 200:
        raise OAuthError(f"Device authorization request failed: {data}")
    return data


async def _poll_device_token(
    device_code: str,
    interval: int,
    expires_in: int,
) -> OAuthToken:
    """Poll token endpoint until authorization completes or device code expires.

    `expires_in` is the server-provided lifetime in seconds (from device auth response).
    Sleep is placed AFTER the first request so an instant user approval returns immediately.
    """
    deadline = time.time() + max(expires_in, 60)
    async with new_client_session() as session:
        first = True
        while time.time() < deadline:
            if not first:
                await asyncio.sleep(interval)
            first = False

            try:
                async with session.post(
                    f"{OPENAI_CODEX_AUTH_HOST}/oauth/token",
                    data=urlencode(
                        {
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "client_id": OPENAI_CODEX_CLIENT_ID,
                            "device_code": device_code,
                        }
                    ),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ) as resp:
                    status = resp.status
                    raw = await resp.json(content_type=None)
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                logger.warning(
                    "Device polling transient network error, retrying: {error}", error=exc
                )
                continue

            if not isinstance(raw, dict):
                logger.warning("Unexpected device polling response shape, retrying")
                continue

            data = cast(dict[str, Any], raw)
            if status == 200 and "access_token" in data:
                return OAuthToken.from_response(data)

            error = str(data.get("error") or "")
            if error == "authorization_pending":
                continue
            elif error == "slow_down":
                # Honor server-provided interval if present; otherwise increment by 5.
                server_interval = int(data.get("interval") or 0)
                interval = max(server_interval, min(interval + 5, 30))
            elif error in ("expired_token", "access_denied"):
                raise OAuthError(f"Device authorization failed: {error}")
            else:
                raise OAuthError(f"Unexpected polling response: {data}")

    raise OAuthError("Device code expired.")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _apply_openai_codex_config(
    config: Config,
    *,
    oauth_ref: OAuthRef,
    account_id: str | None = None,
) -> None:
    from pydantic import SecretStr

    provider_key = managed_provider_key(OPENAI_CODEX_PLATFORM_ID)
    custom_headers: dict[str, str] | None = (
        {"ChatGPT-Account-Id": account_id} if account_id else None
    )
    config.providers[provider_key] = LLMProvider(
        type="openai_responses",
        base_url=OPENAI_CODEX_API_BASE_URL,
        api_key=SecretStr(""),
        oauth=oauth_ref,
        custom_headers=custom_headers,
    )

    for key, model in list(config.models.items()):
        if model.provider == provider_key:
            del config.models[key]

    model_key = managed_model_key(OPENAI_CODEX_PLATFORM_ID, _DEFAULT_MODEL)
    config.models[model_key] = LLMModel(
        provider=provider_key,
        model=_DEFAULT_MODEL,
        max_context_size=_DEFAULT_CONTEXT,
        capabilities={"thinking", "always_thinking"},
        display_name=f"OpenAI Codex ({_DEFAULT_MODEL})",
    )

    # Only change default_model if not currently set to a valid model.
    if not config.default_model or config.default_model not in config.models:
        config.default_model = model_key
    else:
        old_default = config.default_model
        config.default_model = model_key
        logger.info(
            "Set default model to {new}; previous default was {old}",
            new=model_key,
            old=old_default,
        )

    # OpenAI Responses provider does not expose thinking as a toggle; reset to avoid
    # stale default_thinking=True from a prior kimi-thinking session.
    config.default_thinking = False


# ---------------------------------------------------------------------------
# Login / Logout
# ---------------------------------------------------------------------------


async def login_openai_codex(
    config: Config,
    *,
    headless: bool = False,
) -> AsyncIterator[OAuthEvent]:
    if not config.is_from_default_location:
        yield OAuthEvent(
            "error",
            "Login requires the default config file; restart without --config/--config-file.",
        )
        return

    token: OAuthToken
    account_id: str | None = None

    if headless:
        try:
            device_data = await _request_device_code()
        except Exception as exc:
            yield OAuthEvent("error", f"Login failed: {exc}")
            return

        user_code = str(device_data.get("user_code") or "")
        verification_uri = str(device_data.get("verification_uri") or "")
        device_code = str(device_data.get("device_code") or "")
        interval = int(device_data.get("interval") or 5)
        expires_in = int(device_data.get("expires_in") or 600)

        yield OAuthEvent(
            "verification_url",
            f"Visit {verification_uri} and enter code: {user_code}",
            data={"verification_url": verification_uri, "user_code": user_code},
        )
        yield OAuthEvent("info", "Please visit the URL above and enter the code shown.")

        try:
            yield OAuthEvent("waiting", "Waiting for device authorization...")
            token = await _poll_device_token(device_code, interval, expires_in)
        except Exception as exc:
            yield OAuthEvent("error", f"Login failed: {exc}")
            return
    else:
        verifier = _generate_code_verifier()
        state = _generate_state()
        auth_url = _build_authorize_url(verifier, state)

        yield OAuthEvent(
            "info",
            "Please visit the following URL to authorize kimi-cli.",
        )
        yield OAuthEvent(
            "verification_url",
            f"Authorization URL: {auth_url}",
            data={"verification_url": auth_url, "user_code": None},
        )

        try:
            webbrowser.open(auth_url)
        except Exception as exc:
            logger.warning("Failed to open browser: {error}", error=exc)

        try:
            yield OAuthEvent("waiting", "Waiting for browser authorization...")
            code = await _wait_for_callback(state)
            token, id_token = await _exchange_code(code, verifier)
            account_id = _extract_account_id(id_token)
        except Exception as exc:
            yield OAuthEvent("error", f"Login failed: {exc}")
            return

    oauth_ref = OAuthRef(storage="file", key=OPENAI_CODEX_OAUTH_KEY)
    save_tokens(oauth_ref, token)

    _apply_openai_codex_config(config, oauth_ref=oauth_ref, account_id=account_id)
    save_config(config)
    yield OAuthEvent("success", "Logged in to OpenAI Codex successfully.")


async def logout_openai_codex(config: Config) -> AsyncIterator[OAuthEvent]:
    if not config.is_from_default_location:
        yield OAuthEvent(
            "error",
            "Logout requires the default config file; restart without --config/--config-file.",
        )
        return

    delete_tokens(OAuthRef(storage="file", key=OPENAI_CODEX_OAUTH_KEY))

    provider_key = managed_provider_key(OPENAI_CODEX_PLATFORM_ID)
    if provider_key in config.providers:
        del config.providers[provider_key]

    removed_default = False
    for key, model in list(config.models.items()):
        if model.provider != provider_key:
            continue
        del config.models[key]
        if config.default_model == key:
            removed_default = True

    if removed_default:
        config.default_model = next(iter(config.models), "")

    save_config(config)
    yield OAuthEvent("success", "Logged out from OpenAI Codex successfully.")
