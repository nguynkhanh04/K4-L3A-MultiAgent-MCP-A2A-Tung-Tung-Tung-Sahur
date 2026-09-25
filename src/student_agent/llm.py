"""LLM helper — thin wrapper around OpenAI-compatible APIs.

Supports DeepSeek, OpenRouter, Gemini, or any OpenAI-compatible provider.
Configure via environment variables in ``.env``:

- ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``: primary provider.
- ``LLM_FALLBACK_API_KEY`` / ``LLM_FALLBACK_BASE_URL`` / ``LLM_FALLBACK_MODEL``:
  optional, used only when the primary call raises (timeout, 5xx, 429, ...).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, AsyncOpenAI

# Fail fast so a slow provider hands over to the fallback instead of
# stalling the batch; the SDK's own retries still cover transient errors.
REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 1

# OpenRouter routing: only full-precision (bf16) hosts that honour every
# request parameter, and no thinking tokens. In our tests reasoning made
# qwen3.5-9b ~8x slower and non-deterministic with no accuracy gain.
_OPENROUTER_EXTRA: dict[str, Any] = {
    "provider": {"quantizations": ["bf16"], "require_parameters": True},
    "reasoning": {"enabled": False},
}


@dataclass
class _Provider:
    client: AsyncOpenAI
    model: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Gemma on the Gemini API rejects system instructions and JSON mode.
    plain_prompt_only: bool = False


# ── Provider singletons ──────────────────────────────────────────────

_primary: _Provider | None = None
_fallback: _Provider | None = None
_fallback_loaded = False


def _build(prefix: str) -> _Provider | None:
    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
    model = os.getenv(f"{prefix}_MODEL", "").strip()
    if not (api_key and base_url and model):
        return None
    return _Provider(
        client=AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=MAX_RETRIES,
        ),
        model=model,
        extra_body=_OPENROUTER_EXTRA if "openrouter.ai" in base_url else {},
        plain_prompt_only=model.startswith("gemma"),
    )


def _get_primary() -> _Provider:
    global _primary
    if _primary is None:
        _primary = _build("LLM")
        if _primary is None:
            raise RuntimeError(
                "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL must all be set in .env "
                "(e.g. https://openrouter.ai/api/v1 + qwen/qwen3.5-9b)"
            )
    return _primary


def _get_fallback() -> _Provider | None:
    global _fallback, _fallback_loaded
    if not _fallback_loaded:
        _fallback = _build("LLM_FALLBACK")
        _fallback_loaded = True
    return _fallback


async def _complete(
    provider: _Provider,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> str:
    if provider.plain_prompt_only:
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        response_format = None
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = await provider.client.chat.completions.create(
        model=model or provider.model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        extra_body=provider.extra_body or None,
        **kwargs,
    )
    return response.choices[0].message.content or ""


# ── Public API ───────────────────────────────────────────────────────

async def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send a chat completion request and return the assistant's response text.

    Tries the primary provider first; if that call raises an API error and a
    fallback provider is configured, retries once there with its own model.

    Args:
        system_prompt: Instructions for the LLM's role/behaviour.
        user_prompt: The actual question or data to analyse.
        model: Override the primary model from ``.env`` (not the fallback's).
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in the response.
        response_format: Optional OpenAI ``response_format`` payload.

    Returns:
        The assistant's response as a plain string.
    """
    call = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
    }
    try:
        return await _complete(
            _get_primary(), system_prompt, user_prompt, model=model, **call
        )
    except APIError:
        fallback = _get_fallback()
        if fallback is None:
            raise
        return await _complete(
            fallback, system_prompt, user_prompt, model=None, **call
        )


async def ask_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Like ``ask_llm`` but parses the response as JSON.

    The system prompt should instruct the LLM to respond with valid JSON only.
    Pass ``schema`` to have the provider enforce it (strict JSON Schema). Put
    free-text reasoning fields before the decision field: the model writes
    keys in order, and deciding first then explaining was wrong in our tests.
    Providers without JSON mode (Gemma) get the schema only via the prompt,
    so callers must still validate the parsed values.

    Returns:
        Parsed JSON dict. Raises ``ValueError`` on parse failure.
    """
    response_format = None
    if schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }
        system_prompt = (
            f"{system_prompt}\n\nRespond with JSON only, matching this schema:\n"
            f"{json.dumps(schema)}"
        )
    raw = await ask_llm(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM did not return valid JSON.\nRaw response:\n{raw}"
        ) from exc
