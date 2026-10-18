"""Small compatibility layer for chat-completions and Responses providers."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
import httpx
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: Optional[str]
    model: str
    api_style: str


def get_llm_settings(scope: str = "rag") -> LLMSettings:
    """Read provider settings for the requested pipeline scope.

    ``rag`` preserves the original RAG/summary configuration.  ``planner``
    may use a separate vision-capable provider so a text-only RAG model is not
    sent the extracted paper figures.
    """
    load_dotenv(PROJECT_ROOT / "paper2slides" / ".env", override=False)
    scope = scope.strip().lower()
    if scope not in {"rag", "planner"}:
        raise ValueError("LLM scope must be 'rag' or 'planner'.")

    planner_provider = os.getenv("PLANNER_LLM_PROVIDER", "").strip().lower()
    provider = (
        planner_provider
        if scope == "planner" and planner_provider
        else os.getenv("RAG_LLM_PROVIDER", "openai_compatible").strip().lower()
    )

    if provider == "codex_gateway":
        api_key = (
            os.getenv("PLANNER_LLM_API_KEY", "") or os.getenv("CODEX_API_KEY", "")
            if scope == "planner"
            else os.getenv("CODEX_API_KEY", "")
        )
        configured_url = (
            os.getenv("PLANNER_LLM_BASE_URL", "") or os.getenv("CODEX_BASE_URL", "")
            if scope == "planner"
            else os.getenv("CODEX_BASE_URL", "")
        ).rstrip("/")
        model = (
            os.getenv("PLANNER_LLM_MODEL", "") or os.getenv("CODEX_MODEL", "")
            if scope == "planner"
            else os.getenv("CODEX_MODEL", "")
        )
        if not api_key or not configured_url or not model:
            raise ValueError(
                f"{scope.upper()} LLM codex_gateway requires CODEX_API_KEY, "
                "CODEX_BASE_URL, and CODEX_MODEL (or planner-specific overrides)."
            )
        base_url = configured_url if configured_url.endswith("/v1") else f"{configured_url}/v1"
        return LLMSettings(provider, api_key, base_url, model, "responses")

    if provider != "openai_compatible":
        raise ValueError(
            "Unsupported RAG_LLM_PROVIDER. Choose 'openai_compatible' or 'codex_gateway'."
        )

    api_key = (
        os.getenv("PLANNER_LLM_API_KEY", "") or os.getenv("RAG_LLM_API_KEY", "")
        if scope == "planner"
        else os.getenv("RAG_LLM_API_KEY", "")
    )
    if not api_key:
        raise ValueError("RAG_LLM_API_KEY is required for the openai_compatible provider.")
    api_style = os.getenv(
        "PLANNER_LLM_API_STYLE" if scope == "planner" else "RAG_LLM_API_STYLE",
        "chat_completions",
    ).strip().lower()
    if api_style not in {"chat_completions", "responses"}:
        raise ValueError("RAG_LLM_API_STYLE must be 'chat_completions' or 'responses'.")
    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=(
            os.getenv("PLANNER_LLM_BASE_URL") or os.getenv("RAG_LLM_BASE_URL")
            if scope == "planner"
            else os.getenv("RAG_LLM_BASE_URL")
        ) or None,
        model=(
            os.getenv("PLANNER_LLM_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
            if scope == "planner"
            else os.getenv("LLM_MODEL", "gpt-4o-mini")
        ),
        api_style=api_style,
    )


def _to_responses_input(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not message:
            continue
        role = message.get("role", "user")
        if role == "function":
            role = "tool"
        content = message.get("content", "")
        if isinstance(content, str):
            items = [{"type": "input_text", "text": content}]
        else:
            items = []
            for item in content or []:
                if not isinstance(item, dict):
                    items.append({"type": "input_text", "text": str(item)})
                    continue
                kind = item.get("type")
                if kind in {"text", "input_text", "output_text"}:
                    items.append({"type": "input_text", "text": item.get("text", "")})
                elif kind in {"image_url", "input_image"}:
                    image = item.get("image_url", item.get("image", ""))
                    if isinstance(image, dict):
                        image = image.get("url", "")
                    image_item = {"type": "input_image", "image_url": image}
                    detail = item.get("detail")
                    if detail:
                        image_item["detail"] = detail
                    items.append(image_item)
        converted.append({"role": role, "content": items})
    return converted


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    output = getattr(response, "output", None) or []
    text_parts: List[str] = []
    for item in output:
        if isinstance(item, dict):
            content = item.get("content", [])
        else:
            content = getattr(item, "content", []) or []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                value = part.get("text")
            else:
                part_type = getattr(part, "type", None)
                value = getattr(part, "text", None)
            if part_type in {"output_text", "text"} and value:
                text_parts.append(value)
    return "".join(text_parts)


class _CompletionsFacade:
    def __init__(self, owner: "LLMClient"):
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        if self._owner.settings.provider == "codex_gateway":
            messages = kwargs.pop("messages", [])
            text = self._owner._gateway_stream_text(messages, **kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                raw_response=None,
            )
        if self._owner.settings.api_style == "chat_completions":
            return self._owner._client.chat.completions.create(**kwargs)

        request = dict(kwargs)
        request["input"] = _to_responses_input(request.pop("messages", []))
        request["model"] = request.get("model") or self._owner.settings.model
        if "max_tokens" in request:
            request["max_output_tokens"] = request.pop("max_tokens")
        response_format = request.pop("response_format", None)
        if response_format:
            request["text"] = {"format": response_format}
        request.pop("stream", None)
        request.pop("extra_body", None)
        request.pop("seed", None)
        response = self._owner._client.responses.create(**request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_extract_output_text(response)))],
            raw_response=response,
        )


class LLMClient:
    """OpenAI-compatible facade backed by chat completions or Responses API."""

    def __init__(self, settings: Optional[LLMSettings] = None, client: Any = None):
        self.settings = settings or get_llm_settings()
        self._client = client or OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        self.chat = SimpleNamespace(completions=_CompletionsFacade(self))

    def complete_text(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        result = self.chat.completions.create(messages=messages, **kwargs)
        return result.choices[0].message.content or ""

    def _gateway_stream_text(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Call the configured Codex gateway's streaming Responses endpoint.

        The gateway requires a `messages` array and `stream=true` even at its
        `/responses` route, so use a small SSE adapter instead of the SDK's
        non-streaming Responses request shape.
        """
        request = dict(kwargs)
        model = request.pop("model", None) or self.settings.model
        max_output_tokens = request.pop("max_output_tokens", request.pop("max_tokens", None))
        request.pop("response_format", None)
        request.pop("extra_body", None)
        request.pop("seed", None)
        request.pop("stream", None)
        configured_timeout = request.pop("timeout", None)
        timeout_seconds = float(
            configured_timeout
            or os.getenv("CODEX_TIMEOUT_SECONDS", "300")
            or "300"
        )
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        for name in ("temperature", "top_p", "reasoning", "text"):
            if name in request:
                payload[name] = request.pop(name)

        url = f"{self.settings.base_url.rstrip('/')}/responses"
        output: List[str] = []
        with httpx.stream(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=httpx.Timeout(
                connect=min(timeout_seconds, 30.0),
                read=timeout_seconds,
                write=timeout_seconds,
                pool=timeout_seconds,
            ),
        ) as response:
            if response.is_error:
                # Deliberately avoid including request headers or body in errors.
                raise RuntimeError(f"Codex gateway request failed with HTTP {response.status_code}.")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")
                if event_type == "response.output_text.delta":
                    output.append(event.get("delta", ""))
                elif event_type in {"response.failed", "error"}:
                    error = event.get("response", {}).get("error") or event.get("error") or {}
                    code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
                    raise RuntimeError(f"Codex gateway generation failed ({code}).")
        return "".join(output)

    def responses_text(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        if self.settings.provider == "codex_gateway":
            return self._gateway_stream_text(messages, **kwargs)
        if self.settings.api_style == "responses":
            request = dict(kwargs)
            request["model"] = request.get("model") or self.settings.model
            request["input"] = _to_responses_input(messages)
            if "max_tokens" in request:
                request["max_output_tokens"] = request.pop("max_tokens")
            request.pop("response_format", None)
            request.pop("extra_body", None)
            request.pop("seed", None)
            return _extract_output_text(self._client.responses.create(**request))
        return self.complete_text(messages, **kwargs)


def create_llm_client(scope: str = "rag") -> LLMClient:
    return LLMClient(settings=get_llm_settings(scope=scope))
