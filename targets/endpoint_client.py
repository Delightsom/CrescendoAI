"""HTTP client for OpenAI-compatible ``/chat/completions`` endpoints."""

import time
from collections.abc import Sequence
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from core.models import Message, TargetError, TargetResponse


def chat_completions_url(base_url: str) -> str:
    """Return ``base_url`` if it already targets chat completions, otherwise append it."""
    base = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


class EndpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key: SecretStr | None = None
    timeout_seconds: float = 60.0

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an http or https URL")
        return cleaned

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model is required")
        return cleaned

    @field_validator("timeout_seconds")
    @classmethod
    def _check_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value


class EndpointClient:
    """Post the running transcript to a chat-completions endpoint."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config
        self._client: httpx.Client | None = None

    def __enter__(self) -> "EndpointClient":
        self._client = httpx.Client(timeout=self.config.timeout_seconds)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def send(self, messages: Sequence[Message]) -> TargetResponse:
        if not messages:
            raise TargetError("Cannot send an empty transcript")
        client = self._client
        if client is None:
            raise RuntimeError("EndpointClient must be used as a context manager")

        headers: dict[str, str] = {}
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"

        payload = {
            "model": self.config.model,
            "messages": [message.model_dump(mode="json") for message in messages],
        }
        started = time.perf_counter()
        try:
            response = client.post(
                chat_completions_url(self.config.base_url),
                headers=headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise TargetError(
                f"Request failed: {exc}",
                latency_ms=_elapsed_ms(started),
            ) from exc

        latency_ms = _elapsed_ms(started)
        if response.status_code >= 400:
            raise TargetError(
                f"HTTP {response.status_code}: {_error_detail(response)}",
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        content = _read_content(response)
        return TargetResponse(
            content=content,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _snippet(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return _snippet(response.text) or "empty error body"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return _snippet(str(error["message"]))
        if isinstance(error, str):
            return _snippet(error)
    return _snippet(response.text) or "empty error body"


def _read_content(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError as exc:
        raise TargetError(
            f"Response was not JSON: {_snippet(response.text) or 'empty body'}",
            status_code=response.status_code,
        ) from exc

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TargetError(
            "Response did not include choices[0].message",
            status_code=response.status_code,
        ) from exc

    if not isinstance(message, dict):
        raise TargetError(
            "Response message was not an object",
            status_code=response.status_code,
        )

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_parts(content, response.status_code)
    raise TargetError(
        "Response did not include message content",
        status_code=response.status_code,
    )


def _join_parts(parts: list[object], status_code: int) -> str:
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") in {None, "text"}:
            text = part.get("text", "")
            texts.append(text if isinstance(text, str) else str(text))
        else:
            raise TargetError(
                "Response content included an unsupported part",
                status_code=status_code,
            )
    return "".join(texts)
