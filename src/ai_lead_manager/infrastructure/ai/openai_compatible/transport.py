import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import SecretStr

from ai_lead_manager.application.ai.contracts import AIErrorKind, AIProviderError
from ai_lead_manager.infrastructure.ai.openai_compatible.config import (
    OpenAICompatibleProviderConfig,
)


@dataclass(frozen=True, slots=True)
class OpenAIWireResponse:
    payload: dict[str, Any]
    provider_request_id: str | None
    latency_ms: int


class OpenAICompatibleHTTPTransport:
    """Synchronous HTTP boundary. Wire objects never escape infrastructure code."""

    def __init__(
        self,
        config: OpenAICompatibleProviderConfig,
        credential: SecretStr,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._credential = credential
        self._clock = clock
        self._now = now
        self._timeout = httpx.Timeout(
            connect=config.timeouts.connect_seconds,
            read=config.timeouts.read_seconds,
            write=config.timeouts.write_seconds,
            pool=config.timeouts.pool_seconds,
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,
            trust_env=False,
            transport=httpx.HTTPTransport(retries=0),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def post_chat_completion(self, payload: dict[str, Any]) -> OpenAIWireResponse:
        url = f"{str(self._config.base_url).rstrip('/')}{self._config.chat_completions_path}"
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._credential.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = self._clock()
        try:
            with self._client.stream(
                "POST",
                url,
                content=body,
                headers=headers,
                timeout=self._timeout,
            ) as response:
                provider_request_id = self._safe_request_id(
                    response.headers.get("x-request-id")
                )
                if response.status_code < 200 or response.status_code >= 300:
                    raise self._http_error(response, provider_request_id)
                content_type = response.headers.get("content-type", "")
                if content_type.split(";", 1)[0].strip().lower() != "application/json":
                    raise AIProviderError(
                        AIErrorKind.MALFORMED_OUTPUT,
                        "Provider returned an unexpected content type",
                        ambiguous=True,
                        provider_request_id=provider_request_id,
                        cost_known=False,
                    )
                response_body = self._bounded_body(response, provider_request_id)
        except AIProviderError as error:
            if error.latency_ms is None:
                error.latency_ms = self._latency_ms(started)
            raise
        except httpx.ConnectTimeout as error:
            raise AIProviderError(
                AIErrorKind.TIMEOUT_CONNECT,
                "Provider connection timed out",
                safe_to_retry=True,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.ReadTimeout as error:
            raise AIProviderError(
                AIErrorKind.TIMEOUT_READ,
                "Provider response timed out",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.WriteTimeout as error:
            raise AIProviderError(
                AIErrorKind.AMBIGUOUS_RESULT,
                "Provider request write timed out",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.PoolTimeout as error:
            raise AIProviderError(
                AIErrorKind.TRANSPORT,
                "Provider connection pool timed out",
                safe_to_retry=True,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.ConnectError as error:
            raise AIProviderError(
                AIErrorKind.TRANSPORT,
                "Provider connection failed",
                safe_to_retry=True,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.WriteError as error:
            raise AIProviderError(
                AIErrorKind.AMBIGUOUS_RESULT,
                "Provider request transmission failed",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        except (httpx.ReadError, httpx.RemoteProtocolError) as error:
            raise AIProviderError(
                AIErrorKind.AMBIGUOUS_RESULT,
                "Provider response transport failed",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.TimeoutException as error:
            raise AIProviderError(
                AIErrorKind.TIMEOUT_TOTAL,
                "Provider operation timed out",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        except httpx.TransportError as error:
            raise AIProviderError(
                AIErrorKind.TRANSPORT,
                "Provider transport failed",
                ambiguous=True,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error

        try:
            decoded = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AIProviderError(
                AIErrorKind.MALFORMED_OUTPUT,
                "Provider returned malformed JSON",
                ambiguous=True,
                provider_request_id=provider_request_id,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            ) from error
        if not isinstance(decoded, dict):
            raise AIProviderError(
                AIErrorKind.MALFORMED_OUTPUT,
                "Provider returned an unexpected response shape",
                ambiguous=True,
                provider_request_id=provider_request_id,
                cost_known=False,
                latency_ms=self._latency_ms(started),
            )
        return OpenAIWireResponse(
            payload=decoded,
            provider_request_id=provider_request_id,
            latency_ms=self._latency_ms(started),
        )

    def _bounded_body(
        self, response: httpx.Response, provider_request_id: str | None
    ) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self._config.max_response_bytes:
                raise AIProviderError(
                    AIErrorKind.MALFORMED_OUTPUT,
                    "Provider response exceeded the configured size limit",
                    ambiguous=True,
                    provider_request_id=provider_request_id,
                    cost_known=False,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _http_error(
        self, response: httpx.Response, provider_request_id: str | None
    ) -> AIProviderError:
        status = response.status_code
        retry_after = self._retry_after(response.headers.get("retry-after"))
        if status in {401, 403}:
            kind = AIErrorKind.AUTHENTICATION
            message = "Provider authentication was rejected"
        elif status == 408:
            kind = AIErrorKind.TIMEOUT_READ
            message = "Provider reported a request timeout"
        elif status == 429:
            kind = AIErrorKind.RATE_LIMIT
            message = "Provider rate limit was reached"
        elif 500 <= status <= 599:
            kind = AIErrorKind.PROVIDER_5XX
            message = "Provider returned a server error"
        elif 400 <= status <= 499:
            kind = AIErrorKind.INVALID_REQUEST
            message = "Provider rejected the request"
        else:
            kind = AIErrorKind.TRANSPORT
            message = "Provider returned an unexpected HTTP status"

        explicitly_safe = status in self._config.safe_retry_http_statuses
        ambiguous = status == 408 or (500 <= status <= 599 and not explicitly_safe)
        return AIProviderError(
            kind,
            message,
            http_status=status,
            retry_after_seconds=retry_after,
            safe_to_retry=explicitly_safe,
            ambiguous=ambiguous,
            provider_request_id=provider_request_id,
            cost_known=not ambiguous,
        )

    def _retry_after(self, raw_value: str | None) -> float | None:
        if raw_value is None:
            return None
        try:
            return max(0.0, float(raw_value.strip()))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw_value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return max(0.0, (retry_at - self._now()).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _safe_request_id(raw_value: str | None) -> str | None:
        if raw_value is None:
            return None
        value = raw_value.strip()
        if not value or len(value) > 255 or re.fullmatch(r"[A-Za-z0-9._:-]+", value) is None:
            return None
        return value

    def _latency_ms(self, started: float) -> int:
        return max(0, round((self._clock() - started) * 1_000))
