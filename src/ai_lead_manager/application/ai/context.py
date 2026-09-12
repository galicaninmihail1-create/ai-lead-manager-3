import hashlib
import json
from dataclasses import dataclass
from typing import Any


def estimate_tokens(text: str) -> int:
    """Deterministic conservative estimator used before provider-specific tokenizers exist."""

    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True, slots=True)
class BoundedAIContext:
    messages: tuple[str, ...]
    character_count: int
    estimated_tokens: int
    truncated: bool


class BoundedContextBuilder:
    """Keeps newest messages; for one oversized message keeps its newest suffix."""

    def __init__(self, max_messages: int, max_characters: int, max_input_tokens: int) -> None:
        if min(max_messages, max_characters, max_input_tokens) < 1:
            raise ValueError("AI context limits must be positive")
        self.max_messages = max_messages
        self.max_characters = max_characters
        self.max_input_tokens = max_input_tokens

    def build(self, messages: list[str]) -> BoundedAIContext:
        cleaned = [message.strip() for message in messages if message.strip()]
        candidates = cleaned[-self.max_messages :]
        selected_reversed: list[str] = []
        remaining_characters = self.max_characters
        remaining_tokens = self.max_input_tokens

        for message in reversed(candidates):
            if remaining_characters <= 0 or remaining_tokens <= 0:
                break
            candidate = message[-remaining_characters:]
            while candidate and estimate_tokens(candidate) > remaining_tokens:
                candidate = candidate[len(candidate) // 2 :]
            if not candidate:
                continue
            selected_reversed.append(candidate)
            remaining_characters -= len(candidate)
            remaining_tokens -= estimate_tokens(candidate)

        selected = tuple(reversed(selected_reversed))
        characters = sum(len(message) for message in selected)
        tokens = sum(estimate_tokens(message) for message in selected)
        return BoundedAIContext(
            messages=selected,
            character_count=characters,
            estimated_tokens=tokens,
            truncated=list(selected) != cleaned,
        )


def canonical_request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request_bytes(payload)).hexdigest()


def canonical_request_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def estimate_structured_tokens(payload: dict[str, Any]) -> int:
    return estimate_tokens(canonical_request_bytes(payload).decode("utf-8"))
