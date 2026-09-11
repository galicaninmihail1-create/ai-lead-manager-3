import re


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    """Keep an optional leading '+' and digits; remove common visual separators.

    No country code is inferred. Letters, extra plus signs, and other characters are rejected.
    """
    stripped = value.strip()
    normalized = re.sub(r"[\s().-]", "", stripped)
    if not re.fullmatch(r"\+?[0-9]+", normalized):
        raise ValueError("phone must contain only digits, an optional leading '+', and separators")
    if len(normalized.removeprefix("+")) < 3:
        raise ValueError("phone must contain at least three digits")
    return normalized


def normalize_external_user_id(value: str) -> str:
    return value.strip()
