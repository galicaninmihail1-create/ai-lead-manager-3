import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from ai_lead_manager.application.normalization import normalize_phone


class TenantCreate(BaseModel):
    name: str
    slug: str


class CustomerCreate(BaseModel):
    name: str
    email: EmailStr | None = None
    phone: str | None = None


class LeadCreate(BaseModel):
    customer_id: uuid.UUID
    status: str = "new"
    score: int | None = None
    source: str | None = None
    notes: str | None = None


class ModelView(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class InboundHttpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: uuid.UUID
    source: str = Field(min_length=1, max_length=100)
    external_event_id: str = Field(min_length=1, max_length=255)
    external_user_id: str | None = Field(default=None, max_length=500)
    external_thread_id: str | None = Field(default=None, max_length=255)
    name: str | None = Field(default=None, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] | None = None

    @field_validator("source", "external_event_id", "external_user_id", "external_thread_id")
    @classmethod
    def strip_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("identifier must not be blank")
        return stripped

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str | None) -> str | None:
        if value is not None:
            normalize_phone(value)
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class InboundHttpResponse(BaseModel):
    event_id: uuid.UUID
    duplicate: bool
    tenant_id: uuid.UUID
    customer_id: uuid.UUID
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    lead_id: uuid.UUID
    lead_created: bool
    status: Literal["processed"] = "processed"
