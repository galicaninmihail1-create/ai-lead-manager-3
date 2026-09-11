import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ai_lead_manager.api.routes import ai_http_error
from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIExecutionMode,
    AIFinalizationError,
    AIIdempotencyConflictError,
    AIOperationNotReplayableError,
    AIOperationStatus,
    AIOperationType,
    AIProviderError,
    AIUsage,
    BudgetExceededError,
    ProviderDisabledError,
    ProviderMetadata,
    StructuredAIRequest,
    StructuredAIResult,
    UnknownCostError,
)
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.pricing import ModelPricing, PricingCatalog
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import MockQualificationProvider
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import InboundHttpRequest
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import Settings
from ai_lead_manager.domain.models import (
    AIOperation,
    AuditEvent,
    Base,
    LeadAnalysis,
    ProviderCall,
    Tenant,
    TenantAIPolicy,
)
from ai_lead_manager.infrastructure.ai.recording import RecordingStructuredProvider
from ai_lead_manager.infrastructure.ai.repositories import (
    SqlAlchemyAIExecutionRepository,
    SqlAlchemyAIRepositoryFactory,
)
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyUnitOfWork,
)


class ExampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{tmp_path / 'ai-runtime.db'}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    yield factory
    engine.dispose()


def add_policy(
    factory: sessionmaker[Session],
    *,
    mode: str = "MOCK",
    provider: str = "fake",
    model: str = "model-v1",
    monthly: int = 1_000_000,
    per_call: int = 1_000_000,
    max_attempts: int = 2,
) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    with factory.begin() as session:
        session.add(Tenant(id=tenant_id, name="AI runtime", slug=f"ai-{tenant_id.hex}"))
        session.add(
            TenantAIPolicy(
                tenant_id=tenant_id,
                enabled=True,
                mode=mode,
                qualification_provider=provider,
                qualification_model=model,
                draft_provider=provider,
                draft_model=model,
                monthly_limit_microusd=monthly,
                per_call_limit_microusd=per_call,
                max_input_tokens=1_000,
                max_output_tokens=100,
                max_attempts=max_attempts,
                allow_unknown_cost=False,
            )
        )
    return tenant_id


def request() -> StructuredAIRequest:
    return StructuredAIRequest(
        operation_type=AIOperationType.QUALIFICATION,
        schema_name="ExampleResult",
        schema_version="1",
        prompt_version="test-v1",
        payload={"input": "safe"},
        estimated_input_tokens=10,
        max_output_tokens=5,
    )


def success(provider: str = "fake", model: str = "model-v1") -> StructuredAIResult:
    return StructuredAIResult(
        output={"answer": "ok"},
        usage=AIUsage(input_tokens=10, output_tokens=2),
        metadata=ProviderMetadata(
            provider=provider,
            model=model,
            provider_request_id="offline-request",
            finish_reason="stop",
        ),
        latency_ms=1,
    )


def build_executor(
    factory: sessionmaker[Session],
    provider: RecordingStructuredProvider,
    *,
    settings: Settings | None = None,
    prices: list[ModelPricing] | None = None,
) -> AIExecutor:
    return AIExecutor(
        settings or Settings(_env_file=None),
        SqlAlchemyAIRepositoryFactory(factory),
        ProviderRegistry([provider]),
        PricingCatalog(
            prices
            if prices is not None
            else [
                ModelPricing(
                    provider.provider_name,
                    provider.model_name,
                    "test-v1",
                    1_000_000,
                    2_000_000,
                    0,
                )
            ]
        ),
        sleeper=lambda _seconds: None,
    )


def execute(executor: AIExecutor, tenant_id: uuid.UUID, key: str, digest: str = "a" * 64):
    return executor.execute(
        TenantContext.untrusted(tenant_id),
        AIOperationType.QUALIFICATION,
        None,
        key,
        digest,
        request(),
        ExampleResult,
    )


def finalize(factory: sessionmaker[Session], tenant_id: uuid.UUID, operation_id: uuid.UUID) -> None:
    with factory.begin() as session:
        operation = session.get(AIOperation, operation_id)
        operation.status = AIOperationStatus.SUCCEEDED.value
        operation.reserved_cost_microusd = 0


def test_mock_is_default_and_untrusted_context_can_execute(session_factory) -> None:
    assert Settings(_env_file=None).ai_execution_mode == "MOCK"
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())

    outcome = execute(build_executor(session_factory, provider), tenant_id, "mock-default")

    assert outcome.value == ExampleResult(answer="ok")
    assert provider.physical_call_count == 1
    assert provider.requests[0].timeouts is not None
    assert provider.requests[0].max_output_tokens == 5


def test_live_requires_trusted_context_and_global_enablement(session_factory) -> None:
    tenant_id = add_policy(session_factory, mode="LIVE")
    provider = RecordingStructuredProvider(
        "fake",
        "model-v1",
        lambda _request: success(),
        execution_mode=AIExecutionMode.LIVE,
    )
    live_settings = Settings(
        _env_file=None, ai_execution_mode="LIVE", allow_live_ai=True
    )
    executor = build_executor(session_factory, provider, settings=live_settings)

    with pytest.raises(ProviderDisabledError):
        execute(executor, tenant_id, "untrusted-live")
    assert provider.physical_call_count == 0

    outcome = executor.execute(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        None,
        "trusted-live",
        "b" * 64,
        request(),
        ExampleResult,
    )
    assert outcome.value.answer == "ok"
    assert provider.physical_call_count == 1


def test_live_fails_closed_without_global_allow(session_factory) -> None:
    tenant_id = add_policy(session_factory, mode="LIVE")
    provider = RecordingStructuredProvider(
        "fake",
        "model-v1",
        lambda _request: success(),
        execution_mode=AIExecutionMode.LIVE,
    )
    executor = build_executor(
        session_factory,
        provider,
        settings=Settings(_env_file=None, ai_execution_mode="LIVE", allow_live_ai=False),
    )

    with pytest.raises(ProviderDisabledError):
        executor.execute(
            TenantContext.trusted_context(tenant_id),
            AIOperationType.QUALIFICATION,
            None,
            "closed-live",
            "c" * 64,
            request(),
            ExampleResult,
        )
    assert provider.physical_call_count == 0


def test_idempotency_replay_never_calls_provider_twice(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(session_factory, provider)
    first = execute(executor, tenant_id, "same-key")
    finalize(session_factory, tenant_id, first.operation_id)

    replay = execute(executor, tenant_id, "same-key")

    assert replay.replayed is True
    assert replay.operation_id == first.operation_id
    assert provider.physical_call_count == 1

    with session_factory.begin() as session:
        session.get(TenantAIPolicy, tenant_id).enabled = False
    replay_while_disabled = execute(executor, tenant_id, "same-key")
    assert replay_while_disabled.replayed is True
    assert provider.physical_call_count == 1


def test_idempotency_hash_conflict_and_calling_state_do_not_execute(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(session_factory, provider)
    execute(executor, tenant_id, "conflict")

    with pytest.raises(AIOperationNotReplayableError):
        execute(executor, tenant_id, "conflict")
    with pytest.raises(AIIdempotencyConflictError):
        execute(executor, tenant_id, "conflict", "d" * 64)
    assert provider.physical_call_count == 1


def test_budget_rejection_occurs_before_provider_and_is_audited(session_factory) -> None:
    tenant_id = add_policy(session_factory, per_call=39)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())

    with pytest.raises(BudgetExceededError):
        execute(build_executor(session_factory, provider), tenant_id, "over-budget")

    assert provider.physical_call_count == 0
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(AuditEvent).where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.action == "ai.budget_rejected",
            )
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(AIOperation).where(
                AIOperation.tenant_id == tenant_id
            )
        ) == 0


def test_unknown_price_and_disabled_provider_fail_before_call(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(session_factory, provider, prices=[])

    with pytest.raises(UnknownCostError):
        execute(executor, tenant_id, "unknown-price")
    assert provider.physical_call_count == 0

    with session_factory.begin() as session:
        session.get(TenantAIPolicy, tenant_id).enabled = False
    with pytest.raises(ProviderDisabledError):
        execute(build_executor(session_factory, provider), tenant_id, "disabled")
    assert provider.physical_call_count == 0


def test_provider_allowlist_and_global_kill_switch_fail_before_call(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    unlisted = RecordingStructuredProvider("other", "other-model", lambda _request: success())
    with pytest.raises(ProviderDisabledError):
        execute(build_executor(session_factory, unlisted), tenant_id, "not-allowlisted")
    assert unlisted.physical_call_count == 0

    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(
        session_factory,
        provider,
        settings=Settings(_env_file=None, ai_kill_switch=True),
    )
    with pytest.raises(ProviderDisabledError):
        execute(executor, tenant_id, "kill-switch")
    assert provider.physical_call_count == 0


def test_monthly_hard_budget_counts_committed_and_reserved_cost(session_factory) -> None:
    tenant_id = add_policy(session_factory, monthly=79, per_call=100)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(session_factory, provider)

    execute(executor, tenant_id, "first-monthly-reservation")
    with pytest.raises(BudgetExceededError):
        execute(executor, tenant_id, "second-monthly-reservation")

    assert provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.scalar(
            select(AIOperation).where(AIOperation.tenant_id == tenant_id)
        )
        assert operation.actual_cost_microusd + operation.reserved_cost_microusd == 40

def test_retry_policy_retries_connect_timeout_but_not_read_timeout(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    connect_error = AIProviderError(
        AIErrorKind.TIMEOUT_CONNECT, "connect timeout", safe_to_retry=True
    )
    provider = RecordingStructuredProvider(
        "fake", "model-v1", lambda _request: success(), [connect_error, success()]
    )
    outcome = execute(build_executor(session_factory, provider), tenant_id, "connect-retry")
    assert outcome.value.answer == "ok"
    assert provider.physical_call_count == 2

    second_tenant = add_policy(session_factory)
    read_error = AIProviderError(
        AIErrorKind.TIMEOUT_READ,
        "possibly transmitted secret value",
        ambiguous=True,
        cost_known=False,
    )
    read_provider = RecordingStructuredProvider(
        "fake", "model-v1", lambda _request: success(), [read_error]
    )
    with pytest.raises(AIProviderError):
        execute(build_executor(session_factory, read_provider), second_tenant, "read-timeout")
    assert read_provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.scalar(
            select(AIOperation).where(AIOperation.tenant_id == second_tenant)
        )
        assert operation.status == "UNKNOWN"
        assert operation.reserved_cost_microusd == operation.estimated_max_cost_microusd
        assert "secret" not in str(
            list(
                session.scalars(
                    select(AuditEvent).where(AuditEvent.tenant_id == second_tenant)
                )
            )
        )


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        (AIProviderError(AIErrorKind.RATE_LIMIT, "429", retry_after_seconds=0), 2),
        (AIProviderError(AIErrorKind.PROVIDER_5XX, "safe", safe_to_retry=True), 2),
        (AIProviderError(AIErrorKind.PROVIDER_5XX, "unsafe"), 1),
        (AIProviderError(AIErrorKind.INVALID_REQUEST, "400", http_status=400), 1),
        (AIProviderError(AIErrorKind.AUTHENTICATION, "401", http_status=401), 1),
    ],
)
def test_retry_matrix_is_explicit_and_bounded(
    session_factory, error: AIProviderError, expected_calls: int
) -> None:
    tenant_id = add_policy(session_factory)
    outcomes = [error, success()] if expected_calls == 2 else [error]
    provider = RecordingStructuredProvider(
        "fake", "model-v1", lambda _request: success(), outcomes
    )

    if expected_calls == 2:
        result = execute(build_executor(session_factory, provider), tenant_id, error.kind.value)
        assert result.value.answer == "ok"
    else:
        with pytest.raises(AIProviderError):
            execute(build_executor(session_factory, provider), tenant_id, error.kind.value)
    assert provider.physical_call_count == expected_calls


@pytest.mark.parametrize(
    ("outcome", "expected_kind"),
    [
        (
            StructuredAIResult(
                output={"wrong": "shape"},
                usage=AIUsage(input_tokens=10, output_tokens=3),
                metadata=ProviderMetadata(provider="fake", model="model-v1"),
            ),
            AIErrorKind.SCHEMA_VALIDATION,
        ),
        (
            AIProviderError(
                AIErrorKind.MALFORMED_OUTPUT,
                "bad json",
                usage=AIUsage(input_tokens=10, output_tokens=3),
            ),
            AIErrorKind.MALFORMED_OUTPUT,
        ),
    ],
)
def test_invalid_structured_output_records_known_usage_without_retry(
    session_factory, outcome, expected_kind
) -> None:
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider(
        "fake", "model-v1", lambda _request: success(), [outcome]
    )

    with pytest.raises(AIProviderError) as captured:
        execute(build_executor(session_factory, provider), tenant_id, f"invalid-{expected_kind}")

    assert captured.value.kind == expected_kind
    assert provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.scalar(
            select(AIOperation).where(AIOperation.tenant_id == tenant_id)
        )
        call = session.scalar(
            select(ProviderCall).where(ProviderCall.tenant_id == tenant_id)
        )
        assert operation.status == "FAILED"
        assert operation.reserved_cost_microusd == 0
        assert operation.actual_cost_microusd == 16
        assert call.cost_known is True
        assert call.error_kind == expected_kind.value


def test_safe_qualification_finalizes_and_replays_outside_request_transaction(
    session_factory,
) -> None:
    tenant_id = add_policy(
        session_factory, provider="mock", model="deterministic-keyword-v1"
    )
    with session_factory() as service_session:
        service = InboundIngestionService(
            SqlAlchemyInboundRepository(service_session, tenant_id),
            SqlAlchemyUnitOfWork(service_session),
        )
        inbound = service.ingest(
            InboundHttpRequest(
                tenant_id=tenant_id,
                source="safe-ai-test",
                external_event_id="safe-ai-event",
                name="Анна",
                email="anna@example.com",
                message="Нужно автоматизировать заявки. Бюджет 100 000 рублей за 2 недели.",
            )
        )

        def handler(ai_request: StructuredAIRequest) -> StructuredAIResult:
            assert service_session.in_transaction() is False
            from ai_lead_manager.application.providers import QualificationContext

            output = MockQualificationProvider().analyze(
                QualificationContext.model_validate(ai_request.payload)
            ).model_dump(mode="json")
            return StructuredAIResult(
                output=output,
                usage=AIUsage(
                    input_tokens=ai_request.estimated_input_tokens, output_tokens=20
                ),
                metadata=ProviderMetadata(
                    provider="mock", model="deterministic-keyword-v1"
                ),
            )

        provider = RecordingStructuredProvider(
            "mock", "deterministic-keyword-v1", handler
        )
        executor = build_executor(session_factory, provider)
        qualification = QualificationService(
            TenantContext.untrusted(tenant_id),
            SqlAlchemyQualificationRepository(service_session, tenant_id),
            SqlAlchemyUnitOfWork(service_session),
            MockQualificationProvider(),
            executor,
        )
        first = qualification.qualify(inbound.lead_id, "qualification-key")
        replay = qualification.qualify(inbound.lead_id, "qualification-key")

    assert first.ai_operation_id == replay.ai_operation_id
    assert first.replayed is False
    assert replay.replayed is True
    assert provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.get(AIOperation, first.ai_operation_id)
        assert operation.status == "SUCCEEDED"
        assert operation.reserved_cost_microusd == 0
        assert operation.result_analysis_id == first.analysis_id
        assert session.scalar(
            select(func.count()).select_from(LeadAnalysis).where(
                LeadAnalysis.tenant_id == tenant_id
            )
        ) == 1


def test_provider_success_persistence_failure_marks_unknown_without_retry(
    session_factory, monkeypatch
) -> None:
    tenant_id = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    original = SqlAlchemyAIExecutionRepository.add_provider_call
    failed_once = False

    def fail_first_success(self, call):
        nonlocal failed_once
        if not failed_once and call.status == "SUCCEEDED":
            failed_once = True
            raise RuntimeError("simulated persistence failure with secret-value")
        return original(self, call)

    monkeypatch.setattr(
        SqlAlchemyAIExecutionRepository, "add_provider_call", fail_first_success
    )

    with pytest.raises(AIFinalizationError) as captured:
        execute(build_executor(session_factory, provider), tenant_id, "persist-failure")

    assert captured.value.kind == AIErrorKind.AMBIGUOUS_RESULT
    assert "secret-value" not in str(captured.value)
    assert provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.scalar(
            select(AIOperation).where(AIOperation.tenant_id == tenant_id)
        )
        assert operation.status == "UNKNOWN"
        assert operation.reserved_cost_microusd == operation.estimated_max_cost_microusd
        assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0

    with pytest.raises(AIOperationNotReplayableError):
        execute(build_executor(session_factory, provider), tenant_id, "persist-failure")
    assert provider.physical_call_count == 1


def test_ai_repository_is_tenant_scoped_for_trusted_context(session_factory) -> None:
    tenant_a = add_policy(session_factory)
    tenant_b = add_policy(session_factory)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())
    executor = build_executor(session_factory, provider)
    result = executor.execute(
        TenantContext.trusted_context(tenant_a, "verified-test-auth"),
        AIOperationType.QUALIFICATION,
        None,
        "tenant-a-operation",
        "f" * 64,
        request(),
        ExampleResult,
    )

    factory = SqlAlchemyAIRepositoryFactory(session_factory)
    with factory.transaction(tenant_b) as repository:
        assert repository.get_operation_for_update(result.operation_id) is None
    with pytest.raises(IntegrityError):
        with factory.transaction(tenant_b) as repository:
            repository.add_provider_call(
                ProviderCall(
                    tenant_id=tenant_b,
                    operation_id=result.operation_id,
                    attempt=2,
                    provider="fake",
                    model="model-v1",
                    status="FAILED",
                    cost_known=True,
                )
            )


def test_provider_call_happens_with_no_ai_repository_transaction(session_factory) -> None:
    tenant_id = add_policy(session_factory)
    transaction_depth = 0

    class ObservedFactory(SqlAlchemyAIRepositoryFactory):
        @contextmanager
        def transaction(self, tenant_id):
            nonlocal transaction_depth
            transaction_depth += 1
            try:
                with super().transaction(tenant_id) as repository:
                    yield repository
            finally:
                transaction_depth -= 1

    def handler(_request):
        assert transaction_depth == 0
        return success()

    provider = RecordingStructuredProvider("fake", "model-v1", handler)
    executor = AIExecutor(
        Settings(_env_file=None),
        ObservedFactory(session_factory),
        ProviderRegistry([provider]),
        PricingCatalog([ModelPricing("fake", "model-v1", "v1", 1_000_000, 2_000_000, 0)]),
        sleeper=lambda _seconds: None,
    )

    execute(executor, tenant_id, "short-transactions")
    assert provider.physical_call_count == 1


def test_domain_finalization_failure_marks_operation_unknown(session_factory) -> None:
    tenant_id = add_policy(
        session_factory, provider="mock", model="deterministic-keyword-v1"
    )
    with session_factory() as session:
        inbound = InboundIngestionService(
            SqlAlchemyInboundRepository(session, tenant_id),
            SqlAlchemyUnitOfWork(session),
        ).ingest(
            InboundHttpRequest(
                tenant_id=tenant_id,
                source="finalization-test",
                external_event_id="finalization-event",
                email="safe@example.com",
                message="Need a safe qualification with a fixed budget.",
            )
        )

        def handler(ai_request):
            from ai_lead_manager.application.providers import QualificationContext

            output = MockQualificationProvider().analyze(
                QualificationContext.model_validate(ai_request.payload)
            ).model_dump(mode="json")
            return StructuredAIResult(
                output=output,
                usage=AIUsage(
                    input_tokens=ai_request.estimated_input_tokens, output_tokens=10
                ),
                metadata=ProviderMetadata(
                    provider="mock", model="deterministic-keyword-v1"
                ),
            )

        provider = RecordingStructuredProvider(
            "mock", "deterministic-keyword-v1", handler
        )

        class FailingFinalCommitUnitOfWork(SqlAlchemyUnitOfWork):
            def __init__(self, db_session):
                super().__init__(db_session)
                self.commits = 0

            def commit(self) -> None:
                self.commits += 1
                if self.commits == 2:
                    raise RuntimeError("simulated domain persistence failure")
                super().commit()

        service = QualificationService(
            TenantContext.untrusted(tenant_id),
            SqlAlchemyQualificationRepository(session, tenant_id),
            FailingFinalCommitUnitOfWork(session),
            MockQualificationProvider(),
            build_executor(session_factory, provider),
        )
        with pytest.raises(RuntimeError, match="simulated domain persistence failure"):
            service.qualify(inbound.lead_id, "domain-finalization-failure")

    assert provider.physical_call_count == 1
    with session_factory() as session:
        operation = session.scalar(
            select(AIOperation).where(AIOperation.tenant_id == tenant_id)
        )
        assert operation.status == "UNKNOWN"
        assert operation.error_code == "domain_finalization_failed"
        assert operation.reserved_cost_microusd > 0
        assert session.scalar(
            select(func.count()).select_from(ProviderCall).where(
                ProviderCall.operation_id == operation.id
            )
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(LeadAnalysis).where(
                LeadAnalysis.tenant_id == tenant_id
            )
        ) == 0


def test_recording_provider_execution_has_zero_network_calls(
    session_factory, monkeypatch
) -> None:
    tenant_id = add_policy(session_factory)

    def forbid_network(*_args, **_kwargs):
        raise AssertionError("external network access is forbidden")

    monkeypatch.setattr("socket.create_connection", forbid_network)
    provider = RecordingStructuredProvider("fake", "model-v1", lambda _request: success())

    result = execute(build_executor(session_factory, provider), tenant_id, "offline-only")

    assert result.value.answer == "ok"
    assert provider.physical_call_count == 1


def test_ai_error_response_does_not_expose_provider_message() -> None:
    exception = ai_http_error(
        AIProviderError(
            AIErrorKind.AUTHENTICATION,
            "Authorization: Bearer highly-sensitive-secret",
            http_status=401,
        )
    )

    assert exception.status_code == 502
    assert exception.detail == {"code": "authentication"}
    assert "sensitive-secret" not in str(exception.detail)
