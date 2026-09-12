import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, delete, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateSchema, DropSchema

from ai_lead_manager.application.ai.contracts import (
    AIExecutionOutcome,
    AIOperationNotReplayableError,
    AIOperationType,
    AIUsage,
    BudgetExceededError,
    ProviderMetadata,
    StructuredAIRequest,
    StructuredAIResult,
)
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.money import Currency
from ai_lead_manager.application.ai.pricing import (
    ModelPricing,
    PricingCatalog,
    default_mock_pricing_catalog,
)
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import (
    MockQualificationProvider,
    QualificationContext,
)
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import InboundHttpRequest
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import Settings, get_settings
from ai_lead_manager.domain.models import AIOperation, ProviderCall, Tenant, TenantAIPolicy
from ai_lead_manager.infrastructure.ai.recording import RecordingStructuredProvider
from ai_lead_manager.infrastructure.ai.repositories import SqlAlchemyAIRepositoryFactory
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyUnitOfWork,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="PostgreSQL integration tests are opt-in",
)


class PGResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


def ai_request() -> StructuredAIRequest:
    return StructuredAIRequest(
        operation_type=AIOperationType.QUALIFICATION,
        schema_name="PGResult",
        schema_version="1",
        prompt_version="pg-test-v1",
        payload={"input": "offline"},
        estimated_input_tokens=10,
        max_output_tokens=5,
    )


def success() -> StructuredAIResult:
    return StructuredAIResult(
        output={"answer": "ok"},
        usage=AIUsage(input_tokens=10, output_tokens=2),
        metadata=ProviderMetadata(provider="pg-fake", model="pg-model"),
    )


def add_tenant_policy(
    engine,
    *,
    monthly_limit: int,
    provider: str = "pg-fake",
    model: str = "pg-model",
    currency: Currency = Currency.USD,
) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    with Session(engine) as session, session.begin():
        session.add(
            Tenant(
                id=tenant_id,
                name="PostgreSQL AI concurrency",
                slug=f"pg-ai-{tenant_id.hex}",
            )
        )
        session.add(
            TenantAIPolicy(
                tenant_id=tenant_id,
                enabled=True,
                mode="MOCK",
                qualification_provider=provider,
                qualification_model=model,
                draft_provider=provider,
                draft_model=model,
                monthly_limit_micros=monthly_limit,
                per_call_limit_micros=1_000,
                budget_currency=currency.value,
                max_input_tokens=1_000,
                max_output_tokens=100,
                max_attempts=2,
                allow_unknown_cost=False,
            )
        )
    return tenant_id


def executor(
    engine, provider: RecordingStructuredProvider, currency: Currency = Currency.USD
) -> AIExecutor:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return AIExecutor(
        Settings(_env_file=None),
        SqlAlchemyAIRepositoryFactory(factory),
        ProviderRegistry([provider]),
        PricingCatalog(
            [
                ModelPricing(
                    "pg-fake", "pg-model", "pg-v1", currency, 1_000_000, 2_000_000, 0
                )
            ]
        ),
        sleeper=lambda _seconds: None,
    )


def execute(
    runtime: AIExecutor, tenant_id: uuid.UUID, key: str
) -> AIExecutionOutcome[PGResult]:
    return runtime.execute(
        TenantContext.untrusted(tenant_id, "postgres-test"),
        AIOperationType.QUALIFICATION,
        None,
        key,
        "a" * 64,
        ai_request(),
        PGResult,
    )


def cleanup(engine, tenant_ids: list[uuid.UUID]) -> None:
    with Session(engine) as session, session.begin():
        session.execute(delete(Tenant).where(Tenant.id.in_(tenant_ids)))


def test_postgres_currency_migration_preserves_legacy_usd_semantics(monkeypatch) -> None:
    base_url = make_url(os.environ["DATABASE_URL"])
    schema = f"currency_migration_{uuid.uuid4().hex}"
    engine = create_engine(base_url)
    schema_url = base_url.update_query_dict({"options": f"-csearch_path={schema}"})
    migration_config = Config("alembic.ini")
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        migration_url = (
            f"{base_url.render_as_string(hide_password=False)}"
            f"?options=-csearch_path={schema}"
        )
        monkeypatch.setenv("DATABASE_URL", migration_url)
        get_settings.cache_clear()
        command.upgrade(migration_config, "20260912_0008")

        schema_engine = create_engine(schema_url)
        with schema_engine.begin() as connection:
            tenant_id = uuid.uuid4()
            operation_id = uuid.uuid4()
            connection.execute(
                text(
                    "INSERT INTO tenants (id, name, slug) VALUES "
                    "(:tenant_id, 'migration sentinel', :slug)"
                ),
                {"tenant_id": tenant_id, "slug": f"migration-{uuid.uuid4().hex}"},
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_ai_policies "
                    "(tenant_id, enabled, mode, qualification_provider, "
                    "qualification_model, draft_provider, draft_model, "
                    "monthly_limit_microusd, per_call_limit_microusd, "
                    "max_input_tokens, max_output_tokens, max_attempts, allow_unknown_cost) "
                    "VALUES (:tenant_id, true, 'MOCK', 'mock', 'model', 'mock', 'model', "
                    "9000000, 900000, 1000, 100, 2, false)"
                ),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text(
                    "INSERT INTO ai_operations "
                    "(id, tenant_id, operation_type, idempotency_key, request_hash, "
                    "provider, model, pricing_version, status, estimated_max_cost_microusd, "
                    "reserved_cost_microusd, actual_cost_microusd) VALUES "
                    "(:operation_id, :tenant_id, 'QUALIFICATION', 'migration-sentinel', "
                    ":request_hash, 'mock', 'model', 'v1', 'SUCCEEDED', 123, 0, 45)"
                ),
                {
                    "operation_id": operation_id,
                    "tenant_id": tenant_id,
                    "request_hash": "a" * 64,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO provider_calls "
                    "(id, tenant_id, operation_id, attempt, provider, model, status, "
                    "actual_cost_microusd, cost_known) VALUES "
                    "(:call_id, :tenant_id, :operation_id, 1, 'mock', 'model', "
                    "'SUCCEEDED', 45, true)"
                ),
                {
                    "call_id": uuid.uuid4(),
                    "tenant_id": tenant_id,
                    "operation_id": operation_id,
                },
            )

        command.upgrade(migration_config, "head")
        with schema_engine.connect() as connection:
            migrated = connection.execute(
                text(
                    "SELECT monthly_limit_micros, per_call_limit_micros, budget_currency "
                    "FROM tenant_ai_policies WHERE tenant_id = :tenant_id"
                ),
                {"tenant_id": tenant_id},
            ).one()
            assert migrated == (9_000_000, 900_000, "USD")
            migrated_operation = connection.execute(
                text(
                    "SELECT estimated_max_cost_micros, reserved_cost_micros, "
                    "actual_cost_micros, cost_currency FROM ai_operations "
                    "WHERE id = :operation_id"
                ),
                {"operation_id": operation_id},
            ).one()
            assert migrated_operation == (123, 0, 45, "USD")
            migrated_call = connection.execute(
                text(
                    "SELECT actual_cost_micros, cost_currency, "
                    "provider_reported_cost_micros, provider_reported_cost_currency "
                    "FROM provider_calls WHERE operation_id = :operation_id"
                ),
                {"operation_id": operation_id},
            ).one()
            assert migrated_call == (45, "USD", None, None)
        schema_engine.dispose()
    finally:
        get_settings.cache_clear()
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        engine.dispose()


def test_exact_provider_cost_migration_round_trip_and_constraints(monkeypatch) -> None:
    base_url = make_url(os.environ["DATABASE_URL"])
    schema = f"exact_cost_migration_{uuid.uuid4().hex}"
    engine = create_engine(base_url)
    schema_url = base_url.update_query_dict({"options": f"-csearch_path={schema}"})
    migration_config = Config("alembic.ini")
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        migration_url = (
            f"{base_url.render_as_string(hide_password=False)}"
            f"?options=-csearch_path={schema}"
        )
        monkeypatch.setenv("DATABASE_URL", migration_url)
        get_settings.cache_clear()
        command.upgrade(migration_config, "20260912_0009")

        schema_engine = create_engine(schema_url)
        tenant_id = uuid.uuid4()
        operation_id = uuid.uuid4()
        historical_call_id = uuid.uuid4()
        with schema_engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (id, name, slug) VALUES (:id, 'Exact cost', :slug)"),
                {"id": tenant_id, "slug": f"exact-cost-{tenant_id.hex}"},
            )
            connection.execute(
                text(
                    "INSERT INTO ai_operations "
                    "(id, tenant_id, operation_type, idempotency_key, request_hash, provider, "
                    "model, pricing_version, status, estimated_max_cost_micros, "
                    "reserved_cost_micros, actual_cost_micros, cost_currency) VALUES "
                    "(:id, :tenant_id, 'QUALIFICATION', 'exact-cost', :request_hash, 'fake', "
                    "'model', 'v1', 'CALLING', 100, 100, 0, 'RUB')"
                ),
                {"id": operation_id, "tenant_id": tenant_id, "request_hash": "c" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO provider_calls "
                    "(id, tenant_id, operation_id, attempt, provider, model, status, "
                    "cost_currency, cost_known) VALUES "
                    "(:id, :tenant_id, :operation_id, 1, 'fake', 'model', 'FAILED', "
                    "'RUB', true)"
                ),
                {
                    "id": historical_call_id,
                    "tenant_id": tenant_id,
                    "operation_id": operation_id,
                },
            )

        command.upgrade(migration_config, "head")
        column = next(
            item
            for item in inspect(schema_engine).get_columns("provider_calls")
            if item["name"] == "provider_reported_cost_decimal"
        )
        assert column["nullable"] is True
        assert column["type"].precision == 32
        assert column["type"].scale == 18

        insert_reported = text(
            "INSERT INTO provider_calls "
            "(id, tenant_id, operation_id, attempt, provider, model, status, cost_currency, "
            "provider_reported_cost_decimal, provider_reported_cost_micros, "
            "provider_reported_cost_currency, cost_known) VALUES "
            "(:id, :tenant_id, :operation_id, :attempt, 'fake', 'model', 'FAILED', 'RUB', "
            ":decimal, :micros, :currency, true)"
        )
        exact_values = [Decimal("0.04131306"), Decimal("0.123456789012345678")]
        with schema_engine.begin() as connection:
            for attempt, exact in enumerate(exact_values, start=2):
                connection.execute(
                    insert_reported,
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant_id,
                        "operation_id": operation_id,
                        "attempt": attempt,
                        "decimal": exact,
                        "micros": 41_314 if attempt == 2 else 123_457,
                        "currency": "RUB",
                    },
                )
            persisted = connection.execute(
                text(
                    "SELECT provider_reported_cost_decimal FROM provider_calls "
                    "WHERE operation_id = :operation_id AND attempt IN (2, 3) ORDER BY attempt"
                ),
                {"operation_id": operation_id},
            ).scalars().all()
            historical = connection.execute(
                text(
                    "SELECT provider_reported_cost_decimal FROM provider_calls WHERE id = :id"
                ),
                {"id": historical_call_id},
            ).scalar_one()
        assert persisted == exact_values
        assert historical is None

        invalid_states = [
            (Decimal("0.1"), None, None),
            (None, 100_000, None),
            (None, None, "RUB"),
            (Decimal("0.1"), 100_000, None),
            (Decimal("0.1"), None, "RUB"),
            (None, 100_000, "RUB"),
            (Decimal("-0.1"), 100_000, "RUB"),
        ]
        for offset, (decimal, micros, currency) in enumerate(invalid_states, start=4):
            with pytest.raises(IntegrityError), schema_engine.begin() as connection:
                connection.execute(
                    insert_reported,
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant_id,
                        "operation_id": operation_id,
                        "attempt": offset,
                        "decimal": decimal,
                        "micros": micros,
                        "currency": currency,
                    },
                )

        command.downgrade(migration_config, "20260912_0009")
        assert "provider_reported_cost_decimal" not in {
            item["name"] for item in inspect(schema_engine).get_columns("provider_calls")
        }
        with schema_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO provider_calls "
                    "(id, tenant_id, operation_id, attempt, provider, model, status, "
                    "cost_currency, provider_reported_cost_micros, "
                    "provider_reported_cost_currency, cost_known) VALUES "
                    "(:id, :tenant_id, :operation_id, 9, 'fake', 'model', 'FAILED', "
                    "'RUB', 100000, 'RUB', true)"
                ),
                {"id": uuid.uuid4(), "tenant_id": tenant_id, "operation_id": operation_id},
            )
        schema_engine.dispose()
    finally:
        get_settings.cache_clear()
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        engine.dispose()


def test_postgres_concurrent_budget_reservations_never_exceed_hard_limit() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    tenant_id = add_tenant_policy(
        engine, monthly_limit=40, currency=Currency.RUB
    )
    entered_provider = threading.Event()
    release_provider = threading.Event()

    def blocking_success(_request: StructuredAIRequest) -> StructuredAIResult:
        entered_provider.set()
        assert release_provider.wait(timeout=10)
        return success()

    provider = RecordingStructuredProvider("pg-fake", "pg-model", blocking_success)
    runtime = executor(engine, provider, Currency.RUB)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute, runtime, tenant_id, "budget-a")
            assert entered_provider.wait(timeout=10)
            second = pool.submit(execute, runtime, tenant_id, "budget-b")
            with pytest.raises(BudgetExceededError):
                second.result(timeout=10)
            release_provider.set()
            assert first.result(timeout=10).value == PGResult(answer="ok")

        assert provider.physical_call_count == 1
        with Session(engine) as session:
            operations = list(
                session.scalars(
                    select(AIOperation).where(AIOperation.tenant_id == tenant_id)
                )
            )
            assert len(operations) == 1
            assert sum(
                (operation.actual_cost_micros or 0)
                + operation.reserved_cost_micros
                for operation in operations
            ) == 40
    finally:
        release_provider.set()
        cleanup(engine, [tenant_id])
        engine.dispose()


def test_postgres_concurrent_duplicate_creates_one_physical_call() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    tenant_id = add_tenant_policy(engine, monthly_limit=1_000)
    entered_provider = threading.Event()
    release_provider = threading.Event()

    def blocking_success(_request: StructuredAIRequest) -> StructuredAIResult:
        entered_provider.set()
        assert release_provider.wait(timeout=10)
        return success()

    provider = RecordingStructuredProvider("pg-fake", "pg-model", blocking_success)
    runtime = executor(engine, provider)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute, runtime, tenant_id, "duplicate-key")
            assert entered_provider.wait(timeout=10)
            duplicate = pool.submit(execute, runtime, tenant_id, "duplicate-key")
            with pytest.raises(AIOperationNotReplayableError):
                duplicate.result(timeout=10)
            release_provider.set()
            first.result(timeout=10)

        assert provider.physical_call_count == 1
        with Session(engine) as session:
            assert len(
                list(
                    session.scalars(
                        select(AIOperation).where(AIOperation.tenant_id == tenant_id)
                    )
                )
            ) == 1
            assert len(
                list(
                    session.scalars(
                        select(ProviderCall).where(ProviderCall.tenant_id == tenant_id)
                    )
                )
            ) == 1
    finally:
        release_provider.set()
        cleanup(engine, [tenant_id])
        engine.dispose()


def test_postgres_ai_operation_and_call_constraints_enforce_tenant_scope() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    tenant_a = add_tenant_policy(engine, monthly_limit=1_000)
    tenant_b = add_tenant_policy(engine, monthly_limit=1_000)
    operation_id = uuid.uuid4()
    try:
        with Session(engine) as session, session.begin():
            session.add(
                AIOperation(
                    id=operation_id,
                    tenant_id=tenant_a,
                    operation_type="QUALIFICATION",
                    idempotency_key="tenant-constraint",
                    request_hash="b" * 64,
                    provider="pg-fake",
                    model="pg-model",
                    pricing_version="pg-v1",
                    status="CALLING",
                    estimated_max_cost_micros=40,
                    reserved_cost_micros=40,
                    actual_cost_micros=0,
                    cost_currency="USD",
                )
            )

        with pytest.raises(IntegrityError):
            with Session(engine) as session, session.begin():
                session.add(
                    ProviderCall(
                        tenant_id=tenant_b,
                        operation_id=operation_id,
                        attempt=1,
                        provider="pg-fake",
                        model="pg-model",
                        status="FAILED",
                        cost_currency="USD",
                        cost_known=True,
                    )
                )

        factory = SqlAlchemyAIRepositoryFactory(
            sessionmaker(bind=engine, expire_on_commit=False)
        )
        with factory.transaction(tenant_b) as repository:
            assert repository.get_operation_for_update(operation_id) is None
    finally:
        cleanup(engine, [tenant_a, tenant_b])
        engine.dispose()


def test_postgres_offline_stage5a_smoke() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = add_tenant_policy(
        engine,
        monthly_limit=1_000_000,
        provider="mock",
        model="deterministic-keyword-v1",
    )

    def qualification_result(request: StructuredAIRequest) -> StructuredAIResult:
        output = MockQualificationProvider().analyze(
            QualificationContext.model_validate(request.payload)
        ).model_dump(mode="json")
        return StructuredAIResult(
            output=output,
            usage=AIUsage(
                input_tokens=request.estimated_input_tokens,
                output_tokens=20,
            ),
            metadata=ProviderMetadata(
                provider="mock",
                model="deterministic-keyword-v1",
                provider_request_id="offline-pg-smoke",
                finish_reason="stop",
            ),
        )

    provider = RecordingStructuredProvider(
        "mock", "deterministic-keyword-v1", qualification_result
    )
    runtime = AIExecutor(
        Settings(_env_file=None),
        SqlAlchemyAIRepositoryFactory(factory),
        ProviderRegistry([provider]),
        default_mock_pricing_catalog(),
        sleeper=lambda _seconds: None,
    )
    try:
        with factory() as session:
            inbound = InboundIngestionService(
                SqlAlchemyInboundRepository(session, tenant_id),
                SqlAlchemyUnitOfWork(session),
            ).ingest(
                InboundHttpRequest(
                    tenant_id=tenant_id,
                    source="stage5a-smoke",
                    external_event_id=f"smoke-{tenant_id.hex}",
                    name="Offline smoke",
                    email="offline-smoke@example.com",
                    message="Need automation. Budget 100000 RUB, deadline two weeks.",
                )
            )
            service = QualificationService(
                TenantContext.untrusted(tenant_id, "offline-smoke"),
                SqlAlchemyQualificationRepository(session, tenant_id),
                SqlAlchemyUnitOfWork(session),
                MockQualificationProvider(),
                runtime,
            )
            first = service.qualify(inbound.lead_id, "stage5a-smoke-key")
            replay = service.qualify(inbound.lead_id, "stage5a-smoke-key")

        assert replay.replayed is True
        assert replay.ai_operation_id == first.ai_operation_id
        assert provider.physical_call_count == 1

        with factory.begin() as session:
            operation = session.get(AIOperation, first.ai_operation_id)
            assert operation.status == "SUCCEEDED"
            assert operation.result_analysis_id == first.analysis_id
            assert operation.actual_cost_micros > 0
            assert operation.reserved_cost_micros == 0
            call = session.scalar(
                select(ProviderCall).where(ProviderCall.operation_id == operation.id)
            )
            assert call.status == "SUCCEEDED"
            assert call.cost_known is True
            assert call.actual_cost_micros == operation.actual_cost_micros
            session.get(TenantAIPolicy, tenant_id).per_call_limit_micros = 0

        with factory() as session:
            over_budget = QualificationService(
                TenantContext.untrusted(tenant_id, "offline-smoke"),
                SqlAlchemyQualificationRepository(session, tenant_id),
                SqlAlchemyUnitOfWork(session),
                MockQualificationProvider(),
                runtime,
            )
            with pytest.raises(BudgetExceededError):
                over_budget.qualify(inbound.lead_id, "stage5a-over-budget")
        assert provider.physical_call_count == 1
    finally:
        cleanup(engine, [tenant_id])
        engine.dispose()
