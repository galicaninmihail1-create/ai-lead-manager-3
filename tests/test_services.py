import uuid

import pytest
from sqlalchemy.orm import Session

from ai_lead_manager.application.schemas import CustomerCreate, LeadCreate, TenantCreate
from ai_lead_manager.application.services import CustomerService, LeadService, TenantService
from ai_lead_manager.domain.exceptions import EntityNotFoundError, TenantScopeError
from ai_lead_manager.domain.models import Customer
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyCustomerRepository,
    SqlAlchemyLeadRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
)


def create_tenant(session: Session, slug: str) -> uuid.UUID:
    service = TenantService(SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session))
    return service.create(TenantCreate(name=slug.title(), slug=slug)).id


def test_create_tenant(session: Session) -> None:
    repository = SqlAlchemyTenantRepository(session)
    tenant = TenantService(repository, SqlAlchemyUnitOfWork(session)).create(TenantCreate(name="Acme", slug="acme"))
    assert tenant.id is not None
    assert repository.get(tenant.id) == tenant


def test_create_customer(session: Session) -> None:
    tenant_id = create_tenant(session, "acme")
    repository = SqlAlchemyCustomerRepository(session, tenant_id)
    customer = CustomerService(tenant_id, repository, SqlAlchemyUnitOfWork(session)).create(
        CustomerCreate(name="Ada", email="ada@example.com"))
    assert customer.tenant_id == tenant_id
    assert repository.get(customer.id) == customer


def test_create_lead(session: Session) -> None:
    tenant_id = create_tenant(session, "acme")
    customers = SqlAlchemyCustomerRepository(session, tenant_id)
    customer = CustomerService(tenant_id, customers, SqlAlchemyUnitOfWork(session)).create(CustomerCreate(name="Ada"))
    leads = SqlAlchemyLeadRepository(session, tenant_id)
    lead = LeadService(tenant_id, leads, customers, SqlAlchemyUnitOfWork(session)).create(
        LeadCreate(customer_id=customer.id, source="manual", score=80))
    assert lead.tenant_id == tenant_id
    assert lead.customer_id == customer.id
    assert leads.get(lead.id) == lead


def test_tenant_isolation(session: Session) -> None:
    tenant_a = create_tenant(session, "tenant-a")
    tenant_b = create_tenant(session, "tenant-b")
    repo_a = SqlAlchemyCustomerRepository(session, tenant_a)
    repo_b = SqlAlchemyCustomerRepository(session, tenant_b)
    customer = CustomerService(tenant_a, repo_a, SqlAlchemyUnitOfWork(session)).create(
        CustomerCreate(name="Private customer"))
    assert repo_b.get(customer.id) is None
    assert repo_b.list() == []
    with pytest.raises(TenantScopeError):
        repo_b.add(Customer(tenant_id=tenant_a, name="Wrong scope"))
    with pytest.raises(EntityNotFoundError):
        LeadService(tenant_b, SqlAlchemyLeadRepository(session, tenant_b), repo_b,
                    SqlAlchemyUnitOfWork(session)).create(LeadCreate(customer_id=customer.id))
