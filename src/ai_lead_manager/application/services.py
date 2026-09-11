import uuid

from ai_lead_manager.application.ports import (
    CustomerRepository,
    LeadRepository,
    TenantRepository,
    UnitOfWork,
)
from ai_lead_manager.application.schemas import CustomerCreate, LeadCreate, TenantCreate
from ai_lead_manager.domain.exceptions import EntityNotFoundError
from ai_lead_manager.domain.models import Customer, Lead, Tenant


class TenantService:
    def __init__(self, repository: TenantRepository, uow: UnitOfWork) -> None:
        self.repository = repository
        self.uow = uow

    def create(self, data: TenantCreate) -> Tenant:
        tenant = self.repository.add(Tenant(name=data.name, slug=data.slug))
        self.uow.commit()
        return tenant


class CustomerService:
    def __init__(self, tenant_id: uuid.UUID, repository: CustomerRepository, uow: UnitOfWork) -> None:
        self.tenant_id = tenant_id
        self.repository = repository
        self.uow = uow

    def create(self, data: CustomerCreate) -> Customer:
        customer = self.repository.add(Customer(tenant_id=self.tenant_id, **data.model_dump()))
        self.uow.commit()
        return customer


class LeadService:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        repository: LeadRepository,
        customers: CustomerRepository,
        uow: UnitOfWork,
    ) -> None:
        self.tenant_id = tenant_id
        self.repository = repository
        self.customers = customers
        self.uow = uow

    def create(self, data: LeadCreate) -> Lead:
        if self.customers.get(data.customer_id) is None:
            raise EntityNotFoundError("Customer was not found in the current tenant")
        lead = self.repository.add(Lead(tenant_id=self.tenant_id, **data.model_dump()))
        self.uow.commit()
        return lead
