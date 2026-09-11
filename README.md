# AI Lead Manager 3.0

Технический фундамент multi-tenant AI Sales Operator. Этот этап намеренно не содержит AI,
мессенджеров, CRM, фоновых задач, аутентификации или платных интеграций.

## Архитектура

- `domain` — SQLAlchemy-модели и доменные ошибки;
- `application` — сценарии использования, схемы и repository-порты;
- `infrastructure` — PostgreSQL/SQLAlchemy и реализации репозиториев;
- `api` — HTTP transport на FastAPI.

Tenant-scoped репозитории требуют `tenant_id`, фильтруют по нему чтения и отвергают запись сущности
другого tenant. Составные foreign keys обеспечивают tenant-consistency связей также на уровне БД.

## HTTP inbound API

Development/internal endpoint `POST /api/v1/inbound/http` принимает tenant в валидируемом JSON body
и атомарно сохраняет event, customer identity, customer, conversation, message и lead. Ключ
`(tenant_id, source, external_event_id)` обеспечивает идемпотентность; replay возвращает те же ID
с `duplicate: true`.

Email нормализуется через trim + lowercase. Для телефона удаляются пробелы, скобки, дефисы и точки,
сохраняются только цифры и необязательный начальный `+`; код страны не угадывается.

## Docker

```bash
docker compose up --build
curl http://localhost:8000/health
docker compose down
```

Миграции автоматически применяются при старте. Ожидаемый ответ: `{"status":"ok"}`.
Удаление локального volume при необходимости: `docker compose down -v`.

## Локальная разработка

```powershell
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
Copy-Item .env.example .env
alembic upgrade head
uvicorn ai_lead_manager.main:app --reload
```

При запуске приложения с хоста замените hostname `postgres` в `DATABASE_URL` на `localhost`.

## Проверки

```bash
pytest
ruff check .
alembic check
docker compose config
```

Тесты используют SQLite in-memory, не обращаются к сети и не требуют credentials.
