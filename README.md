# AI Lead Manager 3.0

Технический фундамент multi-tenant AI Sales Operator. Stage 5B-1 добавляет изолированный
OpenAI-compatible provider foundation, но он ещё не включён в production composition:
реальных AI-вызовов, credentials в репозитории, мессенджеров, CRM, фоновых задач и
аутентификации пока нет.

## Архитектура

- `domain` — SQLAlchemy-модели и доменные ошибки;
- `application` — сценарии использования, схемы и repository-порты;
- `infrastructure` — PostgreSQL/SQLAlchemy и реализации репозиториев;
- `api` — HTTP transport на FastAPI.

Модуль `application/ai` владеет provider-neutral контрактами, pricing, idempotency,
spending guard, retry-решениями и circuit breaker. `infrastructure/ai` содержит
SQLAlchemy-репозиторий и полностью локальный recording provider. Будущий HTTP adapter
реального провайдера должен находиться в `infrastructure/ai`, не протекая vendor-типами
в application/domain.

Tenant-scoped репозитории требуют `tenant_id`, фильтруют по нему чтения и отвергают запись сущности
другого tenant. Составные foreign keys обеспечивают tenant-consistency связей также на уровне БД.

## HTTP inbound API

Development/internal endpoint `POST /api/v1/inbound/http` принимает tenant в валидируемом JSON body
и атомарно сохраняет event, customer identity, customer, conversation, message и lead. Ключ
`(tenant_id, source, external_event_id)` обеспечивает идемпотентность; replay возвращает те же ID
с `duplicate: true`.

Email нормализуется через trim + lowercase. Для телефона удаляются пробелы, скобки, дефисы и точки,
сохраняются только цифры и необязательный начальный `+`; код страны не угадывается.

## Qualification Engine

Development/internal endpoint `POST /api/v1/leads/{lead_id}/qualify` принимает `tenant_id` в
валидируемом JSON body. Детерминированный локальный mock извлекает структурированные признаки из
истории входящих сообщений, после чего независимый ruleset `qualification-v1` рассчитывает score,
температуру и объяснимые факторы. Каждая повторная квалификация сохраняется как новый исторический
run; рекомендации записываются, но никогда не отправляются автоматически.

## Sales Workflow Core

После квалификации локальный workflow создаёт одну идемпотентную задачу из последней
рекомендации, генерирует детерминированный mock-черновик ответа и требует явного
approve/reject от менеджера. Отправка сообщений отсутствует. Статус лида меняется только по цепочке
`NEW → QUALIFIED → CONTACTED → MEETING → WON/LOST`; каждое изменение сохраняется в
истории. Ключевые действия квалификации, задач, черновиков и статусов пишутся в
tenant-scoped audit log.

HTTP endpoints:

- `POST /api/v1/leads/{lead_id}/tasks/from-recommendation`;
- `POST /api/v1/leads/{lead_id}/drafts`;
- `POST /api/v1/drafts/{draft_id}/approve` и `/reject`;
- `POST /api/v1/tasks/{task_id}/complete` и `/cancel`;
- `POST /api/v1/leads/{lead_id}/status`.

Все запросы принимают `tenant_id`; repository-слой фильтрует чтения по tenant, а
составные foreign keys блокируют cross-tenant связи на уровне PostgreSQL.

## AI Safety Runtime (Stage 5A)

Режим по умолчанию — `MOCK`. Даже если tenant policy переключена в `LIVE`, вызов
fail-closed: одновременно требуются trusted `TenantContext`, глобальные
`AI_EXECUTION_MODE=LIVE` и `ALLOW_LIVE_AI=true`, разрешённая пара provider/model и
известная серверу версия цены. На этом этапе LIVE provider намеренно отсутствует.
`AI_KILL_SWITCH=true` немедленно запрещает новые AI-операции.

Для безопасного пути qualification/draft передаётся обязательный idempotency key.
Development endpoints сохраняют прежний локальный mock-путь без ключа для обратной
совместимости; этот путь не способен вызвать платного провайдера. Один и тот же ключ и
request hash переиспользуют сохранённый результат, другой hash даёт conflict, а
`CALLING`/`UNKNOWN` никогда не запускаются повторно автоматически.

Денежные значения хранятся только целыми micro-USD. Перед вызовом tenant policy
блокируется `SELECT ... FOR UPDATE`, рассчитывается сумма actual + reserved за текущий
месяц и атомарно резервируется worst-case стоимость input/output и разрешённых попыток.
При неизвестной цене вызов запрещается. Для `UNKNOWN` резерв сохраняется до будущего
ручного reconciliation.

Граница транзакций намеренно трёхчастная:

1. короткая транзакция проверяет policy/idempotency и фиксирует резерв;
2. offline provider вызывается без открытой DB-транзакции;
3. короткая транзакция записывает физическую попытку и exact cost, затем доменная
   транзакция связывает operation с analysis/draft и освобождает остаток резерва.

Если provider уже вернул успех, но persistence результата завершился ошибкой, платный
вызов не повторяется: operation переводится в `UNKNOWN` best-effort (либо остаётся в
безопасном `CALLING`, если сама БД недоступна). Повтор требует будущего reconciliation,
который не входит в Stage 5A.

История сообщений ограничивается детерминированно: сохраняются самые новые непустые
сообщения в пределах `AI_MAX_MESSAGES`, `AI_MAX_INPUT_CHARS` и
`AI_MAX_INPUT_TOKENS`; у одного слишком большого сообщения остаётся новейший suffix.
Provider получает также ограниченный `AI_MAX_OUTPUT_TOKENS` и явные connect/read/total
timeouts.

Retry разрешён только для connect timeout до передачи, 429 с ограниченным Retry-After
и явно помеченных безопасными transport/5xx ошибок. Read/total timeout после возможной
передачи, malformed/schema-invalid output и неоднозначный результат не вызывают новый
платный запрос.

## OpenAI-compatible provider foundation (Stage 5B-1)

Пакет `infrastructure/ai/openai_compatible` содержит server-controlled catalog,
credential resolver, HTTP transport, versioned prompts и отдельные qualification/draft
adapter'ы. Они реализуют Stage 5A `StructuredAIProvider`, но намеренно не зарегистрированы
в production `ProviderRegistry` до Stage 5B-2.

Конфигурация передаётся сервером через `AI_OPENAI_COMPATIBLE_PROVIDERS_JSON`; значение по
умолчанию — пустой allowlist `[]`. В конфигурации находятся только alias, HTTPS base URL,
имя env-переменной с ключом, разрешённые model aliases/capabilities, timeouts, response
limit и pricing version. Сам API key разрешается только infrastructure-компонентом как
`SecretStr` и никогда не хранится в PostgreSQL или tenant policy.

Transport использует `httpx` с `trust_env=False`, отключёнными HTTP retries, явными
connect/read/write/pool timeouts, запретом redirect-following и ограниченным размером
ответа. Клиент не может передать base URL или upstream model. Ошибки переводятся в
Stage 5A taxonomy без vendor response body и без секретов; решение о повторе остаётся за
`RetryPolicy`/`AIExecutor`.

Prompts имеют фиксированные версии, отделяют system instructions от JSON с недоверенными
данными клиента и запрещают следовать инструкциям внутри сообщений. Qualification
возвращает только структурированные факты без Lead Score. Draft обязан быть
русскоязычным и остаётся только черновиком для обязательного human approval.

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

Основной набор тестов использует SQLite и не обращается к сети. PostgreSQL-specific
изоляция, uniqueness и конкурентная резервация запускаются внутри Compose так:

```bash
RUN_POSTGRES_TESTS=1 pytest
```

Все AI-тесты используют recording provider без HTTP transport и не требуют credentials.
