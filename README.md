# AI Lead Manager 3.0 татаарараа9ми
Технический фундамент multi-tenant AI Sales Operator. Stage 5B-2 интегрирует
OpenAI-compatible qualification/draft adapters в безопасный AI executor. Режим по умолчанию
остаётся полностью локальным `MOCK`; реальных AI-вызовов, credentials в репозитории,
мессенджеров, CRM, фоновых задач и аутентификации пока нет.

## Архитектура

- `domain` — SQLAlchemy-модели и доменные ошибки;
- `application` — сценарии использования, схемы и repository-порты;
- `infrastructure` — PostgreSQL/SQLAlchemy и реализации репозиториев;
- `api` — HTTP transport на FastAPI.

Модуль `application/ai` владеет provider-neutral контрактами, pricing, idempotency,
spending guard, retry-решениями и circuit breaker. `infrastructure/ai` содержит
SQLAlchemy-репозиторий, composition root, полностью локальный recording provider и
OpenAI-compatible HTTP adapters. Vendor-типы не протекают в application/domain.

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
валидируемом JSON body и поддерживает стандартный заголовок `Idempotency-Key`. Детерминированный локальный mock извлекает структурированные признаки из
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
известная серверу версия цены и доступный server-owned credential.
`AI_KILL_SWITCH=true` немедленно запрещает новые AI-операции.

Для безопасного пути qualification/draft передаётся обязательный idempotency key.
Development endpoints сохраняют прежний локальный mock-путь без ключа для обратной
совместимости; этот путь не способен вызвать платного провайдера. Один и тот же ключ и
request hash переиспользуют сохранённый результат, другой hash даёт conflict, а
`CALLING`/`UNKNOWN` никогда не запускаются повторно автоматически.

Денежные значения хранятся целыми micros с явной валютой `USD` или `RUB`; FX-конвертация
не выполняется, а смешение валют блокируется. Перед вызовом tenant policy
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
adapter'ы. Они реализуют Stage 5A `StructuredAIProvider` и регистрируются composition root
только для явно разрешённых operation/provider/model комбинаций.

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

### Polza provider profile (Stage 5C-2A2.2)

Server-controlled profile `config/polza-provider.json` registers provider alias `polza`
and internal model alias `lead-structured-v1` through the existing OpenAI-compatible
infrastructure. Deployments may pass its JSON value through
`AI_OPENAI_COMPATIBLE_PROVIDERS_JSON`; the credential is referenced only by the
`POLZA_AI_API_KEY` environment-variable name.

The immutable server-controlled pricing snapshot is tracked in
`config/polza-pricing-20260912-r1.json` for loading through `AI_MODEL_PRICING_JSON`.
Version `polza-gpt41mini-rub-20260912-r1` fixes input at `48_540_000` and output at
`194_160_000` micro-RUB per million tokens. Its cached-input rate is deliberately the normal
input rate, `48_540_000`, as a conservative reservation safety rate because Polza's exact
cache-read tariff has not been authoritatively confirmed; this is not a claim about the
provider's actual tariff, and cached input is never assumed free. Any future authoritative
cache price or other rate change must use a new pricing version and must not alter this snapshot.

The profile and snapshot do not activate LIVE for any tenant.

## Executor integration (Stage 5B-2)

Composition root объединяет MOCK и LIVE реализации в operation-aware `ProviderRegistry`,
поэтому одна allowlisted пара provider/model может безопасно иметь разные qualification и
draft adapters. Tenant policy выбирает только серверные aliases; API не принимает base URL,
upstream model или цену. LIVE pricing задаётся отдельным server-controlled JSON catalog через
`AI_MODEL_PRICING_JSON`, а версия цены обязана совпадать с версией allowlisted model config.

Перед построением canonical request service разрешает route и использует его фиксированные
schema/prompt versions. Hash включает operation, lead/analysis, provider/model и ограниченный
semantic payload. HTTP всегда выполняется после commit прикладной транзакции. Каждая физическая
попытка получает отдельный `ProviderCall`; сохраняются upstream model, usage, cached tokens,
request ID, finish reason, latency, exact currency-aware cost, опциональная
provider-reported стоимость и cost-known status.

Аудит связывает reservation, attempt start/result, retry, circuit rejection, budget rejection,
domain finalization и `UNKNOWN`. Malformed output не запускает repair-вызов; неоднозначный
результат сохраняет worst-case reservation. Черновик по-прежнему получает
`APPROVAL_REQUIRED` и не имеет автоматического send-path.

## Tenant API keys (Stage 5C-1A)

Tenant API keys имеют формат `alm_sk_<12 hex chars>_<43 base64url chars>`: открытый
12-символьный prefix используется только для tenant-scoped lookup, а 32-байтный secret
создаётся через `secrets` и возвращается только при создании. В БД сохраняются prefix и
HMAC-SHA256 полного ключа с обязательным server-side `TENANT_API_KEY_PEPPER`; plaintext и
pepper не сохраняются. Поддерживаются только scopes `ai:qualify` и `ai:draft`, отзыв и
опциональное время истечения. LIVE qualification/draft принимают ключ через стандартный
`Authorization: Bearer` (`HTTPBearer`), после успешной проверки создают server-owned trusted
`TenantContext` и проверяют operation scope до budget reservation или provider transport.
Body `tenant_id` остаётся лишь lookup-кандидатом и не может создать trusted context без ключа.

`POST /api/v1/ai/readiness` принимает только operation и lead ID, а tenant получает из
проверенного Bearer key. Preflight ограничивает qualification/draft context, разрешает
server-owned route/pricing, считает currency-aware worst-case стоимость и доступный бюджет,
возвращает денежные объекты `{amount_micros, currency}`,
машиночитаемые blockers и не резервирует средства, не создаёт AI ledger и не вызывает provider.

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

какашке
