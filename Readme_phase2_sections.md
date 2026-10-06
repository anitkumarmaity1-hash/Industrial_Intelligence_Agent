<!-- Paste each block into README.md where indicated. -->

## 1) Under "Multi-tenancy" (append after the manage_tenants.py code block)

Isolation is enforced twice. Every query filters on `tenant_id`, and Postgres
row-level security (migration `0006`) refuses anything else: the API connects as
the least-privilege role `iia_app` (`APP_DATABASE_URL`), and each request sets
the transaction-local `app.tenant_id` from its API key. With no tenant set,
every tenant table returns zero rows and rejects writes. `tenants` is not
readable by the API role; login goes through the `auth_lookup_tenant()`
function. Owner tasks (alembic, seed, loaders, backup/restore, `manage_*`) use
`DATABASE_URL`, a superuser or BYPASSRLS role, on purpose. `tests/test_rls.py`
proves the behavior as `iia_app`.

## 2) Under "Setup > Local (no Docker)" (after `alembic upgrade head`)

```bash
APP_DB_PASSWORD=<choose one> python scripts/provision_app_role.py   # creates LOGIN for iia_app
# then in .env: APP_DATABASE_URL=postgresql://iia_app:<password>@localhost:5432/industrial_intelligence
```

## 3) Under "Setup > Docker"

Set `APP_DB_PASSWORD` as well as `POSTGRES_PASSWORD` and `API_KEY` (all required).
The API container only receives `APP_DATABASE_URL` (the `iia_app` role).

## 4) Append to "Limitations"

- **RLS scope (Phase 2).** RLS protects Postgres rows only. It does not cover Pinecone namespaces, BM25 chunk files on disk, object storage, logs, or backups (a dump contains every tenant). Any superuser or BYPASSRLS connection ignores it, including all owner/batch scripts, so those do not guard against a wrong `tenant_id` stamp. If the API connects with such a role it warns locally and refuses to start against a remote database (`RLS_REQUIRED`).
- **Owner role needs superuser or BYPASSRLS.** `FORCE ROW LEVEL SECURITY` also binds a plain table owner, so the seed load, `pg_dump` and admin scripts need a superuser or BYPASSRLS role. Creating `iia_app` needs `CREATEROLE` or a superuser.
- **Not verified:** `docker compose up` with the new init script, and the new CI steps on GitHub.