#!/bin/sh
# Phase 2 (RLS): give the least-privilege API role `iia_app` a login password.
# Runs once on an empty data directory, after 01-schema.sql created the role's
# grants (iia_grant_app_privileges). APP_DB_PASSWORD is required by compose.
set -e
psql -v ON_ERROR_STOP=1 -v pw="$APP_DB_PASSWORD" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
ALTER ROLE iia_app LOGIN PASSWORD :'pw';
SQL