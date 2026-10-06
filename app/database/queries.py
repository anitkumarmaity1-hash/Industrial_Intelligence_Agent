"""
Database query layer for the Industrial Intelligence Agent.

These functions are the actual data-access surface the Phase 4 FastAPI
endpoints and Phase 5 LangGraph tools will call. Written and tested against
a real PostgreSQL instance *before* FastAPI/LangGraph exist, per the phase
plan, so the query logic is validated in isolation.

Deliberately plain SQL via SQLAlchemy's text(), not an ORM: every function
here is either a straightforward filtered/aggregated read or a bounded
ranking query. An ORM would add a mapping layer with no benefit for
read-mostly reporting queries like these.

Every function takes a Connection as its first argument rather than
opening its own connection, so callers (tests, FastAPI's dependency
injection, agent tools) control connection lifecycle and pooling. Typed
as `Connection`, not the broader `Connectable` (which also matches
`Engine`): every real caller passes a request-scoped Connection already
checked out from the pool (app.api.dependencies.get_conn), and calling
`.execute()` on a bare Engine directly — which Connectable would also
accept — bypasses that pooling/lifecycle control this module's docstring
just promised.

Tenant isolation (production-readiness fix 1). Every function that reads
company data takes a REQUIRED keyword-only `tenant_id` and filters every
table it touches on it, so a caller can only ever see its own tenant's
rows. It is keyword-only and has no default on purpose: forgetting it is a
TypeError at the call site, not a silent cross-tenant read. Adding a query
here without a tenant filter fails tests/test_tenant_isolation.py's static
check. The one exception is `get_ai4i_failure_mode_rates`, which reads the
public UCI AI4I reference dataset — shared reference data, not company data.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


def list_machines(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """All machines in the tenant's fleet with static metadata. Backs GET /machines."""
    rows = conn.execute(text("""
        SELECT machine_id, production_line, type, name, install_date, location
        FROM machines
        WHERE tenant_id = :tenant_id
        ORDER BY machine_id
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def get_machine(conn: Connection, machine_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Single machine's metadata. Backs GET /machines/{machine_id}."""
    row = conn.execute(
        text("""
            SELECT machine_id, production_line, type, name, install_date, location
            FROM machines
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().first()
    return dict(row) if row else None


def get_sensor_summary(
    conn: Connection,
    machine_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 168,  # default: last 7 days of hourly windows
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Hourly sensor history for one machine, most recent first.
    Backs the sensor-history evidence a root-cause investigation needs."""
    query = """
        SELECT window_start, window_end, avg_air_temp_k, avg_process_temp_k,
               avg_rotational_speed_rpm, avg_torque_nm, tool_wear_min,
               avg_production_rate, avg_energy_consumption_kwh, avg_defect_rate,
               reading_count, anomalous_reading_count, max_anomaly_score, health_status
        FROM sensor_summary
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if since is not None:
        query += " AND window_start >= :since"
        params["since"] = since
    if until is not None:
        query += " AND window_start <= :until"
        params["until"] = until
    query += " ORDER BY window_start DESC LIMIT :limit"

    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def get_machine_health(conn: Connection, machine_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Latest known health snapshot for one machine.
    Backs GET /machines/{machine_id}/health.

    Audit F11: avg_defect_rate and tool_wear_min added — the Streamlit
    Overview tab's "key metrics" row previously had nothing to show but
    the window's own start/end timestamps, because this query never
    selected any of sensor_summary's actual measured columns beyond the
    anomaly-count bookkeeping fields.
    """
    row = conn.execute(
        text("""
            SELECT machine_id, window_start, window_end, health_status,
                   max_anomaly_score, anomalous_reading_count, reading_count,
                   avg_defect_rate, tool_wear_min
            FROM sensor_summary
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
            ORDER BY window_start DESC
            LIMIT 1
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().first()
    return dict(row) if row else None


def get_machine_anomalies(
    conn: Connection,
    machine_id: str,
    since: datetime | None = None,
    severity: str | None = None,
    limit: int = 100,
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Flagged anomaly events for one machine, most recent first.
    Backs GET /machines/{machine_id}/anomalies."""
    query = """
        SELECT detected_at, anomaly_score, severity, triggered_reasons
        FROM machine_anomalies
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if since is not None:
        query += " AND detected_at >= :since"
        params["since"] = since
    if severity is not None:
        query += " AND severity = :severity"
        params["severity"] = severity
    query += " ORDER BY detected_at DESC LIMIT :limit"

    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def get_maintenance_records(conn: Connection, machine_id: str, *, tenant_id: str) -> list[dict[str, Any]]:
    """Maintenance history for one machine, most recent first."""
    rows = conn.execute(
        text("""
            SELECT event_date, event_type, technician_notes, resolved
            FROM maintenance_records
            WHERE tenant_id = :tenant_id AND machine_id = :machine_id
            ORDER BY event_date DESC
        """),
        {"tenant_id": tenant_id, "machine_id": machine_id},
    ).mappings().all()
    return [dict(r) for r in rows]


def get_fleet_status(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """One row per machine: its most recent health window. Backs
    "which machines currently show abnormal behavior" — a fleet-wide
    scan using each machine's latest window, not a single global timestamp,
    since machines are independent in principle even though this dataset
    happens to have them all aligned to the same hourly grid."""
    rows = conn.execute(text("""
        SELECT DISTINCT ON (s.machine_id)
               s.machine_id, m.production_line, s.window_start, s.health_status,
               s.max_anomaly_score, s.anomalous_reading_count
        FROM sensor_summary s
        JOIN machines m ON m.tenant_id = s.tenant_id AND m.machine_id = s.machine_id
        WHERE s.tenant_id = :tenant_id
        ORDER BY s.machine_id, s.window_start DESC
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def get_fleet_sensor_trend(
    conn: Connection, windows: int = 24, *, tenant_id: str
) -> dict[str, list[dict[str, Any]]]:
    """The last `windows` hourly sensor_summary rows for every machine at
    once, keyed by machine_id. Same shape/ordering as get_sensor_summary,
    just for the whole fleet in one round trip instead of one query per
    machine.

    Phase 9 fix. Exists so fleet_scan (app/agents/nodes.py) can run each
    machine's rows through risk.assess_risk() — the exact same
    sustained-anomalous-rate rule a single-machine /investigate uses —
    instead of flagging on get_fleet_status's single latest hourly window,
    which is noisy enough that a healthy machine's one-off WATCH hour used
    to get it flagged fleet-wide while the same machine's own investigation
    correctly called it LOW risk.
    """
    rows = conn.execute(text("""
        SELECT machine_id, window_start, window_end, health_status,
               max_anomaly_score, anomalous_reading_count, reading_count
        FROM (
            SELECT s.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY s.machine_id ORDER BY s.window_start DESC
                   ) AS rn
            FROM sensor_summary s
            WHERE s.tenant_id = :tenant_id
        ) ranked
        WHERE rn <= :windows
        ORDER BY machine_id, window_start DESC
    """), {"tenant_id": tenant_id, "windows": windows}).mappings().all()

    trend_by_machine: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trend_by_machine.setdefault(row["machine_id"], []).append(dict(row))
    return trend_by_machine


def get_fleet_recent_high_anomalies(conn: Connection, windows: int = 24, *, tenant_id: str) -> set[str]:
    """machine_ids with a HIGH-severity anomaly inside their own last
    `windows` hourly windows — the fleet-wide equivalent of
    risk.assess_risk()'s `_has_recent_high_anomaly` check.

    Computed as one query (a per-machine trend floor timestamp, joined
    against machine_anomalies) rather than a global "detected_at >= NOW() -
    24h" cutoff: the trend floor is exact for every machine regardless of
    whether its data is clock-aligned to "now", which a synthetic dataset
    generated at a fixed reference time is not guaranteed to be.
    """
    rows = conn.execute(text("""
        WITH trend AS (
            SELECT machine_id, window_start,
                   ROW_NUMBER() OVER (
                       PARTITION BY machine_id ORDER BY window_start DESC
                   ) AS rn
            FROM sensor_summary
            WHERE tenant_id = :tenant_id
        ),
        trend_floor AS (
            SELECT machine_id, MIN(window_start) AS floor_ts
            FROM trend
            WHERE rn <= :windows
            GROUP BY machine_id
        )
        SELECT DISTINCT a.machine_id
        FROM machine_anomalies a
        JOIN trend_floor t ON t.machine_id = a.machine_id
        WHERE a.tenant_id = :tenant_id
          AND a.severity = 'HIGH' AND a.detected_at >= t.floor_ts
    """), {"tenant_id": tenant_id, "windows": windows}).mappings().all()
    return {row["machine_id"] for row in rows}


def get_ai4i_failure_mode_rates(conn: Connection) -> dict[str, dict[str, Any]]:
    """Real-world incidence of each AI4I failure mode (TWF/HDF/PWF/OSF/RNF)
    among ai4i_reference's 10,000 published rows.

    P2 cleanup: ai4i_reference was loaded and indexed (Phase 3/schema.sql)
    but no tool ever queried it — the "real data" story in the audit was
    decorative. synthesis.infer_failure_modes already maps every anomaly's
    triggered_reasons to one of these same five codes, so this is the one
    query needed to attach a real base rate to each root-cause candidate
    (see app.agents.synthesis.build_root_cause_candidates) instead of only
    ever reasoning over the synthetic fleet.

    One row per mode: how many of the 10,000 real machines tripped that
    flag, and what share of the real *machine_failure=1* rows (not all
    10,000 — a mode's rate is only meaningful relative to failures, since
    a machine can trip a flag without an overall failure) each mode
    accounts for. Modes can co-occur, so the per-mode shares don't sum to
    1.0 — that's expected, not a bug.

    Not tenant-scoped, on purpose: ai4i_reference is the public UCI dataset
    (shared read-only reference data), not any company's data.
    """
    total_row = conn.execute(
        text("SELECT COUNT(*) AS n FROM ai4i_reference")).mappings().one()
    total_rows = total_row["n"]
    failed_row = conn.execute(
        text("SELECT COUNT(*) AS n FROM ai4i_reference WHERE machine_failure = 1")
    ).mappings().one()
    total_failures = failed_row["n"]

    rows = conn.execute(text("""
        SELECT
            SUM(twf) AS twf, SUM(hdf) AS hdf, SUM(pwf) AS pwf,
            SUM(osf) AS osf, SUM(rnf) AS rnf
        FROM ai4i_reference
        WHERE machine_failure = 1
    """)).mappings().one()

    rates: dict[str, dict[str, Any]] = {}
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        count = rows[mode.lower()] or 0
        rates[mode] = {
            "failure_count": count,
            "share_of_real_failures": (count / total_failures) if total_failures else 0.0,
            "total_real_rows": total_rows,
            "total_real_failures": total_failures,
        }
    return rates


def rank_machines_for_inspection(
    conn: Connection, since: datetime, top_n: int = 5, *, tenant_id: str
) -> list[dict[str, Any]]:
    """Rank machines by anomaly volume/severity since a given time.
    Backs "which machines should the maintenance team inspect first" —
    deterministic ranking by count and severity-weighted score, computed
    entirely in SQL so the LLM never invents the ordering."""
    rows = conn.execute(
        text("""
            SELECT machine_id,
                   COUNT(*) AS anomaly_count,
                   SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) AS high_severity_count,
                   MAX(detected_at) AS most_recent_anomaly
            FROM machine_anomalies
            WHERE tenant_id = :tenant_id AND detected_at >= :since
            GROUP BY machine_id
            ORDER BY high_severity_count DESC, anomaly_count DESC
            LIMIT :top_n
        """),
        {"tenant_id": tenant_id, "since": since, "top_n": top_n},
    ).mappings().all()
    return [dict(r) for r in rows]


def get_tenant_settings(conn: Connection, *, tenant_id: str) -> dict[str, Any] | None:
    """This tenant's calibration overrides (production-readiness fix 11/12:
    risk thresholds, RAG coverage floor, machine-ID scheme), or None when
    they haven't set any — see app.core.tenant_settings for how the
    documented defaults are filled in around whatever this returns."""
    row = conn.execute(
        text("SELECT config FROM tenant_settings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id},
    ).mappings().first()
    return dict(row["config"]) if row else None


def upsert_tenant_settings(conn: Connection, config: dict[str, Any], *, tenant_id: str) -> None:
    """Replace this tenant's calibration config wholesale (see
    app.core.tenant_settings for its shape). Used by
    scripts/manage_tenants.py `calibrate`, never by request-serving code."""
    conn.execute(
        text("""
            INSERT INTO tenant_settings (tenant_id, config, updated_at)
            VALUES (:tenant_id, CAST(:config AS JSONB), NOW() AT TIME ZONE 'UTC')
            ON CONFLICT (tenant_id)
            DO UPDATE SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at
        """),
        {"tenant_id": tenant_id, "config": json.dumps(config)},
    )


def insert_investigation_audit(
    conn: Connection,
    *,
    tenant_id: str,
    request_id: str | None,
    endpoint: str,
    machine_id: str | None,
    intent: str | None,
    risk_level: str | None,
    authenticated: bool,
    question: str,
) -> None:
    """Record who asked what, of which machine, and what the agent
    concluded — production-readiness fix 19. Called once, after a
    successful /investigate or /chat call (app/api/routes.py); a call that
    raises before returning is never audited here, which is fine — the
    thing worth an audit trail is "tenant X's credential was used to
    investigate machine Y", not the request's outcome."""
    conn.execute(
        text("""
            INSERT INTO investigation_audit_log
                (tenant_id, request_id, endpoint, machine_id, intent,
                 risk_level, authenticated, question)
            VALUES (:tenant_id, :request_id, :endpoint, :machine_id, :intent,
                    :risk_level, :authenticated, :question)
        """),
        {
            "tenant_id": tenant_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "machine_id": machine_id,
            "intent": intent,
            "risk_level": risk_level,
            "authenticated": authenticated,
            "question": question,
        },
    )


def get_ingestion_checkpoint(conn: Connection, source: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Where an incremental ingestion run for this tenant/source left off
    (production-readiness fix 17) — see scripts/incremental_load.py."""
    row = conn.execute(
        text("""
            SELECT source, last_window_end, last_run_at, rows_processed
            FROM ingestion_checkpoints
            WHERE tenant_id = :tenant_id AND source = :source
        """),
        {"tenant_id": tenant_id, "source": source},
    ).mappings().first()
    return dict(row) if row else None


def upsert_ingestion_checkpoint(
    conn: Connection,
    source: str,
    last_window_end: datetime,
    rows_processed: int,
    *,
    tenant_id: str,
) -> None:
    """Advance (or create) this tenant/source's incremental-ingestion
    checkpoint. `rows_processed` is this run's count, not a running total —
    scripts/incremental_load.py logs the total separately."""
    conn.execute(
        text("""
            INSERT INTO ingestion_checkpoints
                (tenant_id, source, last_window_end, last_run_at, rows_processed)
            VALUES (:tenant_id, :source, :last_window_end, NOW() AT TIME ZONE 'UTC', :rows_processed)
            ON CONFLICT (tenant_id, source) DO UPDATE SET
                last_window_end = EXCLUDED.last_window_end,
                last_run_at = EXCLUDED.last_run_at,
                rows_processed = EXCLUDED.rows_processed
        """),
        {
            "tenant_id": tenant_id,
            "source": source,
            "last_window_end": last_window_end,
            "rows_processed": rows_processed,
        },
    )


# ---------------------------------------------------------------------
# Generic sensor model (migration 0005): sensor_registry + sensor_readings
# ---------------------------------------------------------------------

def list_sensor_registry(conn: Connection, *, tenant_id: str) -> list[dict[str, Any]]:
    """This tenant's declared sensors and their anomaly-rule settings."""
    rows = conn.execute(text("""
        SELECT sensor_name, unit, normal_min, normal_max, z_score_threshold, enabled
        FROM sensor_registry
        WHERE tenant_id = :tenant_id
        ORDER BY sensor_name
    """), {"tenant_id": tenant_id}).mappings().all()
    return [dict(r) for r in rows]


def upsert_sensor_registry(
    conn: Connection, sensors: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Create or update this tenant's sensors (idempotent on sensor_name)."""
    if not sensors:
        return 0
    rows = [{
        "tenant_id": tenant_id,
        "sensor_name": s["sensor_name"],
        "unit": s["unit"],
        "normal_min": s.get("normal_min"),
        "normal_max": s.get("normal_max"),
        "z_score_threshold": s.get("z_score_threshold"),
        "enabled": s.get("enabled", True),
    } for s in sensors]
    conn.execute(text("""
        INSERT INTO sensor_registry
            (tenant_id, sensor_name, unit, normal_min, normal_max, z_score_threshold, enabled)
        VALUES (:tenant_id, :sensor_name, :unit, :normal_min, :normal_max,
                :z_score_threshold, :enabled)
        ON CONFLICT (tenant_id, sensor_name) DO UPDATE SET
            unit = EXCLUDED.unit, normal_min = EXCLUDED.normal_min,
            normal_max = EXCLUDED.normal_max,
            z_score_threshold = EXCLUDED.z_score_threshold, enabled = EXCLUDED.enabled
    """), rows)
    return len(rows)


def upsert_machines(
    conn: Connection, machines: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Create or update machine rows (machine_id, production_line, type)."""
    if not machines:
        return 0
    def _clean(value):   # pandas hands back NaN for "no value"; the column wants NULL
        return None if value is None or (isinstance(value, float) and value != value) or value == "" else value

    rows = [{
        "tenant_id": tenant_id,
        "machine_id": m["machine_id"],
        "production_line": _clean(m.get("production_line")) or "UNASSIGNED",
        "type": _clean(m.get("type")),
    } for m in machines]
    conn.execute(text("""
        INSERT INTO machines (tenant_id, machine_id, production_line, type)
        VALUES (:tenant_id, :machine_id, :production_line, :type)
        ON CONFLICT (tenant_id, machine_id) DO UPDATE SET
            production_line = EXCLUDED.production_line, type = EXCLUDED.type
    """), rows)
    return len(rows)


def insert_sensor_readings(
    conn: Connection, readings: list[dict[str, Any]], *, tenant_id: str
) -> int:
    """Bulk-insert long-format readings. Idempotent: the primary key
    (tenant_id, machine_id, sensor_name, ts) is the conflict target and an
    already-present reading is left untouched (first write wins, same as
    scripts/incremental_load.py). Returns the number actually inserted."""
    if not readings:
        return 0
    rows = [{
        "tenant_id": tenant_id, "machine_id": r["machine_id"],
        "sensor_name": r["sensor_name"], "ts": r["ts"], "value": r["value"],
    } for r in readings]
    before = conn.execute(text(
        "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}).scalar()
    conn.execute(text("""
        INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
        VALUES (:tenant_id, :machine_id, :sensor_name, :ts, :value)
        ON CONFLICT (tenant_id, machine_id, sensor_name, ts) DO NOTHING
    """), rows)
    after = conn.execute(text(
        "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}).scalar()
    return after - before


def get_sensor_readings(
    conn: Connection,
    machine_id: str,
    sensor_name: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 1000,
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Raw readings for one machine (optionally one sensor), newest first."""
    query = """
        SELECT sensor_name, ts, value
        FROM sensor_readings
        WHERE tenant_id = :tenant_id AND machine_id = :machine_id
    """
    params: dict[str, Any] = {
        "tenant_id": tenant_id, "machine_id": machine_id, "limit": limit}
    if sensor_name is not None:
        query += " AND sensor_name = :sensor_name"
        params["sensor_name"] = sensor_name
    if since is not None:
        query += " AND ts >= :since"
        params["since"] = since
    if until is not None:
        query += " AND ts <= :until"
        params["until"] = until
    query += " ORDER BY ts DESC, sensor_name LIMIT :limit"
    rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


def merge_tenant_anomaly_settings(
    conn: Connection, anomaly: dict[str, Any], *, tenant_id: str
) -> None:
    """Set ONLY the `anomaly` key of this tenant's settings, leaving the
    risk / RAG / machine-id keys alone (upsert_tenant_settings, by contrast,
    replaces the whole config)."""
    conn.execute(
        text("""
            INSERT INTO tenant_settings (tenant_id, config, updated_at)
            VALUES (:tenant_id, jsonb_build_object('anomaly', CAST(:anomaly AS JSONB)),
                    NOW() AT TIME ZONE 'UTC')
            ON CONFLICT (tenant_id) DO UPDATE SET
                config = tenant_settings.config || EXCLUDED.config,
                updated_at = EXCLUDED.updated_at
        """),
        {"tenant_id": tenant_id, "anomaly": json.dumps(anomaly)},
    )


def get_tenant_by_key_hash(conn: Connection, api_key_hash: str) -> dict[str, Any] | None:
    """The active tenant whose credential hashes to `api_key_hash`, or None.

    This is the authentication lookup (app.api.dependencies.get_tenant), so
    it is the one query here that is keyed by credential rather than by
    tenant_id — the tenant id is its *output*.
    """
    row = conn.execute(
        text("""
            SELECT tenant_id, name
            FROM tenants
            WHERE api_key_hash = :api_key_hash AND active
        """),
        {"api_key_hash": api_key_hash},
    ).mappings().first()
    return dict(row) if row else None
