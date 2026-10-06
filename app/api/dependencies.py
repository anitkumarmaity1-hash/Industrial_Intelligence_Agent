"""
FastAPI dependency providers.

Two resources are shared across requests but must not be rebuilt per
request:

  * The SQLAlchemy Engine — built once in `app.main`'s lifespan and stored
    on `app.state`. `get_conn` opens a short-lived pooled connection per
    request and closes it when the request ends; `queries.py` functions
    already accept any `Connection`, so no query code changes for FastAPI.
  * The retriever — `app.rag.retriever.get_retriever()` is already a
    process-wide singleton built lazily on first use. `get_search_retriever`
    just exposes that existing cache through DI (for overriding in tests),
    it does not add a second cache on top of it.

Both raise a clean 503 if the resource can't be constructed (e.g.
DATABASE_URL unset) instead of letting a raw exception surface from deep
inside a request handler.

Tenant isolation (production-readiness fixes 1-3): `get_tenant` is the single
place a request's tenant is decided, from its X-API-Key. Everything
downstream (queries, retriever) is scoped by the TenantContext it returns —
nothing reads a tenant from the request body, path or query string.
"""

from __future__ import annotations

import hmac
import logging
from typing import Iterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.api.rate_limit import limiter, retry_after_header
from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext, hash_api_key
from app.core.tenant_settings import TenantCalibration, load_tenant_calibration
from app.database import queries
from app.database.tenant_context import bind_tenant, lookup_tenant_by_key_hash
from app.rag.retriever import DocumentRetriever, get_retriever
from app.agents.llm import GeminiSynthesizer, get_synthesizer
from app.agents.planner import get_planner_llm

logger = logging.getLogger(__name__)


def get_db_engine(request: Request) -> Engine:
    """The process-wide Engine, built once at startup (see app.main lifespan)."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503, detail="Database is not configured (DATABASE_URL unset).")
    return engine


def get_conn(request: Request) -> Iterator[Connection]:
    """A request-scoped connection from the shared pooled Engine.

    `Engine` construction (get_db_engine) only proves DATABASE_URL was set
    and well-formed — it does not open a socket. The actual connection
    attempt happens here, on first use, so a wrong password or an
    unreachable/down Postgres surfaces here too. Both are operational
    failures a caller should see as a clean 503, not an unhandled
    exception with a raw driver traceback.
    """
    engine = get_db_engine(request)
    try:
        with engine.connect() as connection:
            yield connection
    except SQLAlchemyError as exc:
        # Audit F10: the raw driver exception used to go straight into the
        # HTTP response body, which leaked internal infrastructure detail
        # to the caller (the DB host:port showed up verbatim in one test
        # run). The exception is still fully logged server-side — nothing
        # is lost for debugging — the caller just gets a generic message.
        logger.error("Database connection failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Could not connect to the database.") from exc


def get_tenant(
    request: Request,
    x_api_key: str | None = Header(default=None),
    conn: Connection = Depends(get_conn),
) -> TenantContext:
    """Authenticate the request, decide which tenant it acts as, and bind
    that tenant to the request's connection for Postgres RLS (Phase 2) —
    the one place the database tenant context is set. See `_authenticate`."""
    tenant = _authenticate(x_api_key, conn)
    bind_tenant(conn, tenant.tenant_id)
    return tenant


def _authenticate(x_api_key: str | None, conn: Connection) -> TenantContext:
    """Authenticate the request and decide which tenant it acts as.

    Order of checks:
      1. X-API-Key present and equal to the legacy bootstrap API_KEY
         -> the demo tenant (constant-time compare). It can never select
         any other tenant.
      2. X-API-Key present -> looked up (by SHA-256 hash) in `tenants`;
         an active match acts as that tenant.
      3. A key was sent but matched nothing -> 401, always (even in demo
         posture: a wrong key is a mistake worth surfacing, not a request
         to quietly downgrade to the demo tenant).
      4. No key, AUTH_REQUIRED off -> the demo tenant, unauthenticated
         (the zero-config `docker compose up` posture).
      5. No key, AUTH_REQUIRED on -> 401.

    The tenant is derived only from the credential. There is deliberately
    no way to name a tenant in the request, so "tenant A's key + tenant B's
    machine_id" can only ever produce a 404 for a machine that doesn't
    exist in A's fleet.
    """
    settings = get_settings()

    if x_api_key:
        if settings.api_key and hmac.compare_digest(
                x_api_key.encode("utf-8"), settings.api_key.encode("utf-8")):
            return TenantContext(DEFAULT_TENANT_ID, name="Demo tenant", authenticated=True)
        # Phase 2: the lookup goes through a SECURITY DEFINER function; the
        # API's role has no direct access to `tenants`. Only reached with a
        # real connection: the demo-posture path below never touches the database, which is what lets tests (and a
        # no-DB deployment's /) run without one.
        try:
            row = lookup_tenant_by_key_hash(conn, hash_api_key(x_api_key))
        except SQLAlchemyError as exc:
            logger.error("Tenant lookup failed: %s", exc)
            raise HTTPException(
                status_code=503, detail="Could not verify credentials.") from exc
        if row is None:
            raise HTTPException(
                status_code=401, detail="Missing or invalid API key.")
        return TenantContext(row["tenant_id"], name=row["name"], authenticated=True)

    if settings.auth_required:
        raise HTTPException(
            status_code=401, detail="Missing or invalid API key.")
    return TenantContext(DEFAULT_TENANT_ID, name="Demo tenant", authenticated=False)


def rate_limit_llm(request: Request, tenant: TenantContext = Depends(get_tenant)) -> None:
    """Throttle the billed LLM-backed endpoints (/investigate, /chat).

    Limit = Settings.rate_limit_per_minute (0 disables), applied per tenant
    for authenticated callers and per (tenant, client IP) for anonymous
    demo traffic. Over the limit -> 429 with a Retry-After header.
    """
    settings = get_settings()
    key = tenant.tenant_id
    if not tenant.authenticated:
        client_host = request.client.host if request.client else "unknown"
        key = f"{tenant.tenant_id}|{client_host}"
    wait = limiter.check(key, settings.rate_limit_per_minute, 60.0)
    if wait is not None:
        logger.warning("rate limit exceeded for %s", key)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"Retry-After": retry_after_header(wait)},
        )


def get_tenant_calibration(
    tenant: TenantContext = Depends(get_tenant),
    conn: Connection = Depends(get_conn),
) -> TenantCalibration:
    """This request's tenant's calibration (production-readiness fixes
    11/12: risk thresholds, RAG coverage floor, machine-ID scheme).

    A plain FastAPI dependency (not a process-wide cache like
    get_search_retriever/get_retriever): reading one small JSONB row is
    cheap, and it means an operator's `manage_tenants.py calibrate` change
    takes effect on the tenant's very next request, not after a restart.
    FastAPI resolves this once per request regardless of how many route
    parameters depend on it, so /investigate and /chat each pay for one
    read, not two.

    Same degrade-not-raise contract as get_gemini_synthesizer/
    get_evidence_planner_llm: a failed lookup (a transient DB hiccup, or —
    in tests — a stubbed get_conn that returns something that isn't a
    live Connection) falls back to the documented defaults rather than
    turning an otherwise-servable request into a 500. Calibration is an
    optimization on top of correct defaults, not something request
    handling should ever depend on being reachable.
    """
    try:
        return load_tenant_calibration(conn, tenant.tenant_id)
    except Exception as exc:  # noqa: BLE001 - degrade to defaults, never 500
        logger.warning(
            "Tenant calibration unavailable for %s (%s); using defaults.",
            tenant.tenant_id, exc)
        return _default_calibration()


def _default_calibration() -> TenantCalibration:
    from app.agents.query_parser import DEFAULT_MACHINE_ID_SCHEME
    from app.agents.risk import DEFAULT_RISK_THRESHOLDS
    return TenantCalibration(
        risk_thresholds=DEFAULT_RISK_THRESHOLDS,
        rag_min_term_coverage_lexical=get_settings().rag_min_term_coverage_lexical,
        machine_id_scheme=DEFAULT_MACHINE_ID_SCHEME,
    )


def get_search_retriever(
    tenant: TenantContext = Depends(get_tenant),
    calibration: TenantCalibration = Depends(get_tenant_calibration),
) -> DocumentRetriever:
    """The process-wide retriever for THIS request's tenant (lexical or
    dense per RAG_BACKEND). Each tenant gets its own index/namespace — see
    app.rag.retriever.get_retriever — so one tenant's documents can never
    be retrieved into another tenant's answer."""
    try:
        return get_retriever(
            tenant_id=tenant.tenant_id,
            min_coverage_override=calibration.rag_min_term_coverage_lexical,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a clean 503, not a 500
        # Audit F10: same reasoning as get_conn above — a Pinecone/Vertex
        # configuration error can carry credential or endpoint detail in
        # its message. Log it server-side, keep the caller-facing detail generic.
        logger.error("Retriever unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="Document retriever is unavailable.") from exc


def get_gemini_synthesizer() -> GeminiSynthesizer | None:
    """The process-wide Gemini synthesizer (Phase 6), or None.

    Deliberately does not raise: `get_synthesizer()` already returns None
    when Vertex isn't configured or `google-genai` isn't installed, and
    that's a valid, expected state for a portfolio deployment running
    without cloud credentials — /investigate still works, just with the
    deterministic template (see app.agents.nodes.build_recommendation).
    """
    return get_synthesizer()


def get_evidence_planner_llm():
    """The process-wide evidence-gathering planner LLM (Phase 10), or None.

    Same degrade-not-raise contract as get_gemini_synthesizer:
    get_planner_llm() already returns None when agentic routing isn't
    configured/enabled or langchain-google-vertexai isn't installed, and
    that's the expected default state — /investigate and /chat fall back
    to the Phase 5 deterministic evidence chain (see
    app.agents.graph.build_graph) rather than failing.
    """
    return get_planner_llm()
