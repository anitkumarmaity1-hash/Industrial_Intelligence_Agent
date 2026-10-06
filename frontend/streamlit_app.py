"""
Streamlit dashboard — Phase 7.

Thin HTTP client over the Phase 4-6 FastAPI backend. No business logic
lives here: risk levels, root-cause candidates, citations and the
narrative report are all computed by the API (deterministic code +
Gemini synthesis). This file only calls endpoints and renders what
comes back — the same "evidence before explanation" discipline the
backend enforces applies to the frontend too: nothing here invents a
number or a verdict that the API didn't return.

Screens (per the master prompt's Streamlit spec):
  1. Machine selector           -> sidebar
  2. Machine health summary     -> Overview tab (health_status AND risk_level,
                                    both from GET /machines/{id}/health — audit F8)
  3. Key metrics                -> Overview tab (from GET /machines/{id}/health)
  4. Anomaly indicators         -> Overview tab (from GET /machines/{id}/anomalies)
  5. Investigation input        -> Investigate tab
  6. Agent investigation results -> Investigate tab
  7. Evidence/citations         -> Investigate tab
  8. Recommendation             -> Investigate tab

Run with: streamlit run frontend/streamlit_app.py
Requires the API running separately: uvicorn app.main:app --reload
Configure the API location with the API_BASE_URL env var if it isn't
on http://localhost:8000.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# Per-tenant API key (sent as X-API-Key). Only needed when the API runs with
# AUTH_REQUIRED=true / API_KEY set, or to act as a tenant other than the demo one.
DEFAULT_API_KEY = os.environ.get("API_KEY", "")

st.set_page_config(
    page_title="Industrial Intelligence Agent",
    page_icon="🏭",
    layout="wide",
)

HEALTH_BADGE = {"HEALTHY": "🟢 HEALTHY",
                "WATCH": "🟡 WATCH", "AT_RISK": "🔴 AT_RISK"}
RISK_BADGE = {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM",
              "HIGH": "🔴 HIGH", "UNKNOWN": "⚪ UNKNOWN"}
SEVERITY_BADGE = {"MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH"}


# --------------------------------------------------------------------------
# API client — every call goes through here so error handling is uniform.
# A backend that isn't running, or a network hiccup, should show one clear
# banner rather than a stack trace or a half-rendered page.
# --------------------------------------------------------------------------

class ApiError(Exception):
    """Raised for any non-2xx response or connection failure, with a
    message already suitable to show the user directly."""


def _request(method: str, path: str, timeout: int = 30, **kwargs) -> dict | list:
    base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    url = f"{base_url.rstrip('/')}{path}"
    api_key = st.session_state.get("api_key", DEFAULT_API_KEY)
    if api_key:
        kwargs.setdefault("headers", {})["X-API-Key"] = api_key
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Can't reach the API at {base_url}. Is `uvicorn app.main:app` running? ({exc})"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"Request to {url} timed out.") from exc

    if resp.status_code == 404:
        raise ApiError(resp.json().get("detail", "Not found."))
    if resp.status_code == 401:
        raise ApiError("The API rejected the credentials — set a valid API key under "
                       "'Backend connection' in the sidebar.")
    if resp.status_code == 429:
        raise ApiError("Rate limit reached — wait a moment and try again "
                       f"(Retry-After: {resp.headers.get('Retry-After', '?')}s).")
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"{resp.status_code} from {path}: {detail}")

    return resp.json()


def api_get(path: str, params: dict | None = None) -> dict | list:
    return _request("GET", path, params=params)


def api_post(path: str, payload: dict, timeout: int = 30) -> dict:
    # Audit F11: /investigate and /chat run the LangGraph agent — sensor +
    # anomaly evidence gathering, optionally a planner round for
    # supplementary evidence, then Gemini synthesis — which can comfortably
    # exceed the 30s the rest of the API's simple DB-read endpoints need.
    # Callers of these two pass a longer timeout explicitly rather than
    # this function's fast-endpoint default.
    return _request("POST", path, json=payload, timeout=timeout)


INVESTIGATE_TIMEOUT_SECONDS = 120


@st.cache_data(ttl=60, show_spinner=False)
def fetch_machines(api_base_url: str) -> list[dict]:
    # api_base_url is only in the signature so the cache key changes if
    # the user points the dashboard at a different backend.
    return api_get("/machines")  # type: ignore[return-value]


def fetch_health(machine_id: str) -> dict | None:
    try:
        # type: ignore[return-value]
        return api_get(f"/machines/{machine_id}/health")
    except ApiError as exc:
        st.info(f"No health data yet for {machine_id}: {exc}")
        return None


def fetch_anomalies(machine_id: str, limit: int = 25) -> list[dict]:
    # type: ignore[return-value]
    return api_get(f"/machines/{machine_id}/anomalies", params={"limit": limit})


def fetch_sensors(machine_id: str, limit: int = 168) -> list[dict]:
    # Audit F11: backs the trend chart — GET /machines/{id}/sensors.
    try:
        # type: ignore[return-value]
        return api_get(f"/machines/{machine_id}/sensors", params={"limit": limit})
    except ApiError:
        return []


# --------------------------------------------------------------------------
# Sidebar — API location + machine selector
# --------------------------------------------------------------------------

st.sidebar.title("🏭 Industrial Intelligence Agent")

with st.sidebar.expander("Backend connection", expanded=False):
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
    st.session_state["api_base_url"] = api_base_url
    st.session_state["api_key"] = st.text_input(
        "API key (X-API-Key)", value=DEFAULT_API_KEY, type="password")
    if st.button("Refresh machine list"):
        fetch_machines.clear()

try:
    machines = fetch_machines(st.session_state.get(
        "api_base_url", DEFAULT_API_BASE_URL))
except ApiError as exc:
    st.sidebar.error(str(exc))
    st.error(str(exc))
    st.stop()

if not machines:
    st.sidebar.warning("No machines returned by the API.")
    st.stop()

machine_options = {m["machine_id"]: m for m in machines}
st.sidebar.subheader("Machine")
selected_id = st.sidebar.selectbox(
    "Select a machine",
    options=list(machine_options.keys()),
    format_func=lambda mid: f"{mid} — {machine_options[mid]['production_line']}",
)
selected_machine = machine_options[selected_id]

st.sidebar.markdown(
    f"""
**Line:** {selected_machine.get('production_line', '—')}
**Type:** {selected_machine.get('type', '—')}
**Name:** {selected_machine.get('name') or '—'}
**Location:** {selected_machine.get('location') or '—'}
"""
)

st.sidebar.caption(f"{len(machines)} machines in the fleet")

st.title("Operations Dashboard")

tab_overview, tab_investigate = st.tabs(
    ["📊 Machine Overview", "🔎 Investigate"])


# --------------------------------------------------------------------------
# Tab 1 — Machine health summary, key metrics, anomaly indicators
# --------------------------------------------------------------------------

with tab_overview:
    st.subheader(f"{selected_id} — Health Summary")

    health = fetch_health(selected_id)

    if health:
        badge = HEALTH_BADGE.get(
            health["health_status"], health["health_status"])
        # Audit F8 (finding 3): health_status is still the raw latest
        # hourly window (kept as-is — see HEALTH_BADGE above), but it used
        # to be the ONLY status this tab showed, while the Investigate tab
        # showed risk_level from the same sustained-rate rule — two
        # different verdicts for "is this machine okay" that could (and
        # did) disagree. GET /machines/{id}/health now also returns
        # risk_level computed by that exact rule, so both tabs show the
        # same figure, side by side rather than picking one.
        risk_level = health.get("risk_level")
        status_cols = st.columns(2) if risk_level else [st]
        with status_cols[0]:
            st.markdown(f"### Latest hourly snapshot: {badge}")
        if risk_level:
            with status_cols[1]:
                st.markdown(
                    f"### Risk (sustained rate): "
                    f"{RISK_BADGE.get(risk_level, risk_level)}"
                )
        st.caption(
            "Latest hourly snapshot is a single window and can be noisy. "
            "Risk is the same sustained-rate assessment the Investigate tab uses."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max anomaly score", health["max_anomaly_score"])
        c2.metric("Anomalous readings",
                  f"{health['anomalous_reading_count']} / {health['reading_count']}")
        c3.metric("Avg defect rate",
                  f"{health.get('avg_defect_rate', 0):.1%}" if health.get("avg_defect_rate") is not None else "—")
        c4.metric("Tool wear (min)",
                  f"{health.get('tool_wear_min'):.0f}" if health.get("tool_wear_min") is not None else "—")
        st.caption(
            f"Latest hourly window: {health['window_start'][:16].replace('T', ' ')} "
            f"to {health['window_end'][:16].replace('T', ' ')}"
        )

    st.divider()
    st.subheader("Sensor trend")

    trend_hours = st.slider(
        "Trailing window (hours)", min_value=24, max_value=336, value=168, step=24)
    sensor_points = fetch_sensors(selected_id, limit=trend_hours)

    if not sensor_points:
        st.info("No sensor trend data yet for this machine.")
    else:
        trend_df = pd.DataFrame(sensor_points).sort_values("window_start")
        trend_df["window_start"] = pd.to_datetime(trend_df["window_start"])
        trend_df = trend_df.set_index("window_start")

        trend_metric = st.selectbox(
            "Metric",
            options=[
                ("avg_defect_rate", "Defect rate"),
                ("avg_torque_nm", "Torque (Nm)"),
                ("avg_process_temp_k", "Process temperature (K)"),
                ("avg_rotational_speed_rpm", "Rotational speed (rpm)"),
                ("tool_wear_min", "Tool wear (min)"),
                ("avg_energy_consumption_kwh", "Energy consumption (kWh)"),
                ("max_anomaly_score", "Max anomaly score"),
            ],
            format_func=lambda pair: pair[1],
        )
        metric_col, metric_label = trend_metric
        if metric_col in trend_df.columns:
            st.line_chart(trend_df[metric_col].rename(metric_label))
        else:
            st.caption("This metric isn't present in the returned windows.")

    st.divider()
    st.subheader("Anomaly indicators")

    anomaly_limit = st.slider(
        "Show most recent N anomalies", min_value=5, max_value=100, value=25, step=5)
    try:
        anomalies = fetch_anomalies(selected_id, limit=anomaly_limit)
    except ApiError as exc:
        st.error(str(exc))
        anomalies = []

    if not anomalies:
        st.info("No flagged anomalies for this machine in the selected window.")
    else:
        df = pd.DataFrame(anomalies)
        df["severity"] = df["severity"].map(lambda s: SEVERITY_BADGE.get(s, s))
        df = df.rename(columns={
            "detected_at": "Detected at",
            "anomaly_score": "Anomaly score",
            "severity": "Severity",
            "triggered_reasons": "Triggered reasons",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Tab 2 — Investigation input, agent results, evidence/citations, recommendation
# --------------------------------------------------------------------------

with tab_investigate:
    st.subheader("Ask the agent")

    question = st.text_area(
        "Operations question",
        placeholder='e.g. "Why is this machine underperforming?" or '
                    '"Which machines currently show abnormal behavior?"',
        height=90,
    )

    scope_to_machine = st.checkbox(
        f"Scope to selected machine ({selected_id})",
        value=True,
        help="Uncheck for a fleet-wide question — the agent will scan all machines instead of one.",
    )

    if st.button("Investigate", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            payload = {
                "question": question.strip(),
                "machine_id": selected_id if scope_to_machine else None,
            }
            with st.spinner("Running the agent — gathering evidence, assessing risk, synthesizing report..."):
                try:
                    result = api_post("/investigate", payload, timeout=INVESTIGATE_TIMEOUT_SECONDS)
                except ApiError as exc:
                    st.error(str(exc))
                    result = None

            if result:
                st.session_state.setdefault("history", []).insert(0, {
                    "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "result": result,
                })

    def _strip_narrative_sections(narrative: str, headers_to_drop: set[str]) -> str:
        """Audit F11: `narrative` already contains '## Possible causes' and
        '## Supporting evidence' sections (see app/agents/synthesis.py) —
        this tab then rendered those same two lists again, structured,
        right below it. Drop the duplicated sections from the markdown
        here rather than removing the structured widgets: the widgets are
        the better UI for scanning a list, the narrative is the better
        read for the rest of the report, so each fact should only show up
        in the one of the two it reads best in.
        """
        sections = narrative.split("\n\n## ")
        kept = [sections[0]] if sections else []
        for section in sections[1:]:
            header = section.split("\n", 1)[0].strip()
            if header not in headers_to_drop:
                kept.append(section)
        return "\n\n## ".join(kept)

    def render_result(result: dict) -> None:
        header_bits = [f"**Intent:** `{result['intent']}`"]
        if result.get("machine_id"):
            header_bits.append(f"**Machine:** `{result['machine_id']}`")
        if result.get("risk_level"):
            header_bits.append(
                f"**Risk:** {RISK_BADGE.get(result['risk_level'], result['risk_level'])}")
        st.markdown(" &nbsp;|&nbsp; ".join(header_bits))

        if result.get("limitations_note"):
            st.warning(f"⚠️ {result['limitations_note']}")

        st.markdown(_strip_narrative_sections(
            result["narrative"], {"Possible causes", "Supporting evidence"}))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "**Possible causes** _(hypotheses, not confirmed diagnoses)_")
            candidates = result.get("root_cause_candidates") or []
            if candidates:
                for c in candidates:
                    st.markdown(f"- {c}")
            else:
                st.caption("None identified from current evidence.")
        with col_b:
            st.markdown("**Supporting citations**")
            citations = result.get("supporting_citations") or []
            if citations:
                for c in citations:
                    st.markdown(f"- {c}")
            else:
                st.caption("No document citations for this investigation.")

        if result.get("fleet_ranking"):
            st.markdown("**Fleet ranking**")
            fdf = pd.DataFrame(result["fleet_ranking"])
            if "health_status" in fdf.columns:
                fdf["health_status"] = fdf["health_status"].map(
                    lambda s: HEALTH_BADGE.get(s, s))
            st.dataframe(fdf, use_container_width=True, hide_index=True)

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.markdown(f"**Latest result** — asked {history[0]['asked_at']}")
        render_result(history[0]["result"])

        if len(history) > 1:
            with st.expander(f"Previous investigations ({len(history) - 1})"):
                for entry in history[1:]:
                    st.markdown(
                        f"---\n**{entry['asked_at']}** — _{entry['result']['question']}_")
                    render_result(entry["result"])
    else:
        st.caption("No investigations run yet this session.")
