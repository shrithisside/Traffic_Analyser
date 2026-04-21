"""
Dashboard page – overview KPIs and key insights.
"""

import streamlit as st
import pandas as pd
from modules.insights import generate_insights
from modules.charts import violations_by_area_chart, violations_by_hour_chart, violations_by_type_chart


def _metric(icon, value, label, accent="linear-gradient(90deg,#a78bfa,#60a5fa)"):
    return f"""
    <div class="metric-card" style="--card-accent:{accent}">
        <span class="metric-icon">{icon}</span>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>🚦 Dashboard</h1>
            <p>Live overview of traffic violation patterns and risk signals</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ins = generate_insights(df)

    if ins["total"] == 0:
        st.info("📭 No data yet. Head to **Add Violation** to enter some records.")
        return

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _metric("📋", ins["total"], "Total Violations",
                    "linear-gradient(90deg,#a78bfa,#7c3aed)"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _metric("📍", ins["high_risk_area"], "Most Violations Area",
                    "linear-gradient(90deg,#ef4444,#b91c1c)"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _metric("⚡", ins["common_violation"], "Top Violation",
                    "linear-gradient(90deg,#f59e0b,#d97706)"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _metric("🕐", ins["peak_hour"], "Peak Hour",
                    "linear-gradient(90deg,#34d399,#059669)"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── second row ────────────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            _metric("📅", ins["peak_day"], "Busiest Day",
                    "linear-gradient(90deg,#60a5fa,#2563eb)"),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _metric("🗓️", ins["peak_month"], "Busiest Month",
                    "linear-gradient(90deg,#f472b6,#db2777)"),
            unsafe_allow_html=True,
        )
    with c7:
        sev = ins["avg_severity"]
        sev_label = "Low" if sev < 1.7 else "Medium" if sev < 2.4 else "High"
        st.markdown(
            _metric("🚨", f"{sev} ({sev_label})", "Avg Severity Score",
                    "linear-gradient(90deg,#fb923c,#ea580c)"),
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            _metric("📊", ins["busiest_area_count"], "Records in Top Area",
                    "linear-gradient(90deg,#a78bfa,#38bdf8)"),
            unsafe_allow_html=True,
        )

    # ── charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Visual Overview</div>', unsafe_allow_html=True)
    ca, cb = st.columns([3, 2])
    with ca:
        st.plotly_chart(violations_by_area_chart(df), width="stretch")
    with cb:
        st.plotly_chart(violations_by_type_chart(df), width="stretch")

    st.plotly_chart(violations_by_hour_chart(df), width="stretch")

    # ── quick insight summary ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(
            f"""<div class="insight-card">
                <div class="ic-label">🔴 Highest Risk Zone</div>
                <div class="ic-value">{ins['high_risk_area']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            f"""<div class="insight-card">
                <div class="ic-label">⚡ Most Frequent Violation</div>
                <div class="ic-value">{ins['common_violation']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with i3:
        st.markdown(
            f"""<div class="insight-card">
                <div class="ic-label">🕐 Peak Violation Window</div>
                <div class="ic-value">{ins['peak_hour']}</div>
            </div>""",
            unsafe_allow_html=True,
        )