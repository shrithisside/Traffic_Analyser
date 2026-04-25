                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Analytics page – deep-dive charts and drill-downs.
"""

import streamlit as st
import pandas as pd
from modules.charts import (
    violations_by_area_chart,
    violations_by_type_chart,
    violations_by_hour_chart,
    violations_by_day_chart,
    month_trend_chart,
)


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>📊 Analytics</h1>
            <p>Drill down into violation patterns across time, area, and type</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No data yet – please add some violation records first.")
        return

    # ── filters ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔎 Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        areas = ["All"] + sorted(df["area"].unique().tolist())
        sel_area = st.selectbox("Filter by Area", areas, key="an_area")
    with fc2:
        vtypes = ["All"] + sorted(df["violation_type"].unique().tolist())
        sel_vtype = st.selectbox("Filter by Violation Type", vtypes, key="an_vtype")
    with fc3:
        months = ["All"] + sorted(df["month"].unique().tolist())
        sel_month = st.selectbox("Filter by Month", months, key="an_month")

    fdf = df.copy()
    if sel_area != "All":
        fdf = fdf[fdf["area"] == sel_area]
    if sel_vtype != "All":
        fdf = fdf[fdf["violation_type"] == sel_vtype]
    if sel_month != "All":
        fdf = fdf[fdf["month"] == sel_month]

    st.markdown(
        f"<span style='color:#64748b;font-size:0.8rem;'>Showing <b style='color:#a78bfa'>{len(fdf)}</b> records</span>",
        unsafe_allow_html=True,
    )

    if fdf.empty:
        st.warning("No records match the selected filters.")
        return

    # ── charts row 1 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📍 Area & Type Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(violations_by_area_chart(fdf), width="stretch")
    with c2:
        st.plotly_chart(violations_by_type_chart(fdf), width="stretch")

    # ── charts row 2 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⏱️ Temporal Patterns</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(violations_by_hour_chart(fdf), width="stretch")
    with c4:
        st.plotly_chart(violations_by_day_chart(fdf), width="stretch")

    # ── monthly trend ─────────────────────────────────────────────────────────
    st.plotly_chart(month_trend_chart(fdf), width="stretch")

    # ── area × violation heatmap ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🗺️ Area × Violation Heatmap</div>', unsafe_allow_html=True)
    pivot = fdf.pivot_table(
        index="area", columns="violation_type", values="id", aggfunc="count", fill_value=0
    )

    import plotly.express as px
    fig = px.imshow(
        pivot,
        color_continuous_scale="Plasma",
        title="Violation Frequency Heatmap",
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    # ── top 5 table ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Top 5 Areas by Violations</div>', unsafe_allow_html=True)
    top5 = (
        fdf.groupby("area")
        .agg(count=("id", "count"), avg_severity=("severity", "mean"))
        .sort_values("count", ascending=False)
        .head(5)
        .reset_index()
    )
    top5.columns = ["Area", "Violations", "Avg Severity"]
    top5["Avg Severity"] = top5["Avg Severity"].round(2)
    st.dataframe(top5, width="stretch", hide_index=True)