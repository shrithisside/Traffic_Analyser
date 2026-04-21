"""
Mining Engine page – K-Means clustering and risk zone analysis.
"""

import streamlit as st
import pandas as pd
from modules.clustering import run_kmeans, RISK_COLORS
from modules.preprocessing import preprocess
from modules.charts import cluster_scatter_chart, risk_gauge


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>🔬 Mining Engine</h1>
            <p>K-Means Clustering — classify areas into High / Medium / Low risk zones</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty or len(df) < 10:
        st.warning("⚠️ Need at least 10 records to run clustering. Add more data first.")
        return

    # ── controls ──────────────────────────────────────────────────────────────
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        st.markdown('<div class="section-title">⚙️ Parameters</div>', unsafe_allow_html=True)
        k = st.slider("Number of Clusters (k)", min_value=2, max_value=5, value=3, step=1)
        run_btn = st.button("🚀 Run K-Means Clustering", width="stretch")

    with col_info:
        st.markdown('<div class="section-title">ℹ️ How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-card">
                <ol style="color:#94a3b8;font-size:0.85rem;line-height:2;margin:0;padding-left:18px;">
                    <li>Data is <b style="color:#a78bfa">preprocessed</b> — nulls dropped, categoricals encoded</li>
                    <li>Per-area features computed: <b style="color:#60a5fa">total violations, avg severity, peak hour, unique types</b></li>
                    <li>Features <b style="color:#34d399">normalized</b> via StandardScaler</li>
                    <li><b style="color:#f59e0b">K-Means</b> groups areas into k clusters</li>
                    <li>Clusters ranked by violation count → <b style="color:#ef4444">Risk label assigned</b></li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── run on load (default) OR button ──────────────────────────────────────
    if "cluster_results" not in st.session_state or run_btn:
        with st.spinner("🔄 Running K-Means..."):
            processed = preprocess(df)
            area_df, labeled_df = run_kmeans(df, k=k)
            st.session_state["cluster_results"] = (area_df, labeled_df)
            st.session_state["cluster_k"] = k

    area_df, labeled_df = st.session_state["cluster_results"]

    if area_df.empty:
        st.error("Clustering failed — not enough distinct areas.")
        return

    # ── scatter plot ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Cluster Visualization</div>', unsafe_allow_html=True)
    st.plotly_chart(cluster_scatter_chart(area_df), width="stretch")

    # ── risk table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗂️ Area Risk Classification</div>', unsafe_allow_html=True)

    area_display = area_df[["area", "total_violations", "avg_severity", "unique_types", "risk_level"]].copy()
    area_display["avg_severity"] = area_display["avg_severity"].round(2)
    area_display = area_display.sort_values("total_violations", ascending=False).reset_index(drop=True)
    area_display.columns = ["Area", "Violations", "Avg Severity", "Violation Types", "Risk Level"]

    st.dataframe(area_display, width="stretch", hide_index=True)

    # ── risk gauges ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Per-Area Risk Gauges</div>', unsafe_allow_html=True)

    max_v = area_df["total_violations"].max()
    min_v = area_df["total_violations"].min()

    cols = st.columns(min(5, len(area_df)))
    for i, row in area_df.iterrows():
        score = int(
            (row["total_violations"] - min_v) / max(max_v - min_v, 1) * 100
        )
        with cols[i % len(cols)]:
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:#94a3b8;'>{row['area']}</div>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(score), width="stretch", key=f"gauge_{i}")

    # ── preprocessing showcase ────────────────────────────────────────────────
    with st.expander("🔍 View Preprocessed Dataset"):
        processed = preprocess(df)
        st.dataframe(
            processed[["area", "violation_type", "hour", "area_enc", "violation_enc", "hour_sin", "hour_cos", "severity"]].head(50),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Total preprocessed records: {len(processed)}")