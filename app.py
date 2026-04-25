                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Traffic Violation Analysis & Risk Prediction System
Data Mining and Data Warehousing - College Project
"""
#Jai Shree Ram

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ── page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="TrafficMine | DMW Project",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local imports ────────────────────────────────────────────────────────────
from modules.data_store import DataStore
from modules.preprocessing import preprocess
from modules.clustering import run_kmeans
from modules.insights import generate_insights
from modules.charts import (
    violations_by_area_chart,
    violations_by_type_chart,
    violations_by_hour_chart,
    cluster_scatter_chart,
    risk_gauge,
)
from modules.styles import inject_css

# ── inject custom CSS ────────────────────────────────────────────────────────
inject_css()

# ── initialise data store ────────────────────────────────────────────────────
store = DataStore()

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  ─  navigation
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <span class="logo-icon">🚦</span>
            <div>
                <div class="logo-title">TrafficMine</div>
                <div class="logo-sub">DMW Project</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📝 Add Violation",
            "📊 Analytics",
            "🔬 Mining Engine",
            "🔗 Association Rules",
            "🚨 Anomaly Detection",
            "📦 OLAP Operations",
            "🗄️ Data Warehouse",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── quick stats in sidebar ──────────────────────────────────────────────
    df = store.load()
    total = len(df)
    st.markdown(
        f"""
        <div class="sidebar-stats">
            <div class="stat-item">
                <span class="stat-num">{total}</span>
                <span class="stat-label">Total Records</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">{df['area'].nunique() if total else 0}</span>
                <span class="stat-label">Unique Areas</span>
            </div>
            <div class="stat-item">
                <span class="stat-num">{df['violation_type'].nunique() if total else 0}</span>
                <span class="stat-label">Violation Types</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#6b7280;text-align:center;'>©DMW College Project by Shriwayanta Maiti & Harshal Nanekar</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ════════════════════════════════════════════════════════════════════════════

# ── reload fresh df for each page ───────────────────────────────────────────
df = store.load()

# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    from pages.dashboard import render
    render(df)

elif page == "📝 Add Violation":
    from pages.add_violation import render
    render(store)

elif page == "📊 Analytics":
    from pages.analytics import render
    render(df)

elif page == "🔬 Mining Engine":
    from pages.mining import render
    render(df)

elif page == "🔗 Association Rules":
    from pages.association import render
    render(df)

elif page == "🚨 Anomaly Detection":
    from pages.anomaly import render
    render(df)

elif page == "📦 OLAP Operations":
    from pages.olap import render
    render(df)

elif page == "🗄️ Data Warehouse":
    from pages.warehouse import render
    render(df)