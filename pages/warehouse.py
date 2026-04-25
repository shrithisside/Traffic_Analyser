                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Data Warehouse page – schema explorer and raw data viewer.
Demonstrates Star Schema concept.
"""

import streamlit as st
import pandas as pd


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>🗄️ Data Warehouse</h1>
            <p>Star Schema explorer — Fact table & Dimension tables</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── schema diagram ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌟 Star Schema</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns([1, 0.2, 2, 0.2, 1])

    with s1:
        st.markdown(
            """
            <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;text-align:center;">
                <div style="color:#60a5fa;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">DIM_AREA</div>
                <div style="color:#94a3b8;font-size:0.75rem;margin-top:8px;line-height:1.9;">
                    area_id (PK)<br>area_name<br>latitude<br>longitude<br>zone_type
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;color:#a78bfa;font-size:1.5rem;margin-top:30px;'>→</div>", unsafe_allow_html=True)

    with s3:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#1a1040,#0f172a);border:2px solid #7c3aed;border-radius:16px;padding:24px;text-align:center;">
                <div style="color:#a78bfa;font-weight:800;font-size:1rem;text-transform:uppercase;letter-spacing:0.1em;">⭐ FACT_VIOLATIONS</div>
                <div style="color:#94a3b8;font-size:0.78rem;margin-top:12px;line-height:2.1;">
                    <span style="color:#e2e8f0">violation_id (PK)</span><br>
                    <span style="color:#60a5fa">area_id (FK → DIM_AREA)</span><br>
                    <span style="color:#f59e0b">time_id (FK → DIM_TIME)</span><br>
                    <span style="color:#34d399">violation_type_id (FK → DIM_VIOLATION)</span><br>
                    <span style="color:#f472b6">severity_score</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s4:
        st.markdown("<div style='display:flex;flex-direction:column;align-items:center;justify-content:space-around;height:100%;color:#a78bfa;font-size:1.2rem;margin-top:10px;gap:20px;'><span>→</span><span>→</span></div>", unsafe_allow_html=True)

    with s5:
        st.markdown(
            """
            <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;text-align:center;margin-bottom:10px;">
                <div style="color:#f59e0b;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">DIM_TIME</div>
                <div style="color:#94a3b8;font-size:0.75rem;margin-top:8px;line-height:1.9;">
                    time_id (PK)<br>date<br>hour<br>day_of_week<br>month<br>year
                </div>
            </div>
            <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;text-align:center;">
                <div style="color:#34d399;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">DIM_VIOLATION</div>
                <div style="color:#94a3b8;font-size:0.75rem;margin-top:8px;line-height:1.9;">
                    violation_id (PK)<br>violation_name<br>category<br>base_severity
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── dimension tables ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 Dimension Tables</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Fact Table", "📍 DIM_AREA", "⏱️ DIM_TIME", "🚫 DIM_VIOLATION"])

    with tab1:
        st.markdown("**FACT_VIOLATIONS** — the central table in our star schema")
        if not df.empty:
            st.dataframe(df.head(100), width="stretch", hide_index=True)
            st.caption(f"Total records: {len(df)}")
        else:
            st.info("No data yet.")

    with tab2:
        st.markdown("**DIM_AREA** — area dimension table")
        from modules.data_store import AREA_COORDS
        dim_area = pd.DataFrame(
            [
                {"area_id": i + 1, "area_name": a, "latitude": c[0], "longitude": c[1]}
                for i, (a, c) in enumerate(AREA_COORDS.items())
            ]
        )
        st.dataframe(dim_area, width="stretch", hide_index=True)

    with tab3:
        st.markdown("**DIM_TIME** — time dimension table (sample)")
        if not df.empty:
            dim_time = (
                df[["date", "hour", "day_of_week", "month"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            dim_time.insert(0, "time_id", range(1, len(dim_time) + 1))
            st.dataframe(dim_time.head(50), width="stretch", hide_index=True)
            st.caption(f"Unique time records: {len(dim_time)}")
        else:
            st.info("No data yet.")

    with tab4:
        st.markdown("**DIM_VIOLATION** — violation type dimension table")
        from modules.data_store import VIOLATION_SEVERITY
        dim_v = pd.DataFrame(
            [
                {
                    "violation_id": i + 1,
                    "violation_name": v,
                    "category": ("Critical" if s == 3 else "Moderate" if s == 2 else "Minor"),
                    "base_severity": s,
                }
                for i, (v, s) in enumerate(VIOLATION_SEVERITY.items())
            ]
        )
        st.dataframe(dim_v, width="stretch", hide_index=True)

    # ── aggregation query results ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🔄 OLAP Aggregations (Simulated Queries)</div>', unsafe_allow_html=True)
    if not df.empty:
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**GROUP BY area → violation count**")
            agg1 = df.groupby("area")["id"].count().reset_index()
            agg1.columns = ["Area", "Violation Count"]
            agg1 = agg1.sort_values("Violation Count", ascending=False)
            st.dataframe(agg1, width="stretch", hide_index=True)

        with q2:
            st.markdown("**GROUP BY violation_type → avg severity**")
            agg2 = df.groupby("violation_type")["severity"].mean().round(2).reset_index()
            agg2.columns = ["Violation Type", "Avg Severity"]
            agg2 = agg2.sort_values("Avg Severity", ascending=False)
            st.dataframe(agg2, width="stretch", hide_index=True)

        st.markdown("**GROUP BY area × violation_type → count (CUBE)**")
        cube = df.pivot_table(
            index="area", columns="violation_type", values="id", aggfunc="count", fill_value=0
        )
        st.dataframe(cube, width="stretch")
    else:
        st.info("Add data to see aggregation results.")