                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
OLAP Operations page — Roll-up, Drill-down, Slice, Dice, Pivot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=45, b=10),
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
)

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived time columns needed for OLAP operations."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype("Int64").astype(str).replace("<NA>", "Unknown")
    out["quarter"] = out["date"].dt.quarter.apply(lambda q: f"Q{q}")
    out["time_period"] = out["hour"].apply(
        lambda h: "🌙 Night (0–6)"       if h < 6
        else      "🌅 Morning (6–12)"    if h < 12
        else      "☀️ Afternoon (12–18)" if h < 18
        else      "🌆 Evening (18–24)"
    )
    return out


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>📦 OLAP Operations</h1>
            <p>Online Analytical Processing — Roll-up, Drill-down, Slice, Dice and Pivot on the data warehouse</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No data available.")
        return

    cube = _enrich(df)

    with st.expander("📚 OLAP Operations Explained", expanded=False):
        st.markdown("""
        | Operation | Description | Example |
        |---|---|---|
        | **Roll-up** | Aggregate to a higher granularity level | Hour → Time Period → Day → Month → Year |
        | **Drill-down** | Expand to a lower granularity level | Year → Quarter → Month → Hourly |
        | **Slice** | Fix ONE dimension to a single value | Only `violation_type = 'Drunk Driving'` |
        | **Dice** | Filter on MULTIPLE dimensions simultaneously | Gujarat + January + Drunk Driving |
        | **Pivot** | Rotate dimensions — rows vs columns | Rows = Area, Columns = Violation Type |
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⬆️ Roll-up",
        "⬇️ Drill-down",
        "🔪 Slice",
        "🎲 Dice",
        "🔄 Pivot",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Roll-up
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">⬆️ Roll-up: Aggregate up the Time Hierarchy</div>', unsafe_allow_html=True)
        st.markdown("Move from fine-grained **Hour** up to coarser levels like **Month** or **Year**.")

        level_map = {
            "Hour":         "hour",
            "Time Period":  "time_period",
            "Day of Week":  "day_of_week",
            "Month":        "month",
            "Quarter":      "quarter",
            "Year":         "year",
        }
        rollup_level = st.radio(
            "Select Granularity",
            list(level_map.keys()),
            horizontal=True,
            key="rollup_lvl",
        )
        col = level_map[rollup_level]

        rollup = cube.groupby(col).agg(
            total_violations=("id",            "count"),
            avg_severity    =("severity",       "mean"),
            unique_areas    =("area",           "nunique"),
            unique_types    =("violation_type", "nunique"),
        ).reset_index()
        rollup["avg_severity"] = rollup["avg_severity"].round(2)

        # Apply canonical ordering where applicable
        if rollup_level == "Month":
            rollup[col] = pd.Categorical(rollup[col], categories=MONTH_ORDER, ordered=True)
            rollup = rollup.sort_values(col)
        elif rollup_level == "Day of Week":
            rollup[col] = pd.Categorical(rollup[col], categories=DAY_ORDER, ordered=True)
            rollup = rollup.sort_values(col)
        else:
            rollup = rollup.sort_values(col)

        fig_ru = px.bar(
            rollup, x=col, y="total_violations",
            color="avg_severity",
            color_continuous_scale="RdYlGn_r",
            title=f"Violations Rolled-up by {rollup_level}",
            labels={"total_violations": "Total Violations", "avg_severity": "Avg Severity"},
        )
        fig_ru.update_layout(**_LAYOUT)
        st.plotly_chart(fig_ru, use_container_width=True)

        display_rollup = rollup.copy()
        display_rollup.columns = [rollup_level, "Total Violations", "Avg Severity",
                                   "Unique Areas", "Unique Violation Types"]
        st.dataframe(display_rollup, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Drill-down
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">⬇️ Drill-down: From Year → Quarter → Month → Hour</div>', unsafe_allow_html=True)
        st.markdown("Select progressively finer levels to zoom into a specific period.")

        r1, r2, r3 = st.columns(3)
        with r1:
            years = sorted(cube["year"].dropna().unique().tolist(), key=str)
            sel_yr  = st.selectbox("Year", years, key="dd_yr")
        year_data = cube[cube["year"] == sel_yr]

        with r2:
            quarters  = sorted(year_data["quarter"].unique().tolist())
            sel_q     = st.selectbox("Quarter", ["All"] + quarters, key="dd_q")
        if sel_q != "All":
            year_data = year_data[year_data["quarter"] == sel_q]

        with r3:
            months   = sorted(year_data["month"].unique().tolist())
            sel_mo   = st.selectbox("Month", ["All"] + months, key="dd_mo")
        if sel_mo != "All":
            year_data = year_data[year_data["month"] == sel_mo]

        period_label = " / ".join(filter(lambda x: x != "All", [sel_yr, sel_q, sel_mo]))
        st.markdown(
            f"<span style='color:#64748b;font-size:0.82rem;'>Drilling into <b style='color:#a78bfa'>{period_label}</b> — "
            f"<b style='color:#34d399'>{len(year_data)}</b> records</span>",
            unsafe_allow_html=True,
        )

        if not year_data.empty:
            dc1, dc2 = st.columns(2)

            with dc1:
                # Hourly pattern for selected period
                hourly = year_data.groupby("hour")["id"].count().reset_index()
                hourly.columns = ["Hour", "Violations"]
                all_h = pd.DataFrame({"Hour": range(24)})
                hourly = all_h.merge(hourly, on="Hour", how="left").fillna(0)

                fig_h = px.line(
                    hourly, x="Hour", y="Violations", markers=True,
                    title=f"Hourly Violations — {period_label}",
                    color_discrete_sequence=["#a78bfa"],
                )
                fig_h.update_layout(**_LAYOUT)
                st.plotly_chart(fig_h, use_container_width=True)

            with dc2:
                # Area breakdown
                area_dd = year_data.groupby("area")["id"].count().reset_index()
                area_dd.columns = ["Area", "Violations"]
                area_dd = area_dd.sort_values("Violations", ascending=False).head(15)

                fig_a = px.bar(
                    area_dd, x="Area", y="Violations",
                    color="Violations", color_continuous_scale="Plasma",
                    title=f"Top Areas — {period_label}",
                )
                fig_a.update_layout(**_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_a, use_container_width=True)

            # Violation type breakdown
            vt_dd = year_data.groupby("violation_type")["id"].count().reset_index()
            vt_dd.columns = ["Type", "Violations"]
            fig_vt = px.pie(
                vt_dd, names="Type", values="Violations", hole=0.4,
                title=f"Violation Type Mix — {period_label}",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_vt.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_vt, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Slice
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">🔪 Slice: Fix One Dimension, Analyse the Rest</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            slice_dim = st.selectbox(
                "Dimension to Slice on",
                ["area", "violation_type", "month", "day_of_week", "time_period"],
                key="sl_dim",
            )
        with s2:
            slice_vals = sorted(cube[slice_dim].unique().tolist())
            slice_val  = st.selectbox("Value", slice_vals, key="sl_val")

        sliced = cube[cube[slice_dim] == slice_val]
        st.markdown(
            f"<div style='color:#64748b;font-size:0.82rem;margin:8px 0 16px;'>"
            f"Slice: <b style='color:#a78bfa'>{slice_dim} = {slice_val}</b>"
            f" → <b style='color:#34d399'>{len(sliced)}</b> records</div>",
            unsafe_allow_html=True,
        )

        if sliced.empty:
            st.warning("No records for this slice.")
        else:
            sc1, sc2 = st.columns(2)
            with sc1:
                # Show the "other" key dimension
                if slice_dim != "area":
                    grp = sliced.groupby("area")["id"].count().reset_index()
                    grp.columns = ["Area", "Count"]
                    grp = grp.sort_values("Count", ascending=False)
                    fig_s = px.bar(grp, x="Area", y="Count", color="Count",
                                   color_continuous_scale="Plasma",
                                   title=f"Areas within  [{slice_val}]")
                else:
                    grp = sliced.groupby("violation_type")["id"].count().reset_index()
                    grp.columns = ["Type", "Count"]
                    fig_s = px.pie(grp, names="Type", values="Count", hole=0.4,
                                   title=f"Violation Types in  [{slice_val}]",
                                   color_discrete_sequence=px.colors.qualitative.Vivid)
                fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e2e8f0"),
                                    coloraxis_showscale=False)
                st.plotly_chart(fig_s, use_container_width=True)

            with sc2:
                h_sl = sliced.groupby("hour")["id"].count().reset_index()
                h_sl.columns = ["Hour", "Count"]
                all_h3 = pd.DataFrame({"Hour": range(24)})
                h_sl   = all_h3.merge(h_sl, on="Hour", how="left").fillna(0)
                fig_hs = px.bar(h_sl, x="Hour", y="Count",
                                color="Count", color_continuous_scale="Turbo",
                                title=f"Hourly Pattern  [{slice_val}]")
                fig_hs.update_layout(**_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_hs, use_container_width=True)

            # Monthly trend for this slice
            mo_sl = sliced.groupby("month")["id"].count().reset_index()
            mo_sl.columns = ["Month", "Count"]
            mo_sl["Month"] = pd.Categorical(mo_sl["Month"], categories=MONTH_ORDER, ordered=True)
            mo_sl = mo_sl.sort_values("Month")
            fig_mo = px.line(mo_sl, x="Month", y="Count", markers=True,
                             title=f"Monthly Trend  [{slice_val}]",
                             color_discrete_sequence=["#60a5fa"])
            fig_mo.update_layout(**_LAYOUT)
            st.plotly_chart(fig_mo, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Dice
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">🎲 Dice: Intersect Multiple Dimensions</div>', unsafe_allow_html=True)
        st.markdown("Apply **several filters at once** to carve out a sub-cube.")

        da, db, dc, dd_ = st.columns(4)
        with da:
            d_areas   = st.multiselect("Areas",
                sorted(cube["area"].unique()), default=sorted(cube["area"].unique())[:3], key="d_a")
        with db:
            d_months  = st.multiselect("Months",
                MONTH_ORDER, default=MONTH_ORDER[:3], key="d_m")
        with dc:
            d_vtypes  = st.multiselect("Violation Types",
                sorted(cube["violation_type"].unique()),
                default=sorted(cube["violation_type"].unique())[:3], key="d_v")
        with dd_:
            d_periods = st.multiselect("Time Period",
                sorted(cube["time_period"].unique()),
                default=list(cube["time_period"].unique()), key="d_p")

        diced = cube.copy()
        if d_areas:   diced = diced[diced["area"].isin(d_areas)]
        if d_months:  diced = diced[diced["month"].isin(d_months)]
        if d_vtypes:  diced = diced[diced["violation_type"].isin(d_vtypes)]
        if d_periods: diced = diced[diced["time_period"].isin(d_periods)]

        st.markdown(
            f"<div style='color:#64748b;font-size:0.82rem;margin:8px 0 16px;'>"
            f"Sub-cube contains <b style='color:#a78bfa'>{len(diced)}</b> records</div>",
            unsafe_allow_html=True,
        )

        if diced.empty:
            st.warning("No records match the selected filters.")
        else:
            dk1, dk2, dk3, dk4 = st.columns(4)
            with dk1:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#a78bfa,#7c3aed)">
                    <span class="metric-icon">📋</span>
                    <div class="metric-value">{len(diced)}</div>
                    <div class="metric-label">Records in Sub-cube</div>
                </div>""", unsafe_allow_html=True)
            with dk2:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#ef4444,#b91c1c)">
                    <span class="metric-icon">📍</span>
                    <div class="metric-value">{diced['area'].nunique()}</div>
                    <div class="metric-label">Areas</div>
                </div>""", unsafe_allow_html=True)
            with dk3:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#f59e0b,#d97706)">
                    <span class="metric-icon">⚡</span>
                    <div class="metric-value">{diced['violation_type'].nunique()}</div>
                    <div class="metric-label">Violation Types</div>
                </div>""", unsafe_allow_html=True)
            with dk4:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#34d399,#059669)">
                    <span class="metric-icon">⚠️</span>
                    <div class="metric-value">{diced['severity'].mean():.2f}</div>
                    <div class="metric-label">Avg Severity</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            stacked = (
                diced.groupby(["area", "violation_type"])["id"]
                .count()
                .reset_index()
                .rename(columns={"id": "Count"})
            )
            fig_dice = px.bar(
                stacked, x="area", y="Count", color="violation_type",
                barmode="stack",
                title="Diced Sub-cube: Area × Violation Type (stacked)",
                labels={"area": "Area", "violation_type": "Type"},
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_dice.update_layout(**_LAYOUT)
            st.plotly_chart(fig_dice, use_container_width=True)

            with st.expander("📄 View Raw Sub-cube Data"):
                st.dataframe(
                    diced[["area", "violation_type", "date", "hour",
                           "month", "severity", "time_period"]].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

    # ════════════════════════════════════════════════════════════════
    # TAB 5 — Pivot
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">🔄 Pivot Table — Rotate Dimensions</div>', unsafe_allow_html=True)

        dim_options = ["area", "violation_type", "month", "day_of_week",
                       "time_period", "quarter", "year"]

        p1, p2, p3 = st.columns(3)
        with p1:
            row_dim = st.selectbox("Row Dimension", dim_options, index=0, key="pv_row")
        with p2:
            col_options = [d for d in dim_options if d != row_dim]
            col_dim = st.selectbox("Column Dimension", col_options, index=0, key="pv_col")
        with p3:
            agg_label = st.selectbox("Aggregation", ["Count", "Mean Severity", "Sum Severity"], key="pv_agg")

        agg_map = {
            "Count":         ("id",       "count"),
            "Mean Severity": ("severity", "mean"),
            "Sum Severity":  ("severity", "sum"),
        }
        val_col, agg_fn = agg_map[agg_label]
        pivot = cube.pivot_table(
            index=row_dim, columns=col_dim,
            values=val_col, aggfunc=agg_fn, fill_value=0,
        )
        if agg_fn == "mean":
            pivot = pivot.round(2)

        fig_pv = px.imshow(
            pivot,
            color_continuous_scale="Viridis",
            title=f"Pivot: {row_dim} × {col_dim}  ({agg_label})",
            aspect="auto",
        )
        fig_pv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=max(400, len(pivot) * 26 + 80),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_pv, use_container_width=True)

        st.markdown('<div class="section-title">📋 Raw Pivot Table</div>', unsafe_allow_html=True)
        st.dataframe(pivot, use_container_width=True)