                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Add Violation page – data input layer (Fact Table entry).
"""

import streamlit as st
from datetime import date
from modules.data_store import DataStore, AREA_COORDS, VIOLATION_SEVERITY


AREAS = list(AREA_COORDS.keys())
VIOLATIONS = list(VIOLATION_SEVERITY.keys())
HOURS = [f"{h:02d}:00" for h in range(24)]


def render(store: DataStore):
    st.markdown(
        """
        <div class="page-header">
            <h1>📝 Add Violation Record</h1>
            <p>Input new traffic violation data into the warehouse (Fact Table entry)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_form, col_info = st.columns([2, 1], gap="large")

    with col_form:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        st.markdown("#### 📍 Location Details")
        area = st.selectbox("Select Area", AREAS, key="area_select")

        st.markdown("#### 🚫 Violation Details")
        violation = st.selectbox("Violation Type", VIOLATIONS, key="viol_select")

        sev = VIOLATION_SEVERITY[violation]
        sev_color = "#ef4444" if sev == 3 else "#f59e0b" if sev == 2 else "#22c55e"
        sev_label = "High" if sev == 3 else "Medium" if sev == 2 else "Low"
        st.markdown(
            f"<span style='color:{sev_color};font-weight:600;font-size:0.85rem;'>"
            f"⚠ Severity: {sev_label} ({sev}/3)</span>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 📅 Time Details")
        c1, c2 = st.columns(2)
        with c1:
            violation_date = st.date_input("Date", value=date.today(), key="date_in")
        with c2:
            hour_str = st.selectbox("Hour of Day", HOURS, index=8, key="hour_sel")
            hour_val = int(hour_str.split(":")[0])

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.button("➕ Add to Warehouse", width="stretch")

        if submitted:
            store.add(area, violation, str(violation_date), hour_val)
            st.success(f"✅ Violation recorded! **{violation}** at **{area}** on {violation_date} at {hour_str}")
            st.balloons()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-title">🗄️ Warehouse Schema</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-card">
                <div class="ic-label">FACT TABLE</div>
                <div class="ic-value" style="font-size:0.9rem">violations</div>
            </div>
            <div style="padding-left:12px;color:#64748b;font-size:0.82rem;line-height:2;">
                ├─ <span style="color:#a78bfa">area</span> → DIM_AREA<br>
                ├─ <span style="color:#60a5fa">violation_type</span> → DIM_VIOLATION<br>
                ├─ <span style="color:#34d399">date / hour</span> → DIM_TIME<br>
                └─ <span style="color:#f59e0b">severity</span> (derived)
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">📌 Severity Scale</div>', unsafe_allow_html=True)
        for v, s in VIOLATION_SEVERITY.items():
            color = "#ef4444" if s == 3 else "#f59e0b" if s == 2 else "#22c55e"
            label = "High" if s == 3 else "Med" if s == 2 else "Low"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                f"border-bottom:1px solid #1e293b;font-size:0.8rem;'>"
                f"<span style='color:#94a3b8;'>{v}</span>"
                f"<span style='color:{color};font-weight:700;'>{label}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown('<div class="section-title">🗑️ Danger Zone</div>', unsafe_allow_html=True)
        if st.button("🗑 Clear All Data", width="stretch"):
            store.clear()
            st.warning("All data cleared. Sample data will re-seed on next load.")