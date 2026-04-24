"""
Anomaly Detection page — Z-Score method.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.anomaly import (
    compute_zscore_anomalies,
    compute_temporal_anomalies,
    compute_hour_anomalies,
    compute_violation_type_anomalies,
)

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=45, b=10),
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
)


def _anomaly_bar(x_vals, y_vals, z_scores, threshold, title, x_label):
    """Helper: bar chart coloured by anomaly status."""
    colors = [
        "#ef4444" if z > threshold else "#3b82f6" if z < -threshold else "#475569"
        for z in z_scores
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals, y=y_vals,
        marker_color=colors,
        customdata=z_scores,
        hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y}}<br>Z-Score: %{{customdata:.3f}}<extra></extra>",
    ))
    mean_v = sum(y_vals) / len(y_vals) if y_vals else 0
    std_v  = pd.Series(y_vals).std() or 1
    fig.add_hline(y=mean_v + threshold * std_v, line_dash="dash",
                  line_color="#ef4444", annotation_text=f"+{threshold}σ")
    fig.add_hline(y=mean_v, line_dash="dot",
                  line_color="#a78bfa", annotation_text="Mean")
    if mean_v - threshold * std_v > 0:
        fig.add_hline(y=mean_v - threshold * std_v, line_dash="dash",
                      line_color="#3b82f6", annotation_text=f"−{threshold}σ")
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=x_label, yaxis_title="Count")
    return fig


def _zscore_bar(x_vals, z_scores, threshold, title):
    """Helper: Z-score bar chart."""
    colors = [
        "#ef4444" if z > threshold else "#3b82f6" if z < -threshold else "#475569"
        for z in z_scores
    ]
    
    fig = go.Figure(go.Bar(x=x_vals, y=z_scores, marker_color=colors))
    fig.add_hline(y=threshold,  line_dash="dash", line_color="#ef4444",
                  annotation_text=f"+{threshold} (upper bound)")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3b82f6",
                  annotation_text=f"−{threshold} (lower bound)")
    fig.add_hline(y=0, line_dash="dot", line_color="#a78bfa")
    fig.update_layout(**_LAYOUT, title=title, yaxis_title="Z-Score")
    return fig


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>🚨 Anomaly Detection</h1>
            <p>Detect unusual traffic violation patterns using the Z-Score statistical method</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty or len(df) < 10:
        st.warning("Need at least 10 records for anomaly detection.")
        return

    # ── Theory expander ───────────────────────────────────────────────────────
    with st.expander("📚 How Z-Score Anomaly Detection Works", expanded=False):
        st.markdown("""
        **Z-Score** measures how far a value deviates from the population mean:

        ```
        Z = (X − μ) / σ
        ```

        | Z-Score | Status |
        |---|---|
        | −threshold < Z < +threshold | ✅ Normal |
        | Z ≥ +threshold | 🔴 High Anomaly — unusually high violation count |
        | Z ≤ −threshold | 🔵 Low Anomaly — unusually low violation count |

        A common threshold is **2.0**, which flags roughly the top/bottom 2.5% of values.
        Raise it to **3.0** for only extreme outliers.
        """)

    # ── Threshold slider ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Z-Score Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Threshold  (higher = stricter, fewer anomalies flagged)",
        min_value=1.0, max_value=3.5, value=2.0, step=0.1, key="anom_thresh",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📍 By Area", "📅 By Date", "🕐 By Hour", "🚫 By Violation Type"]
    )

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Area anomalies
    # ════════════════════════════════════════════════════════════════
    with tab1:
        area_df = compute_zscore_anomalies(df, group_by="area", threshold=threshold)
        if area_df.empty:
            st.info("No data.")
        else:
            n_anom = int(area_df["is_anomaly"].sum())
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#ef4444,#b91c1c)">
                    <span class="metric-icon">🚨</span>
                    <div class="metric-value">{n_anom}</div>
                    <div class="metric-label">Anomalous Areas</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#a78bfa,#7c3aed)">
                    <span class="metric-icon">📍</span>
                    <div class="metric-value">{len(area_df)}</div>
                    <div class="metric-label">Total Areas</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                max_z = area_df["z_score"].abs().max()
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#f59e0b,#d97706)">
                    <span class="metric-icon">📈</span>
                    <div class="metric-value">{max_z:.2f}</div>
                    <div class="metric-label">Max |Z-Score|</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.plotly_chart(
                _anomaly_bar(
                    area_df["area"].tolist(), area_df["count"].tolist(), area_df["z_score"].tolist(),
                    threshold, "Violation Count per Area  (🔴 = High Anomaly, 🔵 = Low Anomaly)", "Area"
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                _zscore_bar(area_df["area"].tolist(), area_df["z_score"].tolist(), threshold, "Z-Score per Area"),
                use_container_width=True,
            )

            st.markdown('<div class="section-title">📋 Anomaly Summary</div>', unsafe_allow_html=True)
            anom_rows = area_df[area_df["is_anomaly"]]
            if anom_rows.empty:
                st.success("✅ No area anomalies detected at this threshold.")
            else:
                st.dataframe(
                    anom_rows[["area", "count", "z_score", "anomaly_type"]].rename(columns={
                        "area": "Area", "count": "Count",
                        "z_score": "Z-Score", "anomaly_type": "Status",
                    }),
                    use_container_width=True, hide_index=True,
                )

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Date anomalies
    # ════════════════════════════════════════════════════════════════
    with tab2:
        date_df = compute_temporal_anomalies(df, threshold=threshold)
        if date_df.empty:
            st.info("No data.")
        else:
            n_anom_d = int(date_df["is_anomaly"].sum())
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#ef4444,#b91c1c)">
                    <span class="metric-icon">📅</span>
                    <div class="metric-value">{n_anom_d}</div>
                    <div class="metric-label">Anomalous Days</div>
                </div>""", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#60a5fa,#2563eb)">
                    <span class="metric-icon">📊</span>
                    <div class="metric-value">{len(date_df)}</div>
                    <div class="metric-label">Total Days Tracked</div>
                </div>""", unsafe_allow_html=True)
            with d3:
                pct = f"{n_anom_d / len(date_df) * 100:.1f}%" if len(date_df) else "0%"
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#f59e0b,#d97706)">
                    <span class="metric-icon">📉</span>
                    <div class="metric-value">{pct}</div>
                    <div class="metric-label">Anomaly Rate</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Time-series line with anomaly markers
            normal   = date_df[~date_df["is_anomaly"]]
            spikes   = date_df[date_df["z_score"] > threshold]
            drops    = date_df[date_df["z_score"] < -threshold]

            mean_d = date_df["count"].mean()
            std_d  = date_df["count"].std() or 1

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=normal["date"].astype(str), y=normal["count"],
                mode="lines", name="Normal",
                line=dict(color="#475569", width=1.5),
            ))
            if not spikes.empty:
                fig_ts.add_trace(go.Scatter(
                    x=spikes["date"].astype(str), y=spikes["count"],
                    mode="markers", name="🔴 Spike",
                    marker=dict(color="#ef4444", size=10, symbol="x"),
                ))
            if not drops.empty:
                fig_ts.add_trace(go.Scatter(
                    x=drops["date"].astype(str), y=drops["count"],
                    mode="markers", name="🔵 Drop",
                    marker=dict(color="#3b82f6", size=10, symbol="triangle-down"),
                ))
            fig_ts.add_hline(y=mean_d + threshold * std_d, line_dash="dash",
                             line_color="#ef4444", annotation_text=f"+{threshold}σ")
            fig_ts.add_hline(y=mean_d, line_dash="dot",
                             line_color="#a78bfa", annotation_text="Mean")
            fig_ts.update_layout(**_LAYOUT, title="Daily Violations Over Time (Anomalies Highlighted)")
            st.plotly_chart(fig_ts, use_container_width=True)

            if n_anom_d > 0:
                st.markdown('<div class="section-title">🚨 Anomalous Days Detail</div>', unsafe_allow_html=True)
                st.dataframe(
                    date_df[date_df["is_anomaly"]][["date", "count", "z_score", "anomaly_type"]].rename(
                        columns={"date": "Date", "count": "Count",
                                 "z_score": "Z-Score", "anomaly_type": "Status"}
                    ),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("✅ No date anomalies detected at this threshold.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Hour anomalies
    # ════════════════════════════════════════════════════════════════
    with tab3:
        hour_df = compute_hour_anomalies(df, threshold=threshold)
        if hour_df.empty:
            st.info("No data.")
        else:
            n_anom_h = int(hour_df["is_anomaly"].sum())
            h1, h2 = st.columns(2)
            with h1:
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#ef4444,#b91c1c)">
                    <span class="metric-icon">🕐</span>
                    <div class="metric-value">{n_anom_h}</div>
                    <div class="metric-label">Anomalous Hours</div>
                </div>""", unsafe_allow_html=True)
            with h2:
                peak_h = int(hour_df.loc[hour_df["count"].idxmax(), "hour"])
                st.markdown(f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#a78bfa,#7c3aed)">
                    <span class="metric-icon">⏰</span>
                    <div class="metric-value">{peak_h:02d}:00</div>
                    <div class="metric-label">Peak Hour</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.plotly_chart(
                _anomaly_bar(
                    hour_df["hour"].tolist(), hour_df["count"].tolist(), hour_df["z_score"].tolist(),
                    threshold, "Violations by Hour  (🔴 = Anomalous Peak)", "Hour of Day"
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                _zscore_bar(hour_df["hour"].tolist(), hour_df["z_score"].tolist(), threshold, "Z-Score by Hour"),
                use_container_width=True,
            )

            st.markdown('<div class="section-title">📋 Hourly Z-Score Table</div>', unsafe_allow_html=True)
            st.dataframe(
                hour_df[["hour", "count", "z_score", "anomaly_type"]].rename(columns={
                    "hour": "Hour", "count": "Count",
                    "z_score": "Z-Score", "anomaly_type": "Status",
                }),
                use_container_width=True, hide_index=True,
            )

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Violation type anomalies
    # ════════════════════════════════════════════════════════════════
    with tab4:
        vtype_df = compute_violation_type_anomalies(df, threshold=threshold)
        if vtype_df.empty:
            st.info("No data.")
        else:
            st.plotly_chart(
                _anomaly_bar(
                    vtype_df["violation_type"].tolist(), vtype_df["count"].tolist(), vtype_df["z_score"].tolist(),
                    threshold, "Violation Type Frequency  (Anomalies Highlighted)", "Violation Type"
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                _zscore_bar(
                    vtype_df["violation_type"].tolist(), vtype_df["z_score"].tolist(),
                    threshold, "Z-Score by Violation Type"
                ),
                use_container_width=True,
            )

            anom_vt = vtype_df[vtype_df["is_anomaly"]]
            st.markdown('<div class="section-title">📋 Anomalous Violation Types</div>', unsafe_allow_html=True)
            if anom_vt.empty:
                st.success("✅ No violation type anomalies detected at this threshold.")
            else:
                st.dataframe(
                    anom_vt[["violation_type", "count", "z_score", "anomaly_type"]].rename(columns={
                        "violation_type": "Violation Type", "count": "Count",
                        "z_score": "Z-Score", "anomaly_type": "Status",
                    }),
                    use_container_width=True, hide_index=True,
                )
                