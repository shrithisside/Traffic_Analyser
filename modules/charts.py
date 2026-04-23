                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Charts module – Plotly chart builders.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PALETTE = px.colors.qualitative.Vivid

# common layout tweaks
_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def violations_by_area_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["area"].value_counts().reset_index()
    counts.columns = ["area", "count"]

    fig = px.bar(
        counts,
        x="area",
        y="count",
        color="count",
        color_continuous_scale="Plasma",
        labels={"count": "Violations", "area": "Area"},
        title="Violations per Area",
    )
    fig.update_layout(**_layout, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def violations_by_type_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["violation_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]

    fig = px.pie(
        counts,
        names="type",
        values="count",
        color_discrete_sequence=PALETTE,
        title="Violation Type Distribution",
        hole=0.45,
    )
    fig.update_layout(**_layout)
    fig.update_traces(textfont_size=12)
    return fig


def violations_by_hour_chart(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("hour").size().reset_index(name="count")
    all_hours = pd.DataFrame({"hour": range(24)})
    counts = all_hours.merge(counts, on="hour", how="left").fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=counts["hour"],
            y=counts["count"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#a78bfa", width=3),
            marker=dict(color="#7c3aed", size=6),
            name="Violations",
        )
    )
    fig.update_layout(
        **_layout,
        title="Violations by Hour of Day",
        xaxis=dict(
            title="Hour",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}h" for h in range(24)],
            gridcolor="#1e293b",
        ),
        yaxis=dict(title="Count", gridcolor="#1e293b"),
    )
    return fig


def cluster_scatter_chart(area_df: pd.DataFrame) -> go.Figure:
    color_map = {
        "🔴 High Risk": "#ef4444",
        "🟡 Medium Risk": "#f59e0b",
        "🟢 Low Risk": "#22c55e",
        "🔴 Very High Risk": "#dc2626",
        "🟠 High Risk": "#ea580c",
        "🟠 Medium-High Risk": "#f97316",
        "🟡 Medium-Low Risk": "#eab308",
        "🔵 Very Low Risk": "#3b82f6",
    }

    fig = px.scatter(
        area_df,
        x="total_violations",
        y="avg_severity",
        color="risk_level",
        size="total_violations",
        text="area",
        color_discrete_map=color_map,
        labels={
            "total_violations": "Total Violations",
            "avg_severity": "Average Severity",
            "risk_level": "Risk Level",
        },
        title="K-Means Clustering – Area Risk Zones",
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(
        **_layout,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def risk_gauge(score: float) -> go.Figure:
    """score 0-100"""
    if score < 33:
        color, label = "#22c55e", "Low Risk"
    elif score < 66:
        color, label = "#f59e0b", "Medium Risk"
    else:
        color, label = "#ef4444", "High Risk"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Risk Score – {label}", "font": {"color": "#e2e8f0", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#e2e8f0"},
                "bar": {"color": color},
                "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0, 33], "color": "rgba(34,197,94,0.15)"},
                    {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,0.15)"},
                ],
            },
            number={"suffix": "/100", "font": {"color": "#e2e8f0"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def violations_by_day_chart(df: pd.DataFrame) -> go.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = df["day_of_week"].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = ["day", "count"]

    fig = px.bar(
        counts,
        x="day",
        y="count",
        color="count",
        color_continuous_scale="Turbo",
        title="Violations by Day of Week",
        labels={"count": "Violations", "day": "Day"},
    )
    fig.update_layout(**_layout, coloraxis_showscale=False)
    return fig


def month_trend_chart(df: pd.DataFrame) -> go.Figure:
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    counts = df["month"].value_counts().reindex(month_order, fill_value=0).reset_index()
    counts.columns = ["month", "count"]

    fig = go.Figure(
        go.Bar(
            x=counts["month"],
            y=counts["count"],
            marker=dict(
                color=counts["count"],
                colorscale="Viridis",
            ),
        )
    )
    fig.update_layout(**_layout, title="Monthly Violation Trend",
                      xaxis=dict(gridcolor="#1e293b"),
                      yaxis=dict(gridcolor="#1e293b"))
    return fig