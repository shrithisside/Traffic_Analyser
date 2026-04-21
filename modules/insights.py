"""
Insights Generator – extracts key findings from the dataset.
"""
import pandas as pd


def generate_insights(df: pd.DataFrame) -> dict:
    """
    Returns a dict of insight keys/values for the dashboard.
    """
    if df.empty:
        return {
            "total": 0,
            "high_risk_area": "N/A",
            "common_violation": "N/A",
            "peak_hour": "N/A",
            "peak_day": "N/A",
            "peak_month": "N/A",
            "avg_severity": 0,
            "busiest_area_count": 0,
        }

    high_risk_area = df["area"].value_counts().idxmax()
    common_violation = df["violation_type"].value_counts().idxmax()
    peak_hour_val = int(df["hour"].value_counts().idxmax())
    peak_hour_str = f"{peak_hour_val:02d}:00 – {(peak_hour_val+1)%24:02d}:00"
    peak_day = df["day_of_week"].value_counts().idxmax()
    peak_month = df["month"].value_counts().idxmax()
    avg_severity = round(df["severity"].mean(), 2)
    busiest_count = int(df["area"].value_counts().max())

    return {
        "total": len(df),
        "high_risk_area": high_risk_area,
        "common_violation": common_violation,
        "peak_hour": peak_hour_str,
        "peak_day": peak_day,
        "peak_month": peak_month,
        "avg_severity": avg_severity,
        "busiest_area_count": busiest_count,
    }