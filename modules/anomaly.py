                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163

import pandas as pd
import numpy as np

def _add_anomaly_labels(df, threshold):
    mean = df["count"].mean()
    std = df["count"].std()
    if std == 0 or pd.isna(std):
        df["z_score"] = 0.0
    else:
        df["z_score"] = (df["count"] - mean) / std
    df["is_anomaly"] = df["z_score"].abs() > threshold
    df["anomaly_type"] = df.apply(
        lambda r: "🔴 High Anomaly" if r["z_score"] > threshold
        else "🔵 Low Anomaly" if r["z_score"] < -threshold
        else "✅ Normal", axis=1,
    )
    df["z_score"] = df["z_score"].round(3)
    return df

def compute_zscore_anomalies(df, group_by="area", threshold=2.0):
    if df.empty: return pd.DataFrame()
    counts = df.groupby(group_by)["id"].count().reset_index()
    counts.columns = [group_by, "count"]
    counts = _add_anomaly_labels(counts, threshold)
    return counts.sort_values("z_score", ascending=False).reset_index(drop=True)

def compute_temporal_anomalies(df, time_col="date", threshold=2.0):
    if df.empty: return pd.DataFrame()
    temp = df.copy()
    temp["date_only"] = pd.to_datetime(temp[time_col]).dt.date
    daily = temp.groupby("date_only")["id"].count().reset_index()
    daily.columns = ["date", "count"]
    daily = _add_anomaly_labels(daily, threshold)
    return daily.sort_values("date").reset_index(drop=True)

def compute_hour_anomalies(df, threshold=2.0):
    if df.empty: return pd.DataFrame()
    hourly = df.groupby("hour")["id"].count().reset_index()
    hourly.columns = ["hour", "count"]
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
    hourly["count"] = hourly["count"].astype(int)
    hourly = _add_anomaly_labels(hourly, threshold)
    return hourly.sort_values("hour").reset_index(drop=True)

def compute_violation_type_anomalies(df, threshold=2.0):
    if df.empty: return pd.DataFrame()
    counts = df.groupby("violation_type")["id"].count().reset_index()
    counts.columns = ["violation_type", "count"]
    counts = _add_anomaly_labels(counts, threshold)
    return counts.sort_values("z_score", ascending=False).reset_index(drop=True)
