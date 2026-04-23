                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
K-Means Clustering Module
Groups areas into High / Medium / Low risk zones.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RISK_LABELS_BY_K = {
    2: ["🔴 High Risk", "🟢 Low Risk"],
    3: ["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
    4: ["🔴 High Risk", "🟠 Medium-High Risk", "🟡 Medium-Low Risk", "🟢 Low Risk"],
    5: ["🔴 Very High Risk", "🟠 High Risk", "🟡 Medium Risk", "🟢 Low Risk", "🔵 Very Low Risk"]
}

RISK_COLORS = {
    "🔴 High Risk": "#ef4444",
    "🟡 Medium Risk": "#f59e0b",
    "🟢 Low Risk": "#22c55e",
    "🔴 Very High Risk": "#dc2626",
    "🟠 High Risk": "#ea580c",
    "🟠 Medium-High Risk": "#f97316",
    "🟡 Medium-Low Risk": "#eab308",
    "🔵 Very Low Risk": "#3b82f6",
}


def run_kmeans(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      area_risk_df  – per-area cluster assignment
      raw_labeled   – original df with cluster column
    """
    if df.empty or len(df) < k:
        return pd.DataFrame(), df

    # Build per-area feature matrix
    area_features = (
        df.groupby("area")
        .agg(
            total_violations=("id", "count"),
            avg_severity=("severity", "mean"),
            avg_hour=("hour", "mean"),
            unique_types=("violation_type", "nunique"),
        )
        .reset_index()
    )

    feature_cols = ["total_violations", "avg_severity", "avg_hour", "unique_types"]
    X = area_features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Map cluster ids to risk levels based on total_violations (desc = high risk)
    cluster_totals = {}
    for c in range(k):
        mask = labels == c
        cluster_totals[c] = area_features.loc[mask, "total_violations"].sum()

    sorted_clusters = sorted(cluster_totals, key=cluster_totals.get, reverse=True)
    labels_list = RISK_LABELS_BY_K.get(k, RISK_LABELS_BY_K[3])
    cluster_to_risk = {sorted_clusters[i]: labels_list[i] for i in range(k)}

    area_features["cluster"] = labels
    area_features["risk_level"] = area_features["cluster"].map(cluster_to_risk)

    # Attach risk to raw df
    raw_labeled = df.merge(
        area_features[["area", "risk_level"]], on="area", how="left"
    )

    return area_features, raw_labeled