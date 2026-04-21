"""
Preprocessing – clean & encode data for mining.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned, encoded copy of the dataframe ready for mining.
    Steps:
      1. Drop nulls / duplicates
      2. Encode categorical → numeric
      3. Feature engineering
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Clean
    df.dropna(subset=["area", "violation_type", "hour"], inplace=True)
    df.drop_duplicates(subset=["area", "violation_type", "date", "hour"], inplace=True)

    # 2. Encode
    le_area = LabelEncoder()
    le_vtype = LabelEncoder()
    le_day = LabelEncoder()

    df["area_enc"] = le_area.fit_transform(df["area"])
    df["violation_enc"] = le_vtype.fit_transform(df["violation_type"])
    df["day_enc"] = le_day.fit_transform(df["day_of_week"])

    # 3. Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df