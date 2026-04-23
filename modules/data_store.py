                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
DataStore – CSV-backed warehouse simulation.
Acts as the Fact Table layer.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "violations.csv")

# Canonical schema ─ mirrors star-schema fact table
COLUMNS = [
    "id",
    "area",
    "violation_type",
    "date",
    "hour",
    "day_of_week",
    "month",
    "severity",       # derived: 1-Low / 2-Medium / 3-High
    "lat",            # mock coordinates for scatter
    "lon",
]

AREA_COORDS = {
    # Original Bengaluru areas
    "MG Road": (12.9757, 77.6011),
    "Koramangala": (12.9352, 77.6245),
    "Indiranagar": (12.9719, 77.6412),
    "Whitefield": (12.9698, 77.7499),
    "HSR Layout": (12.9116, 77.6389),
    "Jayanagar": (12.9308, 77.5838),
    "BTM Layout": (12.9166, 77.6101),
    "Hebbal": (13.0354, 77.5970),
    "Electronic City": (12.8456, 77.6603),
    "Yeshwanthpur": (13.0218, 77.5511),
    # Indian states / cities from the new dataset
    "Karnataka": (15.3173, 75.7139),
    "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6139, 77.2090),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Punjab": (31.1471, 75.3412),
    "Tamil Nadu": (11.1271, 78.6569),
    "Gujarat": (22.2587, 71.1924),
    "West Bengal": (22.9868, 87.8550),
    "Rajasthan": (27.0238, 74.2179),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Bihar": (25.0961, 85.3131),
    "Andhra Pradesh": (15.9129, 79.7400),
    "Telangana": (18.1124, 79.0193),
    "Kerala": (10.8505, 76.2711),
    "Odisha": (20.9517, 85.0985),
    "Chhattisgarh": (21.2787, 81.8661),
    "Jharkhand": (23.6102, 85.2799),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Uttarakhand": (30.0668, 79.0193),
    "Assam": (26.2006, 92.9376),
    "Goa": (15.2993, 74.1240),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Sikkim": (27.5330, 88.5122),
    "Tripura": (23.9408, 91.9882),
    "Arunachal Pradesh": (28.2180, 94.7278),
}

VIOLATION_SEVERITY = {
    # Original types
    "Red Light Jump": 3,
    "Over Speeding": 3,
    "Wrong Side Driving": 3,
    "Drunk Driving": 3,
    "No Helmet": 2,
    "No Seatbelt": 2,
    "Mobile Usage": 2,
    "Illegal Parking": 1,
    "No PUC": 1,
    "Lane Violation": 2,
    # New dataset violation types
    "Over-speeding": 3,
    "Signal Jumping": 3,
    "Wrong Parking": 1,
    "Using Mobile Phone": 2,
    "Driving Without License": 3,
    "Overloading": 2,
}


def _get_severity(violation_type: str) -> int:
    """Return severity score 1-3 for a violation type."""
    return VIOLATION_SEVERITY.get(str(violation_type).strip(), 1)


def _get_coords(area: str):
    """Return (lat, lon) for a known area, or a random India-ish coord."""
    if area in AREA_COORDS:
        return AREA_COORDS[area]
    # Fallback: centre of India with small hash-based offset so same area→same coords
    seed = abs(hash(area)) % 1000
    return (20.5937 + (seed % 10) * 0.3, 78.9629 + (seed % 7) * 0.4)


def _normalize_new_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the rich 34-column dataset into the 10-column internal schema.

    New CSV columns (sample):
        Violation_ID, Violation_Type, Fine_Amount, Location, Date, Time,
        Vehicle_Type, ... (many more)
    """
    out = pd.DataFrame()

    out["id"] = df["Violation_ID"].astype(str)

    # Location → area
    out["area"] = df["Location"].astype(str).str.strip()

    # Violation_Type → violation_type
    out["violation_type"] = df["Violation_Type"].astype(str).str.strip()

    # Date → date (parse flexibly)
    out["date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Time (HH:MM string) → hour integer
    def _parse_hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return 0

    out["hour"] = df["Time"].apply(_parse_hour)

    # Derived time fields
    out["day_of_week"] = out["date"].dt.day_name()
    out["month"] = out["date"].dt.strftime("%B")

    # Severity from violation type
    out["severity"] = out["violation_type"].apply(_get_severity)

    # Coordinates
    coords = out["area"].apply(_get_coords)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.05, 0.05, size=len(out))
    out["lat"] = coords.apply(lambda c: c[0]) + noise
    out["lon"] = coords.apply(lambda c: c[1]) + noise

    return out


class DataStore:
    def __init__(self):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        if not os.path.exists(DATA_PATH):
            self._seed()

    # ── public API ───────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(DATA_PATH)

            # ── Detect schema: new rich format vs internal format ────────────
            if "Violation_ID" in df.columns or "Location" in df.columns:
                # New rich dataset – normalise on the fly
                df = _normalize_new_schema(df)
            else:
                # Internal schema – just parse dates
                df["date"] = pd.to_datetime(df.get("date", pd.Series()), errors="coerce")

            # Ensure all expected columns exist
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = None

            return df[COLUMNS]

        except Exception:
            return pd.DataFrame(columns=COLUMNS)

    def add(self, area: str, violation_type: str, date: str, hour: int):
        df = self.load()
        dt = pd.to_datetime(date)
        lat, lon = _get_coords(area)
        noise = lambda: float(np.random.uniform(-0.01, 0.01))
        new_row = {
            "id": str(len(df) + 1),
            "area": area,
            "violation_type": violation_type,
            "date": dt.strftime("%Y-%m-%d"),
            "hour": hour,
            "day_of_week": dt.day_name(),
            "month": dt.strftime("%B"),
            "severity": _get_severity(violation_type),
            "lat": lat + noise(),
            "lon": lon + noise(),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

    def clear(self):
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_PATH, index=False)

    # ── seed with realistic sample data ─────────────────────────────────────
    def _seed(self):
        rng = np.random.default_rng(42)
        areas = list(AREA_COORDS.keys())
        violations = list(VIOLATION_SEVERITY.keys())
        n = 320

        records = []
        for i in range(1, n + 1):
            area = rng.choice(areas[:10])   # original Bengaluru areas for seeding
            vtype = rng.choice(violations)
            month = rng.integers(1, 13)
            day = rng.integers(1, 29)
            try:
                dt = datetime(2024, int(month), int(day))
            except Exception:
                dt = datetime(2024, 1, 1)
            hour = int(rng.choice(range(24), p=self._hour_weights()))
            lat, lon = AREA_COORDS[area]
            records.append({
                "id": i,
                "area": area,
                "violation_type": vtype,
                "date": dt.strftime("%Y-%m-%d"),
                "hour": hour,
                "day_of_week": dt.strftime("%A"),
                "month": dt.strftime("%B"),
                "severity": _get_severity(vtype),
                "lat": lat + float(rng.uniform(-0.01, 0.01)),
                "lon": lon + float(rng.uniform(-0.01, 0.01)),
            })

        pd.DataFrame(records).to_csv(DATA_PATH, index=False)

    @staticmethod
    def _area_weights(areas):
        # MG Road and Whitefield get higher probability
        base = np.ones(len(areas))
        base[0] *= 2.5   # MG Road
        base[3] *= 2.0   # Whitefield
        base[2] *= 1.8   # Indiranagar
        return (base / base.sum()).tolist()

    @staticmethod
    def _hour_weights():
        w = np.ones(24)
        # rush hours: 8-10, 17-20
        w[8:11] *= 3
        w[17:21] *= 4
        w[0:6] *= 0.3
        return (w / w.sum()).tolist()