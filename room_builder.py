import numpy as np
import pandas as pd


def build_room_contour(
    lidar_points,
    angle_bin_deg=1.0,
    min_distance_mm=130,
    max_distance_mm=8000,
):
    df = pd.DataFrame(lidar_points)

    if df.empty:
        return pd.DataFrame(columns=["angle", "distance", "x", "y"])

    df = df[
        (df["distance"] >= min_distance_mm) &
        (df["distance"] <= max_distance_mm)
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["angle", "distance", "x", "y"])

    df["angle_bin"] = (df["angle"] / angle_bin_deg).round() * angle_bin_deg

    room_df = (
        df.groupby("angle_bin")["distance"]
        .max()
        .reset_index()
        .rename(columns={"angle_bin": "angle"})
        .sort_values("angle")
    )

    room_df["distance"] = (
        room_df["distance"]
        .rolling(window=7, center=True, min_periods=1)
        .median()
    )

    angle_rad = np.radians(room_df["angle"])

    room_df["x"] = room_df["distance"] * np.cos(angle_rad)
    room_df["y"] = room_df["distance"] * np.sin(angle_rad)

    return room_df