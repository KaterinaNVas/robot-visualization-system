import math
import pandas as pd


def lidar_to_xy(lidar_data):
    points = []

    for point in lidar_data:
        angle = point.get("angle")
        distance = point.get("distance")

        if angle is None or distance is None:
            continue

        angle_rad = math.radians(angle)

        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)

        points.append({
            "angle": angle,
            "distance": distance,
            "x": x,
            "y": y
        })

    return points


def lidar_to_dataframe(lidar_data):
    points = lidar_to_xy(lidar_data)

    df = pd.DataFrame(points, columns=["angle", "distance", "x", "y"])

    if df.empty:
        return df

    return df