import cv2
import numpy as np
import pandas as pd


def build_occupancy_map(
    map_x,
    map_y,
    map_distance,
    map_size_mm=12000,
    image_size_px=600,
    danger_radius_mm=1000,
    min_distance_mm=100,
    max_distance_mm=6000
):
    df = pd.DataFrame({
        "x": map_x,
        "y": map_y,
        "distance": map_distance
    })

    result = np.zeros((image_size_px, image_size_px, 3), dtype=np.uint8)

    stats = {
        "obstacle_count": 0,
        "nearest_obstacle_mm": None,
        "danger_detected": False
    }

    if df.empty:
        return result, stats

    df = df[
        (df["distance"] >= min_distance_mm) &
        (df["distance"] <= max_distance_mm)
    ]

    if df.empty:
        return result, stats

    df["range_from_robot"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)

    nearest = df["range_from_robot"].min()
    stats["nearest_obstacle_mm"] = round(float(nearest), 1)
    stats["danger_detected"] = bool(nearest <= danger_radius_mm)

    scale = image_size_px / map_size_mm
    center = image_size_px // 2

    df["px"] = ((df["x"] + map_size_mm / 2) * scale).astype(int)
    df["py"] = ((map_size_mm / 2 - df["y"]) * scale).astype(int)

    df = df[
        (df["px"] >= 0) &
        (df["px"] < image_size_px) &
        (df["py"] >= 0) &
        (df["py"] < image_size_px)
    ]

    binary = np.zeros((image_size_px, image_size_px), dtype=np.uint8)

    for _, row in df.iterrows():
        distance = row["range_from_robot"]

        if distance <= danger_radius_mm:
            color = (255, 0, 0)      # красный, опасно близко
        elif distance <= danger_radius_mm * 2:
            color = (255, 255, 0)    # жёлтый, средняя дистанция
        else:
            color = (0, 255, 0)      # зелёный, далеко

        point = (int(row["px"]), int(row["py"]))

        cv2.circle(result, point, 2, color, -1)
        cv2.circle(binary, point, 2, 255, -1)

    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 10
    ]

    stats["obstacle_count"] = len(filtered_contours)

    cv2.drawContours(result, filtered_contours, -1, (255, 255, 255), 1)

    danger_radius_px = int(danger_radius_mm * scale)

    danger_color = (255, 0, 0) if stats["danger_detected"] else (0, 255, 255)

    cv2.circle(result, (center, center), danger_radius_px, danger_color, 1)
    cv2.circle(result, (center, center), 5, (0, 0, 255), -1)

    return result, stats