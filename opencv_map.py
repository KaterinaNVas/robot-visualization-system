import cv2
import numpy as np
import pandas as pd


def build_occupancy_map(
    map_x,
    map_y,
    map_distance,
    map_size_mm=12000,
    image_size_px=600
):
    df = pd.DataFrame({
        "x": map_x,
        "y": map_y,
        "distance": map_distance
    })

    image = np.zeros((image_size_px, image_size_px), dtype=np.uint8)

    if df.empty:
        return image, 0

    scale = image_size_px / map_size_mm

    df["px"] = ((df["x"] + map_size_mm / 2) * scale).astype(int)
    df["py"] = ((map_size_mm / 2 - df["y"]) * scale).astype(int)

    df = df[
        (df["px"] >= 0) &
        (df["px"] < image_size_px) &
        (df["py"] >= 0) &
        (df["py"] < image_size_px)
    ]

    # рисуем точки лидара как маленькие круги
    for _, row in df.iterrows():
        cv2.circle(
            image,
            (int(row["px"]), int(row["py"])),
            radius=2,
            color=255,
            thickness=-1
        )

    kernel = np.ones((3, 3), np.uint8)

    # расширяем препятствия, чтобы их было видно
    image = cv2.dilate(image, kernel, iterations=2)

    # сглаживаем карту
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # контуры препятствий зелёным
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

    # робот в центре карты
    center = image_size_px // 2
    cv2.circle(result, (center, center), 5, (255, 0, 0), -1)

    return result, len(contours)