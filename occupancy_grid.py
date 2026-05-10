import cv2
import numpy as np


UNKNOWN = 80
FREE = 190
OCCUPIED = 0
ROBOT_COLOR = (255, 0, 0)  # синий в OpenCV BGR


def world_to_grid(x, y, map_size_mm, image_size_px):
    scale = image_size_px / map_size_mm
    center = image_size_px // 2

    px = int(center + x * scale)
    py = int(center - y * scale)

    return px, py

def bresenham_line(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return points


def init_occupancy_grid(image_size_px=600):
    return np.full(
        (image_size_px, image_size_px),
        UNKNOWN,
        dtype=np.uint8
    )


def update_occupancy_grid(
    grid,
    lidar_df,
    robot_x=0,
    robot_y=0,
    map_size_mm=12000,
    image_size_px=600,
    min_distance_mm=130,
    max_distance_mm=8000,
    free_step_px=16
):
    if lidar_df.empty:
        return grid

    if "global_x" not in lidar_df.columns or "global_y" not in lidar_df.columns:
        return grid
    
    grid[grid == FREE] = 170

    robot_px, robot_py = world_to_grid(
        robot_x,
        robot_y,
        map_size_mm,
        image_size_px
    )

    for _, row in lidar_df.iterrows():
        distance = row.get("distance", 0)

        if distance < min_distance_mm or distance > max_distance_mm:
            continue

        if distance > 5000:
            continue

        obstacle_px, obstacle_py = world_to_grid(
            row["global_x"],
            row["global_y"],
            map_size_mm,
            image_size_px
        )

        if not (0 <= obstacle_px < image_size_px and 0 <= obstacle_py < image_size_px):
            continue

        line_points = bresenham_line(
            robot_px,
            robot_py,
            obstacle_px,
            obstacle_py
        )

        for px, py in line_points[::free_step_px][5:-8]:
            if 0 <= px < image_size_px and 0 <= py < image_size_px:
                if grid[py, px] != OCCUPIED:
                    grid[py, px] = FREE

        cv2.circle(
            grid,
            (obstacle_px, obstacle_py),
            2,
            OCCUPIED,
            -1
        )

    # kernel = np.ones((3, 3), np.uint8)

    # occupied_mask = (grid == OCCUPIED).astype(np.uint8) * 255
    # occupied_mask = cv2.dilate(occupied_mask, kernel, iterations=1)

    # grid[occupied_mask > 0] = OCCUPIED

    occupied_mask = (grid == OCCUPIED).astype(np.uint8)

    occupied_mask = cv2.medianBlur(
        occupied_mask,
        3
    )

    grid[grid == OCCUPIED] = FREE
    grid[occupied_mask > 0] = OCCUPIED

    return grid


def render_occupancy_grid(
    grid,
    robot_x=0,
    robot_y=0,
    map_size_mm=12000,
    image_size_px=600
):
    image = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    robot_px, robot_py = world_to_grid(
        robot_x,
        robot_y,
        map_size_mm,
        image_size_px
    )

    cv2.circle(
        image,
        (robot_px, robot_py),
        6,
        ROBOT_COLOR,
        -1
    )

    return image