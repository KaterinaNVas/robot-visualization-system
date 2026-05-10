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
    free_step_px=8
):
    if lidar_df.empty:
        return grid

    if "global_x" not in lidar_df.columns or "global_y" not in lidar_df.columns:
        return grid

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

        obstacle_px, obstacle_py = world_to_grid(
            row["global_x"],
            row["global_y"],
            map_size_mm,
            image_size_px
        )

        if not (0 <= obstacle_px < image_size_px and 0 <= obstacle_py < image_size_px):
            continue

        line_points = np.linspace(
            [robot_px, robot_py],
            [obstacle_px, obstacle_py],
            num=max(abs(obstacle_px - robot_px), abs(obstacle_py - robot_py), 1),
            dtype=int
        )

        for px, py in line_points[::free_step_px][:-3]:
            if 0 <= px < image_size_px and 0 <= py < image_size_px:
                if grid[py, px] != OCCUPIED:
                    grid[py, px] = FREE

        cv2.circle(
            grid,
            (obstacle_px, obstacle_py),
            1,
            OCCUPIED,
            -1
        )

    # kernel = np.ones((3, 3), np.uint8)

    # occupied_mask = (grid == OCCUPIED).astype(np.uint8) * 255
    # occupied_mask = cv2.dilate(occupied_mask, kernel, iterations=1)

    # grid[occupied_mask > 0] = OCCUPIED

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