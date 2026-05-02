import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from receiver import read_robot_state
from lidar_processing import lidar_to_xy


trajectory_x = []
trajectory_y = []

plt.ion()
fig, ax = plt.subplots()


def draw_robot(ax, x, y, yaw):
    robot_length = 400  # мм
    robot_width = 250   # мм

    rect = Rectangle(
        (-robot_length / 2, -robot_width / 2),
        robot_length,
        robot_width,
        fill=False,
        linewidth=2
    )

    transform = (
        Affine2D()
        .rotate_deg(yaw)
        .translate(x, y)
        + ax.transData
    )

    rect.set_transform(transform)
    ax.add_patch(rect)

    # стрелка направления
    yaw_rad = math.radians(yaw)
    arrow_x = x + math.cos(yaw_rad) * 300
    arrow_y = y + math.sin(yaw_rad) * 300

    ax.arrow(
        x, y,
        arrow_x - x,
        arrow_y - y,
        head_width=80,
        length_includes_head=True
    )


while True:
    state = read_robot_state()
    points = lidar_to_xy(state["lidar"])

    robot_x = state["x"] * 1000
    robot_y = state["y"] * 1000
    yaw = state["yaw"]

    trajectory_x.append(robot_x)
    trajectory_y.append(robot_y)

    lidar_x = [p["x"] + robot_x for p in points]
    lidar_y = [p["y"] + robot_y for p in points]

    ax.clear()

    ax.scatter(lidar_x, lidar_y, s=20, label="Lidar points")
    ax.plot(trajectory_x, trajectory_y, label="Trajectory")
    draw_robot(ax, robot_x, robot_y, yaw)

    ax.set_title(
        f"Robot Map | speed={state['speed']} m/s | battery={state['battery']}%"
    )
    ax.set_xlabel("X, mm")
    ax.set_ylabel("Y, mm")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    ax.set_xlim(-4000, 4000)
    ax.set_ylim(-4000, 4000)

    plt.pause(0.1)