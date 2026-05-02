import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from receiver import read_robot_state
from lidar_processing import lidar_to_xy
from telemetry_filter import TelemetryFilter


global_map_x = []
global_map_y = []
global_map_colors = []

telemetry_filter = TelemetryFilter()

show_lidar = True
show_trajectory = True
show_metrics = True
running = True

plt.ion()
fig, ax = plt.subplots()
colorbar = None


def on_key(event):
    global show_lidar, show_trajectory, show_metrics, running

    if event.key == "l":
        show_lidar = not show_lidar
    elif event.key == "t":
        show_trajectory = not show_trajectory
    elif event.key == "m":
        show_metrics = not show_metrics
    elif event.key == "c":
        global_map_x.clear()
        global_map_y.clear()
        global_map_colors.clear()
    elif event.key == "q":
        running = False


fig.canvas.mpl_connect("key_press_event", on_key)


def draw_robot(ax, x, y, yaw):
    robot_length = 400
    robot_width = 250

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

    yaw_rad = math.radians(yaw)

    ax.arrow(
        x,
        y,
        math.cos(yaw_rad) * 350,
        math.sin(yaw_rad) * 350,
        head_width=90,
        length_includes_head=True
    )


def local_to_global(local_x, local_y, robot_x, robot_y, yaw):
    yaw_rad = math.radians(yaw)

    global_x = robot_x + local_x * math.cos(yaw_rad) - local_y * math.sin(yaw_rad)
    global_y = robot_y + local_x * math.sin(yaw_rad) + local_y * math.cos(yaw_rad)

    return global_x, global_y


while running:
    raw_state = read_robot_state()

    telemetry_filter.update_filter_settings({
        "show_lidar": show_lidar,
        "show_trajectory": show_trajectory,
        "show_metrics": show_metrics
    })

    state = telemetry_filter.process_telemetry(raw_state)

    metrics = state.get("metrics", {})

    robot_x_m = raw_state.get("x") or 0
    robot_y_m = raw_state.get("y") or 0

    robot_x = robot_x_m * 1000
    robot_y = robot_y_m * 1000

    yaw = raw_state.get("yaw") or 0
    speed = raw_state.get("speed") or 0
    battery = raw_state.get("battery") or 0

    points = lidar_to_xy(state.get("lidar", []))

    if show_lidar:
        for p in points:
            gx, gy = local_to_global(
                p["x"],
                p["y"],
                robot_x,
                robot_y,
                yaw
            )

            global_map_x.append(gx)
            global_map_y.append(gy)
            global_map_colors.append(p["distance"])

    max_points = 800
    global_map_x = global_map_x[-max_points:]
    global_map_y = global_map_y[-max_points:]
    global_map_colors = global_map_colors[-max_points:]

    trajectory = state.get("trajectory", [])
    trajectory_x = [p[0] * 1000 for p in trajectory]
    trajectory_y = [p[1] * 1000 for p in trajectory]

    ax.clear()

    if show_lidar and global_map_x and global_map_y:
        scatter = ax.scatter(
            global_map_x,
            global_map_y,
            c=global_map_colors,
            s=6,
            alpha=0.8,
            cmap="viridis",
            label="Global map"
        )

        if colorbar is None:
            colorbar = fig.colorbar(scatter, ax=ax, label="Distance (mm)")

    if show_trajectory and trajectory:
        ax.plot(
            trajectory_x,
            trajectory_y,
            linewidth=2,
            label="Trajectory"
        )

    draw_robot(ax, robot_x, robot_y, yaw)

    if show_metrics:
        title = (
            f"WOW Map | speed={speed} m/s | battery={battery}% | "
            f"L:{show_lidar} T:{show_trajectory} M:{show_metrics}"
        )
    else:
        title = (
            f"WOW Map | "
            f"L:{show_lidar} T:{show_trajectory} M:{show_metrics}"
        )

    ax.set_title(title)

    ax.set_xlabel("X, mm")
    ax.set_ylabel("Y, mm")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)

    plt.pause(0.1)

plt.ioff()
plt.show()