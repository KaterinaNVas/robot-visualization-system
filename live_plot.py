import matplotlib.pyplot as plt
from receiver import read_robot_state
from lidar_processing import lidar_to_xy

plt.ion()  # интерактивный режим

fig, ax = plt.subplots()

while True:
    state = read_robot_state()
    points = lidar_to_xy(state["lidar"])

    x = [p["x"] for p in points]
    y = [p["y"] for p in points]

    ax.clear()

    ax.scatter(x, y, label="Lidar")
    ax.scatter(0, 0, label="Robot")

    ax.set_title("Live Lidar Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.pause(0.1)