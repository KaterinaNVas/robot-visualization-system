import matplotlib.pyplot as plt
from receiver import read_robot_state
from lidar_processing import lidar_to_xy

state = read_robot_state()
points = lidar_to_xy(state["lidar"])

x = [p["x"] for p in points]
y = [p["y"] for p in points]

plt.scatter(x, y, label="Lidar points")
plt.scatter(0, 0, label="Robot")

plt.title("Lidar Map")
plt.xlabel("X, mm")
plt.ylabel("Y, mm")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()