from receiver import read_robot_state
from lidar_processing import lidar_to_xy

state = read_robot_state()
points = lidar_to_xy(state["lidar"])

print(points)