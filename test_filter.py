from receiver import read_robot_state
from telemetry_filter import TelemetryFilter
from lidar_processing import lidar_to_xy

telemetry_filter = TelemetryFilter()

raw_state = read_robot_state()
filtered_state = telemetry_filter.process_telemetry(raw_state)

print("RAW STATE:")
print(raw_state)

print("\nFILTERED STATE:")
print(filtered_state)

points = lidar_to_xy(filtered_state["lidar"])

print("\nLIDAR POINTS:")
print(points[:5])
print(f"Всего точек: {len(points)}")