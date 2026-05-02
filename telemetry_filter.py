from collections import deque


class TelemetryFilter:
    def __init__(self, max_trail_points=200):
        self.trail = deque(maxlen=max_trail_points)

        self.settings = {
            "show_lidar": True,
            "show_trajectory": True,
            "show_metrics": True
        }

    def update_filter_settings(self, new_settings):
        self.settings.update(new_settings)

    def process_telemetry(self, raw_data):
        display_packet = {}

        current_x = raw_data.get("x") or 0
        current_y = raw_data.get("y") or 0

        if self.settings["show_trajectory"]:
            self.trail.append((current_x, current_y))
            display_packet["trajectory"] = list(self.trail)
        else:
            self.trail.clear()
            display_packet["trajectory"] = []

        if self.settings["show_metrics"]:
            display_packet["metrics"] = {
                "speed": raw_data.get("speed") or 0,
                "yaw": raw_data.get("yaw") or 0,
                "battery": raw_data.get("battery") or 0,
                "tilt": raw_data.get("tilt") or 0,
                "position": (current_x, current_y)
            }
        else:
            display_packet["metrics"] = {}
        
        
        if self.settings["show_lidar"]:
            display_packet["lidar"] = raw_data.get("lidar", [])
        else:
            display_packet["lidar"] = []

        return display_packet