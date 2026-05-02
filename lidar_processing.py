import math


def lidar_to_xy(lidar_data):
    points = []

    for point in lidar_data:
        angle = point["angle"]
        distance = point["distance"]

        if distance is None:
            continue

        angle_rad = math.radians(angle)

        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)

        points.append({
            "x": x,
            "y": y,
            "distance": distance,
            "angle": angle
        })

    return points