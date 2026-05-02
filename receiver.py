import requests

BASE_URL = "http://127.0.0.1:8000"


def get_data(num: int):
    try:
        response = requests.get(
            f"{BASE_URL}/api/get",
            params={"num": num},
            timeout=3
        )

        text = response.text.strip().replace(",", ".")
        # print(f"data{num} = {text}")  # временно для проверки

        return float(text)

    except Exception as e:
        return None


def set_data(num: int, value: int):
    try:
        response = requests.get(
            f"{BASE_URL}/api/set",
            params={"num": num, "val": value},
            timeout=0.3
        )
        return response.text
    except Exception:
        return "ERROR"


def read_robot_state():
    state = {
        "speed": get_data(1),
        "yaw": get_data(2),
        "battery": get_data(3),
        "x": get_data(4),
        "y": get_data(5),
        "tilt": get_data(6),
        "lidar": []
    }

    for i in range(7, 31):
        angle = (i - 7) * 15
        distance = get_data(i)
        state["lidar"].append({
            "angle": angle,
            "distance": distance
        })

    return state


if __name__ == "__main__":
    import time

    while True:
        print(read_robot_state())
        time.sleep(1)