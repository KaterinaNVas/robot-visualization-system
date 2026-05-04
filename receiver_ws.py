import socketio

BASE_URL = "http://127.0.0.1:8000"

sio = socketio.Client()


def connect_socket():
    if not sio.connected:
        sio.connect(BASE_URL)


def disconnect_socket():
    if sio.connected:
        sio.disconnect()


def read_robot_state():
    try:
        connect_socket()
        state = sio.call("request_state", timeout=3)
        return state
    except Exception:
        return {
            "x": 0,
            "y": 0,
            "yaw": 0,
            "speed": 0,
            "battery": 0,
            "tilt": 0,
            "lidar": []
        }


def set_data(num: int, value: int):
    try:
        connect_socket()
        response = sio.call(
            "set_data",
            {"num": num, "val": value},
            timeout=3
        )
        return response
    except Exception:
        return {
            "status": "ERROR",
            "message": "WebSocket connection error"
        }