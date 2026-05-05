from receiver import read_robot_state as read_http_state
from receiver import set_data as set_http_data

from receiver_ws import read_robot_state as read_ws_state
from receiver_ws import set_data as set_ws_data


def read_robot_state(source="WebSocket"):
    if source == "HTTP":
        return read_http_state()

    return read_ws_state()


def send_set_data(num: int, value: int, source="WebSocket"):
    if source == "HTTP":
        return set_http_data(num, value)

    return set_ws_data(num, value)