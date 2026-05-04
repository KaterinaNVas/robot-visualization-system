from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import random
import math
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

setters = {
    "set1": 0,
    "set2": 0,
    "set3": 0,
    "set4": 0,
}

start_time = time.time()


def generate_robot_state():
    t = time.time() - start_time

    state = {
        "speed": round(abs(math.sin(t)) * 0.8, 2),
        "yaw": round((t * 20) % 360, 1),
        "battery": random.randint(70, 100),
        "x": round(math.sin(t / 3) * 2, 2),
        "y": round(math.cos(t / 3) * 2, 2),
        "tilt": round(random.uniform(-5, 5), 2),
        "lidar": []
    }

    for i in range(7, 31):
        angle = (i - 7) * 15
        distance = 1200 + 400 * math.sin(t + i)

        state["lidar"].append({
            "angle": angle,
            "distance": round(distance, 1)
        })

    return state


@app.route("/api/get")
def api_get_data():
    num = int(request.args.get("num", 0))
    state = generate_robot_state()

    mapping = {
        1: state["speed"],
        2: state["yaw"],
        3: state["battery"],
        4: state["x"],
        5: state["y"],
        6: state["tilt"],
    }

    if num in mapping:
        return str(mapping[num])

    if 7 <= num <= 30:
        return str(state["lidar"][num - 7]["distance"])

    return "0"


@app.route("/api/set")
def set_data_http():
    num = request.args.get("num")
    val = request.args.get("val")

    if num in ["1", "2", "3", "4"]:
        setters[f"set{num}"] = int(val)
        return f"OK: set_data{num} = {val}"

    return "ERROR"


@app.route("/api/get_setters")
def get_setters():
    return jsonify(setters)


@socketio.on("request_state")
def handle_request_state():
    state = generate_robot_state()
    return state


@socketio.on("set_data")
def handle_set_data(data):
    num = str(data.get("num"))
    val = data.get("val")

    if num in ["1", "2", "3", "4"]:
        setters[f"set{num}"] = int(val)
        return {
            "status": "OK",
            "message": f"set_data{num} = {val}"
        }

    return {
        "status": "ERROR",
        "message": "Invalid setter number"
    }


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)