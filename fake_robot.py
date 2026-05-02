from flask import Flask, request, jsonify
import random
import math
import time

app = Flask(__name__)

setters = {
    "set1": 0,
    "set2": 0,
    "set3": 0,
    "set4": 0,
}

start_time = time.time()

@app.route("/api/get")
def api_get_data():
    num = int(request.args.get("num", 0))
    t = time.time() - start_time

    if num == 1:
        return str(round(abs(math.sin(t)) * 0.8, 2))

    elif num == 2:
        return str(round((t * 20) % 360, 1))

    elif num == 3:
        return str(random.randint(70, 100))

    elif num == 4:
        return str(round(math.sin(t / 3) * 2, 2))

    elif num == 5:
        return str(round(math.cos(t / 3) * 2, 2))

    elif num == 6:
        return str(round(random.uniform(-5, 5), 2))

    elif 7 <= num <= 30:
        distance = 1200 + 400 * math.sin(t + num)
        return str(round(distance, 1))

    return "0"


@app.route("/api/set")
def set_data():
    num = request.args.get("num")
    val = request.args.get("val")

    if num in ["1", "2", "3", "4"]:
        setters[f"set{num}"] = int(val)
        return f"OK: set_data{num} = {val}"

    return "ERROR"


@app.route("/api/get_setters")
def get_setters():
    return jsonify(setters)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)