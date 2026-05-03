import time
import math
import streamlit as st
import plotly.graph_objects as go

from receiver import read_robot_state
from lidar_processing import lidar_to_xy


st.set_page_config(
    page_title="Robot Visualization System",
    layout="wide"
)

if "trajectory" not in st.session_state:
    st.session_state.trajectory = []

if "global_map_x" not in st.session_state:
    st.session_state.global_map_x = []

if "global_map_y" not in st.session_state:
    st.session_state.global_map_y = []

if "global_map_colors" not in st.session_state:
    st.session_state.global_map_colors = []

if "running" not in st.session_state:
    st.session_state.running = False
if "connected" not in st.session_state:
    st.session_state.connected = False


st.title("Robot Visualization System")

with st.sidebar:
    st.header("Панель управления")

    status = "RUNNING" if st.session_state.running else "PAUSED"
    connection_status = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
    work_status = "RUNNING" if st.session_state.running else "PAUSED"

    st.markdown(f"### Connection: `{connection_status}`")
    st.markdown(f"### System status: `{work_status}`")

    col_conn, col_disc = st.columns(2)

    with col_conn:
        if st.button("Connect"):
            st.session_state.connected = True
            st.session_state.running = True
            st.rerun()

    with col_disc:
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.running = False
            st.rerun()

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button("Старт"):
            st.session_state.running = True
            st.rerun()

    with col_stop:
        if st.button("Стоп"):
            st.session_state.running = False
            st.rerun()

    if st.button("Обновить кадр"):
        st.session_state.running = False
        st.rerun()

    show_lidar = st.checkbox("Показывать лидар", value=True)
    show_trajectory = st.checkbox("Показывать траекторию", value=True)
    show_robot = st.checkbox("Показывать робота", value=True)
    show_metrics = st.checkbox("Показывать метрики", value=True)

    max_points = st.slider(
        "Количество точек карты",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100
    )

    refresh_rate = st.slider(
        "Частота обновления, сек",
        min_value=1.0,
        max_value=4.0,
        value=1.0,
        step=0.1
    )

    if st.button("Очистить карту"):
        st.session_state.global_map_x = []
        st.session_state.global_map_y = []
        st.session_state.global_map_colors = []
        st.session_state.trajectory = []


placeholder_metrics = st.empty()
placeholder_map = st.empty()


def local_to_global(local_x, local_y, robot_x, robot_y, yaw):
    yaw_rad = math.radians(yaw)

    global_x = robot_x + local_x * math.cos(yaw_rad) - local_y * math.sin(yaw_rad)
    global_y = robot_y + local_x * math.sin(yaw_rad) + local_y * math.cos(yaw_rad)

    return global_x, global_y


def robot_shape(x, y, yaw):
    length = 400
    width = 250

    corners = [
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2),
        (-length / 2, -width / 2),
    ]

    yaw_rad = math.radians(yaw)
    xs = []
    ys = []

    for cx, cy in corners:
        gx = x + cx * math.cos(yaw_rad) - cy * math.sin(yaw_rad)
        gy = y + cx * math.sin(yaw_rad) + cy * math.cos(yaw_rad)
        xs.append(gx)
        ys.append(gy)

    return xs, ys


if st.session_state.connected:
    raw_state = read_robot_state()
else:
    raw_state = {
        "x": 0,
        "y": 0,
        "yaw": 0,
        "speed": 0,
        "battery": 0,
        "tilt": 0,
        "lidar": []
    }

robot_x = (raw_state.get("x") or 0) * 1000
robot_y = (raw_state.get("y") or 0) * 1000
yaw = raw_state.get("yaw") or 0
speed = raw_state.get("speed") or 0
battery = raw_state.get("battery") or 0
tilt = raw_state.get("tilt") or 0

st.session_state.trajectory.append((robot_x, robot_y))

points = lidar_to_xy(raw_state.get("lidar", []))

if show_lidar:
    for p in points:
        gx, gy = local_to_global(
            p["x"],
            p["y"],
            robot_x,
            robot_y,
            yaw
        )

        st.session_state.global_map_x.append(gx)
        st.session_state.global_map_y.append(gy)
        st.session_state.global_map_colors.append(p["distance"])

st.session_state.global_map_x = st.session_state.global_map_x[-max_points:]
st.session_state.global_map_y = st.session_state.global_map_y[-max_points:]
st.session_state.global_map_colors = st.session_state.global_map_colors[-max_points:]
st.session_state.trajectory = st.session_state.trajectory[-300:]


if show_metrics:
    col1, col2, col3, col4 = placeholder_metrics.columns(4)

    col1.metric("Battery", f"{battery}%")
    col2.metric("Speed", f"{speed} m/s")
    col3.metric("Yaw", f"{yaw}°")
    col4.metric("Tilt", f"{tilt}°")


fig = go.Figure()

if show_lidar and st.session_state.global_map_x:
    fig.add_trace(
        go.Scatter(
            x=st.session_state.global_map_x,
            y=st.session_state.global_map_y,
            mode="markers",
            marker=dict(
                size=5,
                color=st.session_state.global_map_colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Distance, mm")
            ),
            name="Global map"
        )
    )

if show_trajectory and st.session_state.trajectory:
    tx = [p[0] for p in st.session_state.trajectory]
    ty = [p[1] for p in st.session_state.trajectory]

    fig.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            mode="lines",
            name="Trajectory"
        )
    )

if show_robot:
    rx, ry = robot_shape(robot_x, robot_y, yaw)

    fig.add_trace(
        go.Scatter(
            x=rx,
            y=ry,
            mode="lines",
            name="Robot body",
            line=dict(width=3)
        )
    )

    yaw_rad = math.radians(yaw)

    fig.add_trace(
        go.Scatter(
            x=[robot_x, robot_x + math.cos(yaw_rad) * 450],
            y=[robot_y, robot_y + math.sin(yaw_rad) * 450],
            mode="lines+markers",
            name="Direction",
            line=dict(width=3)
        )
    )

fig.update_layout(
    title="Real-Time Robot Map",
    xaxis_title="X, mm",
    yaxis_title="Y, mm",
    height=700,
    xaxis=dict(range=[-5000, 5000], scaleanchor="y"),
    yaxis=dict(range=[-5000, 5000]),
    legend=dict(x=0.02, y=0.98)
)

placeholder_map.plotly_chart(fig, use_container_width=True)

if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()