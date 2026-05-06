import time
import math
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import cv2

from datetime import datetime
from data_receiver import read_robot_state, send_set_data
from lidar_processing import lidar_to_dataframe
from opencv_map import build_occupancy_map


st.set_page_config(
    page_title="Robot Visualization System",
    layout="wide"
)

# ----------------------------
# Session state
# ----------------------------

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

if "danger_events" not in st.session_state:
    st.session_state.danger_events = []

if "last_danger_event_time" not in st.session_state:
    st.session_state.last_danger_event_time = None

# ----------------------------
# Sidebar
# ----------------------------

st.title("Система Визуализации Робота")

with st.sidebar:
    st.header("Панель управления")

    data_source = st.radio(
        "Источник данных",
        ["WebSocket", "HTTP"],
        index=0
    )

    connection_status = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
    work_status = "RUNNING" if st.session_state.running else "PAUSED"

    st.markdown(f"**Подключение:** `{connection_status}`")
    st.markdown(f"**Состояние системы:** `{work_status}`")

    col_conn, col_disc = st.columns(2)

    with col_conn:
        if st.button("Подключить"):
            st.session_state.connected = True
            st.session_state.running = True
            st.rerun()

    with col_disc:
        if st.button("Отключить"):
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
    show_cv_map = st.checkbox("Показывать OpenCV-карту", value=True)

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

    st.subheader("Настройки OpenCV")

    danger_radius_mm = st.slider(
        "Радиус опасной зоны, мм",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100
    )

    min_distance_mm = st.slider(
        "Минимальная дистанция лидара, мм",
        min_value=0,
        max_value=1000,
        value=100,
        step=50
    )

    max_distance_mm = st.slider(
        "Максимальная дистанция лидара, мм",
        min_value=1000,
        max_value=10000,
        value=6000,
        step=500
    )

    st.subheader("Команды роботу")

    set1 = st.number_input("set_data1", value=0, step=1)
    set2 = st.number_input("set_data2", value=0, step=1)
    set3 = st.number_input("set_data3", value=0, step=1)
    set4 = st.number_input("set_data4", value=0, step=1)

    if st.button("Отправить команды"):
        responses = []

        responses.append(send_set_data(1, int(set1), data_source))
        responses.append(send_set_data(2, int(set2), data_source))
        responses.append(send_set_data(3, int(set3), data_source))
        responses.append(send_set_data(4, int(set4), data_source))

        st.success("Команды отправлены")
        st.write(responses)
    

    if st.button("Очистить карту"):
        st.session_state.global_map_x = []
        st.session_state.global_map_y = []
        st.session_state.global_map_colors = []
        st.session_state.trajectory = []
        st.rerun()
    if st.button("Сохранить карту"):
        import os

        map_df = pd.DataFrame({
            "x": st.session_state.global_map_x,
            "y": st.session_state.global_map_y,
            "distance": st.session_state.global_map_colors
        })

        map_df.to_csv("map_data.csv", index=False)

        path = os.path.abspath("map_data.csv")
        st.success(f"Карта сохранена: {path}")


# ----------------------------
# Helper functions
# ----------------------------


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


# ----------------------------
# Read data
# ----------------------------

if st.session_state.connected:
    raw_state = read_robot_state(data_source)
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

if raw_state is None:
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


# ----------------------------
# Update map data
# ----------------------------

if st.session_state.connected:
    st.session_state.trajectory.append((robot_x, robot_y))

lidar_df = lidar_to_dataframe(raw_state.get("lidar", []))

if show_lidar and st.session_state.connected and not lidar_df.empty:
    yaw_rad = np.radians(yaw)

    lidar_df["global_x"] = (
        robot_x
        + lidar_df["x"] * np.cos(yaw_rad)
        - lidar_df["y"] * np.sin(yaw_rad)
    )

    lidar_df["global_y"] = (
        robot_y
        + lidar_df["x"] * np.sin(yaw_rad)
        + lidar_df["y"] * np.cos(yaw_rad)
    )

    st.session_state.global_map_x.extend(lidar_df["global_x"].tolist())
    st.session_state.global_map_y.extend(lidar_df["global_y"].tolist())
    st.session_state.global_map_colors.extend(lidar_df["distance"].tolist())

st.session_state.global_map_x = st.session_state.global_map_x[-max_points:]
st.session_state.global_map_y = st.session_state.global_map_y[-max_points:]
st.session_state.global_map_colors = st.session_state.global_map_colors[-max_points:]
st.session_state.trajectory = st.session_state.trajectory[-300:]


# ----------------------------
# Metrics
# ----------------------------

if show_metrics:

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    sensor_status = "OK" if st.session_state.connected and not lidar_df.empty else "NO DATA"

    col1.metric("Батарея", f"{battery}%")
    col2.metric("Скорость", f"{speed} м/с")
    col3.metric("Рыскание", f"{yaw}°")
    col4.metric("Наклон", f"{tilt}°")
    col5.metric("Сенсоры", sensor_status)
    col6.metric("Канал", data_source)

if show_metrics and st.session_state.connected and not lidar_df.empty:
    with st.expander("Таблица данных лидара"):
        st.dataframe(lidar_df, use_container_width=True)


# ----------------------------
# Visualization
# ----------------------------

fig = go.Figure()

if show_lidar and st.session_state.global_map_x:
    point_count = len(st.session_state.global_map_x)

    opacity_values = [
        0.2 + 0.8 * (i / max(point_count - 1, 1))
        for i in range(point_count)
    ]

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
                colorbar=dict(title="Distance, mm"),
                opacity=opacity_values
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
    title=f"Real-Time Robot Map | Points: {len(st.session_state.global_map_x)}",
    xaxis_title="X, mm",
    yaxis_title="Y, mm",
    height=700,
    xaxis=dict(
        range=[-6000, 6000],
        scaleanchor="y",
        fixedrange=True
    ),
    yaxis=dict(
        range=[-6000, 6000],
        fixedrange=True
    ),
    dragmode=False,
    legend=dict(x=0.02, y=0.98),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="white")
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": False,
        "displayModeBar": False
    }
)

if show_cv_map and st.session_state.global_map_x:

    cv_map, cv_stats = build_occupancy_map(
        st.session_state.global_map_x,
        st.session_state.global_map_y,
        st.session_state.global_map_colors,
        danger_radius_mm=danger_radius_mm,
        min_distance_mm=min_distance_mm,
        max_distance_mm=max_distance_mm
    )

    st.subheader("OpenCV Occupancy Map")

    c1, c2, c3 = st.columns(3)

    c1.metric("Контуры препятствий", cv_stats["obstacle_count"])

    nearest = cv_stats["nearest_obstacle_mm"]

    if cv_stats["danger_detected"]:
        now = datetime.now()
        should_log_event = False

        if st.session_state.last_danger_event_time is None:
            should_log_event = True
        else:
            time_delta = (now - st.session_state.last_danger_event_time).total_seconds()
            should_log_event = time_delta >= 5

        if should_log_event:
            event = {
                "time": now.strftime("%H:%M:%S"),
                "nearest_obstacle_mm": nearest,
                "danger_radius_mm": danger_radius_mm
            }

            st.session_state.danger_events.append(event)
            st.session_state.danger_events = st.session_state.danger_events[-20:]
            st.session_state.last_danger_event_time = now

    c2.metric(
        "Ближайшее препятствие",
        f"{nearest} мм" if nearest is not None else "нет данных"
    )

    if cv_stats["danger_detected"]:
        c3.error("Опасная зона: препятствие близко")
    else:
        c3.success("Опасная зона свободна")

    st.image(
        cv_map,
        caption="OpenCV-карта: красный — опасно близко, жёлтый — средне, зелёный — далеко",
        use_container_width=False
    )

    if st.button("Сохранить OpenCV-карту PNG"):
        filename = f"opencv_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        cv2.imwrite(filename, cv2.cvtColor(cv_map, cv2.COLOR_RGB2BGR))

        st.success(f"OpenCV-карта сохранена: {filename}")

    if st.session_state.danger_events:
        with st.expander("Журнал опасных событий"):
            danger_df = pd.DataFrame(st.session_state.danger_events)

            st.dataframe(
                danger_df,
                use_container_width=True
            )

            col_save_log, col_clear_log = st.columns(2)

            with col_save_log:
                if st.button("Сохранить журнал CSV"):
                    filename = f"danger_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    danger_df.to_csv(filename, index=False)
                    st.success(f"Журнал сохранён: {filename}")

            with col_clear_log:
                if st.button("Очистить журнал"):
                    st.session_state.danger_events = []
                    st.session_state.last_danger_event_time = None
                    st.rerun()


# ----------------------------
# Auto update
# ----------------------------

if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()