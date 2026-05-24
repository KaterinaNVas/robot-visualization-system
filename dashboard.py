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
from lidar_parser import parse_lidar_data
from room_builder import build_room_contour
from occupancy_grid import init_occupancy_grid, update_occupancy_grid, render_occupancy_grid
from enhanced_occupancy_grid import EnhancedOccupancyGrid


st.set_page_config(
    page_title="Robot Visualization System",
    layout="wide"
)

# ============================================
# Session state initialization
# ============================================

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

if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = []

if "occupancy_grid" not in st.session_state:
    st.session_state.occupancy_grid = init_occupancy_grid()

# Enhanced occupancy grid for clean 2D map
if "enhanced_grid" not in st.session_state:
    st.session_state.enhanced_grid = EnhancedOccupancyGrid(
        map_size_mm=12000,
        cell_size_mm=50,
        robot_radius_mm=250,
        min_distance_mm=150,
        max_distance_mm=6000
    )

if "cleaned_trajectory" not in st.session_state:
    st.session_state.cleaned_trajectory = []


# ============================================
# Sidebar
# ============================================

st.title("Система Визуализации Робота")

with st.sidebar:
    st.header("Панель управления")

    data_source = st.radio(
        "Источник данных",
        ["WebSocket", "HTTP"],
        index=0
    )

    lidar_model = st.radio(
        "Модель лидара",
        ["Delta2A", "Delta2D"],
        index=0
    )

    uploaded_lidar_file = st.file_uploader(
        "Загрузить HEX-файл Delta2D", type=["txt"],
        help="Загрузите файл с данными лидара в HEX-формате для визуализации без подключения к роботу"
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

    st.divider()

    # Visualization toggles
    show_lidar = st.checkbox("Показывать лидар", value=True)
    show_trajectory = st.checkbox("Показывать траекторию", value=True)
    show_robot = st.checkbox("Показывать робота", value=True)
    show_metrics = st.checkbox("Показывать метрики", value=True)
    show_cv_map = st.checkbox("Показывать OpenCV-карту", value=True)
    show_room_contour = st.checkbox("Показывать контур комнаты", value=True)
    show_occupancy_grid = st.checkbox("Показывать Occupancy Grid", value=True)

    st.divider()

    # Enhanced map settings
    st.subheader("🗺️ Настройки четкой карты")
    show_enhanced_map = st.checkbox("Показывать четкую 2D карту", value=True)

    col_cell_size, col_robot_radius = st.columns(2)
    with col_cell_size:
        cell_size = st.select_slider(
            "Размер ячейки (мм)",
            options=[25, 50, 75, 100],
            value=50,
            help="Меньше = детальнее, но медленнее"
        )
    with col_robot_radius:
        robot_radius = st.slider(
            "Радиус робота (мм)",
            min_value=150,
            max_value=400,
            value=250,
            step=25
        )

    # Update enhanced grid parameters if changed
    if cell_size != st.session_state.enhanced_grid.cell_size_mm or \
       robot_radius != st.session_state.enhanced_grid.robot_radius_mm:
        st.session_state.enhanced_grid = EnhancedOccupancyGrid(
            map_size_mm=12000,
            cell_size_mm=cell_size,
            robot_radius_mm=robot_radius,
            min_distance_mm=150,
            max_distance_mm=6000
        )
        if st.session_state.cleaned_trajectory:
            st.session_state.cleaned_trajectory = st.session_state.cleaned_trajectory.copy()
        st.rerun()

    col_clear_enhanced, col_export_enhanced = st.columns(2)
    with col_clear_enhanced:
        if st.button("🗑️ Очистить четкую карту"):
            st.session_state.enhanced_grid.clear()
            st.session_state.cleaned_trajectory = []
            st.rerun()
    with col_export_enhanced:
        if st.button("💾 Экспортировать четкую карту"):
            png_file, csv_file = st.session_state.enhanced_grid.export_map("robot_map")
            st.success(f"Сохранено: {png_file} и {csv_file}")

    st.divider()

    # Map settings
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

    record_telemetry = st.checkbox("Записывать телеметрию в CSV", value=False)

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
        st.session_state.occupancy_grid = init_occupancy_grid()
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


# ============================================
# Helper functions
# ============================================

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


# ============================================
# Read data
# ============================================

read_start_time = time.time()

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

# Если загружен HEX-файл, используем его данные вместо данных от робота
if uploaded_lidar_file is not None:
    hex_text = uploaded_lidar_file.read().decode("utf-8")
    parsed_lidar = parse_lidar_data(hex_text, lidar_model)
    raw_state["lidar"] = parsed_lidar
    # При загрузке файла позиционируем робота в центр
    if not st.session_state.connected:
        raw_state["x"] = 0
        raw_state["y"] = 0
        raw_state["yaw"] = 0

lidar_data_available = (
    st.session_state.connected
    or uploaded_lidar_file is not None
)

read_latency_ms = round((time.time() - read_start_time) * 1000, 1)

last_packet_time = datetime.now().strftime("%H:%M:%S")
lidar_points_count = len(raw_state.get("lidar", [])) if raw_state else 0

robot_x = (raw_state.get("x") or 0) * 1000
robot_y = (raw_state.get("y") or 0) * 1000
yaw = raw_state.get("yaw") or 0
speed = raw_state.get("speed") or 0
battery = raw_state.get("battery") or 0
tilt = raw_state.get("tilt") or 0


# ============================================
# Telemetry logging
# ============================================

if st.session_state.connected:
    st.session_state.telemetry_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "speed": speed,
        "battery": battery,
        "yaw": yaw,
        "tilt": tilt,
        "latency_ms": read_latency_ms,
        "lidar_points": lidar_points_count
    })

    st.session_state.telemetry_history = st.session_state.telemetry_history[-100:]

if record_telemetry and st.session_state.connected:
    telemetry_row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "x_mm": robot_x,
        "y_mm": robot_y,
        "yaw_deg": yaw,
        "speed_m_s": speed,
        "battery_percent": battery,
        "tilt_deg": tilt,
        "lidar_points": lidar_points_count,
        "data_source": data_source,
        "read_latency_ms": read_latency_ms
    }])

    telemetry_row.to_csv(
        "telemetry_log.csv",
        mode="a",
        header=not pd.io.common.file_exists("telemetry_log.csv"),
        index=False
    )


# ============================================
# Update map data
# ============================================

if st.session_state.connected:
    st.session_state.trajectory.append((robot_x, robot_y))

lidar_df = lidar_to_dataframe(raw_state.get("lidar", []))
room_df = build_room_contour(
    raw_state.get("lidar", []),
    min_distance_mm=min_distance_mm,
    max_distance_mm=max_distance_mm
)

# Обработка данных лидара для глобальной карты
if lidar_data_available and not lidar_df.empty:
    if uploaded_lidar_file is not None and not st.session_state.connected:
        # Режим загрузки файла: точки уже в локальных координатах робота
        # Робот находится в центре (0,0)
        lidar_df["global_x"] = lidar_df["x"]
        lidar_df["global_y"] = lidar_df["y"]
        
        st.session_state.global_map_x.extend(lidar_df["global_x"].tolist())
        st.session_state.global_map_y.extend(lidar_df["global_y"].tolist())
        st.session_state.global_map_colors.extend(lidar_df["distance"].tolist())
        
        # Добавляем начальную точку траектории (центр)
        if not st.session_state.trajectory:
            st.session_state.trajectory.append((0, 0))
    else:
        # Режим реального робота: преобразуем координаты с учетом положения и поворота
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


# ============================================
# Update enhanced occupancy grid
# ============================================

if st.session_state.connected and not lidar_df.empty:
    st.session_state.cleaned_trajectory.append((robot_x, robot_y))
    max_trajectory = 500
    if len(st.session_state.cleaned_trajectory) > max_trajectory:
        st.session_state.cleaned_trajectory = st.session_state.cleaned_trajectory[-max_trajectory:]

    try:
        st.session_state.enhanced_grid.update(
            lidar_df,
            robot_x,
            robot_y,
            yaw
        )
    except Exception as e:
        st.error(f"Ошибка обновления четкой карты: {e}")


# ============================================
# Metrics display
# ============================================

if show_metrics:
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    sensor_status = "OK" if lidar_data_available and not lidar_df.empty else "NO DATA"
    
    # Для загруженного файла показываем особые метрики
    if uploaded_lidar_file is not None and not st.session_state.connected:
        col1.metric("Режим", "HEX файл")
        col2.metric("Точек лидара", lidar_points_count)
        col3.metric("Диапазон", f"{min_distance_mm}-{max_distance_mm} мм")
        col4.metric("Всего точек на карте", len(st.session_state.global_map_x))
    else:
        col1.metric("Батарея", f"{battery}%")
        col2.metric("Скорость", f"{speed} м/с")
        col3.metric("Рыскание", f"{yaw}°")
        col4.metric("Наклон", f"{tilt}°")
    
    col5.metric("Сенсоры", sensor_status)
    col6.metric("Канал", "FILE" if uploaded_lidar_file else data_source)
    col7.metric("Точек лидара", lidar_points_count)
    col8.metric("Задержка", f"{read_latency_ms} мс")
    col9.metric("Последний пакет", last_packet_time)

if show_metrics and lidar_data_available and not lidar_df.empty:
    with st.expander("Таблица данных лидара"):
        st.dataframe(lidar_df, use_container_width=True)

if show_metrics and st.session_state.telemetry_history:
    with st.expander("Графики телеметрии во времени"):
        telemetry_df = pd.DataFrame(st.session_state.telemetry_history)

        st.line_chart(
            telemetry_df,
            x="time",
            y=["speed", "battery", "latency_ms"],
            use_container_width=True
        )

        st.line_chart(
            telemetry_df,
            x="time",
            y=["yaw", "tilt"],
            use_container_width=True
        )

if show_metrics:
    with st.expander("Состояние сенсоров и телеметрии"):
        connection_ok = st.session_state.connected or uploaded_lidar_file is not None
        lidar_ok = not lidar_df.empty
        position_ok = raw_state.get("x") is not None and raw_state.get("y") is not None
        battery_ok = battery is not None and battery > 0
        telemetry_ok = raw_state is not None and isinstance(raw_state, dict)

        sensor_df = pd.DataFrame([
            {"Модуль": "Connection", "Статус": "OK" if connection_ok else "NO DATA"},
            {"Модуль": "LiDAR", "Статус": "OK" if lidar_ok else "NO DATA"},
            {"Модуль": "Position", "Статус": "OK" if position_ok else "NO DATA"},
            {"Модуль": "Battery", "Статус": "OK" if battery_ok else "NO DATA"},
            {"Модуль": "Telemetry", "Статус": "OK" if telemetry_ok else "NO DATA"},
        ])

        st.dataframe(sensor_df, use_container_width=True)


# ============================================
# Main visualization (Plotly)
# ============================================

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

if show_room_contour and not room_df.empty:
    fig.add_trace(
        go.Scatter(
            x=robot_x + room_df["x"],
            y=robot_y + room_df["y"],
            mode="lines",
            name="Room contour",
            line=dict(width=3)
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
    # Определяем координаты робота для отображения
    if uploaded_lidar_file is not None and not st.session_state.connected:
        robot_x_display = 0
        robot_y_display = 0
        yaw_display = 0
    else:
        robot_x_display = robot_x
        robot_y_display = robot_y
        yaw_display = yaw
    
    rx, ry = robot_shape(robot_x_display, robot_y_display, yaw_display)

    fig.add_trace(
        go.Scatter(
            x=rx,
            y=ry,
            mode="lines",
            name="Robot body",
            line=dict(width=3)
        )
    )

    yaw_rad = math.radians(yaw_display)

    fig.add_trace(
        go.Scatter(
            x=[robot_x_display, robot_x_display + math.cos(yaw_rad) * 450],
            y=[robot_y_display, robot_y_display + math.sin(yaw_rad) * 450],
            mode="lines+markers",
            name="Direction",
            line=dict(width=3)
        )
    )

if (show_cv_map and lidar_data_available and not lidar_df.empty
    and "global_x" in lidar_df.columns and "global_y" in lidar_df.columns):

    lidar_df["range_from_robot"] = np.sqrt(
        (lidar_df["global_x"] - robot_x) ** 2 +
        (lidar_df["global_y"] - robot_y) ** 2
    )

    nearest_idx = lidar_df["range_from_robot"].idxmin()
    nearest_point = lidar_df.loc[nearest_idx]

    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=robot_x - danger_radius_mm,
        y0=robot_y - danger_radius_mm,
        x1=robot_x + danger_radius_mm,
        y1=robot_y + danger_radius_mm,
        line=dict(width=2, dash="dash")
    )

    fig.add_trace(
        go.Scatter(
            x=[nearest_point["global_x"]],
            y=[nearest_point["global_y"]],
            mode="markers",
            marker=dict(size=14, symbol="x"),
            name="Nearest current obstacle"
        )
    )

fig.update_layout(
    title=f"Real-Time Robot Map | Points: {len(st.session_state.global_map_x)}",
    xaxis_title="X, mm",
    yaxis_title="Y, mm",
    height=700,
    xaxis=dict(
        range=[-8000, 8000],
        scaleanchor="y",
        fixedrange=True
    ),
    yaxis=dict(
        range=[-8000, 8000],
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


# ============================================
# OpenCV Map
# ============================================

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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Контуры препятствий", cv_stats["obstacle_count"])

    nearest = cv_stats["nearest_obstacle_mm"]

    collision_risk = "LOW"

    if nearest is not None:
        if nearest <= danger_radius_mm and speed > 0.4:
            collision_risk = "HIGH"
        elif nearest <= danger_radius_mm:
            collision_risk = "MEDIUM"

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

    if collision_risk == "HIGH":
        c4.error("Риск столкновения: HIGH")
    elif collision_risk == "MEDIUM":
        c4.warning("Риск столкновения: MEDIUM")
    else:
        c4.success("Риск столкновения: LOW")

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


# ============================================
# Occupancy Grid (original)
# ============================================

if show_occupancy_grid and not lidar_df.empty and "global_x" in lidar_df.columns:
    st.session_state.occupancy_grid = update_occupancy_grid(
        st.session_state.occupancy_grid,
        lidar_df,
        robot_x=robot_x,
        robot_y=robot_y,
        min_distance_mm=min_distance_mm,
        max_distance_mm=max_distance_mm
    )

    occupancy_image = render_occupancy_grid(
        st.session_state.occupancy_grid,
        robot_x=robot_x,
        robot_y=robot_y
    )

    st.subheader("Occupancy Grid Map")

    st.caption(
        "Тёмное — неизвестная область, "
        "серое — свободное пространство, "
        "чёрное — препятствия, "
        "синее — робот"
    )

    st.image(
        occupancy_image,
        caption="Накопительная карта помещения",
        use_container_width=False
    )


# ============================================
# Enhanced Clean 2D Map
# ============================================

if show_enhanced_map:
    st.markdown("---")
    st.header("🗺️ Четкая 2D карта помещения")

    map_stats = st.session_state.enhanced_grid.get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Размер сетки", f"{map_stats['grid_size']}×{map_stats['grid_size']}")
    with col2:
        st.metric("Разрешение", f"{map_stats['cell_size_mm']} мм/пикс")
    with col3:
        st.metric("Препятствий", map_stats['obstacle_count'])
    with col4:
        st.metric("Покрытие", f"{map_stats['coverage_percent']:.1f}%")

    try:
        # Определяем координаты робота для отображения
        if uploaded_lidar_file is not None and not st.session_state.connected:
            robot_x_display = 0
            robot_y_display = 0
            yaw_display = 0
            trajectory_display = [(0, 0)] if st.session_state.trajectory else None
        else:
            robot_x_display = robot_x
            robot_y_display = robot_y
            yaw_display = yaw
            trajectory_display = st.session_state.cleaned_trajectory if show_trajectory else None
        
        enhanced_map_img = st.session_state.enhanced_grid.render_as_image(
            show_robot=True,
            robot_x=robot_x_display if st.session_state.connected or uploaded_lidar_file else None,
            robot_y=robot_y_display if st.session_state.connected or uploaded_lidar_file else None,
            robot_yaw_deg=yaw_display if st.session_state.connected or uploaded_lidar_file else None,
            show_trajectory=show_trajectory,
            trajectory=trajectory_display,
            image_scale=3
        )

        st.image(enhanced_map_img, use_container_width=True)

        with st.expander("📖 Легенда карты"):
            st.markdown("""
            - **⬛ Черный** — препятствия (стены, объекты)
            - **⬜ Белый** — свободное пространство
            - **⬜ Светло-серый** — неизвестная область
            - **🔵 Синий круг** — положение робота
            - **🟡 Желтая линия** — траектория движения
            """)

        col_png, col_csv, col_stats = st.columns(3)

        with col_png:
            if st.button("📸 Экспортировать как PNG"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clean_map_{timestamp}.png"
                img_to_save = st.session_state.enhanced_grid.render_as_image(
                    show_robot=False, image_scale=1
                )
                cv2.imwrite(filename, img_to_save)
                st.success(f"Сохранено: {filename}")

        with col_csv:
            if st.button("📊 Экспортировать данные CSV"):
                _, csv_file = st.session_state.enhanced_grid.export_map("robot_map")
                st.success(f"Сохранено: {csv_file}")

        with col_stats:
            if st.button("📈 Детальная статистика"):
                st.json(map_stats)

    except Exception as e:
        st.error(f"Ошибка отображения четкой карты: {e}")


# ============================================
# Auto update
# ============================================

if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()