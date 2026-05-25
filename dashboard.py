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
from clean_map import CleanMap
import matplotlib.pyplot as plt


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

# Новая чистая карта - увеличенный размер
if "clean_map" not in st.session_state:
    st.session_state.clean_map = CleanMap(map_size_mm=20000, cell_size_mm=25)


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
        index=1
    )

    uploaded_lidar_file = st.file_uploader(
        "Загрузить HEX-файл Delta2D", type=["txt"],
        help="Загрузите файл с данными лидара в HEX-формате"
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

    # Настройки четкой карты
    st.subheader("🗺️ Настройки четкой карты")
    show_clean_map = st.checkbox("Показывать четкую 2D карту", value=True)

    col_cell, col_clear = st.columns(2)
    with col_cell:
        cell_size = st.select_slider(
            "Размер ячейки (мм)",
            options=[25, 50, 75, 100],
            value=25,
            help="Меньше = детальнее"
        )
    with col_clear:
        if st.button("🗑️ Очистить карту"):
            st.session_state.clean_map.clear()
            st.rerun()

    # Обновление размера ячейки
    if cell_size != st.session_state.clean_map.cell_size_mm:
        st.session_state.clean_map = CleanMap(map_size_mm=20000, cell_size_mm=cell_size)

    col_export_png, col_export_csv = st.columns(2)
    with col_export_png:
        if st.button("📸 Экспорт PNG"):
            png_file, _ = st.session_state.clean_map.export_map("clean_map")
            st.success(f"Сохранено: {png_file}")
    with col_export_csv:
        if st.button("📊 Экспорт CSV"):
            _, csv_file = st.session_state.clean_map.export_map("clean_map")
            st.success(f"Сохранено: {csv_file}")

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

    if st.button("Очистить глобальную карту"):
        st.session_state.global_map_x = []
        st.session_state.global_map_y = []
        st.session_state.global_map_colors = []
        st.session_state.trajectory = []
        st.session_state.occupancy_grid = init_occupancy_grid()
        st.rerun()

    if st.button("Сохранить глобальную карту"):
        import os
        map_df = pd.DataFrame({
            "x": st.session_state.global_map_x,
            "y": st.session_state.global_map_y,
            "distance": st.session_state.global_map_colors
        })
        map_df.to_csv("map_data.csv", index=False)
        st.success(f"Карта сохранена: {os.path.abspath('map_data.csv')}")


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
    xs, ys = [], []
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
        "x": 0, "y": 0, "yaw": 0, "speed": 0,
        "battery": 0, "tilt": 0, "lidar": []
    }

if raw_state is None:
    raw_state = {
        "x": 0, "y": 0, "yaw": 0, "speed": 0,
        "battery": 0, "tilt": 0, "lidar": []
    }

# Если загружен HEX-файл
if uploaded_lidar_file is not None:
    hex_text = uploaded_lidar_file.read().decode("utf-8")
    parsed_lidar = parse_lidar_data(hex_text, lidar_model)
    raw_state["lidar"] = parsed_lidar
    if not st.session_state.connected:
        raw_state["x"] = 0
        raw_state["y"] = 0
        raw_state["yaw"] = 0

lidar_data_available = st.session_state.connected or uploaded_lidar_file is not None

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
        "speed": speed, "battery": battery, "yaw": yaw,
        "tilt": tilt, "latency_ms": read_latency_ms,
        "lidar_points": lidar_points_count
    })
    st.session_state.telemetry_history = st.session_state.telemetry_history[-100:]

if record_telemetry and st.session_state.connected:
    telemetry_row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "x_mm": robot_x, "y_mm": robot_y, "yaw_deg": yaw,
        "speed_m_s": speed, "battery_percent": battery,
        "tilt_deg": tilt, "lidar_points": lidar_points_count,
        "data_source": data_source, "read_latency_ms": read_latency_ms
    }])
    telemetry_row.to_csv("telemetry_log.csv", mode="a", 
                         header=not pd.io.common.file_exists("telemetry_log.csv"), 
                         index=False)


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

# Обработка данных для глобальной карты
if lidar_data_available and not lidar_df.empty:
    if uploaded_lidar_file is not None and not st.session_state.connected:
        lidar_df["global_x"] = lidar_df["x"]
        lidar_df["global_y"] = lidar_df["y"]
        st.session_state.global_map_x.extend(lidar_df["global_x"].tolist())
        st.session_state.global_map_y.extend(lidar_df["global_y"].tolist())
        st.session_state.global_map_colors.extend(lidar_df["distance"].tolist())
        if not st.session_state.trajectory:
            st.session_state.trajectory.append((0, 0))
    else:
        yaw_rad = np.radians(yaw)
        lidar_df["global_x"] = robot_x + lidar_df["x"] * np.cos(yaw_rad) - lidar_df["y"] * np.sin(yaw_rad)
        lidar_df["global_y"] = robot_y + lidar_df["x"] * np.sin(yaw_rad) + lidar_df["y"] * np.cos(yaw_rad)
        st.session_state.global_map_x.extend(lidar_df["global_x"].tolist())
        st.session_state.global_map_y.extend(lidar_df["global_y"].tolist())
        st.session_state.global_map_colors.extend(lidar_df["distance"].tolist())

st.session_state.global_map_x = st.session_state.global_map_x[-max_points:]
st.session_state.global_map_y = st.session_state.global_map_y[-max_points:]
st.session_state.global_map_colors = st.session_state.global_map_colors[-max_points:]
st.session_state.trajectory = st.session_state.trajectory[-300:]


# ============================================
# Чистая карта - обновление
# ============================================

if show_clean_map and lidar_data_available and not lidar_df.empty:
    try:
        filtered_df = lidar_df[lidar_df['distance'] <= max_distance_mm]
        st.session_state.clean_map.update(filtered_df)
    except Exception as e:
        st.error(f"Ошибка обновления карты: {e}")


# ============================================
# Metrics display
# ============================================

if show_metrics:
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    sensor_status = "OK" if lidar_data_available and not lidar_df.empty else "NO DATA"
    
    if uploaded_lidar_file is not None and not st.session_state.connected:
        col1.metric("Режим", "HEX файл")
        col2.metric("Точек лидара", lidar_points_count)
        col3.metric("Диапазон", f"{min_distance_mm}-{max_distance_mm} мм")
        col4.metric("Всего точек", len(st.session_state.global_map_x))
    else:
        col1.metric("Батарея", f"{battery}%")
        col2.metric("Скорость", f"{speed} м/с")
        col3.metric("Рыскание", f"{yaw}°")
        col4.metric("Наклон", f"{tilt}°")
    
    col5.metric("Сенсоры", sensor_status)
    col6.metric("Канал", "FILE" if uploaded_lidar_file else data_source)
    col7.metric("Точек", lidar_points_count)
    col8.metric("Задержка", f"{read_latency_ms} мс")
    col9.metric("Пакет", last_packet_time)


# ============================================
# Main visualization (Plotly)
# ============================================

fig = go.Figure()

if show_lidar and st.session_state.global_map_x:
    point_count = len(st.session_state.global_map_x)
    opacity_values = [0.2 + 0.8 * (i / max(point_count - 1, 1)) for i in range(point_count)]
    fig.add_trace(go.Scatter(
        x=st.session_state.global_map_x, y=st.session_state.global_map_y,
        mode="markers",
        marker=dict(size=5, color=st.session_state.global_map_colors, colorscale="Viridis",
                   showscale=True, colorbar=dict(title="Distance, mm"), opacity=opacity_values),
        name="Global map"
    ))

if show_room_contour and not room_df.empty:
    fig.add_trace(go.Scatter(
        x=robot_x + room_df["x"], y=robot_y + room_df["y"],
        mode="lines", name="Room contour", line=dict(width=3)
    ))

if show_trajectory and st.session_state.trajectory:
    tx = [p[0] for p in st.session_state.trajectory]
    ty = [p[1] for p in st.session_state.trajectory]
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", name="Trajectory"))

if show_robot:
    if uploaded_lidar_file is not None and not st.session_state.connected:
        robot_x_display, robot_y_display, yaw_display = 0, 0, 0
    else:
        robot_x_display, robot_y_display, yaw_display = robot_x, robot_y, yaw
    
    rx, ry = robot_shape(robot_x_display, robot_y_display, yaw_display)
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name="Robot body", line=dict(width=3)))
    
    yaw_rad = math.radians(yaw_display)
    fig.add_trace(go.Scatter(
        x=[robot_x_display, robot_x_display + math.cos(yaw_rad) * 450],
        y=[robot_y_display, robot_y_display + math.sin(yaw_rad) * 450],
        mode="lines+markers", name="Direction", line=dict(width=3)
    ))

if (show_cv_map and lidar_data_available and not lidar_df.empty and "global_x" in lidar_df.columns):
    lidar_df["range_from_robot"] = np.sqrt((lidar_df["global_x"] - robot_x) ** 2 + (lidar_df["global_y"] - robot_y) ** 2)
    nearest_idx = lidar_df["range_from_robot"].idxmin()
    nearest_point = lidar_df.loc[nearest_idx]
    
    fig.add_shape(type="circle", xref="x", yref="y",
                 x0=robot_x - danger_radius_mm, y0=robot_y - danger_radius_mm,
                 x1=robot_x + danger_radius_mm, y1=robot_y + danger_radius_mm,
                 line=dict(width=2, dash="dash"))
    fig.add_trace(go.Scatter(x=[nearest_point["global_x"]], y=[nearest_point["global_y"]],
                            mode="markers", marker=dict(size=14, symbol="x"), name="Nearest obstacle"))

fig.update_layout(
    title=f"Robot Map | Points: {len(st.session_state.global_map_x)}",
    xaxis_title="X, mm", yaxis_title="Y, mm", height=700,
    xaxis=dict(range=[-10000, 10000], scaleanchor="y", fixedrange=True),
    yaxis=dict(range=[-10000, 10000], fixedrange=True),
    dragmode=False, legend=dict(x=0.02, y=0.98),
    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})


# ============================================
# OpenCV Map
# ============================================

if show_cv_map and st.session_state.global_map_x:
    cv_map, cv_stats = build_occupancy_map(
        st.session_state.global_map_x, st.session_state.global_map_y, st.session_state.global_map_colors,
        danger_radius_mm=danger_radius_mm, min_distance_mm=min_distance_mm, max_distance_mm=max_distance_mm
    )

    st.subheader("OpenCV Occupancy Map")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Контуры препятствий", cv_stats["obstacle_count"])

    nearest = cv_stats["nearest_obstacle_mm"]
    if nearest and nearest <= danger_radius_mm and speed > 0.4:
        collision_risk = "HIGH"
    elif nearest and nearest <= danger_radius_mm:
        collision_risk = "MEDIUM"
    else:
        collision_risk = "LOW"

    if cv_stats["danger_detected"]:
        now = datetime.now()
        if st.session_state.last_danger_event_time is None or (now - st.session_state.last_danger_event_time).total_seconds() >= 5:
            st.session_state.danger_events.append({"time": now.strftime("%H:%M:%S"), "nearest_obstacle_mm": nearest, "danger_radius_mm": danger_radius_mm})
            st.session_state.danger_events = st.session_state.danger_events[-20:]
            st.session_state.last_danger_event_time = now

    c2.metric("Ближайшее препятствие", f"{nearest} мм" if nearest else "нет данных")

    if cv_stats["danger_detected"]:
        c3.error("Опасная зона!")
    else:
        c3.success("Зона свободна")
    
    if collision_risk == "HIGH":
        c4.error("Риск: HIGH")
    elif collision_risk == "MEDIUM":
        c4.warning("Риск: MEDIUM")
    else:
        c4.success("Риск: LOW")

    st.image(cv_map, caption="OpenCV-карта", use_container_width=False)


# ============================================
# Occupancy Grid
# ============================================

if show_occupancy_grid and not lidar_df.empty and "global_x" in lidar_df.columns:
    st.session_state.occupancy_grid = update_occupancy_grid(
        st.session_state.occupancy_grid, lidar_df,
        robot_x=robot_x, robot_y=robot_y,
        min_distance_mm=min_distance_mm, max_distance_mm=max_distance_mm
    )
    occupancy_image = render_occupancy_grid(st.session_state.occupancy_grid, robot_x=robot_x, robot_y=robot_y)
    st.subheader("Occupancy Grid Map")
    st.image(occupancy_image, use_container_width=False)







# ============================================
# ОТЛАДКА - что реально в данных
# ============================================

if lidar_data_available and not lidar_df.empty:
    with st.expander("🔍 Отладка данных лидара", expanded=True):
        st.write(f"**Всего точек:** {len(lidar_df)}")
        st.write(f"**X (мм):** мин={lidar_df['x'].min():.1f}, макс={lidar_df['x'].max():.1f}")
        st.write(f"**Y (мм):** мин={lidar_df['y'].min():.1f}, макс={lidar_df['y'].max():.1f}")
        st.write(f"**Расстояния (мм):** мин={lidar_df['distance'].min():.1f}, макс={lidar_df['distance'].max():.1f}")
        
        # Простой scatter plot точек
        fig_debug, ax_debug = plt.subplots(figsize=(8, 8))
        scatter = ax_debug.scatter(lidar_df['x'], lidar_df['y'], c=lidar_df['distance'], 
                                   cmap='viridis', s=1, alpha=0.5)
        ax_debug.set_title("Сырые точки лидара (X, Y)")
        ax_debug.set_xlabel("X (мм)")
        ax_debug.set_ylabel("Y (мм)")
        ax_debug.set_aspect('equal')
        plt.colorbar(scatter, ax=ax_debug, label='Расстояние (мм)')
        st.pyplot(fig_debug)
        plt.close(fig_debug)
        
        # Полярный график
        fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        angles_rad = np.radians(lidar_df['angle'])
        ax_polar.scatter(angles_rad, lidar_df['distance'], c=lidar_df['distance'], 
                        cmap='viridis', s=1, alpha=0.5)
        ax_polar.set_title("Полярный график (угол vs расстояние)")
        ax_polar.set_rmax(8000)
        st.pyplot(fig_polar)
        plt.close(fig_polar)
        
        # Проверка распределения углов
        st.write("### 📐 Проверка углов")
        
        fig_angles, ax_angles = plt.subplots(figsize=(10, 4))
        ax_angles.hist(lidar_df['angle'], bins=36, range=(0, 360), color='purple', alpha=0.7)
        ax_angles.set_title("Распределение углов (должно быть равномерно 0-360°)")
        ax_angles.set_xlabel("Угол (градусы)")
        ax_angles.set_ylabel("Количество точек")
        st.pyplot(fig_angles)
        plt.close(fig_angles)
        
        # Проверка: все ли углы присутствуют
        unique_angles = lidar_df['angle'].unique()
        st.write(f"**Уникальных углов:** {len(unique_angles)} из 360 возможных")
        st.write(f"**Диапазон углов:** {lidar_df['angle'].min():.1f}° - {lidar_df['angle'].max():.1f}°")
        
        if lidar_df['angle'].max() - lidar_df['angle'].min() < 350:
            st.error(f"⚠️ Углы покрывают только {lidar_df['angle'].max() - lidar_df['angle'].min():.0f}°! Нужно 360°")
        else:
            st.success("✅ Углы покрывают полные 360°")
        
        # Выборка точек
        st.write("**Первые 20 точек:**")
        st.dataframe(lidar_df[['angle', 'distance', 'x', 'y']].head(20))


# ============================================
# АНАЛИЗ ФОРМЫ КОМНАТЫ
# ============================================

if lidar_data_available and not lidar_df.empty:
    with st.expander("📐 Анализ формы комнаты", expanded=True):
        st.write("### Анализ данных лидара")
        
        # Основная статистика
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего точек", len(lidar_df))
        with col2:
            st.metric("Мин. расстояние", f"{lidar_df['distance'].min():.0f} мм")
        with col3:
            st.metric("Макс. расстояние", f"{lidar_df['distance'].max():.0f} мм")
        with col4:
            st.metric("Среднее расстояние", f"{lidar_df['distance'].mean():.0f} мм")
        
        # 1. Ближние точки (0.5-2 м) - возможно мебель
        st.write("### 1. Ближние объекты (0.5 - 2 метра)")
        lidar_near = lidar_df[(lidar_df['distance'] > 500) & (lidar_df['distance'] < 2000)]
        
        if len(lidar_near) > 0:
            fig_near, ax_near = plt.subplots(figsize=(8, 8))
            ax_near.scatter(lidar_near['x'], lidar_near['y'], s=10, alpha=0.7, c='red')
            ax_near.set_title(f"Ближние объекты ({len(lidar_near)} точек)\nрасстояние 0.5-2 м")
            ax_near.set_xlabel("X (мм)")
            ax_near.set_ylabel("Y (мм)")
            ax_near.set_aspect('equal')
            ax_near.grid(True, alpha=0.3)
            st.pyplot(fig_near)
            plt.close(fig_near)
        else:
            st.info("Нет ближних объектов (0.5-2 м)")
        
        # 2. Средние точки (2-5 м) - основные стены
        st.write("### 2. Основные стены (2 - 5 метров)")
        lidar_mid = lidar_df[(lidar_df['distance'] >= 2000) & (lidar_df['distance'] < 5000)]
        
        if len(lidar_mid) > 0:
            fig_mid, ax_mid = plt.subplots(figsize=(8, 8))
            ax_mid.scatter(lidar_mid['x'], lidar_mid['y'], s=5, alpha=0.5, c='green')
            ax_mid.set_title(f"Основные стены ({len(lidar_mid)} точек)\nрасстояние 2-5 м")
            ax_mid.set_xlabel("X (мм)")
            ax_mid.set_ylabel("Y (мм)")
            ax_mid.set_aspect('equal')
            ax_mid.grid(True, alpha=0.3)
            st.pyplot(fig_mid)
            plt.close(fig_mid)
        else:
            st.info("Нет точек в диапазоне 2-5 м")
        
        # 3. Дальние точки (5-8 м) - дальние стены/шум
        st.write("### 3. Дальние объекты (5 - 8 метров)")
        lidar_far = lidar_df[(lidar_df['distance'] >= 5000) & (lidar_df['distance'] < 8000)]
        
        if len(lidar_far) > 0:
            fig_far, ax_far = plt.subplots(figsize=(8, 8))
            ax_far.scatter(lidar_far['x'], lidar_far['y'], s=3, alpha=0.3, c='blue')
            ax_far.set_title(f"Дальние объекты ({len(lidar_far)} точек)\nрасстояние 5-8 м")
            ax_far.set_xlabel("X (мм)")
            ax_far.set_ylabel("Y (мм)")
            ax_far.set_aspect('equal')
            ax_far.grid(True, alpha=0.3)
            st.pyplot(fig_far)
            plt.close(fig_far)
        else:
            st.info("Нет точек в диапазоне 5-8 м")
        
        # 4. Анализ углов - где есть стены
        st.write("### 4. Распределение расстояний по направлениям")
        
        # Разбиваем на 8 секторов по 45 градусов
        sectors = []
        sector_names = []
        for i in range(8):
            start_angle = i * 45
            end_angle = (i + 1) * 45
            sector_data = lidar_df[(lidar_df['angle'] >= start_angle) & (lidar_df['angle'] < end_angle)]
            if not sector_data.empty:
                avg_dist = sector_data['distance'].mean()
                sectors.append(avg_dist)
                sector_names.append(f"{start_angle}°-{end_angle}°")
            else:
                sectors.append(0)
                sector_names.append(f"{start_angle}°-{end_angle}°")
        
        # Барчарт направлений
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        bars = ax_bar.bar(sector_names, sectors, color='skyblue')
        ax_bar.set_xlabel("Направление (градусы)")
        ax_bar.set_ylabel("Среднее расстояние (мм)")
        ax_bar.set_title("Среднее расстояние по направлениям")
        ax_bar.set_ylim(0, max(sectors) * 1.2 if sectors else 8000)
        
        # Подписываем значения на барах
        for bar, dist in zip(bars, sectors):
            if dist > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                           f'{dist:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
        
        # 5. Вывод рекомендаций
        st.write("### 📋 Рекомендации")
        
        # Определяем тип данных
        if len(lidar_near) > 100:
            st.success("✅ **Обнаружены ближние объекты** - возможно мебель или стены близко")
        else:
            st.info("ℹ️ **Мало ближних объектов** - комната может быть пустой или лидар в центре")
        
        if len(lidar_mid) > 500:
            st.success("✅ **Обнаружены стены** на расстоянии 2-5 метров")
        else:
            st.warning("⚠️ **Мало данных о стенах** - возможно лидар стоит в центре большого помещения")
        
        # Определяем форму
        max_dist = max(sectors) if sectors else 0
        min_dist = min([d for d in sectors if d > 0]) if sectors else 0
        ratio = max_dist / min_dist if min_dist > 0 else 1
        
        if ratio < 1.2:
            st.info("📊 **Форма: круглая** - данные показывают почти одинаковые расстояния во всех направлениях")
        elif ratio < 1.8:
            st.info("📊 **Форма: квадратная/прямоугольная** - есть небольшая разница в расстояниях")
        else:
            st.info("📊 **Форма: вытянутая** - сильная разница в расстояниях по направлениям")
        
        st.caption(f"Соотношение макс/мин расстояний: {ratio:.2f}")
        
        # 6. Предложение преобразования
        st.write("### 🔧 Что делать дальше?")
        
        if ratio < 1.3:
            st.markdown("""
            **Ваши данные показывают круглую форму.** Это может означать:
            1. Лидар стоит в центре круглой комнаты
            2. Лидар сканировал на открытом пространстве
            3. Нет препятствий близко к лидару
            
            **Рекомендации:**
            - Поставьте лидар в **угол комнаты** и запишите новые данные
            - Или поставьте **препятствия** (коробки, стулья) вокруг лидара
            - Или используйте эти данные как базовую карту открытого пространства
            """)
        else:
            st.markdown("""
            **Ваши данные показывают форму помещения!** 
            - Используйте настройки фильтрации выше для улучшения карты
            - Экспортируйте карту в PNG/CSV для дальнейшего использования
            """)


# ============================================
# ПРЕОБРАЗОВАНИЕ КРУГА В КВАДРАТНУЮ КОМНАТУ (ДЕМО)
# ============================================

if lidar_data_available and not lidar_df.empty:
    with st.expander("🏠 Демо: Преобразование в квадратную комнату", expanded=True):
        st.warning("⚠️ Это демонстрационное преобразование! Для реальной карты нужны новые данные.")
        
        # Преобразуем круговые данные в квадратные
        lidar_demo = lidar_df.copy()
        
        # Искусственно создаем квадратную форму
        x_demo = []
        y_demo = []
        
        for idx, row in lidar_demo.iterrows():
            angle_rad = np.radians(row['angle'])
            dist = row['distance']
            
            # Преобразуем круг в квадрат
            corner_factor = 1 / (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad)))
            new_dist = dist * min(1.0, corner_factor * 0.7)
            
            x_demo.append(new_dist * np.cos(angle_rad))
            y_demo.append(new_dist * np.sin(angle_rad))
        
        # Ограничиваем размер комнаты
        x_demo = np.array(x_demo)
        y_demo = np.array(y_demo)
        mask = (np.abs(x_demo) < 4000) & (np.abs(y_demo) < 4000)
        x_demo = x_demo[mask]
        y_demo = y_demo[mask]
        
        fig_demo, ax_demo = plt.subplots(figsize=(8, 8))
        ax_demo.scatter(x_demo, y_demo, s=2, alpha=0.5, c='blue')
        ax_demo.set_title("ДЕМО: Искусственная квадратная комната\n(для визуализации, не реальные данные!)")
        ax_demo.set_xlabel("X (мм)")
        ax_demo.set_ylabel("Y (мм)")
        ax_demo.set_aspect('equal')
        ax_demo.set_xlim(-4500, 4500)
        ax_demo.set_ylim(-4500, 4500)
        ax_demo.grid(True, alpha=0.3)
        st.pyplot(fig_demo)
        plt.close(fig_demo)
        
        st.info("""
        **📌 Пояснение:**
        - Это **демонстрация**, а не реальные данные лидара
        - Исходные данные были круглыми, мы преобразовали их в квадратную форму
        - **Для реальной карты комнаты** нужно:
          1. Установить лидар в УГЛУ комнаты
          2. Записать новые данные
          3. Повторить сканирование
        """)

# ============================================
# Чистая 2D карта - улучшенное отображение
# ============================================

if show_clean_map:
    st.markdown("---")
    st.header("🗺️ Чистая 2D карта помещения")
    
    if lidar_data_available and not lidar_df.empty:
        try:
            stats = st.session_state.clean_map.get_statistics()
            
            # Показываем реальные размеры комнаты
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Точек на карте", stats['points_count'])
            with col_b:
                if stats['room_width_mm'] > 0:
                    st.metric("Ширина комнаты", f"{stats['room_width_mm']/1000:.1f} м")
                else:
                    st.metric("Ширина комнаты", "—")
            with col_c:
                if stats['room_height_mm'] > 0:
                    st.metric("Высота комнаты", f"{stats['room_height_mm']/1000:.1f} м")
                else:
                    st.metric("Высота комнаты", "—")
            with col_d:
                st.metric("Препятствий", stats['obstacle_count'])
            
            # Отображаем карту с автоцентрированием
            fig = st.session_state.clean_map.render_with_matplotlib(figsize=(8, 8))
            st.pyplot(fig)
            plt.close(fig)
            
            # Легенда
            st.caption("""
            📖 **Легенда:**
            - ⬛ **Черный** — стены и препятствия (данные лидара)
            - ⬜ **Белый/Серый** — свободное/неизвестное пространство
            - 🔴 **Красные линии** — оси X и Y (центр карты)
            """)
            
            # Кнопки управления
            col_exp, col_clr = st.columns(2)
            with col_exp:
                if st.button("💾 Экспортировать карту (PNG)"):
                    png_file, _ = st.session_state.clean_map.export_map("clean_map")
                    st.success(f"Сохранено: {png_file}")
            with col_clr:
                if st.button("🗑️ Очистить карту и начать заново"):
                    st.session_state.clean_map.clear()
                    st.rerun()
                
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.code(f"{e}")
    else:
        st.info("📡 Нет данных лидара. Загрузите HEX файл или подключитесь к роботу.")


# ============================================
# Auto update
# ============================================

if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()