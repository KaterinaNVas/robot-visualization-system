# test_map.py
import streamlit as st
import numpy as np
import pandas as pd
from enhanced_occupancy_grid import EnhancedOccupancyGrid

st.set_page_config(page_title="Тест карты", layout="wide")
st.title("🧪 Тестирование Enhanced Occupancy Grid")

# Инициализация
if 'grid' not in st.session_state:
    st.session_state.grid = EnhancedOccupancyGrid(
        map_size_mm=12000,
        cell_size_mm=50,
        robot_radius_mm=250
    )

# Генерируем тестовые данные
if 'test_points' not in st.session_state:
    # Создаем квадратную комнату
    points = []
    for angle in range(0, 360, 3):
        rad = np.radians(angle)
        # Имитируем стены на расстоянии 2500 мм
        distance = 2500 / max(abs(np.cos(rad)), abs(np.sin(rad)))
        points.append({
            'angle': angle,
            'distance': min(distance, 3000),
            'x': distance * np.cos(rad),
            'y': distance * np.sin(rad),
            'global_x': distance * np.cos(rad),
            'global_y': distance * np.sin(rad)
        })
    st.session_state.test_points = pd.DataFrame(points)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("➕ Обновить карту"):
        for _ in range(10):
            st.session_state.grid.update(st.session_state.test_points, 0, 0, 0)
        st.rerun()

with col2:
    if st.button("🗑️ Очистить карту"):
        st.session_state.grid.clear()
        st.rerun()

with col3:
    if st.button("💾 Экспорт"):
        png, csv = st.session_state.grid.export_map("test_map")
        st.success(f"Сохранено: {png}, {csv}")

with col4:
    if st.button("📊 Статистика"):
        stats = st.session_state.grid.get_statistics()
        st.json(stats)

# Отображаем карту
img = st.session_state.grid.render_as_image(
    show_robot=True,
    robot_x=0,
    robot_y=0,
    robot_yaw_deg=45,
    image_scale=4
)

st.image(img, use_container_width=True)

# Информация
st.info("""
**Как это работает:**
- Карта создается на основе тестовых данных (квадратная комната 5×5 м)
- Каждое нажатие "Обновить" добавляет 10 слоев наблюдений
- Черный = препятствия, белый = свободно, серый = неизвестно
- Синий круг = положение робота
""")