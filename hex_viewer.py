import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from lidar_parser import parse_lidar_data

st.set_page_config(
    page_title="HEX LiDAR Viewer",
    layout="wide"
)

st.title("📡 Визуализация HEX-файла лидара Delta-2D")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    lidar_model = st.radio(
        "Модель лидара",
        ["Delta2A", "Delta2D"],
        index=1,
        help="Для HEX файлов используйте Delta2D"
    )
    
    st.divider()
    
    st.subheader("🎨 Отображение")
    
    show_robot = st.checkbox("Показывать робота", value=True)
    show_grid = st.checkbox("Показывать сетку", value=True)
    
    point_size = st.slider("Размер точек", min_value=2, max_value=10, value=5)
    
    st.divider()
    
    st.subheader("📊 Фильтры")
    
    min_distance = st.slider("Мин. дистанция (мм)", 0, 1000, 100)
    max_distance = st.slider("Макс. дистанция (мм)", 1000, 10000, 8000)
    
    st.divider()
    
    st.subheader("🎯 Границы карты")
    map_limit = st.slider("Граница карты (мм)", 2000, 10000, 6000, step=500)

# Загрузка файла
uploaded_file = st.file_uploader(
    "📁 Загрузите HEX-файл Delta2D",
    type=["txt", "hex"],
    help="Файл должен содержать HEX-данные в формате Delta2D"
)

if uploaded_file is not None:
    # Читаем и парсим файл
    hex_text = uploaded_file.read().decode("utf-8")
    
    with st.spinner("🔄 Парсинг данных..."):
        points = parse_lidar_data(hex_text, lidar_model)
    
    if points:
        st.success(f"✅ Успешно загружено **{len(points)}** точек лидара")
        
        # Создаем DataFrame
        df = pd.DataFrame(points)
        
        # Фильтруем по дистанции
        df_filtered = df[
            (df["distance"] >= min_distance) & 
            (df["distance"] <= max_distance)
        ].copy()
        
        st.info(f"📊 После фильтрации: **{len(df_filtered)}** точек")
        
        # Вычисляем декартовы координаты
        df_filtered["x"] = df_filtered["distance"] * np.cos(np.radians(df_filtered["angle"]))
        df_filtered["y"] = df_filtered["distance"] * np.sin(np.radians(df_filtered["angle"]))
        
        # Статистика
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Всего точек", len(df))
        with col2:
            st.metric("📏 Мин. дистанция", f"{df['distance'].min():.0f} мм")
        with col3:
            st.metric("📐 Макс. дистанция", f"{df['distance'].max():.0f} мм")
        with col4:
            st.metric("📈 Средняя дистанция", f"{df['distance'].mean():.0f} мм")
        with col5:
            st.metric("🔄 Угловой диапазон", f"{df['angle'].min():.0f}° - {df['angle'].max():.0f}°")
        
        st.markdown("---")
        
        # Основная карта
        st.subheader("🗺️ Карта лидара")
        
        fig = go.Figure()
        
        # Добавляем точки лидара
        fig.add_trace(go.Scatter(
            x=df_filtered["x"],
            y=df_filtered["y"],
            mode="markers",
            marker=dict(
                size=point_size,
                color=df_filtered["distance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Distance, mm", x=1.02),
                opacity=0.8,
                line=dict(width=0)
            ),
            text=[f"Угол: {a:.1f}°<br>Дистанция: {d:.0f} мм" 
                  for a, d in zip(df_filtered["angle"], df_filtered["distance"])],
            hoverinfo="text",
            name="LiDAR точки"
        ))
        
        # Добавляем робота в центр
        if show_robot:
            # Рисуем робота
            robot_radius = 200
            theta = np.linspace(0, 2*np.pi, 50)
            robot_x = robot_radius * np.cos(theta)
            robot_y = robot_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=robot_x,
                y=robot_y,
                mode="lines",
                line=dict(color="red", width=3),
                fill="toself",
                fillcolor="rgba(255,0,0,0.3)",
                name="Робот"
            ))
            
            # Стрелка направления (вверх, угол 0°)
            arrow_length = 350
            fig.add_trace(go.Scatter(
                x=[0, arrow_length],
                y=[0, 0],
                mode="lines+markers",
                line=dict(color="red", width=4),
                marker=dict(size=10, symbol="arrow", angleref="previous"),
                name="Направление"
            ))
        
        # Добавляем опорные окружности
        for radius in [1000, 2000, 3000, 4000, 5000]:
            if radius <= map_limit:
                circle_x = radius * np.cos(theta)
                circle_y = radius * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(dash="dot", color="gray", width=1),
                    showlegend=False,
                    hoverinfo="none"
                ))
        
        # Настройки графика
        fig.update_layout(
            title=dict(
                text=f"LiDAR Scan | {len(df_filtered)} точек | Модель: {lidar_model}",
                font=dict(size=16)
            ),
            xaxis=dict(
                title="X, мм",
                range=[-map_limit, map_limit],
                scaleanchor="y",
                scaleratio=1,
                gridcolor="lightgray",
                showgrid=show_grid
            ),
            yaxis=dict(
                title="Y, мм",
                range=[-map_limit, map_limit],
                gridcolor="lightgray",
                showgrid=show_grid
            ),
            height=700,
            hovermode="closest",
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Полярный график
        st.subheader("📊 Полярный график")
        
        fig_polar = go.Figure()
        
        # Группируем по углам для полярного графика
        df_polar = df_filtered.groupby("angle")["distance"].mean().reset_index()
        
        fig_polar.add_trace(go.Scatterpolar(
            r=df_polar["distance"],
            theta=df_polar["angle"],
            mode="markers+lines",
            marker=dict(
                size=8,
                color=df_polar["distance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Distance, mm")
            ),
            line=dict(width=2),
            name="LiDAR scan"
        ))
        
        fig_polar.update_layout(
            title="Полярная диаграмма",
            polar=dict(
                radialaxis=dict(
                    range=[0, df_filtered["distance"].max() + 500],
                    angle=0,
                    side="clockwise"
                ),
                angularaxis=dict(
                    direction="clockwise",
                    tickmode="linear",
                    tick0=0,
                    dtick=45
                )
            ),
            height=600,
            paper_bgcolor="#0E1117",
            font=dict(color="white")
        )
        
        st.plotly_chart(fig_polar, use_container_width=True)
        
        # Таблица с данными
        with st.expander("📋 Показать таблицу данных"):
            st.dataframe(
                df_filtered[["angle", "distance", "x", "y"]].sort_values("angle"),
                use_container_width=True,
                height=400
            )
            
            # Кнопка экспорта
            csv = df_filtered[["angle", "distance", "x", "y"]].to_csv(index=False)
            st.download_button(
                label="📥 Скачать данные как CSV",
                data=csv,
                file_name="lidar_data.csv",
                mime="text/csv"
            )
        
        # Гистограмма дистанций
        st.subheader("📊 Распределение дистанций")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_filtered["distance"],
            nbinsx=30,
            marker_color="blue",
            opacity=0.7
        ))
        fig_hist.update_layout(
            title="Гистограмма дистанций",
            xaxis_title="Дистанция, мм",
            yaxis_title="Количество точек",
            height=400,
            paper_bgcolor="#0E1117",
            font=dict(color="white")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.error("❌ Не удалось распарсить файл. Проверьте формат данных.")
        st.info("""
        **Ожидаемый формат:**
        - Файл должен содержать HEX-данные
        - Каждый пакет начинается с `AA 55`
        - Данные в формате Delta2D
        """)
else:
    # Отображаем инструкцию
    st.info("👈 Загрузите HEX-файл в боковой панели")
    
    with st.expander("📖 Инструкция"):
        st.markdown("""
        ### Как использовать:
        1. Загрузите HEX-файл с данными лидара Delta2D
        2. Выберите модель "Delta2D"
        3. Настройте фильтры и отображение
        4. Изучайте карту, полярный график и таблицу
        
        ### Формат данных:
        - Пакет начинается с `AA 55`
        - Длина блока: `1E` (30 байт)
        - 15 расстояний (по 2 байта, Little Endian)
        - Углы кодируются в стартовом и конечном пакете
        """)
    
    # Пример данных
    st.subheader("📝 Пример HEX-данных:")
    st.code("""
    AA 55 1E 00 00 00 07 B4 D5 CE 07 D8 07 CE 07 DC 
    07 D9 07 CA 07 D8 07 E4 07 E4 07 E3 07 F2 07 F5 
    07 F6 07 05 08 07 08 AA 55 1E 80 07 80 0E B5 DC 
    11 08 26 08 35 08 3A 08 4C 08 54 08 5A 08 63 08 
    86 08 8F 08 9B 08 BA 08 C5 08 CB 08 E9 08
    """, language="text")