import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lidar_parser import parse_lidar_data
import math

st.set_page_config(page_title="HEX LiDAR Viewer", layout="wide")
st.title("📡 Визуализация HEX-файла лидара")

uploaded_file = st.file_uploader("Загрузите HEX-файл Delta2D", type=["txt"])

if uploaded_file is not None:
    hex_text = uploaded_file.read().decode("utf-8")
    
    lidar_model = st.radio("Модель лидара", ["Delta2A", "Delta2D"], index=1)
    
    if st.button("Парсить и показать"):
        with st.spinner("Парсинг данных..."):
            points = parse_lidar_data(hex_text, lidar_model)
        
        st.success(f"✅ Получено {len(points)} точек")
        
        if points:
            # Преобразуем в DataFrame
            df = pd.DataFrame(points)
            
            # Добавляем декартовы координаты
            df["x"] = df["distance"] * np.cos(np.radians(df["angle"]))
            df["y"] = df["distance"] * np.sin(np.radians(df["angle"]))
            
            # Отображаем таблицу
            with st.expander("Таблица данных"):
                st.dataframe(df)
            
            # Статистика
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Количество точек", len(df))
            col2.metric("Мин. дистанция", f"{df['distance'].min():.0f} мм")
            col3.metric("Макс. дистанция", f"{df['distance'].max():.0f} мм")
            col4.metric("Средняя дистанция", f"{df['distance'].mean():.0f} мм")
            
            # 2D график
            st.subheader("Карта точек лидара")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df["distance"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Distance, mm")
                ),
                text=[f"Угол: {a:.1f}°, Дист: {d:.0f} мм" for a, d in zip(df["angle"], df["distance"])],
                hoverinfo="text"
            ))
            
            # Добавляем робота в центр
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=15, symbol="circle", color="red"),
                name="Робот"
            ))
            
            fig.update_layout(
                title="LiDAR Scan",
                xaxis_title="X, мм",
                yaxis_title="Y, мм",
                height=700,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Полярный график
            st.subheader("Полярный график")
            fig_polar = go.Figure()
            fig_polar.add_trace(go.Scatterpolar(
                r=df["distance"],
                theta=df["angle"],
                mode="markers",
                marker=dict(size=5)
            ))
            fig_polar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, df["distance"].max() + 500])
                ),
                height=600
            )
            st.plotly_chart(fig_polar, use_container_width=True)
        else:
            st.error("Не удалось распарсить данные. Проверьте формат файла.")

# Инструкция
with st.expander("📖 Инструкция"):
    st.markdown("""
    1. Выберите файл `Delta2D_HEX_Data.txt`
    2. Выберите модель "Delta2D"
    3. Нажмите "Парсить и показать"
    
    **Формат ожидаемых данных:**
    - Пакеты начинаются с `AA 55`
    - Каждый пакет содержит 15 измерений
    - Расстояния в миллиметрах (Little Endian)
    """)