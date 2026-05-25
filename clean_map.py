#!/usr/bin/env python3
"""
Четкая 2D карта помещения - автоцентрирование и фильтрация шума
"""

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


class CleanMap:
    def __init__(self, map_size_mm=12000, cell_size_mm=25):
        self.map_size_mm = map_size_mm
        self.cell_size_mm = cell_size_mm
        self.grid_size = int(map_size_mm / cell_size_mm)
        self.grid = np.full((self.grid_size, self.grid_size), 0, dtype=np.uint8)
        self.offset = map_size_mm // 2
        self.update_count = 0
        self.all_points_x = []
        self.all_points_y = []
    
    def update(self, points_df):
        if points_df is None or points_df.empty:
            return
        
        # Сохраняем все точки для автоцентрирования
        self.all_points_x.extend(points_df['x'].tolist())
        self.all_points_y.extend(points_df['y'].tolist())
        
        # Автоцентрирование: находим центр масс точек
        if len(self.all_points_x) > 100:
            center_x = np.median(self.all_points_x)
            center_y = np.median(self.all_points_y)
            
            # Сдвигаем offset, чтобы центр оказался в середине карты
            self.offset = self.map_size_mm // 2 - center_x
            self.offset_y = self.map_size_mm // 2 - center_y
        else:
            self.offset = self.map_size_mm // 2
            self.offset_y = self.map_size_mm // 2
        
        wall_count = 0
        for _, point in points_df.iterrows():
            ix = int((point['x'] + self.offset) / self.cell_size_mm)
            iy = int((point['y'] + self.offset_y) / self.cell_size_mm)
            
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                if self.grid[iy, ix] != 100:
                    self.grid[iy, ix] = 100
                    wall_count += 1
        
        self.update_count += 1
    
    def clear(self):
        self.grid.fill(0)
        self.all_points_x = []
        self.all_points_y = []
        self.update_count = 0
    
    def render_with_matplotlib(self, figsize=(10, 10)):
        """Визуализация с автоцентрированием и ограничением по данным"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Находим реальные границы по точкам
        if len(self.all_points_x) > 0:
            x_min = np.percentile(self.all_points_x, 2)
            x_max = np.percentile(self.all_points_x, 98)
            y_min = np.percentile(self.all_points_y, 2)
            y_max = np.percentile(self.all_points_y, 98)
            
            # Добавляем отступ 20%
            x_margin = (x_max - x_min) * 0.2
            y_margin = (y_max - y_min) * 0.2
            x_min, x_max = x_min - x_margin, x_max + x_margin
            y_min, y_max = y_min - y_margin, y_max + y_margin
        else:
            x_min, x_max = -5000, 5000
            y_min, y_max = -5000, 5000
        
        # Показываем карту
        im = ax.imshow(self.grid, cmap='gray', origin='lower',
                       extent=[-self.map_size_mm/2, self.map_size_mm/2,
                              -self.map_size_mm/2, self.map_size_mm/2])
        
        # Устанавливаем границы по реальным данным
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('X (мм)', fontsize=12)
        ax.set_ylabel('Y (мм)', fontsize=12)
        ax.set_title(f'Карта помещения\n{self.grid_size}×{self.grid_size} | {self.cell_size_mm} мм/пикс', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='r', linestyle='-', alpha=0.3, linewidth=0.5)
        
        return fig
    
    def get_statistics(self):
        walls = np.sum(self.grid == 100)
        total = self.grid_size * self.grid_size
        
        if len(self.all_points_x) > 0:
            room_width = np.percentile(self.all_points_x, 98) - np.percentile(self.all_points_x, 2)
            room_height = np.percentile(self.all_points_y, 98) - np.percentile(self.all_points_y, 2)
        else:
            room_width = 0
            room_height = 0
        
        return {
            'grid_size': self.grid_size,
            'cell_size_mm': self.cell_size_mm,
            'obstacle_count': walls,
            'walls_percent': (walls / total) * 100,
            'room_width_mm': room_width,
            'room_height_mm': room_height,
            'points_count': len(self.all_points_x)
        }
    
    def export_map(self, filename_prefix="clean_map"):
        fig = self.render_with_matplotlib(figsize=(12, 10))
        fig.savefig(f"{filename_prefix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        csv_file = f"{filename_prefix}.csv"
        pd.DataFrame(self.grid).to_csv(csv_file, index=False)
        
        return f"{filename_prefix}.png", csv_file