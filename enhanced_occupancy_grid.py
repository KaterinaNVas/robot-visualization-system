# enhanced_occupancy_grid.py
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

class EnhancedOccupancyGrid:
    def __init__(self, 
                 map_size_mm=12000,      # размер карты в мм
                 cell_size_mm=50,        # размер ячейки в мм (чем меньше, тем детальнее)
                 robot_radius_mm=250,    # радиус робота для расширения препятствий
                 min_distance_mm=150,    # минимальная дистанция лидара
                 max_distance_mm=6000):  # максимальная дистанция лидара
        
        self.map_size_mm = map_size_mm
        self.cell_size_mm = cell_size_mm
        self.robot_radius_mm = robot_radius_mm
        self.min_distance_mm = min_distance_mm
        self.max_distance_mm = max_distance_mm
        
        # Размер сетки в ячейках
        self.grid_size = int(map_size_mm / cell_size_mm)
        
        # Инициализация карты: -1 = неизвестно, 0-100 = вероятность занятости
        self.probability_grid = -np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Счетчик наблюдений для каждой ячейки
        self.hit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)
        self.miss_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)
        
        # История для сглаживания
        self.map_history = []
        self.history_size = 3
        
        # Статистика
        self.total_updates = 0
        self.obstacle_count = 0
        
    def world_to_grid(self, x_mm, y_mm):
        """Перевод мировых координат (мм) в индексы сетки"""
        # Сдвигаем так, чтобы центр карты (0,0) был в центре сетки
        grid_x = int((x_mm + self.map_size_mm/2) / self.cell_size_mm)
        grid_y = int((self.map_size_mm/2 - y_mm) / self.cell_size_mm)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Перевод индексов сетки в мировые координаты (мм)"""
        x_mm = (grid_x + 0.5) * self.cell_size_mm - self.map_size_mm/2
        y_mm = self.map_size_mm/2 - (grid_y + 0.5) * self.cell_size_mm
        return x_mm, y_mm
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Алгоритм Брезенхема для получения всех точек на линии"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points
    
    def update_cell_probability(self, x, y, occupied):
        """Обновление вероятности ячейки с использованием логического Байеса"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return
        
        # Параметры для обновления вероятности
        p_occ = 0.85 if occupied else 0.3  # Вероятность правильного измерения
        p_free = 1 - p_occ
        
        # Текущая вероятность (преобразуем из -1..100 в 0..1)
        if self.probability_grid[x, y] == -1:
            current_prob = 0.5  # Априорная вероятность
        else:
            current_prob = self.probability_grid[x, y] / 100.0
        
        # Байесовское обновление
        if occupied:
            new_prob = (p_occ * current_prob) / (p_occ * current_prob + p_free * (1 - current_prob))
            self.hit_count[x, y] += 1
        else:
            new_prob = ((1 - p_occ) * current_prob) / ((1 - p_occ) * current_prob + (1 - p_free) * (1 - current_prob))
            self.miss_count[x, y] += 1
        
        # Преобразуем обратно в шкалу 0-100
        new_value = new_prob * 100
        
        # Сглаживание с предыдущими значениями
        if self.probability_grid[x, y] != -1:
            new_value = 0.7 * new_value + 0.3 * self.probability_grid[x, y]
        
        self.probability_grid[x, y] = np.clip(new_value, 0, 100)
    
    def update(self, lidar_df, robot_x, robot_y, robot_yaw_deg=0):
        """Обновление карты по данным лидара"""
        if lidar_df.empty or 'global_x' not in lidar_df.columns:
            return
        
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        
        # Проверка границ робота
        if not (0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size):
            return
        
        points_processed = 0
        
        for _, point in lidar_df.iterrows():
            distance = point.get('distance', 0)
            
            # Фильтрация по дистанции
            if distance < self.min_distance_mm or distance > self.max_distance_mm:
                continue
            
            # Получаем координаты препятствия
            obs_gx, obs_gy = self.world_to_grid(point['global_x'], point['global_y'])
            
            # Проверка границ
            if not (0 <= obs_gx < self.grid_size and 0 <= obs_gy < self.grid_size):
                continue
            
            # Рисуем луч от робота до препятствия
            line_points = self._bresenham_line(robot_gx, robot_gy, obs_gx, obs_gy)
            
            # Все точки на луче до препятствия - свободное пространство
            for x, y in line_points[:-1]:
                self.update_cell_probability(x, y, occupied=False)
            
            # Конечная точка - препятствие
            self.update_cell_probability(obs_gx, obs_gy, occupied=True)
            points_processed += 1
        
        # Обновляем статистику
        self.total_updates += 1
        self.obstacle_count = np.sum(self.probability_grid >= 70)
        
        # Сохраняем в историю для сглаживания
        if len(self.map_history) >= self.history_size:
            self.map_history.pop(0)
        self.map_history.append(self.probability_grid.copy())
        
        # Применяем морфологическое расширение для препятствий
        self._dilate_obstacles()
        
        return points_processed
    
    def _dilate_obstacles(self):
        """Расширение препятствий на радиус робота"""
        # Создаем маску препятствий
        obstacle_mask = (self.probability_grid >= 70).astype(np.uint8)
        
        # Рассчитываем размер ядра в зависимости от радиуса робота
        kernel_size = max(1, int(self.robot_radius_mm / self.cell_size_mm))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Расширяем препятствия
        dilated = cv2.dilate(obstacle_mask, kernel, iterations=1)
        
        # Обновляем вероятности для расширенных областей
        self.probability_grid[dilated > 0] = np.maximum(
            self.probability_grid[dilated > 0], 
            70
        )
    
    def get_clean_map(self):
        """Получение чистой карты (0=свободно, 100=занято, -1=неизвестно)"""
        return self.probability_grid.copy()
    
    def get_binary_map(self, threshold=50):
        """Получение бинарной карты (True=препятствие)"""
        return self.probability_grid >= threshold
    
    def render_as_image(self, show_robot=True, robot_x=None, robot_y=None, 
                       robot_yaw_deg=None, show_trajectory=None, trajectory=None,
                       image_scale=4):
        """
        Визуализация карты как цветное изображение
        
        Параметры:
        - show_robot: показывать положение робота
        - robot_x, robot_y: координаты робота в мм
        - robot_yaw_deg: угол поворота робота в градусах
        - show_trajectory: показывать траекторию
        - trajectory: список точек траектории (x, y) в мм
        - image_scale: масштаб для отображения (увеличение)
        """
        # Создаем RGB-изображение
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Занятые ячейки (препятствия) - черные
        occupied_mask = self.probability_grid >= 70
        img[occupied_mask] = [0, 0, 0]
        
        # Частично занятые - темно-серые
        partial_mask = (self.probability_grid >= 30) & (self.probability_grid < 70) & (self.probability_grid != -1)
        img[partial_mask] = [80, 80, 80]
        
        # Свободные ячейки - белые
        free_mask = (self.probability_grid >= 0) & (self.probability_grid < 30) & (self.probability_grid != -1)
        img[free_mask] = [255, 255, 255]
        
        # Неизвестные ячейки - светло-серые
        unknown_mask = self.probability_grid == -1
        img[unknown_mask] = [200, 200, 200]
        
        # Рисуем траекторию (если передана)
        if show_trajectory and trajectory and len(trajectory) > 1:
            # Конвертируем траекторию в координаты сетки
            trajectory_points = []
            for x_mm, y_mm in trajectory:
                gx, gy = self.world_to_grid(x_mm, y_mm)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    trajectory_points.append((gx, gy))
            
            # Рисуем линию траектории
            for i in range(len(trajectory_points) - 1):
                cv2.line(img, 
                        trajectory_points[i], 
                        trajectory_points[i+1], 
                        (0, 255, 255),  # Желтый цвет
                        2)
            
            # Рисуем точки траектории
            for gx, gy in trajectory_points:
                cv2.circle(img, (gx, gy), 1, (0, 255, 255), -1)
        
        # Рисуем робота (если передан)
        if show_robot and robot_x is not None and robot_y is not None:
            robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
            
            if 0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size:
                # Рисуем круг робота
                robot_radius_px = max(3, int(self.robot_radius_mm / self.cell_size_mm))
                cv2.circle(img, (robot_gx, robot_gy), robot_radius_px, (255, 0, 0), 2)
                cv2.circle(img, (robot_gx, robot_gy), 2, (255, 0, 0), -1)
                
                # Рисуем направление (если есть угол)
                if robot_yaw_deg is not None:
                    yaw_rad = np.radians(robot_yaw_deg)
                    arrow_end_x = robot_gx + int(robot_radius_px * 2 * np.cos(yaw_rad))
                    arrow_end_y = robot_gy + int(robot_radius_px * 2 * np.sin(yaw_rad))
                    cv2.arrowedLine(img, (robot_gx, robot_gy), (arrow_end_x, arrow_end_y), 
                                   (255, 0, 0), 2, tipLength=0.3)
        
        # Увеличиваем изображение для лучшего просмотра
        if image_scale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w * image_scale, h * image_scale), 
                           interpolation=cv2.INTER_NEAREST)
        
        return img
    
    def export_map(self, filename_prefix="occupancy_map"):
        """Экспорт карты в различные форматы"""
        # Сохраняем PNG
        img = self.render_as_image(show_robot=False)
        png_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(png_filename, img)
        
        # Сохраняем CSV с данными
        csv_filename = f"{filename_prefix}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        grid_data = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.probability_grid[i, j] != -1:
                    x_mm, y_mm = self.grid_to_world(i, j)
                    grid_data.append({
                        'x_mm': x_mm,
                        'y_mm': y_mm,
                        'probability': self.probability_grid[i, j],
                        'is_obstacle': self.probability_grid[i, j] >= 70,
                        'hit_count': self.hit_count[i, j],
                        'miss_count': self.miss_count[i, j]
                    })
        
        df = pd.DataFrame(grid_data)
        df.to_csv(csv_filename, index=False)
        
        return png_filename, csv_filename
    
    def clear(self):
        """Очистка карты"""
        self.probability_grid = -np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.hit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)
        self.miss_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)
        self.map_history = []
        self.total_updates = 0
        self.obstacle_count = 0
    
    def get_statistics(self):
        """Получение статистики карты"""
        total_cells = self.grid_size * self.grid_size
        occupied_cells = np.sum(self.probability_grid >= 70)
        free_cells = np.sum((self.probability_grid >= 0) & (self.probability_grid < 30))
        unknown_cells = np.sum(self.probability_grid == -1)
        
        return {
            'grid_size': self.grid_size,
            'cell_size_mm': self.cell_size_mm,
            'map_size_mm': self.map_size_mm,
            'total_updates': self.total_updates,
            'obstacle_count': self.obstacle_count,
            'occupied_cells': occupied_cells,
            'free_cells': free_cells,
            'unknown_cells': unknown_cells,
            'coverage_percent': (occupied_cells + free_cells) / total_cells * 100
        }