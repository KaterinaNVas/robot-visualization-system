#!/usr/bin/env python3
"""
Парсеры для лидаров Delta2A, Delta2D, Delta2B
"""

import struct
import numpy as np
import pandas as pd


def parse_delta2a_packet(raw_data):
    """
    Заглушка под Delta2A.
    В ТЗ указано, что парсинг Delta2A выполняется на микроконтроллере.
    Поэтому на ПК обычно приходят уже готовые angle/distance.
    """
    return raw_data


def parse_delta2d_hex(hex_text):
    """
    Парсер HEX-данных лидара Delta2D.

    Формат пакета:
    0-1   AA 55       заголовок
    2     1E          длина блока данных
    3-4   start angle начальный угол, Little Endian, *0.01 градуса
    5-6   end angle   конечный угол, Little Endian, *0.01 градуса
    7-8   checksum    контрольная сумма
    9-38  distances   15 расстояний по 2 байта, Little Endian
    """
    hex_values = hex_text.replace("\n", " ").split()
    data = []

    for value in hex_values:
        try:
            data.append(int(value, 16))
        except ValueError:
            continue

    points = []
    i = 0

    while i < len(data) - 39:
        if data[i] == 0xAA and data[i + 1] == 0x55:
            block_length = data[i + 2]

            if block_length != 0x1E:
                i += 1
                continue

            packet = data[i:i + 39]

            if len(packet) < 39:
                break

            start_angle_raw = packet[3] | (packet[4] << 8)
            end_angle_raw = packet[5] | (packet[6] << 8)

            start_angle = start_angle_raw / 100.0
            end_angle = end_angle_raw / 100.0

            if end_angle < start_angle:
                end_angle += 360.0

            distances = []

            for j in range(9, 39, 2):
                distance = packet[j] | (packet[j + 1] << 8)
                distances.append(distance)

            points_count = len(distances)

            if points_count > 1:
                angle_step = (end_angle - start_angle) / (points_count - 1)
            else:
                angle_step = 0

            for idx, distance in enumerate(distances):
                angle = (start_angle + idx * angle_step) % 360

                if 100 <= distance <= 8000:  # мм
                    points.append({
                        "angle": angle,
                        "distance": distance  # в мм
                    })

            i += 39
        else:
            i += 1

    return points


def parse_delta2b_hex(hex_text):
    """
    Парсер HEX-данных лидара Delta2B.
    
    Формат пакета (из документации iArduino):
    0     0xAA         заголовок пакета
    1-2   length       длина пакета
    3     0x01         версия протокола
    4     0x61         тип пакета
    5     0xAD         заголовок полезных данных
    6-7   data_length  длина данных
    8     speed        скорость мотора *0.05 об/с
    9-10  zero_offset  сдвиг нулевого градуса *0.01°
    11-12 start_angle  стартовый угол *0.01°
    13    signal1      уровень сигнала 1-й пробы
    14-15 distance1    расстояние 1-й пробы *0.25мм
    16    signal2      уровень сигнала 2-й пробы
    17-18 distance2    расстояние 2-й пробы *0.25мм
    ...                ...
    последние 2 байта  контрольная сумма
    """
    hex_values = hex_text.replace("\n", " ").split()
    data = []
    
    for value in hex_values:
        try:
            data.append(int(value, 16))
        except ValueError:
            continue
    
    points = []
    i = 0
    
    while i < len(data) - 20:  # Минимум 20 байт на пакет
        # Поиск заголовка 0xAA
        if data[i] == 0xAA:
            # Читаем длину пакета (байты 1-2, little-endian)
            if i + 2 >= len(data):
                break
            
            packet_length = data[i+1] | (data[i+2] << 8)
            
            # Проверка валидности длины
            if packet_length < 20 or packet_length > 1024:
                i += 1
                continue
            
            # Проверяем, что пакет помещается в данные
            if i + packet_length > len(data):
                i += 1
                continue
            
            # Проверяем заголовок данных (байт 5 должен быть 0xAD)
            if i + 5 >= len(data) or data[i+5] != 0xAD:
                i += 1
                continue
            
            packet = data[i:i+packet_length]
            
            # Парсим метаданные
            # data_length (байты 6-7)
            data_length = packet[6] | (packet[7] << 8)
            
            # Стартовый угол (байты 11-12)
            start_angle_raw = packet[11] | (packet[12] << 8)
            start_angle = start_angle_raw * 0.01  # градусы
            
            # Количество точек в пакете
            # data_length = 5 служебных байтов + 3 * N
            N = int((data_length - 5) / 3) if data_length > 5 else 0
            
            # Парсим точки (начинаются с байта 13)
            for n in range(N):
                base_idx = 13 + n * 3
                
                if base_idx + 2 >= len(packet):
                    break
                
                signal = packet[base_idx]
                distance_raw = packet[base_idx+1] | (packet[base_idx+2] << 8)
                distance_mm = distance_raw * 0.25
                
                # Расчет угла для точки
                if N > 1:
                    angle = start_angle + 22.5 * (n - 1) / N
                else:
                    angle = start_angle
                
                angle = angle % 360
                
                # Фильтр по спецификации Delta2B: 130-8000 мм
                if 130 <= distance_mm <= 8000:
                    points.append({
                        "angle": angle,
                        "distance": distance_mm,  # в мм
                        "intensity": signal
                    })
            
            i += packet_length
        else:
            i += 1
    
    return points


def parse_lidar_data(hex_text, model="Delta2A"):
    """
    Главная функция парсинга данных лидара.
    
    Args:
        hex_text: строка с HEX данными
        model: модель лидара ("Delta2A", "Delta2D", "Delta2B")
    
    Returns:
        list of dict: [{"angle": float, "distance": float, "intensity": int}, ...]
    """
    if model == "Delta2D":
        return parse_delta2d_hex(hex_text)
    elif model == "Delta2B":
        return parse_delta2b_hex(hex_text)
    else:  # Delta2A
        return parse_delta2a_packet(hex_text)


def lidar_to_dataframe(lidar_points):
    """
    Преобразование точек лидара в DataFrame с координатами.
    
    Args:
        lidar_points: список точек от parse_lidar_data
    
    Returns:
        pandas DataFrame с колонками: angle, distance, intensity, x, y
    """
    if not lidar_points:
        return pd.DataFrame()
    
    df = pd.DataFrame(lidar_points)
    
    # Преобразование угла в радианы
    angles_rad = np.radians(df['angle'])
    
    # Декартовы координаты
    df['x'] = df['distance'] * np.cos(angles_rad)
    df['y'] = df['distance'] * np.sin(angles_rad)
    
    return df


# Пример использования
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python lidar_parser.py <файл.hex> [модель]")
        print("Модели: Delta2A, Delta2D, Delta2B")
        sys.exit(1)
    
    hex_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "Delta2D"
    
    with open(hex_file, 'r') as f:
        hex_text = f.read()
    
    points = parse_lidar_data(hex_text, model)
    
    print(f"Модель: {model}")
    print(f"Всего точек: {len(points)}")
    
    if points:
        distances = [p['distance'] for p in points]
        print(f"Расстояния: мин={min(distances):.1f} мм, макс={max(distances):.1f} мм")
        print(f"\nПервые 5 точек:")
        for p in points[:5]:
            print(f"  угол={p['angle']:.1f}°, расстояние={p['distance']:.1f} мм")