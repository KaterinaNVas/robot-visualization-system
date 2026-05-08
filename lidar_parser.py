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

            # Для этого формата ожидаем 0x1E = 30 байт измерений
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

            # Если угол прошёл через 360 градусов
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

                # Фильтр неадекватных значений
                if 100 <= distance <= 8000:
                    points.append({
                        "angle": angle,
                        "distance": distance
                    })

            i += 39
        else:
            i += 1

    return points


def parse_lidar_data(raw_data, lidar_model="Delta2A"):
    if lidar_model == "Delta2D":
        return parse_delta2d_hex(raw_data)

    return parse_delta2a_packet(raw_data)