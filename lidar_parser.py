def parse_delta2a_packet(raw_data):
    """
    Заглушка под Delta2A.
    В ТЗ указано, что парсинг Delta2A выполняется на микроконтроллере.
    Поэтому на ПК обычно приходят уже готовые angle/distance.
    """
    return raw_data


def parse_delta2d_hex(hex_text):
    """
    Базовый парсер Delta2D HEX-потока.
    Ищет пакеты AA 55 и достаёт расстояния.
    """
    hex_values = hex_text.replace("\n", " ").split()
    data = [int(x, 16) for x in hex_values]

    points = []
    i = 0

    while i < len(data) - 30:
        if data[i] == 0xAA and data[i + 1] == 0x55:
            packet = data[i:i + 30]

            if len(packet) < 30:
                break

            start_angle_raw = packet[4] | (packet[5] << 8)
            end_angle_raw = packet[6] | (packet[7] << 8)

            start_angle = start_angle_raw / 100
            end_angle = end_angle_raw / 100

            distances = []

            for j in range(10, 30, 2):
                distance = packet[j] | (packet[j + 1] << 8)
                distances.append(distance)

            if len(distances) > 1:
                angle_step = (end_angle - start_angle) / (len(distances) - 1)
            else:
                angle_step = 0

            for idx, distance in enumerate(distances):
                angle = start_angle + idx * angle_step

                points.append({
                    "angle": angle % 360,
                    "distance": distance
                })

            i += 30
        else:
            i += 1

    return points


def parse_lidar_data(raw_data, lidar_model="Delta2A"):
    if lidar_model == "Delta2D":
        return parse_delta2d_hex(raw_data)

    return parse_delta2a_packet(raw_data)