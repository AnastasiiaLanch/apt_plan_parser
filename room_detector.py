import cv2
import numpy as np


def detect_rooms(image):
    """
    Детекция комнат на плане
    """
    # Преобразование в черно-белое
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Если изображение светлое - инвертация
    mean_brightness = np.mean(gray)
    if mean_brightness > 127:
        gray = cv2.bitwise_not(gray)

    # Бинаризация - превращение в черно-белое изображение без полутонов
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Заполнение мелких разрывов в стенах
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Нахождение границ комнат
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rooms = []
    room_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Игнорирование очень маленьких и очень больших областей
        if area < 1000 or area > (image.shape[0] * image.shape[1] * 0.8):
            continue

        # Аппроксимиация контура, оставляю только основные углы
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Установка минимум 3х точек для полигона
        if len(approx) >= 3:
            polygon = []
            for point in approx:
                x, y = point[0]
                polygon.append([int(x), int(y)])

            rooms.append({
                "id": f"r{room_id}",
                "polygon": polygon
            })
            room_id += 1

    return rooms