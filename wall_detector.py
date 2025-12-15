import cv2
import numpy as np
import math


def detect_walls(image: np.array):
    """
    Детекция стен на плане
    Взала за основу "стены на планах - толстые линии, образующие прямоугольники".
    """
    # Преобразование в черно-белое
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Определение светлое ли изображение
    mean_brightness = np.mean(gray)

    # Если план светлый, инвертация
    if mean_brightness > 127:
        gray = cv2.bitwise_not(gray)

    # Бинаризация для четкого разделения стен и фона
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Утолщение линий стен, чтобы соединить возможные разрывы
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Все контуры (включая внутренние)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        return []

    walls = []
    wall_id = 1

    # Проход по всем контурам
    for i, contour in enumerate(contours):
        # Только внешние
        if hierarchy[0][i][3] != -1:
            continue  # Пропуск внутренних контуров

        # Упрощение контура до многоугольника
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # На архитектурных планах стены образуют многоугольники
        if len(approx) >= 4:
            # Точки в список координат
            points = [[int(p[0][0]), int(p[0][1])] for p in approx]

            # Разбивка многоугольника на сегменты
            for j in range(len(points)):
                p1 = points[j]
                p2 = points[(j + 1) % len(points)]

                # Проверка длины сегмента
                length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

                # Условие от какой длины добавляется стена
                if length > 50:
                    walls.append({
                        "id": f"w{wall_id}",
                        "points": [p1, p2]
                    })
                    wall_id += 1

    # Удаление дубликатов
    unique_walls = []
    used = [False] * len(walls)

    for i in range(len(walls)):
        if used[i]:
            continue

        wall1 = walls[i]
        unique_walls.append(wall1)

        # Отметка всех похожих стен как использованные
        for j in range(i + 1, len(walls)):
            if used[j]:
                continue

            wall2 = walls[j]
            if walls_are_similar(wall1, wall2):
                used[j] = True

    # Переиндексация id стен
    for i, wall in enumerate(unique_walls, 1):
        wall["id"] = f"w{i}"

    return unique_walls[:40]  # Ограничение количества стен


def walls_are_similar(wall1, wall2, threshold=20):
    """Проверка, являются ли две стены одинаковыми (в пределах порога)"""

    # Получаем точки в стандартном порядке
    def normalize_points(points):
        """
        Нормализация порядка точек в стене
        """
        p1, p2 = points
        if p1[0] == p2[0]:  # Вертикальная стена
            return (p1 if p1[1] < p2[1] else p2, p2 if p1[1] < p2[1] else p1)
        else:  # Горизонтальная или диагональная
            return (p1 if p1[0] < p2[0] else p2, p2 if p1[0] < p2[0] else p1)

    p1_start, p1_end = normalize_points(wall1["points"])
    p2_start, p2_end = normalize_points(wall2["points"])

    # Проверка расстояния между соответствующими точками
    dist_start = math.sqrt((p1_start[0] - p2_start[0]) ** 2 +
                           (p1_start[1] - p2_start[1]) ** 2)
    dist_end = math.sqrt((p1_end[0] - p2_end[0]) ** 2 +
                         (p1_end[1] - p2_end[1]) ** 2)

    return dist_start < threshold and dist_end < threshold