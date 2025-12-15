import os
import cv2
import json


class Data:
    """
    Класс для управления данными: загрузка изображений и сохранение результатов
    """
    def __init__(self, input_dir, output_json_path):
        self.input_dir = input_dir
        self.output_json_path = output_json_path

        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)

    def prepare_image(self, filename):
        """
        Загрузка изображения из файла
        """
        image_path = os.path.join(self.input_dir, filename)
        image = cv2.imread(image_path)
        return image

    def save_json(self, filename, walls_sample, rooms_sample):
        """
        Формирование словаря с результатами для одного изображения
        """
        if walls_sample is None:
            walls_sample = []
        if rooms_sample is None:
            rooms_sample = []

        return {
            "meta": {"source": filename},
            "walls": walls_sample,
            "rooms": rooms_sample
        }

    def save_to_file(self, data_list):
        """
        Сохранение результатов в JSON файл с нужным форматированием
        """

        result = "[\n"

        for i, item in enumerate(data_list):
            if i > 0:
                result += ",\n"

            result += "{"
            result += '"meta": { "source": "' + item["meta"]["source"] + '" },\n'
            result += '"walls": [\n'

            # Добавление стен
            for j, wall in enumerate(item["walls"]):
                if j > 0:
                    result += ",\n"
                # Форматирование координат
                points_str = ",".join(f"[{p[0]},{p[1]}]" for p in wall["points"])
                result += f'{{"id": "{wall["id"]}", "points": [{points_str}] }}'

            result += "\n]"

            # Добавление комнат, при наличии
            if item["rooms"]:
                result += ",\n"
                result += '"rooms": [\n'

                for j, room in enumerate(item["rooms"]):
                    if j > 0:
                        result += ",\n"
                    # Форматирование полигона комнаты
                    polygon_str = ",".join(f"[{p[0]},{p[1]}]" for p in room["polygon"])
                    result += f'{{"id": "{room["id"]}", "polygon": [{polygon_str}] }}'

                result += "\n]"

            result += "\n}"

        result += "\n]"

        with open(self.output_json_path, "w", encoding='utf-8') as f:
            f.write(result)
        print(f"\nСохранено в {self.output_json_path}")


