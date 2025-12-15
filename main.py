import os
import cv2
import json
from data_workflow import Data
from wall_detector import detect_walls
from room_detector import detect_rooms

# Конфигурация путей
input_dir = "data"
output_dir = "output/result.json"


def main():
    data = Data(input_dir, output_dir)
    all_results = []

    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(filename)

    image_files.sort()

    # Обработка каждого изображения
    for filename in image_files:
        image = data.prepare_image(filename)
        if image is None:
            print(f"Невозможно прочитать {filename}")
            continue

        # Детекция стен и комнат
        walls = detect_walls(image) or []
        rooms = detect_rooms(image) or []

        image_result = data.save_json(filename, walls, rooms)
        all_results.append(image_result)

        print(f"{filename} обработан, {len(walls)} стен найдено, {len(rooms)} комнат найдено")

    data.save_to_file(all_results)


if __name__ == "__main__":
    main()
