"""
Полностью навайбкоженный код для рисования квадратиков из файлов разметки
"""
import os
import cv2
from tqdm import tqdm

TRAIN_DIR = "dl-lab-2-stuff-detection/yolo_dataset/yolo_dataset/train"
IMAGES_DIR = f"{TRAIN_DIR}/images"
LABELS_DIR = f"{TRAIN_DIR}/labels"
OUTPUT_DIR = "annotated_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


CLASSES = {0: "customer", 1: "staff"}
# Цвета в формате BGR (Blue, Green, Red) для OpenCV
COLORS = {
    0: (0, 255, 0),   # Зеленый для customer
    1: (0, 0, 255)    # Красный для staff
}


image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]

print(f"Найдено изображений: {len(image_files)}. Начинаем отрисовку...")

for img_name in tqdm(image_files):
    # Пути до текущей картинки и лейбла
    img_path = os.path.join(IMAGES_DIR, img_name)
    # Имя лейбла такое же, как у картинки, но .txt
    label_name = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(LABELS_DIR, label_name)
    
    # Читаем картинку
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    height, width, _ = img.shape
    
    # Проверяем, есть ли файл с разметкой для этой картинки
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                # Читаем нормализованные координаты
                cx, cy, w, h = map(float, parts[1:])
                
                # Переводим из YOLO формата (центр_x, центр_y, ширина, высота) 
                # в пиксельные координаты углов (x_min, y_min, x_max, y_max)
                x_min = int((cx - w / 2) * width)
                y_min = int((cy - h / 2) * height)
                x_max = int((cx + w / 2) * width)
                y_max = int((cy + h / 2) * height)
                
                color = COLORS.get(class_id, (255, 255, 255))
                label_text = CLASSES.get(class_id, "unknown")
                
                # Рисуем сам прямоугольник (толщина линии 2 пикселя)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                
                # --- Рисуем красивую плашку для текста, чтобы он не сливался с фоном ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Плашка под текст
                cv2.rectangle(img, (x_min, y_min - text_h - baseline - 5), (x_min + text_w, y_min), color, -1)
                # Сам текст (белым цветом)
                cv2.putText(img, label_text, (x_min, y_min - 5), font, font_scale, (255, 255, 255), thickness)

    # Сохраняем картинку
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print(f"Готово! Все размеченные картинки лежат в папке: {OUTPUT_DIR}")