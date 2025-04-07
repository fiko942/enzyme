"""
File: clean_dataset.py
Author: Wiji Fiko Teren
Email: tobellord@gmail.com

Description:
Process raw dataset images using YOLOv8n-face to detect faces, crop them,
convert to grayscale, and save the cleaned images into the 'cleaned/' directory.
Uses multithreading for concurrent processing.
"""


import cv2
import os
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ultralytics import YOLO

# === Load YOLOv8n-face model once ===
print("🔍 Loading YOLOv8n-face model...")
face_model = YOLO("yolov8n-face.pt")

# === Configs ===
DATASET_DIR = "dataset"
CLEANED_DIR = "cleaned"
CONF_THRESHOLD = 0.5
MAX_THREADS_PER_FOLDER = 5

def process_image(img_path, output_folder):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return f"❌ Failed to load {filename}"

    h, w = img.shape[:2]
    results = face_model(img, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    face_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(output_folder, f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_path, gray_face)
        face_count += 1

    return f"✅ {face_count} face(s) from {filename}" if face_count > 0 else f"⚠️ No face detected: {filename}"


def process_folder(category):
    input_folder = os.path.join(DATASET_DIR, category)
    output_folder = os.path.join(CLEANED_DIR, category)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    image_files = [str(p) for p in Path(input_folder).glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS_PER_FOLDER) as executor:
        futures = {executor.submit(process_image, img_path, output_folder): img_path for img_path in image_files}

        for future in tqdm(as_completed(futures), total=len(image_files), desc=f"[{category.upper()}] Processing"):
            results.append(future.result())

    for res in results:
        print(res)


if __name__ == "__main__":
    categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"🔍 Found categories: {categories}")
    print(f"🧵 Max threads per category: {MAX_THREADS_PER_FOLDER}\n")

    with ThreadPoolExecutor(max_workers=len(categories)) as executor:
        futures = [executor.submit(process_folder, cat) for cat in categories]
        for future in futures:
            future.result()

    print("\n✅ All processing complete.")
