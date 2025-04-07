import cv2
import os
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load Face Detector (once, outside thread)
FACE_MODEL = {
    "proto": "face_detector/deploy.prototxt",
    "weights": "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
}
face_net = cv2.dnn.readNetFromCaffe(FACE_MODEL["proto"], FACE_MODEL["weights"])

# Configs
DATASET_DIR = "dataset"
CLEANED_DIR = "cleaned"
CONF_THRESHOLD = 0.7
MAX_THREADS_PER_FOLDER = 10

def process_image(img_path, output_folder):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return f"❌ Failed to load {filename}"

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            face = img[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            output_path = os.path.join(output_folder, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(output_path, gray_face)
            return f"✅ Processed: {filename}"

    return f"⚠️ No face detected: {filename}"


def process_folder(category):
    input_folder = os.path.join(DATASET_DIR, category)
    output_folder = os.path.join(CLEANED_DIR, category)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    image_files = [str(p) for p in Path(input_folder).glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS_PER_FOLDER) as executor:
        futures = {executor.submit(process_image, img_path, output_folder): img_path for img_path in image_files}

        for future in tqdm(as_completed(futures), total=len(image_files), desc=f"[{category.upper()}] Processing"):
            result = future.result()
            results.append(result)

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
