"""
File: predict_video.py
Author: Wiji Fiko Teren
Email: tobellord@gmail.com

Description:
Perform gender classification from video input using the trained model.
Processes each frame, predicts gender, and displays result in real-time.
"""


import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm

# === Load Models ===
face_model = YOLO("yolov8n-face.pt")  # Face-specific YOLOv8 model
gender_model = load_model("enzyme_model.h5")
IMG_SIZE = 128
LABELS = ['Female', 'Male']

# === Load Video ===
input_path = "input_video.mp4"
output_path = "output_face_gender.mp4"

cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
print(f"🎬 Processing {input_path} → {output_path} ({w}x{h} at {fps} FPS)")

# === Process Each Frame ===
for _ in tqdm(range(total_frames), desc="🔍 Detecting & Classifying"):
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame, imgsz=640, conf=0.3, verbose=False)[0]

    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w, box_h = x2 - x1, y2 - y1
        box_area = box_w * box_h

        # Filter invalid or small boxes
        if box_w <= 0 or box_h <= 0 or box_area < 500:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # === Preprocess for gender prediction ===
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))

        # === Predict ===
        prediction = gender_model.predict(input_data)[0][0]
        label_index = int(prediction > 0.5)
        gender = LABELS[label_index]
        confidence = prediction if label_index == 1 else 1 - prediction

        # === Annotate ===
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{gender} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

# === Cleanup ===
cap.release()
out.release()
print("✅ Done. Saved to:", output_path)
