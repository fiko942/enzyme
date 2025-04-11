"""
File: collect_dataset.py
Author: Wiji Fiko Teren
Email: tobellord@gmail.com

Description:
Capture images using webcam or screen (real-time) and categorize them as 'male' or 'female'.
The collected faces are saved in grayscale format inside the 'dataset/' directory.
"""

import cv2
import os
import uuid
import numpy as np
from ultralytics import YOLO
from mss import mss

# === Load YOLOv8n-face model ===
print("🔍 Loading YOLOv8n-face model...")
face_model = YOLO("yolov8n-face.pt")

# === Ask for user type and source ===
user_type = input("Enter type (e.g., male/female): ").strip().lower()
source = input("Choose source [webcam/screen]: ").strip().lower()

# === Create output directory ===
dataset_path = f'dataset/{user_type}'
os.makedirs(dataset_path, exist_ok=True)

# === Screen capture setup (for macOS) ===
if source == "screen":
    sct = mss()
    monitors = sct.monitors
    print("\nAvailable Monitors:")
    for idx, m in enumerate(monitors[1:], 1):
        print(f"{idx}: {m}")
    screen_index = int(input("Select monitor number (1, 2, ...): "))
    monitor = monitors[screen_index]
    print(f"🖥️ Capturing from screen {screen_index}: {monitor}")

    def get_frame():
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# === Webcam setup ===
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to access webcam.")

    def get_frame():
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("❌ Failed to grab webcam frame.")
        return frame

print("📸 Ready! Press 'S' to save face, 'ESC' to exit.")

while True:
    try:
        frame = get_frame()
    except RuntimeError as e:
        print(e)
        break

    h, w, _ = frame.shape
    results = face_model(frame, imgsz=640, conf=0.5, verbose=False)[0]

    faces = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_w, box_h = x2 - x1, y2 - y1
        area = box_w * box_h

        if area > 500:
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)
            faces.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Live Feed - Press 'S' to Save", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not faces:
            print("⚠️ No face detected.")
        else:
            for (x, y, w_box, h_box) in faces:
                face_crop = frame[y:y + h_box, x:x + w_box]
                if face_crop.size == 0:
                    print("⚠️ Invalid crop. Skipped.")
                    continue

                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (200, 200))
                filename = os.path.join(dataset_path, f"{uuid.uuid4()}.jpg")
                cv2.imwrite(filename, resized_face)
                print(f"✅ Saved: {filename}")

    elif key == 27:  # ESC
        break

if source == "webcam":
    cap.release()
cv2.destroyAllWindows()
print("📁 Dataset collection ended.")
