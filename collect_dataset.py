import cv2
import os
import uuid
import numpy as np
from ultralytics import YOLO

# === Load YOLOv8n-face model ===
print("🔍 Loading YOLOv8n-face model...")
face_model = YOLO("yolov8n-face.pt")

# === Create output directory ===
user_type = input("Enter type (e.g., male/female): ").strip().lower()
dataset_path = f'dataset/{user_type}'
os.makedirs(dataset_path, exist_ok=True)

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'S' to save face, 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
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

    # Show the frame
    cv2.imshow("Webcam - Press S to Save", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not faces:
            print("⚠️ No face detected. Try again.")
        else:
            for (x, y, w_box, h_box) in faces:
                face_crop = frame[y:y + h_box, x:x + w_box]
                if face_crop.size == 0:
                    print("⚠️ Invalid face crop. Skipped.")
                    continue

                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (200, 200))
                filename = os.path.join(dataset_path, f"{uuid.uuid4()}.jpg")
                cv2.imwrite(filename, resized_face)
                print(f"✅ Saved: {filename}")

    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("📁 Dataset collection ended.")
