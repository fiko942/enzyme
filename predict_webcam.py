import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# === Load Models ===
face_model = YOLO("yolov8n-face.pt")  # Replace with your YOLOv8 face model
gender_model = load_model("enzyme_model.h5")
labels = ['Female', 'Male']
IMG_SIZE = 128

# === Start Webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Webcam started (face detection)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run face detection
    results = face_model(frame, imgsz=640, conf=0.3, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Preprocess for gender classification
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))  # shape: (1, 128, 128, 1)

        # Predict gender
        prediction = gender_model.predict(input_data)[0][0]
        label_index = int(prediction > 0.5)
        label = labels[label_index]
        confidence = prediction if label_index == 1 else 1 - prediction

        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Detection + Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
