import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm
import os
import subprocess

# === Load Models ===
face_model = YOLO("yolov8n-face.pt")  # Face-specific YOLOv8 model
gender_model = load_model("enzyme_model.h5")
IMG_SIZE = 128
LABELS = ['Female', 'Male']

# === Load Video ===
input_path = "input_video.mp4"
temp_video_path = "temp_output_video.mp4"  # No audio
final_output_path = "output_face_gender.mp4"  # With audio

cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
print(f"🎬 Processing {input_path} → {temp_video_path} ({w}x{h} at {fps} FPS)")

# === Process Each Frame ===
for _ in tqdm(range(total_frames), desc="🔍 Detecting & Classifying"):
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame, imgsz=640, conf=0.3, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_w, box_h = x2 - x1, y2 - y1
        box_area = box_w * box_h

        if box_w <= 0 or box_h <= 0 or box_area < 500:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))

        prediction = gender_model.predict(input_data)[0][0]
        label_index = int(prediction > 0.5)
        gender = LABELS[label_index]
        confidence = prediction if label_index == 1 else 1 - prediction

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{gender} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

# === Cleanup OpenCV ===
cap.release()
out.release()
print("🎞️ Video frames processed.")

# === Use FFmpeg to add original audio back ===
print("🔊 Adding original audio back using ffmpeg...")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-i", temp_video_path,
    "-i", input_path,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    final_output_path
]

subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === Optional Cleanup ===
os.remove(temp_video_path)

print("✅ Done. Final output with audio saved to:", final_output_path)
