from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import io
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load gender classification model
model = load_model('enzyme_model.h5')
labels = ['Female', 'Male']
IMG_SIZE = 128

# Load YOLOv8 face detection model
face_detector = YOLO("yolov8n-face.pt")  # Replace with your path if needed

def detect_faces_from_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        # Run YOLO face detection
        results = face_detector(img_bgr, imgsz=640, conf=0.3, verbose=False)[0]
        detections_output = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = img_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1, 128, 128, 1)

            prediction = model.predict(face_input)[0][0]
            label_index = int(prediction > 0.5)
            label = labels[label_index]
            confidence = prediction if label_index == 1 else 1 - prediction

            detections_output.append({
                'position': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
                'label': label,
                'confidence': round(float(confidence), 4)
            })

        return detections_output

    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        detections = detect_faces_from_bytes(image_bytes)
        return jsonify({'detections': detections}), 200
    except ValueError as ve:
        print(str(ve))
        return jsonify({'error': str(ve)}), 500
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=62101)
