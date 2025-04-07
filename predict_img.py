import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load your trained gender model
model = load_model('enzyme_model.h5')
IMG_SIZE = 128
labels = ['Female', 'Male']

# Load YOLOv8 face detection model
face_detector = YOLO("yolov8n-face.pt")  # Replace with your face model path

def extract_and_classify_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or can't be opened.")

    h, w, _ = img.shape
    output = []

    # Run YOLOv8 inference
    results = face_detector(img, imgsz=640, conf=0.3, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = img[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Preprocess for gender classification
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))  # shape: (1, 128, 128, 1)

        # Predict
        prediction = model.predict(input_data)[0][0]
        label_index = int(prediction > 0.5)
        label = labels[label_index]
        confidence = prediction if label_index == 1 else 1 - prediction

        output.append({
            'position': (x1, y1, x2 - x1, y2 - y1),
            'label': label,
            'confidence': float(confidence)
        })

    return output


# Example usage
result = extract_and_classify_face('/Users/fiko/Downloads/beautiful-girl-model-outdoors-nature-wallpaper-preview.jpg')
print(result)
