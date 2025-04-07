
# ENZYME
## The Open-Source Engine for Adaptive AI.
### Enzyme is an open-source machine learning model designed for flexibility, transparency, and experimentation. Whether you're training on custom datasets, fine-tuning existing logic, or exploring new frontiers in AI, ENZYME provides a fully modifiable Python-based framework.

### Ideal for researchers, developers, and curious minds - with ENZYME, you don't just use the model, you make it your own.

---

# Gender Classification from Face Images

A machine learning project to classify gender (Male/Female) from face images using deep learning and computer vision. The system includes data collection, preprocessing using YOLOv8n-face, CNN training, prediction, and API deployment.

---

## 📁 Project Structure

```
project-root/
├── collect_dataset.py        # Collect raw face dataset via webcam
├── clean_dataset.py          # Clean dataset using YOLOv8n-face (grayscale, cropped faces)
├── trainer.py                # Train CNN model with cleaned dataset
├── predict_img.py            # Predict gender from a single image
├── predict_video.py          # Predict gender from video
├── predict_webcam.py         # Real-time gender prediction via webcam
├── server.py                 # Flask API for prediction
│
├── dataset/                  # Raw images
│   ├── male/
│   └── female/
│
├── cleaned/                  # Preprocessed (cropped + grayscale)
│   ├── male/
│   └── female/
```

---

## 🧠 Workflow Overview

### 1. 📸 Data Collection
Use webcam to collect images categorized into `male` or `female`:

```bash
python collect_dataset.py
```

Images are saved under the `dataset/` directory.

---

### 2. 🧼 Dataset Cleaning with YOLOv8n-face
Detect and crop faces using **Ultralytics YOLOv8n-face**, convert to grayscale, and save:

```bash
python clean_dataset.py
```

- Uses **YOLOv8n-face** (`yolov8n-face.pt`)
- Crops faces from images and saves grayscale versions
- Processes multiple categories and images concurrently with multithreading
- Cleaned images are saved in `cleaned/`

---

### 3. 🧠 Training the Model
Train a CNN model to classify gender based on the cleaned face dataset:

```bash
python trainer.py
```

Model will be saved as `model.h5`.

---

### 4. 🔍 Prediction
Predict gender using the trained model:

- From image:
  ```bash
  python predict_img.py
  ```
- From video:
  ```bash
  python predict_video.py
  ```
- From webcam:
  ```bash
  python predict_webcam.py
  ```

---

### 5. 🌐 API Server
Start the Flask server for remote prediction:

```bash
python server.py
```

**Endpoint:**
- `POST /predict` – Accepts image uploads, returns predicted gender.

---

## ⚙️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

Libraries used:
- `opencv-python`
- `ultralytics`
- `tqdm`
- `flask`
- `tensorflow` or `keras`

---

## 📦 Notes

- YOLOv8n-face model file (`yolov8n-face.pt`) must be available in the same directory or specify the correct path in `clean_dataset.py`.
- The system uses multithreading to speed up image preprocessing.
- Supports `.jpg`, `.jpeg`, `.png` images.

---

## 👤 Author Info

<img src="https://avatars.githubusercontent.com/u/84434815" width="100" alt="Wiji Fiko Teren" />

**Wiji Fiko Teren**  
🌐 [wijifikoteren.streampeg.com](https://wijifikoteren.streampeg.com)  
📧 Email: [tobellord@gmail.com](mailto:tobellord@gmail.com) / [wijifikoteren@streampeg.com](mailto:wijifikoteren@streampeg.com)  
📺 YouTube: [@wijifikoteren](https://www.youtube.com/@wijifikoteren)  
☕ Donate: [PayPal - paypal.me/wijifikoteren](http://paypal.me/wijifikoteren)

---

## 🪪 License

MIT License — use freely for learning and development.
