# ENZYME
## The Open-Source Engine for Adaptive AI.
### Enzyme is an open-source machine learning model designed for flexibility, transparency, and experimentation. Whether you're training on custom datasets, fine-tuning existing logic, or exploring new frontiers in AI, ENZYME provides a fully modifiable Python-based framework.

### Ideal for researchers, developers, and curious minds - with ENZYME, you don't just use the model, you make it your own.


---

## 🧠 Workflow Overview

1. **Data Collection**
   - Run `collect_dataset.py` to capture images using webcam.
   - Images are stored in `dataset/male` or `dataset/female`.

2. **Data Cleaning**
   - Run `clean_dataset.py` to:
     - Detect face using Haar Cascade
     - Crop the face region
     - Convert to grayscale
     - Save to `cleaned/` directory

3. **Training**
   - Run `trainer.py` to train a CNN model using data from `cleaned/`.
   - Model will be saved as `model.h5`.

4. **Prediction**
   - Use `predict_img.py` for single image prediction.
   - Use `predict_video.py` for video files.
   - Use `predict_webcam.py` for real-time webcam predictions.

5. **API Deployment**
   - Run `server.py` to start a Flask API server.
   - Endpoint `/predict` accepts image uploads and returns predicted class.


## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
