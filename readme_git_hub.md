# 🐟 Fish Classifier (MobileNetV2)

A **deep learning image classification project** that identifies fish categories from images using **MobileNetV2 (transfer learning)**.  
Built with **TensorFlow 2.13 + Keras 2.13** and deployed using **Streamlit**.

---

## 🚀 Features

- Image classification using **MobileNetV2**
- Single image prediction
- Folder (batch) prediction
- Confidence score for predictions
- Simple **Streamlit web app** for inference
- Clean project structure, beginner‑friendly

---

## 🧠 Model Overview

- Base model: **MobileNetV2 (ImageNet weights)**
- Input size: `224 × 224 × 3`
- Top layers: Global Average Pooling + Dense (Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## 📂 Project Structure

```
fish_predict/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── fish_classifier.keras
│
├── str.py              # Streamlit app
├── train.py            # Model training script
├── predict.py          # Prediction utilities
├── requirements.txt
├── README.md
```

---

## 🛠️ Requirements

> ⚠️ **Important:** This project works only with the versions below. Do not upgrade.

### Python
```
Python 3.9.x
```

### Libraries

```
tensorflow-intel==2.13.0
keras==2.13.1
numpy==1.23.5
scipy==1.10.1
pillow==10.0.0
pandas==2.0.3
matplotlib==3.7.5
scikit-learn==1.3.2
streamlit==1.29.0
ipykernel==6.29.0
```

---

## ⚙️ Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```bash
python train.py
```

After training, the model is saved as:

```
models/fish_classifier.keras
```

---

## 🔍 Prediction

### ✅ Single Image

```python
predict_path(
    "data/val/fish sea_food sea_bass/img1.jpg",
    mobilenet_model,
    train_data
)
```

### 📁 Folder Prediction

```python
predict_path(
    "data/val/fish sea_food sea_bass",
    mobilenet_model,
    train_data
)
```

The function automatically detects **file or folder** input.

---

## 🌐 Streamlit App

Run the web app:

```bash
streamlit run str.py
```

### App Features
- Upload single image
- Upload folder of images
- Displays predicted class with confidence

---

## ❗ Common Issues

### TensorFlow import error
- Ensure Python version is **3.9**
- Ensure NumPy is **1.23.5**

### SciPy error

```bash
pip install scipy==1.10.1
```

### Model file not found
- Ensure `models/fish_classifier.keras` exists
- Check model path in `str.py`

---

## ✅ Best Practices

- Do not mix `tensorflow.keras` and `keras`
- Use `keras.*` imports for TensorFlow ≥ 2.13
- Restart Jupyter / Streamlit after installing packages

---

## 👤 Author

**Fish Classifier Project**  
Built for learning and academic purposes.

---

## 📜 License

This project is intended for **educational use only**.

