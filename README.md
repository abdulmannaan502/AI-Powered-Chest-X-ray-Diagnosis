# AI-Powered Chest X-ray Diagnosis with ResNet-50 and Grad-CAM

This project builds an AI-based medical diagnosis web application that uses a **ResNet-50** convolutional neural network to classify chest X-ray images into **Normal** or **Pneumonia** categories. The predictions are visually explained using **Grad-CAM**, highlighting regions that contributed most to the model’s decision.

The system includes a complete pipeline — from training on Kaggle to local web deployment using Flask.

---

## 📌 Features

- ✅ Trained CNN model (ResNet-50) for pneumonia detection  
- ✅ Grad-CAM heatmap visualization for model explainability  
- ✅ Web-based image upload interface using Flask  
- ✅ Data preprocessing and augmentation  
- ✅ Performance evaluation using Accuracy, Precision, Recall, F1-score  
- ✅ Clean and modular codebase

---

## 🧠 Model Architecture

- Base Model: ResNet-50 (ImageNet pretrained backbone)  
- Classification Head: Fully connected layer for binary classification  
- Classes: Normal, Pneumonia  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam

---

## 🧪 Model Performance (Test Set)

| Metric    | Score (%) |
|------------|------------|
| Accuracy   | 84.46 |
| Precision  | 87.32 |
| Recall     | 82.94 |
| F1-Score   | 85.07 |

---

## 🚀 Local Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/abdulmannaan502/AI-Powered-Chest-X-ray-Diagnosis.git
cd AI-Powered-Chest-X-ray-Diagnosis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Trained Model

Download `model.pth` from Kaggle training output and place inside project:

```text
AI-Powered-Chest-X-ray-Diagnosis/
├── app.py
├── model.pth
├── requirements.txt
└── templates/
    └── index.html
```

### 5. Run Web App
```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

Upload any chest X‑ray image to see prediction and Grad‑CAM visualization.

---

## 📦 Dataset

Kaggle Chest X-Ray Pneumonia Dataset  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset structure:
train / val / test with NORMAL and PNEUMONIA folders.

---

## 🌐 Live Demo

Try the complete AI-powered Chest X-ray diagnosis system directly in your browser:

👉 **Hugging Face Space:**  
[huggingface](https://huggingface.co/spaces/abdulmannaan1/chest-xray-diagnosis)]

### Features in the Demo:
- Upload chest X-ray images directly
- Get real-time predictions:
  - **Normal**
  - **Pneumonia**
- View probability confidence scores
- See **Grad-CAM heatmap overlays** explaining where the CNN focused
- No installation required – runs completely online

> ⚠️ *This demo is intended for educational and demonstration purposes only and must not be used for medical diagnosis.*


## ⚠️ Disclaimer

This project is intended strictly for educational and demonstration purposes.  
It is **NOT** a certified medical diagnostic tool.

Always consult a medical professional for clinical diagnosis.
