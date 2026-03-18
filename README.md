# Facial Emotion Recognition System  
**ResNet50 · DDAMFN · ResEmoteNet (PyTorch, FastAPI, Docker)**

This project benchmarks multiple deep learning architectures on the **FER+ (FER2013Plus)** dataset and deploys the best-performing model as a **REST API for real-time emotion recognition**.

---

## Dataset

**FER2013Plus (FER+)**
- Source: https://www.kaggle.com/datasets/subhaditya/fer2013plus
- Grayscale facial images (48 × 48)
- 8 emotion classes

---

## Models Evaluated

- **ResNet50 (Pretrained)**  
- **DDAMFN (Dual Dynamic Attention Model with Feature Normalization)**  
- **ResEmoteNet**

All models are trained and evaluated using **PyTorch** under a unified preprocessing pipeline.

---

## Data Preprocessing

To ensure fair comparison, all models use identical preprocessing:

- Random horizontal flip  
- Color jitter (illumination variation)  
- Random rotation (±10°) and cropping (p=0.2)  
- Random erasing for regularization  
- Standard normalization (ImageNet mean/std)  

---

## Performance Results

| Model | Train Accuracy (%) | Test Accuracy (%) |
|------|-------------------:|------------------:|
| **ResNet50 (Pretrained)** | **97.41** | **83.77** |
| **DDAMFN** | 94.80 | 82.15 |
| **ResEmoteNet** | 94.99 | 80.63 |

> ResNet50 achieved the best generalization performance and was selected for deployment.

---

## Model Deployment

The best-performing model (ResNet50) is deployed as a **REST API** using **FastAPI** and containerized with **Docker**.

### API Endpoint

**POST `/predict`**

**Input:**  
- Image file (face)

**Output:**
```json
{
  "class_index": 7,
  "class_name": "Surprise",
  "confidence": 0.92
}
````

---

## 🚀 Running the Project

### 1. Download Pretrained Weights

Download the trained model weights:

👉 [https://drive.google.com/file/d/1tEcJ25IcZ3IVfUnYxrWTfPa9W_tsQAE5/view?usp=sharing](https://drive.google.com/file/d/1tEcJ25IcZ3IVfUnYxrWTfPa9W_tsQAE5/view?usp=sharing)

Place the file in:

```
weights/best_model.pth
```

---

### 2. Run with Docker

```bash
docker build -t fer-api .
docker run -p 8000:8000 fer-api
```

---

## 🚀 Test the API

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

Or send a request via command line:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/image.png"
```

---

## ⚙️ Training Details

* **Framework:** PyTorch

* **Loss Function:** Cross-Entropy Loss

* **Optimizer:** SGD with momentum

* **Learning Rate Scheduling:** Exponential / Step-based decay

* **Regularization:** Data augmentation (flip, rotation, cropping, color jitter, random erasing)

* **Early Stopping:** Based on validation accuracy

* **Batch Size:** Configurable

* **Image Size:** 224 × 224

* **Dataset:** FER2013Plus (FER+)

* **Evaluation:**

  * Training and validation accuracy/loss per epoch
  * Confusion matrix on test set

---

## 📊 Visualizations

The repository includes:

* Training and validation loss/accuracy curves across epochs
* Confusion matrices for model evaluation
* Model architecture diagrams for each network

Location:

```
figures/
```

---

## 📌 Summary

* Benchmarked multiple deep learning architectures on the FER+ dataset
* Achieved up to **83.77% test accuracy** using a pretrained ResNet50 model
* Designed a reproducible training and evaluation pipeline
* Deployed the best-performing model as a **containerized REST API** using FastAPI and Docker for real-time inference