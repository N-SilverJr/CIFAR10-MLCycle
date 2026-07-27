# CIFAR-10 Image Classification – End-to-End ML Cycle

A complete end-to-end machine learning pipeline for classifying **32×32 RGB images** into the **10 CIFAR-10 classes**, covering the entire ML lifecycle from data preparation to deployment.

**Classes**

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

The project demonstrates:

- Data acquisition and preprocessing
- Custom CNN development and evaluation
- Flask REST API
- Interactive web dashboard
- Safety-gated model retraining
- Docker + Nginx deployment
- Load testing with Locust
- Cloud deployment using ONNX Runtime

---

# Demo

| Resource | Link |
|-----------|------|
| 🎥 Video Demo | **YOUR_YOUTUBE_LINK_HERE** |
| 🌐 Live Application | https://cifar10-mlcycle.onrender.com |
| 📂 GitHub Repository | https://github.com/N-SilverJr/CIFAR10-MLCycle |

> **Note:** The free cloud deployment may require **30–60 seconds** to wake up after periods of inactivity.

> **Note:** Retraining is available only in the local/Docker deployment. The cloud version is inference-only using ONNX Runtime.

---

# Project Overview

This project implements a production-style machine learning workflow rather than just a trained model.

It includes:

- Custom CNN designed specifically for CIFAR-10
- Model training and evaluation
- REST API for inference
- Browser-based dashboard
- Upload-based retraining
- Safety gate preventing degraded models from replacing the production model
- Cloud deployment
- Performance testing
- Monitoring visualizations

---

# Model Architecture

The selected model is a custom Convolutional Neural Network containing approximately **1.2 million parameters**.

Architecture:

- Conv → BatchNorm
- Conv → BatchNorm
- MaxPooling
- Dropout

(repeated three times)

Followed by:

- Global Average Pooling
- Dense output layer

The network is optimized using:

- Batch Normalization
- Dropout
- EarlyStopping
- ReduceLROnPlateau
- Optional Keras data augmentation layers

Unlike transfer learning approaches, the network is built specifically for **32×32 CIFAR-10 images**.

---

# Safety-Gated Retraining

The application supports controlled retraining using newly uploaded labeled images.

Workflow:

1. Upload labeled images
2. Fine-tune the existing model
3. Evaluate on the held-out test set
4. Compare against the baseline model
5. Promote only if performance remains within the allowed tolerance

If the retrained model performs worse than the baseline, the deployment is rejected automatically and the previous production model remains active.

---

# Closed-Set Inference & Out-of-Distribution Handling

The classifier is a **closed-set classifier**, meaning it always predicts one of the ten CIFAR-10 classes.

To improve reliability, the API evaluates prediction confidence.

The application detects:

- Low confidence predictions
- Small decision margins
- Possible out-of-scope images

If confidence is high on an unrelated image, the application explains that this is normal neural-network overconfidence and that confidence only reflects certainty **among the ten known classes**.

---

# Runtime Options

The project supports two deployment modes.

### Local / Docker

- TensorFlow
- Full retraining support
- Full dashboard functionality

### Cloud Deployment

- ONNX Runtime
- Lightweight inference
- Optimized for free-tier hosting
- Inference only

---

# Model Performance

| Experiment | Architecture | Test Accuracy | Macro F1 |
|------------|--------------|--------------:|----------:|
| Baseline CNN | Conv-Pool ×2 + Dense | ~71% | ~0.71 |
| **Enhanced CNN (Selected)** | Conv-BN Blocks + GAP + Dropout | **~85.7%** | **~0.85** |
| Enhanced + Augmentation | Same + RandomFlip / Translation / Rotation | ~78.7% | ~0.78 |

The production model is selected based on the highest **Macro F1 Score**.

Saved model:

```
models/cifar10_classifier.h5
```

Cloud deployment uses:

```
models/cifar10_classifier.onnx
```

Evaluation metrics are stored in:

```
models/eval_metrics.json
```

Metrics include:

- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix
- Per-Class F1

---

# Load Testing

The Flask API was stress tested using **Locust**.

Configuration:

- 10 concurrent users
- Spawn rate: 2 users/second
- Duration: 30 seconds
- Synthetic 32×32 RGB images

Results:

| Setup | Requests | Median Latency | Average Latency | Throughput | Max Latency | Failures |
|--------|---------:|---------------:|----------------:|------------:|------------:|----------:|
| Local (1 Process) | 246 | **85 ms** | 148 ms | ~8.5 req/s | 1636 ms | **0** |

Scaling is supported using Docker Compose and Nginx.

```bash
docker compose up -d --scale app=1
docker compose up -d --scale app=2
docker compose up -d --scale app=4
```

Example Locust command:

```bash
locust \
-f locust/locustfile.py \
--host=http://localhost:8080 \
--users 50 \
--spawn-rate 10 \
--run-time 60s \
--headless \
--csv=locust/results/scale_N
```

Generated CSV reports are saved inside:

```
locust/results/
```

---

# Repository Structure

```text
CIFAR10-MLCycle/
├── README.md
├── app.py                 # Full Flask API (TensorFlow)
├── app_render.py          # Lightweight ONNX API
├── requirements.txt
├── requirements-render.txt
├── Dockerfile
├── Dockerfile.render
├── docker-compose.yml
├── nginx.conf
├── notebook/
│   └── cifar10_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── prediction_onnx.py
├── ui/
│   ├── index.html
│   └── script.js
├── data/
│   ├── train/
│   ├── test/
│   └── retrain/
├── models/
│   ├── cifar10_classifier.h5
│   ├── cifar10_classifier.onnx
│   └── eval_metrics.json
└── locust/
    ├── locustfile.py
    └── results/
```

---

# Quick Start

## 1. Create a Virtual Environment

```powershell
cd CIFAR10-MLCycle

python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Prepare Data & Model

Train the model inside the notebook (or Google Colab), then place the generated files into the project.

Required model files:

```
models/cifar10_classifier.h5
models/eval_metrics.json
```

Required datasets:

```
data/train/*.npy
data/test/*.npy
```

---

## 3. Run the Application

```powershell
python app.py
```

Open:

```
http://127.0.0.1:5000
```

Available dashboard pages:

| Page | Description |
|------|-------------|
| Overview | Health, uptime, quick prediction |
| Inference | Single image classification |
| Analytics | Class distribution, F1 scores, confusion matrix |
| Retrain | Upload images and retrain |
| Activity | Retraining audit history |

---

## 4. Docker Deployment

Build

```bash
docker compose build
```

Run

```bash
docker compose up -d --scale app=1
```

Application:

```
http://localhost:8080
```

Stop

```bash
docker compose down
```

---

## 5. Cloud Deployment

The cloud version uses:

- Dockerfile.render
- app_render.py
- models/cifar10_classifier.onnx

Example deployment:

```
https://cifar10-mlcycle.onrender.com
```

---

# API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health status, uptime and model information |
| POST | `/predict` | Predict image class |
| POST | `/upload` | Upload labeled images or ZIP archive |
| POST | `/retrain` | Fine-tune model using uploaded images |
| GET | `/retrain-history` | View retraining history |
| GET | `/visualizations` | Retrieve dashboard metrics |

---

# Prediction Response

Important response fields include:

| Field | Description |
|-------|-------------|
| `predicted_class` | Predicted CIFAR-10 class |
| `confidence` | Top-1 prediction probability |
| `decision_margin` | Difference between top-1 and top-2 probabilities |
| `in_domain` | Indicates whether prediction appears in-domain |
| `warning` | Confidence or ambiguity warning |
| `note` | Closed-set explanation for high-confidence predictions |

---

# Handling Images Outside CIFAR-10

Since the model is a closed-set classifier, it cannot return an "unknown" class.

Instead, it behaves as follows:

| Situation | Behavior |
|-----------|----------|
| Confidence < 0.45 | Prediction flagged as likely out-of-scope |
| Small decision margin | Prediction marked as ambiguous |
| High confidence on unrelated image | User informed that confidence only reflects certainty among the ten known classes |

---

# Edge Cases

| Scenario | Protection |
|----------|------------|
| Missing image | Client and server validation |
| Invalid file | Only PNG, JPG, JPEG, BMP and ZIP accepted |
| Retraining without uploads | UI blocks request and API returns HTTP 400 |
| Retraining decreases accuracy | Safety gate rejects the updated model |

---

# Notebook Highlights

The training notebook demonstrates:

- Loading the CIFAR-10 dataset
- Dataset exploration
- Class balance visualization
- Mean image visualization
- Data normalization
- One-hot encoding
- Stratified train/validation split
- Three CNN experiments
- EarlyStopping
- ReduceLROnPlateau
- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix
- Saving the trained model
- Saving evaluation metrics
- Safety-gated retraining demonstration

---

# Technologies Used

### Machine Learning

- TensorFlow
- Keras
- ONNX Runtime
- NumPy
- scikit-learn
- Pillow

### Backend

- Flask
- Flask-CORS
- Gunicorn

### Frontend

- HTML
- JavaScript
- Tailwind CSS
- Chart.js
- IBM Plex

### Deployment

- Docker
- Docker Compose
- Nginx

### Testing

- Locust

---

# Summary

This project demonstrates a complete production-oriented machine learning workflow using the CIFAR-10 dataset. It combines model development, evaluation, deployment, retraining, monitoring, cloud inference, and performance testing into a single end-to-end application while incorporating practical safeguards such as confidence-based warnings and safety-gated retraining.