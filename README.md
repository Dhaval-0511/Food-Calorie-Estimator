# 🍔 Food Calorie Estimator

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.3.3-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-blueviolet?style=for-the-badge"/>
</p>

<p align="center">
  <b>AI-powered web application that detects food from images and estimates calorie content using Deep Learning.</b>
</p>

---

## 🚀 Overview

The **Food Calorie Estimator** is an end-to-end Machine Learning application that enables users to upload a food image and instantly receive:

* 🍽️ Food classification
* 🔥 Calorie estimation per serving
* 📊 Confidence score

This project demonstrates a **complete ML pipeline**, from training a CNN model to deploying it via a web interface.

---

## 🎯 Problem Statement

Understanding calorie intake is essential for maintaining a healthy lifestyle, but manual estimation is often inaccurate and inconvenient.

This project solves that by:

* Automating food recognition
* Providing instant calorie estimates
* Making AI accessible through a simple web interface

---

## 📸 Screenshots

<p align="center">
  <img src="screenshots/home.png" width="700"/>
</p>

<p align="center"><b>Home Interface</b></p>

<p align="center">
  <img src="screenshots/pizza_result.png" width="700"/>
</p>

<p align="center"><b>Pizza Prediction — 285 kcal | 99.97% Confidence</b></p>

<p align="center">
  <img src="screenshots/french_fries_result.png" width="700"/>
</p>

<p align="center"><b>French Fries — 312 kcal | 59.87% Confidence</b></p>

---

## ✨ Features

* 📷 Drag & Drop Image Upload (JPG, PNG, WebP)
* 🧠 AI-powered Food Recognition (MobileNetV2)
* 🔥 Calorie Estimation using JSON mapping
* 📊 Confidence Score Visualization
* 🌐 Flask Web Backend (Jinja2 Templates)
* 🎨 Responsive UI with modern design
* 🐳 Docker Support for easy deployment

---

## 🍽️ Supported Food Classes

| Food             | Calories (per serving) |
| ---------------- | ---------------------- |
| 🍔 Hamburger     | 295 kcal               |
| 🍕 Pizza         | 285 kcal               |
| 🍟 French Fries  | 312 kcal               |
| 🥪 Club Sandwich | 250 kcal               |
| 🥗 Caesar Salad  | 150 kcal               |

---

## 🧠 Machine Learning Details

| Aspect                 | Detail                                        |
| ---------------------- | --------------------------------------------- |
| Technique              | Transfer Learning                             |
| Base Model             | MobileNetV2 (ImageNet pretrained, frozen)     |
| Dataset                | Food-101 (subset: 5 classes, 200 images each) |
| Input Size             | 224 × 224 × 3                                 |
| Architecture           | GAP → Dense(128) → Dense(5 Softmax)           |
| Loss                   | Categorical Crossentropy                      |
| Optimizer              | Adam                                          |
| Epochs                 | 5                                             |
| Train/Validation Split | 80% / 20%                                     |
| Trainable Parameters   | ~164K                                         |

---

## 🔄 How It Works (Inference Pipeline)

```
User uploads image
       ↓
Flask server receives request
       ↓
Image preprocessing (resize + normalize)
       ↓
MobileNetV2 extracts features
       ↓
Dense layer predicts class probabilities
       ↓
Highest probability → predicted food
       ↓
Calories fetched from JSON database
       ↓
Result displayed with confidence score
```

---

## 🗂️ Project Structure

```
ml-project/
├── app.py
├── train_model.py
├── calories.json
├── requirements.txt
├── Dockerfile
├── model/
│   └── food_model.h5
├── templates/
│   └── index.html
├── static/
│   └── uploads/
├── screenshots/
└── dataset/ (excluded)
```

---

## ⚙️ Tech Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| ML Framework     | TensorFlow / Keras    |
| CNN Model        | MobileNetV2           |
| Backend          | Flask                 |
| Frontend         | HTML, CSS, JavaScript |
| Image Processing | Pillow                |
| Deployment       | Docker                |
| Language         | Python                |

---

## 🚀 How to Run

### ▶️ Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

### 🐳 Run with Docker

```bash
docker build -t food-calorie-estimator .
docker run -p 5000:5000 food-calorie-estimator
```

---

## 📊 Key Highlights

* End-to-end ML system (training → deployment)
* Lightweight CNN optimized for performance
* Real-world applicable use case (health + AI)
* Clean UI with interactive experience
* Portable deployment using Docker

---

## 📌 Notes

* Dataset (Food-101) not included due to size
* Pre-trained model included for direct use
* Runs efficiently on CPU

---

## 🔮 Future Improvements

* Add more food categories
* Improve model accuracy with larger dataset
* Deploy on cloud (AWS / GCP)
* Mobile app integration

---

## 👤 Author

**Dhaval Prajapati**
🔗 GitHub: https://github.com/Dhaval-0511
🔗 LinkedIn: https://linkedin.com/in/dhaval-prajapati-a62401292

---

<p align="center">
⭐ Star this repo if you found it useful!
</p>
