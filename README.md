# 🌱 Basic Crop Recommendation System

A beginner-friendly Machine Learning web application designed to recommend the most suitable crop for cultivation based on soil nutrients and meteorological parameters. Built with **Python**, **Flask**, **Scikit-learn**, **HTML5**, **CSS3**, and **JavaScript**, specifically tailored for **BE AIML / CSE** academic projects and live laboratory demonstrations.

---

## 📌 Project Overview

* **Title:** Basic Crop Recommendation System
* **Domain:** Artificial Intelligence & Machine Learning / Smart Agriculture
* **Objective:** Assist farmers and agricultural students by classifying optimal crop varieties based on quantitative soil chemistry and climate conditions.
* **Core Algorithm:** Random Forest Classification (Ensemble Learning)

---

## 🗂️ Project Directory Structure

```text
shreyas yadav/
│
├── data/
│   └── crop_data.csv        # Agronomic dataset (2,200 records across 22 crops)
│
├── model/
│   └── crop_model.pkl       # Serialized trained Random Forest model
│
├── templates/
│   └── index.html           # Responsive frontend web template
│
├── static/
│   ├── style.css            # Agricultural green & white styling
│   └── script.js            # Client-side input validation & AJAX handler
│
├── app.py                   # Flask web server and prediction REST API
├── train_model.py           # Model training, evaluation & serialization script
├── generate_data.py         # Dataset generator script
├── test_app.py              # Automated test verification suite
├── requirements.txt         # Project dependencies
└── README.md                # Comprehensive documentation & Viva guide
```

---

## 📊 Input Features (7-Dimensional Vector)

| # | Feature Name | Unit / Scale | Agronomic Range | Description |
|---|--------------|--------------|-----------------|-------------|
| 1 | **Nitrogen (N)** | kg/ha | 0 – 140 | Ratio of Nitrogen content in soil |
| 2 | **Phosphorus (P)** | kg/ha | 5 – 145 | Ratio of Phosphorus content in soil |
| 3 | **Potassium (K)** | kg/ha | 5 – 205 | Ratio of Potassium content in soil |
| 4 | **Temperature** | °C | 5 – 50 | Average ambient temperature |
| 5 | **Humidity** | % | 10 – 100 | Relative atmospheric humidity |
| 6 | **Soil pH** | 0 – 14 | 3.5 – 10.0 | Acidity/Alkalinity level of the soil |
| 7 | **Rainfall** | mm | 15 – 350 | Cumulative seasonal rainfall |

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
Open your terminal or command prompt in the project root directory and execute:
```bash
pip install -r requirements.txt
```
*(On Windows systems with the Python launcher, you may also use: `py -3.11 -m pip install -r requirements.txt`)*

### Step 2: Train the Machine Learning Model
Train the Random Forest Classifier on the agronomic dataset and save the serialized model artifact:
```bash
python train_model.py
```
*(or `py -3.11 train_model.py`)*

You will see output detailing the dataset dimensions, training/testing split, validation accuracy, classification report, and a sample sanity prediction.

### Step 3: Run the Web Application
Start the Flask development server:
```bash
python app.py
```
*(or `py -3.11 app.py`)*

### Step 4: Open in Web Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Running Automated Tests
To run the automated endpoint and validation test suite:
```bash
python test_app.py
```

---

## 🎓 BE AIML Presentation & Viva Reference Guide

### 1. Problem Statement
Traditional crop selection relies predominantly on historical tradition, intuition, or regional peer choices without scientific validation of soil nutrient status and climate trends. This frequently results in suboptimal crop-soil matching, excessive fertilizer expenditure, soil degradation, and harvest losses.

### 2. Proposed Solution
An intelligent, data-driven web advisory system that takes seven real-time agricultural indicators (soil NPK levels and local weather parameters) and predicts the most biologically compatible crop using an ensemble machine learning classifier.

### 3. Project Objectives
1. Implement a multidimensional classification pipeline for agricultural decision support.
2. Build an intuitive, mobile-friendly interface for rural and academic users.
3. Provide instantaneous feedback with cultivation guidelines (growing season, water demand, soil type).
4. Perform robust server-side and client-side input validation to reject anomalous readings.

### 4. Methodology
1. **Data Collection & Preparation:** Synthesized and structured a dataset of 2,200 observations representing 22 major crops (cereals, pulses, fruits, commercial cash crops).
2. **Feature Engineering:** Mapped 7 continuous agronomic variables: $X = [N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}]$.
3. **Data Partitioning:** 80% training set and 20% hold-out test set using stratified sampling to preserve class balance.
4. **Model Selection:** Trained a **Random Forest Classifier** with 100 decision trees (`n_estimators=100`, `max_depth=15`).
5. **Model Serialization:** Pickled the trained model into `crop_model.pkl`.
6. **Backend Service:** Implemented RESTful endpoints using Flask (`/predict` and `/health`).
7. **Frontend Engineering:** Vanilla HTML5/CSS3/JavaScript utilizing asynchronous `fetch()` API calls to prevent screen reloads.

### 5. Why Random Forest Classifier?
* **Ensemble Learning:** Constructs an ensemble of diverse decision trees using bootstrap aggregating (bagging) and random feature subspace selection.
* **Resilience to Overfitting:** Individual decision trees can easily overfit noisy agricultural data; averaging predictions across the forest significantly reduces variance.
* **Non-Linear Relationships:** Soil nutrients and climate interact non-linearly (e.g., high temperature combined with high rainfall creates a tropical humid microclimate distinct from high temperature with low humidity).
* **High Interpretability & Stability:** Provides robust multiclass predictions across 22 classes without requiring complex neural architecture or GPU acceleration.

### 6. Key Advantages
* **Lightweight & Fast:** Sub-millisecond inference time on commodity hardware.
* **Zero Configuration for Users:** One-click sample autofill presets for live demonstration.
* **Comprehensive Validation:** Catches out-of-range pH, negative nutrient values, or implausible weather figures before running inference.
* **Accessible Clean UI:** Green and white agricultural theme adhering to responsive card-based layout principles.

### 7. Limitations
* **Sample Dataset Scope:** Trained on benchmark agronomic distributions; does not encompass hyper-localized micro-varieties.
* **Static Snapshot:** Evaluates instantaneous seasonal averages rather than day-to-day weather forecasts.
* **Excludes Market Economics:** Recommends biological suitability rather than market wholesale price (MSP/Mandi rates).

### 8. Future Enhancements
* **IoT Sensor Integration:** Real-time ESP32/Arduino integration to ingest NPK and soil moisture readings directly from field probes.
* **Live Weather API Integration:** Automatic temperature, rainfall, and humidity fetching via OpenWeatherMap API using GPS geolocation.
* **Multilingual Voice Interface:** Vernacular language support (Hindi, Kannada, Marathi, Tamil, Telugu, etc.) with voice input for local farming communities.
* **Fertilizer Recommendation Module:** Suggesting specific chemical/organic fertilizer dosages (Urea, DAP, MOP) to rectify deficient soil nutrients.

---

## ⚠️ Educational Disclaimer
> **Notice:** This application and its machine learning model are developed strictly for educational, demonstration, and academic project evaluation purposes. Crop recommendations represent statistical estimates derived from a sample dataset. Actual agricultural viability is subject to local soil lab test certificates, verified seed varieties, irrigation infrastructure, pest pressures, and official government agricultural advisories.
