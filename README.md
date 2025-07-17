# AQI Prediction Using Time Series Forecasting and MLflow Tracking

This project presents a full-stack machine learning pipeline to forecast **Air Quality Index (AQI)** using:
- Multi-step pollutant prediction via **Echo State Networks (ESNs)**
- Future AQI classification using **XGBoost**
- Experiment tracking and metric logging with **MLflow**

---

## 📌 Objective

- Predict AQI categories (`0–4`, mapped to Good → Very Poor) for **next 6 time steps**
- Use previously forecasted **pollutant values** to make these predictions
- Log and compare all results using **MLflow**

---
## 🔧 Pipeline Overview

### ✅ Step 1: Pollution Forecasting (ESN-based)
- Built individual **Echo State Network models** per pollutant (`pm2.5`, `pm10`, `co`, etc.)
- Trained models can be found inside "models" file
- Forecasted each pollutant **12 steps ahead** using delay-embedded temporal inputs
- Visualized each forecast vs ground truth

### ✅ Step 2: AQI Classification (XGBoost-based)
- Used predicted pollutant values as inputs
- Trained **multi-output XGBoost classifier** to predict AQI levels at `t+1` to `t+6`
- Labels categorized into AQI classes: `Good (0)` to `Very Poor (4)`

### ✅ Step 3: Experiment Tracking with MLflow
- Logged all training metrics (`f1_macro`, `precision`, `recall`, `accuracy`) for each step
- Saved classification reports and confusion matrices
- Auto-generated CSV summary of all MLflow runs

---

## 📁 Dataset

The dataset consists of pollution measurements for various Indian cities, with features:

date, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3

---

## 🧪 Future Work
- Address class imbalance using:
- SMOTE / RandomOverSampler
- Class-weighted loss

---

📦 MLflow Run Summary

- mlflow_summary_baseline.csv
- Includes metrics, params, tags for every model run.
---
